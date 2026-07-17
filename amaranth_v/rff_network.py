"""Full RFF-INR network in Amaranth, mirroring the qkeras model in
``qkeras_v.qkeras_model`` and built from a training pickle (the pattern is
lifted from cdcc's ``QbNetwork``).

Datapath (one scalar phase + ``in_d-1`` embedding channels per sample)::

    i.payload[0]        -> RFF LUT -> [cos_0..cos_{K-1}, sin_0..sin_{K-1}]  (io)
    i.payload[1:in_d]   -> latched embed registers                          (io)
                              |
        concat(rff, embed)  ==>  mlp0 (io -> NNQ, relu)
                                   -> mlp1 (NNQ -> NNQ, relu)
                                   -> ...
                                   -> y_pred (NNQ -> io, no relu) -> o

Weights come from the qkeras pickle: ``weights[name]["weights"] == (kernel, bias)``
for each dense layer (``mlp0 .. mlp{N-1}``, ``y_pred``) plus an ``rff`` entry
(``{"B", "b_bits", "b_integer", "io_bits", "io_integer"}``) used to build the
LUT feature layer.  Because the kernels/biases/tables are the very integers the
numpy golden model uses, the hardware stays bit-exact with it.
"""

import pickle

import numpy as np
from amaranth import Module, Signal
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from amaranth_future import fixed

from . import NNQ
from .dense_layer import QDenseLayer
from .rff import RandomFourierFeaturesLUT


class RffNetwork(wiring.Component):

    @staticmethod
    def build(weights_pkl: str, **kwargs):
        with open(weights_pkl, "rb") as f:
            data = pickle.load(f)
        return RffNetwork(data, **kwargs)

    def __init__(
        self,
        qkeras_weights: dict,
        lut_size: int = 1024,
        relu_upper_bound: float = 8.0,
    ):
        """
        Args:
            qkeras_weights    dict from the qkeras pickle (dense layers + "rff")
            lut_size          cos/sin ROM depth for the RFF layer
            relu_upper_bound  upper bound of the MLP relu activations
        """
        self.qkeras_weights = qkeras_weights
        self.lut_size = int(lut_size)
        self.relu_upper_bound = float(relu_upper_bound)

        # dense layers in order: every "mlp{idx}" then the "y_pred" regressor.
        self.mlp_names = sorted(
            (k for k in qkeras_weights if k.startswith("mlp")),
            key=lambda k: int(k[len("mlp") :]),
        )
        assert self.mlp_names, "expected at least one mlp* layer in weights"
        assert "y_pred" in qkeras_weights, "expected a y_pred layer in weights"
        assert "rff" in qkeras_weights, "expected an rff entry in weights"

        # io fixed-point shape from the rff entry (qkeras quantized_bits: 1 sign
        # bit + `integer` int bits + the rest fractional).
        rff = qkeras_weights["rff"]
        io_bits, io_integer = int(rff["io_bits"]), int(rff["io_integer"])
        self.io_shape = fixed.SQ(io_integer + 1, io_bits - io_integer - 1)

        # number of fourier features (B is (in_dim=1, num_features)).
        self.num_features = int(np.asarray(rff["B"]).reshape(-1).shape[0])

        # first mlp consumes 2*num_features rff outputs + the embedding channels.
        w0, _b0 = self.dense_weights_biases_for(self.mlp_names[0])
        self.MLP_WIDTH = int(w0.shape[1])
        self.EMBED_D = int(w0.shape[0]) - 2 * self.num_features
        assert (
            self.EMBED_D >= 0
        ), f"mlp0 in_d={w0.shape[0]} < 2*num_features={2 * self.num_features}"
        self.IN_D = self.EMBED_D + 1  # + scalar phase

        wy, _by = self.dense_weights_biases_for("y_pred")
        self.OUT_D = int(wy.shape[1])

        print(
            f">RffNetwork in_d={self.IN_D} embed_d={self.EMBED_D}"
            f" num_features={self.num_features} mlp_layers={len(self.mlp_names)}"
            f" mlp_width={self.MLP_WIDTH} out_d={self.OUT_D}"
            f" io_shape={self.io_shape!r} lut_size={self.lut_size}"
        )

        super().__init__(
            {
                "i": In(stream.Signature(data.ArrayLayout(self.io_shape, self.IN_D))),
                "o": Out(stream.Signature(data.ArrayLayout(self.io_shape, self.OUT_D))),
            }
        )

    def dense_weights_biases_for(self, name: str):
        w, b = self.qkeras_weights[name]["weights"]
        return np.asarray(w), np.asarray(b)

    def elaborate(self, platform):
        m = Module()

        # ---- feature layer -------------------------------------------------
        rff = RandomFourierFeaturesLUT.from_rff(
            self.qkeras_weights["rff"], lut_size=self.lut_size
        )
        m.submodules["rff"] = rff
        num_rff = 2 * self.num_features

        # ---- dense layers --------------------------------------------------
        mlps = []
        for idx, name in enumerate(self.mlp_names):
            w, b = self.dense_weights_biases_for(name)
            mlp = QDenseLayer(
                w,
                b,
                apply_relu=True,
                relu_upper_bound=self.relu_upper_bound,
                in_shape=self.io_shape if idx == 0 else NNQ,
                out_shape=NNQ,
            )
            m.submodules[name] = mlp
            mlps.append(mlp)

        wy, by = self.dense_weights_biases_for("y_pred")
        y_pred = QDenseLayer(
            wy,
            by,
            apply_relu=False,
            in_shape=NNQ,
            out_shape=self.io_shape,
        )
        m.submodules["y_pred"] = y_pred

        # ---- input split: phase -> rff, embed -> latched registers ---------
        # the scalar phase drives the rff stream; the embedding channels are
        # captured on the same input handshake and held until the rff features
        # are ready to be concatenated with them.
        embed_reg = [
            Signal(self.io_shape, name=f"embed_reg_{j}") for j in range(self.EMBED_D)
        ]

        m.d.comb += [
            rff.i.payload.eq(self.i.payload[0].as_value()),
            rff.i.valid.eq(self.i.valid),
            self.i.ready.eq(rff.i.ready),
        ]
        with m.If(self.i.valid & self.i.ready):
            for j in range(self.EMBED_D):
                m.d.sync += embed_reg[j].as_value().eq(self.i.payload[1 + j].as_value())

        # ---- concat(rff, embed) -> mlp0 ------------------------------------
        mlp0 = mlps[0]
        m.d.comb += [
            mlp0.i.valid.eq(rff.o.valid),
            rff.o.ready.eq(mlp0.i.ready),
        ]
        for k in range(num_rff):
            m.d.comb += mlp0.i.payload[k].as_value().eq(rff.o.payload[k])
        for j in range(self.EMBED_D):
            m.d.comb += (
                mlp0.i.payload[num_rff + j].as_value().eq(embed_reg[j].as_value())
            )

        # ---- mlp chain -> y_pred -> output ---------------------------------
        for a, b in zip(mlps, mlps[1:]):
            wiring.connect(m, a.o, b.i)
        wiring.connect(m, mlps[-1].o, y_pred.i)
        wiring.connect(m, y_pred.o, wiring.flipped(self.o))

        return m
