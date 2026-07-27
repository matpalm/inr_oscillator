"""Full RFF-INR network in Amaranth, mirroring the FiLM qkeras model in
qkeras_v.qkeras_model and built from a training pickle. first layer, and
possibly more, use film modulation.


    i.payload[0]        -> RFF LUT -> [cos_0..cos_{K-1}, sin_0..sin_{K-1}]  (io)
    i.payload[1:in_d]   -> embed ---> film{i}_gamma / film{i}_beta (io -> NNQ)
                              |                    |
        rff ==> mlp0 (io -> NNQ, no relu) -> FiLMCombine0 [(1+g)*h + b, relu]
                  -> mlp1 (NNQ -> NNQ) -> FiLMCombine1
                  -> ...
                  -> y_pred (NNQ -> io, no relu) -> o

The mlp layers apply NO relu; the relu lives in the FiLMCombine tail after the
affine, matching the qkeras QDense -> FiLM -> quantized_relu order.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from amaranth import Module
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from amaranth_future import fixed

from . import NNQ
from .dense_layer import QDenseLayer, allocate_mlp_lanes
from .film import FiLMCombine
from .phase_h_lut_ps import PhaseHLutPS
from .rff import RandomFourierFeaturesLUT


def load_weights_and_config(weights_pkl):
    """Load the qkeras weights pickle and the model_config.json"""
    weights_pkl = Path(weights_pkl)
    with open(weights_pkl, "rb") as f:
        weights = pickle.load(f)
    model_config_path = weights_pkl.parents[2] / "model_config.json"
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    quant_sizes_path = weights_pkl.parents[2] / "quant_sizes.json"
    with open(quant_sizes_path, "r") as f:
        quant_sizes = json.load(f)
    return weights, quant_sizes, model_config


class RffNetwork(wiring.Component):

    @staticmethod
    def build(weights_pkl: str, **kwargs):
        return RffNetwork(*load_weights_and_config(weights_pkl), **kwargs)

    def __init__(
        self,
        qkeras_weights: dict,
        quant_sizes: dict,
        model_config: dict,
        mlp_lane_budget: int = 20,
        index_bits: int = 13,
        psram_base: int = 0,
        psram_addr_width: int = 22,
    ):
        """
        Args:
            qkeras_weights    dict from the qkeras pickle (dense layers + "rff").
            quant_sizes       for other config as required
            model_config      for other config as required
            mlp_lane_budget   total parallel MAC lanes (=DSP multipliers) to
                              distribute across the main-path mlp layers. The
                              non-mlp multipliers (rff, film gamma/beta, y_pred,
                              codec cal) sit outside this budget; keep the total
                              under the 28 MULT18X18D on the ECP5.
            index_bits        number of phase bits enumerated by the PSRAM-backed
                              phase->h table (default 13 => 8192 entries). mlp0's
                              pre-activation h depends only on phase, so it is
                              materialised once at startup into PSRAM.
            psram_base        byte offset of the phase->h table in PSRAM.
            psram_addr_width  external (32-bit) PSRAM wishbone address width.
        """

        self.qkeras_weights = qkeras_weights
        self.quant_sizes = quant_sizes
        self.mlp_lane_budget = mlp_lane_budget
        self.index_bits = index_bits

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
        rff_w = qkeras_weights["rff"]
        io_bits, io_integer = quant_sizes["io_bits"], quant_sizes["io_int"]
        self.io_shape = fixed.SQ(io_integer + 1, io_bits - io_integer - 1)
        self.lut_size = model_config["rff"]["lut_size"]

        # number of fourier features (B is (in_dim=1, num_features)).
        self.num_features = int(np.asarray(rff_w["B"]).reshape(-1).shape[0])
        assert self.num_features == model_config["rff"]["num_features"]

        # first mlp consumes ONLY the 2*num_features rff outputs (FiLM does NOT
        # concat the embedding into the main path).
        w0, _b0 = self.dense_weights_biases_for(self.mlp_names[0])
        self.mlp_dim = int(w0.shape[1])
        assert w0.shape[0] == 2 * self.num_features, (
            f"film mlp0 in_d={w0.shape[0]} != 2*num_features={2 * self.num_features}"
            " (expected no embedding concat in the film topology)"
        )

        self.mlp_idxs = [int(k[len("mlp") :]) for k in self.mlp_names]
        self.film_layer_idxs = [
            idx for idx in self.mlp_idxs if f"film{idx}_gamma" in qkeras_weights
        ]
        for idx in self.film_layer_idxs:
            for suffix in ("gamma", "beta"):
                name = f"film{idx}_{suffix}"
                assert name in qkeras_weights, f"expected a {name} layer in weights"
            # w_gamma, _bias = self.dense_weights_biases_for(f"film{idx}_gamma")
            # w_mlp, _bias = self.dense_weights_biases_for(f"mlp{idx}")
            # assert int(w_gamma.shape[1]) == int(
            #     w_mlp.shape[1]
            # ), f"film{idx} gamma out={w_gamma.shape[1]} != mlp{idx} out={w_mlp.shape[1]}"
        w_g0, _b_g0 = self.dense_weights_biases_for(
            f"film{self.film_layer_idxs[0]}_gamma"
        )
        self.embed_dim = int(w_g0.shape[0])
        self.in_d = self.embed_dim + 1  # + scalar phase

        w_y_pred, _bias = self.dense_weights_biases_for("y_pred")
        self.out_d = int(w_y_pred.shape[1])

        # extract mapping from layer id to relu bound o_O
        self.relu_upper_bound = model_config["relu_upper_bound"]

        print(
            f">RffNetwork(film) in_d={self.in_d} embed_dim={self.embed_dim}"
            f" num_features={self.num_features} mlp_layers={len(self.mlp_names)}"
            f" film_layers={self.film_layer_idxs}"
            f" mlp_dim={self.mlp_dim} out_d={self.out_d}"
            f" io_shape={self.io_shape!r} lut_size={self.lut_size}"
        )

        # sanity check the film conditioning strength per embed dim (analogous to
        # the concat version's embed->mlp0 weight magnitudes). a near-zero row
        # means that embed channel barely modulates the network and we're probably
        # seeing the same collapse
        film0 = self.film_layer_idxs[0]
        for suffix in ("gamma", "beta"):
            w_film, _bias = self.dense_weights_biases_for(f"film{film0}_{suffix}")
            for j in range(self.embed_dim):
                row = w_film[j]
                print(
                    f" collapse check; film{film0}_{suffix} embed{j} (net in{j + 1})"
                    f" |w|_mean={float(np.abs(row).mean()):.5f}"
                    f" max={float(np.abs(row).max()):.5f}"
                )

        # main-path MAC-lane allocation across every mlp layer (ranked by cycle
        # cost). mlp0 is materialised into the PSRAM phase->h table so its lanes
        # are only exercised during the startup build, but keep it in the
        # allocation so the fabric DSP footprint is unchanged.
        main_dims = []
        for name in self.mlp_names:
            w, _b = self.dense_weights_biases_for(name)
            main_dims.append((int(w.shape[0]), int(w.shape[1])))
        self.mlp_lanes = allocate_mlp_lanes(main_dims, self.mlp_lane_budget)

        # PSRAM-backed phase->h table: owns rff + mlp0. h = mlp0(RFF(phase))
        # depends only on phase, so the whole table is built once at startup.
        rff = RandomFourierFeaturesLUT.from_rff(
            self.qkeras_weights["rff"]["B"],
            quant_sizes=self.quant_sizes,
            lut_size=self.lut_size,
        )
        w0, b0 = self.dense_weights_biases_for(self.mlp_names[0])
        self.phlut = PhaseHLutPS(
            rff,
            w0,
            b0,
            io_shape=self.io_shape,
            index_bits=self.index_bits,
            addr_width_o=psram_addr_width,
            base=psram_base,
        )
        self.bus_signature = self.phlut.bus_signature

        super().__init__(
            {
                "i": In(stream.Signature(data.ArrayLayout(self.io_shape, self.in_d))),
                "o": Out(stream.Signature(data.ArrayLayout(self.io_shape, self.out_d))),
                "bus_h": Out(self.phlut.bus_signature),
                "ready": Out(1),
            }
        )

    def dense_weights_biases_for(self, name: str):
        w, b = self.qkeras_weights[name]["weights"]
        return np.asarray(w), np.asarray(b)

    def elaborate(self, platform):
        m = Module()

        # PSRAM-backed phase->h table (owns rff + mlp0), replacing the rff->mlp0
        # sub-chain. Exposes a 32-bit wishbone master (bus_h) to PSRAM and a
        # `ready` flag that is high once the startup build has filled the table.
        phlut = self.phlut
        m.submodules["phlut"] = phlut
        wiring.connect(m, phlut.bus, wiring.flipped(self.bus_h))
        m.d.comb += self.ready.eq(phlut.ready)

        film_set = set(self.film_layer_idxs)

        # main-path dense layers mlp1.. (mlp0 lives inside phlut). Lanes were
        # allocated across every mlp layer in __init__; reuse the same slices
        # so the fabric DSP footprint is unchanged.
        print(
            f">RffNetwork(film) mlp lane allocation "
            f"{list(zip(self.mlp_names, self.mlp_lanes))}"
        )
        mlps = {}
        for pos, (idx, name) in enumerate(zip(self.mlp_idxs, self.mlp_names)):
            if pos == 0:
                continue  # mlp0 is inside phlut
            w_mlp, b_mlp = self.dense_weights_biases_for(name)
            filmed = idx in film_set
            mlp = QDenseLayer(
                w_mlp,
                b_mlp,
                apply_relu=not filmed,
                relu_upper_bound=None if filmed else self.relu_upper_bound,
                in_shape=NNQ,
                out_shape=NNQ,
                n_lanes=self.mlp_lanes[pos],
            )
            m.submodules[name] = mlp
            mlps[pos] = mlp

        # per-layer FiLM generators (embed -> gamma/beta) + combine
        gammas, betas, combines = {}, {}, {}
        for idx in self.film_layer_idxs:
            gamma_name = f"film{idx}_gamma"
            beta_name = f"film{idx}_beta"
            w_gamma, b_gamma = self.dense_weights_biases_for(gamma_name)
            w_beta, b_beta = self.dense_weights_biases_for(beta_name)
            dim = int(w_gamma.shape[1])
            gamma = QDenseLayer(
                w_gamma,
                b_gamma,
                apply_relu=False,
                in_shape=self.io_shape,
                out_shape=NNQ,
            )
            beta = QDenseLayer(
                w_beta, b_beta, apply_relu=False, in_shape=self.io_shape, out_shape=NNQ
            )
            combine = FiLMCombine(dim, relu_upper_bound=self.relu_upper_bound)
            m.submodules[gamma_name] = gamma
            m.submodules[beta_name] = beta
            m.submodules[f"film{idx}_combine"] = combine
            gammas[idx] = gamma
            betas[idx] = beta
            combines[idx] = combine

        w_y_pred, b_y_pred = self.dense_weights_biases_for("y_pred")
        y_pred = QDenseLayer(
            w_y_pred,
            b_y_pred,
            apply_relu=False,
            in_shape=NNQ,
            out_shape=self.io_shape,
        )
        m.submodules["y_pred"] = y_pred

        # input fork: phase -> phlut, embed -> every film gamma/beta.
        # the scalar phase drives the phase->h table; the embedding channels
        # drive the film generators. join their handshakes so a sample is only
        # accepted when EVERY consumer is ready (phlut.i.ready stays low until
        # the startup build has finished, stalling inference until then).
        embed_consumers = list(gammas.values()) + list(betas.values())
        all_ready = phlut.i.ready
        for c in embed_consumers:
            all_ready = all_ready & c.i.ready
        in_valid = self.i.valid & all_ready

        m.d.comb += [
            self.i.ready.eq(all_ready),
            phlut.i.payload.eq(self.i.payload[0].as_value()),
            phlut.i.valid.eq(in_valid),
        ]
        for c in embed_consumers:
            m.d.comb += c.i.valid.eq(in_valid)
            for j in range(self.embed_dim):
                m.d.comb += (
                    c.i.payload[j].as_value().eq(self.i.payload[1 + j].as_value())
                )

        # chain everything. a FiLM layer's output is combineI.o; a plain
        # layer's output is mlp.o. mlp0 lives inside phlut, so position 0's h
        # comes from phlut.o straight into the film0 combine.
        prev_out = None
        for pos, idx in enumerate(self.mlp_idxs):
            if pos == 0:
                assert idx in film_set, "film topology expects mlp0 to be filmed"
                wiring.connect(m, phlut.o, combines[idx].i_h)
                wiring.connect(m, gammas[idx].o, combines[idx].i_gamma)
                wiring.connect(m, betas[idx].o, combines[idx].i_beta)
                prev_out = combines[idx].o
                continue
            mlp = mlps[pos]
            wiring.connect(m, prev_out, mlp.i)
            if idx in film_set:
                wiring.connect(m, mlp.o, combines[idx].i_h)
                wiring.connect(m, gammas[idx].o, combines[idx].i_gamma)
                wiring.connect(m, betas[idx].o, combines[idx].i_beta)
                prev_out = combines[idx].o
            else:
                prev_out = mlp.o
        wiring.connect(m, prev_out, y_pred.i)
        wiring.connect(m, y_pred.o, wiring.flipped(self.o))

        return m
