import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import math
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Lambda, Layer
from tensorflow.keras.models import Model
from qkeras import quantized_bits, QDense, QActivation


class QRandomFourierFeatures(Layer):
    """
    Fixed Random Fourier Feature mapping (Tancik et al. 2020), quantised.

    Maps input v -> [cos(2*pi*v*Bq), sin(2*pi*v*Bq)] where B is a fixed
    (non-trainable) Gaussian matrix sampled once from N(0, scale**2) and Bq is
    B passed through the fixed-point quantiser. The transcendental cos/sin are
    kept in float (a LUT on hardware); the layer output is then quantised.
    Output dimensionality is 2 * num_features.
    """

    def __init__(
        self,
        num_features: int,
        scale: float,
        b_quantizer,
        out_quantizer,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale = scale
        self.b_quantizer = b_quantizer
        self.out_quantizer = out_quantizer
        self.seed = seed

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        b_init = tf.random_normal_initializer(stddev=self.scale, seed=self.seed)
        self.B = self.add_weight(
            name="B",
            shape=(in_dim, self.num_features),
            initializer=b_init,
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # quantise the (fixed) frequency matrix, cos/sin stay float (HW LUT)
        b_q = self.b_quantizer(self.B)
        proj = 2.0 * np.pi * tf.matmul(x, b_q)
        feats = tf.concat([tf.cos(proj), tf.sin(proj)], axis=-1)
        return self.out_quantizer(feats)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "scale": self.scale,
                "seed": self.seed,
            }
        )
        return config


class QKerasRFFModelBuilder(object):

    def __init__(self):
        self.layer_info = []
        self.built = False

    # quantiser for kernels / biases
    def quantiser(self, double_width: bool = False):
        nword = self.mlp_n_word
        nint = self.mlp_n_int
        if double_width:
            nword *= 2
            nint *= 2
        return quantized_bits(bits=nword, integer=nint, alpha=1)

    # quantiser for the signal path (inputs, rff output, model output); these
    # all live in [-1, 1] so they only need a sign + fractional bits.
    def io_quantiser(self):
        return quantized_bits(bits=self.io_n_word, integer=self.io_n_int, alpha=1)

    # quantiser for the (fixed) RFF frequency matrix B. B ~ N(0, scale**2) so
    # |B| routinely exceeds the weight integer range; size the integer part to
    # cover ~4 sigma so we do not clip the sampled frequencies.
    def b_quantiser(self, scale: float):
        n_int_b = max(self.mlp_n_int, int(math.ceil(math.log2(max(4.0 * scale, 1.0)))))
        self.layer_info.append({"type": "rff_b_quant", "n_int": n_int_b})
        return quantized_bits(bits=n_int_b + self.mlp_n_frac, integer=n_int_b, alpha=1)

    # relu activation quantiser (string form consumed by QActivation)
    def quant_relu(self, upper_bound: float):
        return f"quantized_relu({self.mlp_n_word},{self.mlp_n_int},relu_upper_bound={upper_bound})"

    def create_rff_inr_model(
        self,
        fp_info: dict,
        in_d: int,
        num_fourier_features: int,
        rff_scale: float,
        mlp_layers: int,
        mlp_width: int,
        out_d: int,
        relu_upper_bound: float,
        rff_seed: int = 0,
    ):
        # phase -> quantised Random Fourier Features -> concat quantised 2D
        # waveform embedding -> quantised ReLU MLP -> quantised output.

        self.mlp_n_int = fp_info["mlp"]["n_int"]
        self.mlp_n_frac = fp_info["mlp"]["n_frac"]
        self.mlp_n_word = self.mlp_n_int + self.mlp_n_frac
        self.io_n_int = fp_info["io"]["n_frac"]
        self.io_n_frac = fp_info["io"]["n_frac"]
        self.io_n_word = self.io_n_int + self.io_n_frac

        self.layer_info = []

        inp = Input((None, in_d))

        # phase angle in [-1, 1); at inference run as: phase += delta; wrap 1 to -1
        phase = Lambda(lambda t: t[..., 0:1], name="phase")(inp)
        embed = Lambda(lambda t: t[..., 1:in_d], name="embed")(inp)

        # quantise the raw inputs (signal-path format)
        phase_q = QActivation(self.io_quantiser(), name="phase_q")(phase)
        embed_q = QActivation(self.io_quantiser(), name="embed_q")(embed)

        rff = QRandomFourierFeatures(
            num_features=num_fourier_features,
            scale=rff_scale,
            b_quantizer=self.b_quantiser(rff_scale),
            out_quantizer=self.io_quantiser(),
            seed=rff_seed,
            name="rff",
        )(phase_q)
        self.layer_info.append({"type": "rff", "num_features": num_fourier_features})

        h = Concatenate(name="rff_embed")([rff, embed_q])
        for i in range(mlp_layers):
            h = QDense(
                mlp_width,
                kernel_quantizer=self.quantiser(),
                bias_quantizer=self.quantiser(double_width=True),
                name=f"mlp{i}",
            )(h)
            self.layer_info.append(
                {"type": "qdense", "id": f"mlp{i}", "width": mlp_width}
            )
            h = QActivation(self.quant_relu(relu_upper_bound), name=f"qrelu{i}")(h)
            self.layer_info.append({"type": "relu", "upper_bound": relu_upper_bound})

        y_pred = QDense(
            out_d,
            kernel_quantizer=self.quantiser(),
            bias_quantizer=self.quantiser(double_width=True),
            name="y_pred",
        )(h)
        self.layer_info.append({"type": "qdense", "id": "y_pred", "width": out_d})

        # TODO: should we keep this as self.quantiser as cdcc did?
        y_pred = QActivation(self.io_quantiser(), name="qout")(y_pred)
        self.layer_info.append(
            {"type": "qout", "n_int": self.io_n_int, "n_frac": self.io_n_frac}
        )

        print("layer_info", self.layer_info)
        self.built = True
        return Model(inp, y_pred)


def create_rff_inr_model_from_config_and_latest_ckpt(run: str):
    run_dir_path = Path("runs") / run
    with open(run_dir_path / "qkeras_model.fp_config.json", "r") as f:
        fp_config = json.load(f)
    builder = QKerasRFFModelBuilder(
        mlp_n_int=fp_config["n_int"],
        mlp_n_frac=fp_config["n_frac"],
        io_n_int=fp_config["io_n_int"],
        io_n_frac=fp_config["io_n_frac"],
    )
    with open(run_dir_path / "model_config.json", "r") as f:
        model_config = json.load(f)
    print("model_config", model_config)
    model = builder.create_rff_inr_model(**model_config)
    ckpts = (run_dir_path / "weights" / "keras").iterdir()
    latest_ckpt = list(sorted(ckpts))[-1]
    print("using ckpt", latest_ckpt)
    model.load_weights(str(latest_ckpt))
    return model
