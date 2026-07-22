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
    https://arxiv.org/abs/2006.10739

    Maps input v -> [cos(2*pi*v*Bq), sin(2*pi*v*Bq)] where B is a fixed
    (non-trainable) Gaussian matrix sampled once from N(0, scale**2) and Bq is
    B passed through the fixed-point quantiser. The transcendental cos/sin are
    kept in float (a LUT on hardware); the layer output is then quantised.
    Output dimensionality is 2 * num_features.
    """

    def __init__(
        self,
        num_features: int,
        scale_min: float,
        scale_max: float,
        b_quantizer,
        out_quantizer,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.b_quantizer = b_quantizer
        self.out_quantizer = out_quantizer
        self.seed = seed

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if self.scale_min <= 0.0 or self.scale_max <= 0.0:
            raise ValueError("RFF scales must be > 0")
        if self.scale_min > self.scale_max:
            raise ValueError("rff scale_min must be <= scale_max")

        rng = np.random.default_rng(seed=self.seed)
        if self.scale_min == self.scale_max:
            col_scales = np.full((self.num_features,), self.scale_min, dtype=np.float32)
        else:
            log_scales = rng.uniform(
                low=np.log(self.scale_min),
                high=np.log(self.scale_max),
                size=(self.num_features,),
            )
            col_scales = np.exp(log_scales).astype(np.float32)

        b_values = (
            rng.standard_normal(size=(in_dim, self.num_features)).astype(np.float32)
            * col_scales[None, :]
        )
        b_init = tf.constant_initializer(b_values)
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
                "scale_min": self.scale_min,
                "scale_max": self.scale_max,
                "seed": self.seed,
            }
        )
        return config


# def b_io_quant_sizes(model_config):
#     return QKerasRFFModelBuilder(**model_config)._get_b_io_quant_sizes()


class QRffInrModel(Model):
    """Quantised RFF-INR model with an optional morph-consistency aux loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lambda_morph_consistency = 0.0
        self._morph_consistency_tracker = tf.keras.metrics.Mean(
            name="morph_consistency"
        )

    def enable_morph_consistency(self, lambda_value: float):
        in_d = int(self.input_shape[-1])
        if in_d != 4:
            raise ValueError("expected (in_d==4), got in_d={in_d}")
        self._lambda_morph_consistency = float(lambda_value)

    @staticmethod
    def _flip_ab_morph(x):
        # [phase, a_cv, b_cv, morph] -> [phase, b_cv, a_cv, -morph]
        phase = x[..., 0:1]
        a_cv = x[..., 1:2]
        b_cv = x[..., 2:3]
        morph = x[..., 3:4]
        return tf.concat([phase, b_cv, a_cv, -morph], axis=-1)

    def _morph_consistency(self, x, y_pred, training):
        if self._lambda_morph_consistency <= 0.0:
            return tf.constant(0.0, dtype=y_pred.dtype)
        y_pred_flip = self(self._flip_ab_morph(x), training=training)
        return tf.reduce_mean(tf.square(y_pred - y_pred_flip))

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses
            )
            consistency = self._morph_consistency(x, y_pred, training=True)
            loss = loss + self._lambda_morph_consistency * consistency
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        self._morph_consistency_tracker.update_state(consistency)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        consistency = self._morph_consistency(x, y_pred, training=False)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        self._morph_consistency_tracker.update_state(consistency)
        return {m.name: m.result() for m in self.metrics}


class QKerasRFFModelBuilder(object):

    def __init__(
        self,
        fp_info: dict,
        in_d: int,
        rff: dict,
        mlp_dims: list[int],
        out_d: int,
        relu_upper_bound: float,
    ):
        # phase -> quantised Random Fourier Features -> concat quantised 2D
        # waveform embedding -> quantised ReLU MLP -> quantised output.
        self.in_d = in_d
        self.rff_num_features = rff["num_features"]
        scale = rff.get("scale")
        self.rff_scale_min = float(rff.get("scale_min", scale))
        self.rff_scale_max = float(rff.get("scale_max", scale))
        self.rff_seed = rff["seed"]
        self.mlp_dims = mlp_dims
        self.out_d = out_d
        self.relu_upper_bound = relu_upper_bound
        self.mlp_n_int = fp_info["mlp"]["n_int"]
        self.mlp_n_frac = fp_info["mlp"]["n_frac"]
        self.mlp_n_word = self.mlp_n_int + self.mlp_n_frac
        self.io_n_int = fp_info["io"]["n_int"]
        self.io_n_frac = fp_info["io"]["n_frac"]
        self.io_n_word = self.io_n_int + self.io_n_frac

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
    def b_quantiser(self):
        max_scale = max(self.rff_scale_min, self.rff_scale_max)
        n_int_b = max(
            self.mlp_n_int, int(math.ceil(math.log2(max(4.0 * max_scale, 1.0))))
        )
        return quantized_bits(bits=n_int_b + self.mlp_n_frac, integer=n_int_b, alpha=1)

    # relu activation quantiser (string form consumed by QActivation)
    def quant_relu(self):
        return f"quantized_relu({self.mlp_n_word},{self.mlp_n_int},relu_upper_bound={self.relu_upper_bound})"

    def get_b_io_quant_sizes(self):
        rff_b_quant = self.b_quantiser()
        rff_io_quant = self.io_quantiser()
        return {
            "b_bits": rff_b_quant.bits,
            "b_int": rff_b_quant.integer,
            "io_bits": rff_io_quant.bits,
            "io_int": rff_io_quant.integer,
        }

    def build(self):

        inp = Input((None, self.in_d))

        # phase angle in [-1, 1); at inference run as: phase += delta; wrap 1 to -1
        phase = Lambda(lambda t: t[..., 0:1], name="phase")(inp)
        embed = Lambda(lambda t: t[..., 1 : self.in_d], name="embed")(inp)

        # quantise the raw inputs (signal-path format)
        phase_q = QActivation(self.io_quantiser(), name="phase_q")(phase)
        embed_q = QActivation(self.io_quantiser(), name="embed_q")(embed)

        rff_b_quant = self.b_quantiser()
        rff_io_quant = self.io_quantiser()
        rff = QRandomFourierFeatures(
            num_features=self.rff_num_features,
            scale_min=self.rff_scale_min,
            scale_max=self.rff_scale_max,
            b_quantizer=rff_b_quant,
            out_quantizer=rff_io_quant,
            seed=self.rff_seed,
            name="rff",
        )(phase_q)

        h = Concatenate(name="rff_embed")([rff, embed_q])
        for i, dim in enumerate(self.mlp_dims):
            h = QDense(
                dim,
                kernel_quantizer=self.quantiser(),
                bias_quantizer=self.quantiser(double_width=True),
                name=f"mlp{i}",
            )(h)
            h = QActivation(self.quant_relu(), name=f"qrelu{i}")(h)

        y_pred = QDense(
            self.out_d,
            kernel_quantizer=self.quantiser(),
            bias_quantizer=self.quantiser(double_width=True),
            name="y_pred",
        )(h)

        # TODO: should we keep this as self.quantiser as cdcc did?
        y_pred = QActivation(self.io_quantiser(), name="qout")(y_pred)

        return QRffInrModel(inp, y_pred)


def build_model_from_config_and_latest_ckpt(run: str):
    run_dir_path = Path("runs") / run
    with open(run_dir_path / "model_config.json", "r") as f:
        model_config = json.load(f)
    print("model_config", model_config)
    model = QKerasRFFModelBuilder(**model_config).build()
    ckpts = (run_dir_path / "weights" / "keras").iterdir()
    latest_ckpt = list(sorted(ckpts))[-1]
    print("using ckpt", latest_ckpt)
    model.load_weights(str(latest_ckpt))
    return model
