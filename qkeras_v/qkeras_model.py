import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import math
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Layer
from tensorflow.keras.models import Model
from qkeras import quantized_bits, QDense, QActivation

from keras_v.model import build_rff_frequency_matrix, FiLM
from amaranth_v.siren_cordic import siren_cordic_output_codes

class NNQSineLUT(Layer):
    """Sine activation via a full signed-code LUT on the NNQ grid.

    Inputs are rounded/clipped to NNQ integer codes, then mapped through a LUT
    storing quantised ``sin(omega_0 * x)`` on that same grid.
    """

    def __init__(self, n_word: int, n_int: int, omega_0: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.n_word = int(n_word)
        self.n_int = int(n_int)
        self.omega_0 = float(omega_0)
        self.keep_negative = 1

    def build(self, input_shape):
        frac = self.n_word - self.n_int - self.keep_negative
        if frac < 0:
            raise ValueError(
                f"invalid NNQ shape for sine LUT: n_word={self.n_word}, n_int={self.n_int}"
            )

        lo = -(1 << (self.n_word - 1))
        hi = (1 << (self.n_word - 1)) - 1
        scale = float(2**frac)
        codes = np.arange(lo, hi + 1, dtype=np.int64)
        y_codes = np.asarray(
            siren_cordic_output_codes(
                [int(c) for c in codes.tolist()],
                width=self.n_word,
                frac_bits=frac,
                omega_0=self.omega_0,
            ),
            dtype=np.int64,
        )
        y_q = y_codes.astype(np.float32) / scale

        self._frac = frac
        self._scale = tf.constant(scale, dtype=tf.float32)
        self._lo = tf.constant(float(lo), dtype=tf.float32)
        self._hi = tf.constant(float(hi), dtype=tf.float32)
        self._offset = tf.constant(-lo, dtype=tf.int32)
        self._lut = tf.constant(y_q, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        # quantise to NNQ codes first so activation matches the hardware-style
        # fixed-point datapath semantics.
        q_codes = tf.round(inputs * self._scale)
        q_codes = tf.clip_by_value(q_codes, self._lo, self._hi)
        q_codes = tf.cast(q_codes, tf.int32)
        idx = q_codes + self._offset
        y_lut = tf.gather(self._lut, idx)
        # Straight-through estimator: the LUT path (round/cast/gather) is
        # non-differentiable and returns zero gradient, which would stall
        # training of every layer feeding this activation. Keep the exact
        # hardware LUT value on the forward pass, but route gradients through
        # the continuous sin(omega_0 * x) so backprop sees omega_0*cos(omega_0*x).
        y_cont = tf.sin(self.omega_0 * inputs)
        return tf.stop_gradient(y_lut - y_cont) + y_cont

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_word": self.n_word,
                "n_int": self.n_int,
                "omega_0": self.omega_0,
            }
        )
        return config


class QRandomFourierFeatures(Layer):
    """
    fixed fourier feature mapping.

    maps phase v -> [cos(2*pi*v*B), sin(2*pi*v*B)] where B is a fixed
    (non-trainable) frequency matrix.

    B init'd either as basis='gaussian' (Tancik et al. 2020)
    or just full set as int harmonics when basis='harmonic' ( 1 -> num_features )

    output always 2 * num_features ( sin & cos done via LUT for hardware version
    """

    def __init__(
        self,
        num_features: int,
        scale_min: float,
        scale_max: float,
        b_quantizer,
        out_quantizer,
        seed: int = 0,
        basis: str = "gaussian",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.b_quantizer = b_quantizer
        self.out_quantizer = out_quantizer
        self.seed = seed
        self.basis = basis

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        b_values = build_rff_frequency_matrix(
            basis=self.basis,
            in_dim=in_dim,
            num_features=self.num_features,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            seed=self.seed,
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
                "basis": self.basis,
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
        film_layers: int = 1,
        mlp_activation: str = "relu",
        siren_omega_0: float = 30.0,
        phase_h_index_bits: int = 13,
    ):
        # phase -> quantised Random Fourier Features -> FiLM-conditioned dense
        # -> quantised ReLU MLP -> quantised output.
        self.in_d = in_d
        self.rff_num_features = rff["num_features"]
        scale = rff.get("scale")
        self.rff_scale_min = float(rff.get("scale_min", scale))
        self.rff_scale_max = float(rff.get("scale_max", scale))
        self.rff_seed = rff["seed"]
        self.rff_basis = rff.get("basis", "gaussian")
        self.mlp_dims = mlp_dims
        self.out_d = out_d
        self.relu_upper_bound = relu_upper_bound
        self.film_layers = int(film_layers)
        self.mlp_activation = str(mlp_activation)
        self.siren_omega_0 = float(siren_omega_0)
        # deployment-only: PSRAM phase->h table index bits (not used by the
        # keras graph, but accepted so build(**model_config) round-trips).
        self.phase_h_index_bits = int(phase_h_index_bits)
        if self.mlp_activation not in {"relu", "siren"}:
            raise ValueError(
                f"unsupported mlp_activation={self.mlp_activation}; expected relu or siren"
            )
        if self.film_layers < 1:
            raise ValueError("film_layers must be >= 1 (concat path removed)")
        self.film_layers = min(self.film_layers, len(mlp_dims))
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

    # quantiser for the (fixed) RFF frequency matrix B.
    # for 'gaussian' basis B ~ N(0, scale**2) so |B| often > the weight integer
    # range; so size the int part to cover ~4 sigma ( to avoid clipping the
    # sampled frequencies)
    # for 'harmonic' basis B is all integer harmonics 1..num_features, so size
    # the int part to represent the largest harmonic exactly.
    def b_quantiser(self):
        if self.rff_basis == "gaussian":
            max_scale = max(self.rff_scale_min, self.rff_scale_max)
            n_int_b = max(
                self.mlp_n_int, int(math.ceil(math.log2(max(4.0 * max_scale, 1.0))))
            )
        elif self.rff_basis == "harmonic":
            # B_k = k/2 for k in 1..num_features, so the largest frequency is
            # num_features/2; size the int part to hold it exactly.
            max_b = self.rff_num_features / 2.0
            n_int_b = max(self.mlp_n_int, int(math.ceil(math.log2(max_b + 1))))
        else:
            raise Exception(self.rff_basis)
        return quantized_bits(bits=n_int_b + self.mlp_n_frac, integer=n_int_b, alpha=1)

    # relu activation quantiser (string form consumed by QActivation)
    def quant_relu(self):
        return f"quantized_relu({self.mlp_n_word},{self.mlp_n_int},relu_upper_bound={self.relu_upper_bound})"

    def _siren_kernel_initializer(self, fan_in: int, is_first: bool):
        fan_in = int(fan_in)
        if fan_in <= 0:
            raise ValueError(f"fan_in must be > 0 for siren init, got {fan_in}")
        if self.siren_omega_0 <= 0.0:
            raise ValueError(f"siren_omega_0 must be > 0, got {self.siren_omega_0}")
        if is_first:
            limit = 1.0 / fan_in
        else:
            limit = math.sqrt(6.0 / fan_in) / self.siren_omega_0
        return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

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
            basis=self.rff_basis,
            name="rff",
        )(phase_q)

        # main path carries only the RFF(phase); embed_q conditions via FiLM.
        h = rff
        for i, dim in enumerate(self.mlp_dims):
            dense_kwargs = {}
            if self.mlp_activation == "siren":
                fan_in = int(h.shape[-1])
                dense_kwargs["kernel_initializer"] = self._siren_kernel_initializer(
                    fan_in=fan_in,
                    is_first=(i == 0),
                )
                dense_kwargs["bias_initializer"] = "zeros"

            h = QDense(
                dim,
                kernel_quantizer=self.quantiser(),
                bias_quantizer=self.quantiser(double_width=True),
                name=f"mlp{i}",
                **dense_kwargs,
            )(h)
            if i < self.film_layers:
                # zero-init so FiLM starts as identity
                # gamma/beta are quantised to the mlp fixed-point format.
                gamma = QDense(
                    dim,
                    kernel_quantizer=self.quantiser(),
                    bias_quantizer=self.quantiser(double_width=True),
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                    name=f"film{i}_gamma",
                )(embed_q)
                gamma = QActivation(self.quantiser(), name=f"film{i}_gamma_q")(gamma)
                beta = QDense(
                    dim,
                    kernel_quantizer=self.quantiser(),
                    bias_quantizer=self.quantiser(double_width=True),
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                    name=f"film{i}_beta",
                )(embed_q)
                beta = QActivation(self.quantiser(), name=f"film{i}_beta_q")(beta)

                h = FiLM(name=f"film{i}")([h, gamma, beta])

            if self.mlp_activation == "siren":
                h = NNQSineLUT(
                    self.mlp_n_word,
                    self.mlp_n_int,
                    omega_0=self.siren_omega_0,
                    name=f"qsiren{i}",
                )(h)
            else:
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

    # no longer required; just select all with qkeras_v
    # def mlp0_row_norms(self, model):
    #     mlp0_kernel = model.get_layer("mlp0").get_weights()[0]
    #     cos_rows = mlp0_kernel[: self.rff_num_features]
    #     sin_rows = mlp0_kernel[self.rff_num_features : 2 * self.rff_num_features]
    #     row_norm = np.sqrt((cos_rows**2).sum(axis=1) + (sin_rows**2).sum(axis=1))
    #     order = np.argsort(row_norm)[::-1]
    #     peak = float(row_norm.max()) if self.rff_num_features else 0.0
    #     threshold = 0.01 * peak
    #     n_dead = int((row_norm < threshold).sum())
    #     n_not_dead = self.rff_num_features - n_dead
    #     with np.printoptions(suppress=True):
    #         print("mlp0 rff row_norms", np.around(row_norm[order], 3))
    #         print(f"n_not_dead={n_not_dead} n_dead={n_dead} (dead = <0.01 of peak)")


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
