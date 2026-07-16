import os

# this package dir is literally named `keras`, which shadows the installed
# `keras` (v3) package. force TF's legacy Keras (tf_keras / Keras 2 API, which
# qkeras also expects) so nothing imports the top-level `keras` package.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Layer
from tensorflow.keras.models import Model
from pathlib import Path
import json


class RandomFourierFeatures(Layer):
    """
    Fixed Random Fourier Feature mapping (Tancik et al. 2020).

    Maps input v -> [cos(2*pi*v*B), sin(2*pi*v*B)] where B is a fixed
    (non-trainable) Gaussian matrix sampled once from N(0, scale**2).
    Output dimensionality is 2 * num_features.
    """

    def __init__(self, num_features: int, scale: float, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale = scale
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
        proj = 2.0 * np.pi * tf.matmul(x, self.B)
        return tf.concat([tf.cos(proj), tf.sin(proj)], axis=-1)

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


def create_rff_inr_model_from_config_and_latest_ckpt(run: str):
    run_dir_path = Path("runs") / run
    with open(run_dir_path / "model_config.json", "r") as f:
        model_config = json.load(f)
    print("model_config", model_config)
    model = create_rff_inr_model(**model_config)
    ckpts = (run_dir_path / "weights" / "keras").iterdir()
    latest_ckpt = list(sorted(ckpts))[-1]
    print("using ckpt", latest_ckpt)
    model.load_weights(str(latest_ckpt))
    return model


def create_rff_inr_model(
    in_d: int,
    num_fourier_features: int,
    rff_scale: float,
    mlp_layers: int,
    mlp_width: int,
    out_d: int,
    rff_seed: int = 0,
):
    # creates an implicit neural representation (INR) model:
    #   reconstruct phase angle from (sin, cos), map through fixed Random
    #   Fourier Features, concat the raw 2D waveform embedding, then regress
    #   the waveshaped output with a standard ReLU MLP.
    #
    # input layout (last axis): [phase_sin, phase_cos, embed0, embed1, ...]

    inp = Input((None, in_d))

    phase_sin = Lambda(lambda t: t[..., 0:1], name="phase_sin")(inp)
    phase_cos = Lambda(lambda t: t[..., 1:2], name="phase_cos")(inp)
    embed = Lambda(lambda t: t[..., 2:in_d], name="embed")(inp)

    # reconstruct the 1D phase angle in (-pi, pi]
    phase = Lambda(lambda t: tf.math.atan2(t[0], t[1]), name="phase_angle")(
        [phase_sin, phase_cos]
    )

    rff = RandomFourierFeatures(
        num_features=num_fourier_features,
        scale=rff_scale,
        seed=rff_seed,
        name="rff",
    )(phase)

    h = Concatenate(name="rff_embed")([rff, embed])
    for i in range(mlp_layers):
        h = Dense(mlp_width, activation="relu", name=f"mlp{i}")(h)

    y_pred = Dense(out_d, activation=None, name="y_pred")(h)

    return Model(inp, y_pred)
