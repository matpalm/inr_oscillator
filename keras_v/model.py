import os

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

    def __init__(
        self,
        num_features: int,
        scale_min: float,
        scale_max: float,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale_min = scale_min
        self.scale_max = scale_max
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
        proj = 2.0 * np.pi * tf.matmul(x, self.B)
        return tf.concat([tf.cos(proj), tf.sin(proj)], axis=-1)

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
    rff: dict,
    mlp_dims: list[int],
    out_d: int,
):
    # creates an implicit neural representation (INR) model:
    #   map the phase angle through fixed Random Fourier
    #   Features, concat the raw 2D waveform embedding, then regress the
    #   waveshaped output with a standard ReLU MLP.
    #
    # input layout (last axis): [phase, embed0, embed1, ...]

    inp = Input((None, in_d))

    # phase angle in [-1, 1); at inference run as: phase += delta; wrap 1 to -1
    phase = Lambda(lambda t: t[..., 0:1], name="phase")(inp)
    embed = Lambda(lambda t: t[..., 1:in_d], name="embed")(inp)

    rff = RandomFourierFeatures(
        num_features=rff["num_features"],
        scale_min=rff["scale_min"],
        scale_max=rff["scale_max"],
        seed=rff["seed"],
        name="rff",
    )(phase)

    h = Concatenate(name="rff_embed")([rff, embed])
    for i, mlp_dim in enumerate(mlp_dims):
        h = Dense(mlp_dim, activation="relu", name=f"mlp{i}")(h)

    y_pred = Dense(out_d, activation=None, name="y_pred")(h)

    return Model(inp, y_pred)
