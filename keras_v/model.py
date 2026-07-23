import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Layer, LeakyReLU
from tensorflow.keras.models import Model
from pathlib import Path
import json


class SparseFeatures(Layer):

    def __init__(self, l1: float, **kwargs):
        super().__init__(**kwargs)
        self.l1 = l1

    def build(self, input_shape):
        # //2 since we want to tie the sin/cos pairs
        self.num_features = int(input_shape[-1]) // 2  #
        self.w = self.add_weight(
            shape=(self.num_features,),
            initializer="ones",
            regularizer=tf.keras.regularizers.l1(self.l1),
            trainable=True,
            name="feature_weights",
        )
        super().build(input_shape)

    def call(self, inputs):
        # tie cos_k & sin_k ( since that's what we actually care about )
        gate = tf.concat([self.w, self.w], axis=-1)
        return inputs * gate

    def get_config(self):
        config = super().get_config()
        config.update({"l1": self.l1})
        return config


def build_rff_frequency_matrix(
    basis: str,
    in_dim: int,
    num_features: int,
    scale_min: float,
    scale_max: float,
    seed: int,
):
    """
    Args:
        basis: gaussian sample or derived harmonic (Tancik et al. 2020).
        id_dim: for asserting just phase
        scale_min: min scale for gaussian sample
        scale_max: max scale for gaussian sample
        seed: for gaussian sample
    """
    if basis == "gaussian":
        if scale_min <= 0.0 or scale_max <= 0.0:
            raise Exception("RFF scales must be > 0")
        if scale_min > scale_max:
            raise Exception("rff scale_min must be <= scale_max")
        rng = np.random.default_rng(seed=seed)
        if scale_min == scale_max:
            col_scales = np.full((num_features,), scale_min, dtype=np.float32)
        else:
            log_scales = rng.uniform(
                low=np.log(scale_min),
                high=np.log(scale_max),
                size=(num_features,),
            )
            col_scales = np.exp(log_scales).astype(np.float32)
        return (
            rng.standard_normal(size=(in_dim, num_features)).astype(np.float32)
            * col_scales[None, :]
        )
    elif basis == "harmonic":
        assert in_dim == 1, "harmonic is just a range over in_dim=1"
        # phase wrapped is [-1, 1) (width 2) over one cycle, so use
        # B_k = k/2 so that the full integer harmonic series is for
        # both odd and even harmonics.
        harmonics = np.arange(1, num_features + 1, dtype=np.float32) * 0.5
        return harmonics[None, :]
    else:
        raise Exception(f"unknown rff basis {basis}")


class RandomFourierFeatures(Layer):
    """
    fixed fourier feature mapping.

    maps phase v -> [cos(2*pi*v*B), sin(2*pi*v*B)] where B is a fixed
    (non-trainable) frequency matrix.

    B init'd either as basis='gaussian' (Tancik et al. 2020)
    or just full set as int harmonics when basis='harmonic' ( 1 -> num_features )

    output always 2 * num_features ( sin & cos )
    """

    def __init__(
        self,
        num_features: int,
        scale_min: float,
        scale_max: float,
        seed: int = 0,
        basis: str = "gaussian",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale_min = scale_min
        self.scale_max = scale_max
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
                "basis": self.basis,
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


class RffInrModel(Model):
    """Functional RFF-INR model with an optional morph-consistency aux loss.

    ( phase, a, b, morph ) should == ( phase, b, a, -morph )
    so add a loss component that checks MSE between y_pred of these
    """

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


def create_rff_inr_model(
    in_d: int,
    rff: dict,
    mlp_dims: list[int],
    mlp_activation: str,
    out_d: int,
    rff_l1: float = 0.0,
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

    rff_t = RandomFourierFeatures(
        num_features=rff["num_features"],
        scale_min=rff["scale_min"],
        scale_max=rff["scale_max"],
        seed=rff["seed"],
        basis=rff["basis"],
        name="rff",
    )(phase)

    # optional per-frequency L1 gate for feature (frequency) selection; folded
    # away by prune_rff_by_l1 before deployment, so training-only.
    if rff_l1 > 0.0:
        rff_t = SparseFeatures(l1=rff_l1, name="rff_gate")(rff_t)

    h = Concatenate(name="rff_embed")([rff_t, embed])
    for i, mlp_dim in enumerate(mlp_dims):
        if mlp_activation == "leaky_relu":
            # use 1/4, instead of 0.3, so can be a shift on device
            print("layer", i, "leaky 0.25")
            h = Dense(mlp_dim, activation=None, name=f"mlp{i}")(h)
            h = LeakyReLU(alpha=0.25)(h)
        else:
            h = Dense(mlp_dim, activation=mlp_activation, name=f"mlp{i}")(h)

    y_pred = Dense(out_d, activation=None, name="y_pred")(h)

    return RffInrModel(inp, y_pred)


def prune_rff_by_l1(model, model_config: dict, keep_k: int):
    """Select the top 'keep_k' RFF frequencies and fold into mlp0.

    ranbk by effective mag |w_k| * ||mlp0 row_k||
    ( since a small L1 can be compensated by large mlp0 weight )

    Returns (pruned_model, pruned_config, selected_indices).
    """
    num_features = int(model_config["rff"]["num_features"])
    if keep_k > num_features:
        raise ValueError(f"keep_k={keep_k} exceeds num_features={num_features}")

    try:
        gate = model.get_layer("rff_gate").w.numpy()  # (nf, )
    except ValueError:
        # source trained without an L1 gate -> plain magnitude pruning
        gate = np.ones(num_features, dtype=np.float32)
    old_B = model.get_layer("rff").B.numpy()  # (in_dim, nf)

    # split the mlp0 kernel into two parts;
    # 1) cos_rows & sin_rows & 2) everything else
    # where the cos and sin rows will have the gating weights folding in
    mlp0 = model.get_layer("mlp0")
    kernel, bias = [w for w in mlp0.get_weights()]  # kernel (in_d, out)
    cos_rows = kernel[:num_features]  # (nf, out)
    sin_rows = kernel[num_features : 2 * num_features]  # (nf, out)
    embed_rows = kernel[2 * num_features :]  # (embed_dim, out)

    # use the norm of these parts of the kernal _and_ the gate amount
    # to decide top entries. can't just use gate since there's a chance
    # the model will learn to compensate
    row_norm = np.sqrt((cos_rows**2).sum(axis=1) + (sin_rows**2).sum(axis=1))
    effective_feature = gate * row_norm  # (nf,)
    selected_idxs = np.argsort(np.abs(effective_feature))[::-1][:keep_k]

    # debug
    with np.printoptions(suppress=True):
        print("row_norm", np.around(row_norm[selected_idxs], 3))
        print("gate", np.around(gate[selected_idxs], 3))
        print("effective_feature", np.around(effective_feature[selected_idxs], 3))

    # clone config with updates and make new model
    pruned_config = copy.deepcopy(model_config)
    pruned_config["rff"]["num_features"] = int(keep_k)
    pruned_config["rff_l1"] = 0.0  # gate folded away
    pruned_model = create_rff_inr_model(**pruned_config)

    # clobber the random init'd B with the selected frequency columns
    pruned_model.get_layer("rff").B.assign(old_B[:, selected_idxs])

    # fold the per-frequency gate into the mlp0 rff rows
    new_cos = cos_rows[selected_idxs] * gate[selected_idxs][:, None]
    new_sin = sin_rows[selected_idxs] * gate[selected_idxs][:, None]
    new_kernel = np.concatenate([new_cos, new_sin, embed_rows], axis=0)
    pruned_model.get_layer("mlp0").set_weights([new_kernel, bias])

    # copy weights for other layers directlry
    dont_copy = {"rff", "rff_gate", "mlp0"}
    for layer in pruned_model.layers:
        if layer.name in dont_copy or not layer.weights:
            continue
        layer.set_weights(model.get_layer(layer.name).get_weights())

    return pruned_model, pruned_config, selected_idxs
