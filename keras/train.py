import os

# this package dir is literally named `keras`, which shadows the installed
# `keras` (v3) package. force TF's legacy Keras (tf_keras / Keras 2 API, which
# qkeras also expects) so nothing imports the top-level `keras` package.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from pathlib import Path
import json

from tensorflow.keras.optimizers import Adam

from .model import create_rff_inr_model
from tf_data.quadrature_data import Embed2DQuadratureData
from common.losses import combined_masked_loss_terms

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-note", type=str, default="A4")
    parser.add_argument("--max-note", type=str, default="A4")
    parser.add_argument("--sample-rate-khz", type=float, default=192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-train-samples", type=int, default=10_000)
    parser.add_argument("--num-validate-samples", type=int, default=100)
    parser.add_argument(
        "--num-fourier-features",
        type=int,
        default=128,
        help="number of Random Fourier Features (output dim is 2x this)",
    )
    parser.add_argument(
        "--rff-scale",
        type=float,
        default=5.0,
        help="Gaussian std (sigma) for the fixed RFF frequency matrix B",
    )
    parser.add_argument("--rff-seed", type=int, default=0)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument(
        "--alpha-mse",
        type=float,
        default=1.0,
        help="weight for masked MSE/Huber in combined loss",
    )
    parser.add_argument(
        "--use-huber-loss",
        action="store_true",
        help="if set use huber instead of MSE",
    )
    parser.add_argument(
        "--beta-stft",
        type=float,
        default=0.1,
        help="STFT-loss weight in combined loss",
    )
    opts = parser.parse_args()
    print("opts", opts)

    print("tf", tf.__version__)
    print("tf devices", tf.config.list_physical_devices())

    tensorboard_dir = Path("runs") / opts.run / "tb"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("runs") / opts.run / "weights" / "keras"
    weights_dir.mkdir(parents=True, exist_ok=True)

    IN_D = 4  # phase_sin, phase_cos, embed0, embed1
    OUT_D = 1  # output wave
    TRAIN_SEQ_LEN = 1024
    TEST_SEQ_LEN = 2048
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    data = Embed2DQuadratureData(
        min_note=opts.min_note,
        max_note=opts.max_note,
        sample_rate_khz=opts.sample_rate_khz,
        seed=opts.seed,
    )
    train_ds = data.tf_dataset(
        batch_size=opts.batch_size,
        seq_len=TRAIN_SEQ_LEN,
        num_samples=opts.num_train_samples,
        emit_endpt_samples=True,
        emit_interpolated_samples=True,
    )
    validate_ds = data.tf_dataset(
        batch_size=opts.batch_size,
        seq_len=TEST_SEQ_LEN,
        num_samples=opts.num_validate_samples,
        emit_endpt_samples=True,
        emit_interpolated_samples=True,
    )

    # make model
    model_config = {
        "in_d": IN_D,
        "num_fourier_features": opts.num_fourier_features,
        "rff_scale": opts.rff_scale,
        "mlp_layers": opts.mlp_layers,
        "mlp_width": opts.mlp_width,
        "out_d": OUT_D,
        "rff_seed": opts.rff_seed,
    }
    print("model_config", model_config)
    with open(Path("runs") / opts.run / "model_config.json", "w") as f:
        json.dump(model_config, f)
    train_model = create_rff_inr_model(**model_config)
    train_model.summary()

    callbacks = []
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_dir / "{epoch:03d}.weights.h5"),
            save_weights_only=True,
        )
    )

    # the INR is pointwise so there is no receptive field to mask
    combined_loss_fn, mse_loss_metric, stft_loss_metric = combined_masked_loss_terms(
        receptive_field_size=None,
        use_huber_loss=opts.use_huber_loss,
        alpha_mse=opts.alpha_mse,
        beta_stft=opts.beta_stft,
        seq_len=TRAIN_SEQ_LEN,
    )
    optimizer = Adam(opts.learning_rate)
    train_model.compile(
        optimizer,
        loss=combined_loss_fn,
        metrics=[mse_loss_metric, stft_loss_metric],
        jit_compile=False,  # XLA problem with STFT ???
    )

    train_model.fit(train_ds, callbacks=callbacks, epochs=opts.epochs)
