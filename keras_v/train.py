import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from pathlib import Path
import json
import argparse

from tensorflow.keras.optimizers import AdamW

from .model import create_rff_inr_model
from .util import CheckYPred
from tf_data.quadrature_data import Embed2DQuadratureData
from common.losses import combined_loss_terms
from common.callbacks import (
    setup_beta_stft_var_and_update_callback,
    LogLrAndBetaStft,
)

def train(opts):

    run_path = Path("runs") / opts.run

    tensorboard_dir = run_path / "tb"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_path / "weights" / "keras"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("opts", opts)
    with open(run_path / "opts.json", "w") as f:
        json.dump(vars(opts), f, default=str)

    IN_D = 3  # phase, embed0, embed1
    OUT_D = 1  # output wave
    TRAIN_SEQ_LEN = 4 * opts.base_stft_win_length
    TEST_SEQ_LEN = 2048
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    data = Embed2DQuadratureData(
        min_note=opts.min_note,
        max_note=opts.max_note,
        sample_rate_khz=opts.sample_rate_khz,
        harsh=opts.harsh,
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
    with open(run_path / "model_config.json", "w") as f:
        json.dump(model_config, f)
    train_model = create_rff_inr_model(**model_config)

    train_model.summary()
    with open(run_path / "train_model_summary.txt", "w") as f:
        train_model.summary(print_fn=lambda line: f.write(line + "\n"))

    ramp_callback, beta_stft = setup_beta_stft_var_and_update_callback(
        opts.epochs, opts.beta_stft_warmup, opts.beta_stft_ramp, opts.beta_stft
    )

    callbacks = []

    # log beta_stft (and lr) into the logs before the TensorBoard callback
    callbacks.append(LogLrAndBetaStft(beta_stft_var=beta_stft))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_dir / "{epoch:03d}.weights.h5"),
            save_weights_only=True,
        )
    )
    callbacks.append(CheckYPred(tb_dir=str(tensorboard_dir), dataset=validate_ds))
    if ramp_callback is not None:
        callbacks.append(ramp_callback)

    def halving_triple(base):
        # o_O
        triple = [base, base // 2, base // 4]
        last = triple[-1]
        last_is_po2 = (last & (last - 1)) == 0
        assert (
            last > 0 and last_is_po2
        ), f"last value {last} (from base {base}) must be a power of 2"
        return triple

    if opts.base_stft_hop_size is None:
        stft_hop_sizes = halving_triple(opts.base_stft_fft_size // 4)
    else:
        stft_hop_sizes = halving_triple(opts.base_stft_hop_size)

    combined_loss_fn, mse_metric, huber_metric, stft_metric = combined_loss_terms(
        alpha_mse=opts.alpha_mse,
        alpha_huber=opts.alpha_huber,
        beta_stft=beta_stft,
        seq_len=TRAIN_SEQ_LEN,
        stft_fft_sizes=halving_triple(opts.base_stft_fft_size),
        stft_win_lengths=halving_triple(opts.base_stft_win_length),
        stft_hop_sizes=stft_hop_sizes,
    )
    optimizer = AdamW(opts.learning_rate, weight_decay=opts.weight_decay)
    train_model.compile(
        optimizer,
        loss=combined_loss_fn,
        metrics=[mse_metric, huber_metric, stft_metric],
        jit_compile=False,  # XLA problem with STFT ???
    )

    train_model.fit(train_ds, callbacks=callbacks, epochs=opts.epochs, verbose=2)


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-note", type=str, default="A3")
    parser.add_argument("--max-note", type=str, default="A5")
    parser.add_argument("--harsh", action="store_true")
    parser.add_argument("--sample-rate-khz", type=float, default=192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-train-samples", type=int, default=10_000)
    parser.add_argument("--num-validate-samples", type=int, default=100)
    parser.add_argument(
        "--num-fourier-features",
        type=int,
        default=64,
        help="number of Random Fourier Features (output dim is 2x this)",
    )
    parser.add_argument(
        "--rff-scale",
        type=float,
        default=5.0,
        help="Gaussian std (sigma) for the fixed RFF frequency matrix B",
    )
    parser.add_argument("--rff-seed", type=int, default=0)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--mlp-width", type=int, default=16)
    parser.add_argument(
        "--alpha-mse",
        type=float,
        default=1.0,
        help="weight for MSE in combined loss",
    )
    parser.add_argument(
        "--alpha-huber",
        type=float,
        default=0.0,
        help="weight for Huber in combined loss",
    )
    parser.add_argument(
        "--beta-stft",
        type=float,
        default=0.0001,
        help="target STFT-loss weight in combined loss (after warmup and ramp)",
    )
    parser.add_argument(
        "--beta-stft-warmup",
        type=int,
        default=0,
        help="keep beta_stft at 0 for this many epochs at start",
    )
    parser.add_argument(
        "--beta-stft-ramp",
        type=int,
        default=0,
        help="linearly ramp beta_stft from 0 to target over this many epochs (post warmup)",
    )
    parser.add_argument(
        "--base-stft-fft-size",
        type=int,
        default=2048,
        help="base FFT size; resolutions are (base, base//2, base//4)",
    )
    parser.add_argument(
        "--base-stft-win-length",
        type=int,
        default=256,
        help="base STFT window length; resolutions are (base, base//2, base//4)",
    )
    parser.add_argument(
        "--base-stft-hop-size",
        type=int,
        default=None,
        help="base STFT hop size. dfts to 1/4 --base-stft-win-length. resolutions are (base, base//2, base//4)",
    )
    return parser


if __name__ == "__main__":
    print("tf", tf.__version__)
    print("tf devices", tf.config.list_physical_devices())
    train(build_parser().parse_args())
