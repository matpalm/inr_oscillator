import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import json
import pickle
import argparse
import warnings
from pathlib import Path

from tensorflow.keras.optimizers import AdamW
from qkeras.utils import model_save_quantized_weights

from .qkeras_model import QKerasRFFModelBuilder
from common.util import CheckYPred
from tf_data.quadrature_data import Embed2DQuadratureData
from common.losses import combined_loss_terms
from common.callbacks import (
    setup_beta_stft_var_and_update_callback,
    LogLrAndBetaStft,
)

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*API should only be used for objects.*"
)


def train(opts):

    run_path = Path("runs") / opts.run

    tensorboard_dir = run_path / "tb"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    keras_weights_dir = run_path / "weights" / "keras"
    keras_weights_dir.mkdir(parents=True, exist_ok=True)
    qkeras_weights_dir = run_path / "weights" / "qkeras"
    qkeras_weights_dir.mkdir(parents=True, exist_ok=True)

    print("opts", opts)
    with open(run_path / "opts.json", "w") as f:
        json.dump(vars(opts), f, default=str)

    in_d = 3  # phase, embed0, embed1
    out_d = 1  # output wave
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

    # TODO: rff_lut_size can be pushed all the way to the RFF generation

    # make model
    model_config = {
        "in_d": in_d,
        "fp_info": {
            "mlp": {"n_int": opts.mlp_fp_int, "n_frac": opts.mlp_fp_frac},
            "io": {"n_int": opts.io_fp_int, "n_frac": opts.io_fp_frac},
        },
        "rff": {
            "num_features": opts.num_fourier_features,
            "scale": opts.rff_scale,
            "lut_size": opts.rff_lut_size,
            "seed": opts.rff_seed,
        },
        "mlp_dims": opts.mlp_dims,
        "out_d": out_d,
        "relu_upper_bound": opts.relu_upper_bound,
    }
    print("model_config", model_config)
    with open(run_path / "model_config.json", "w") as f:
        json.dump(model_config, f)

    builder = QKerasRFFModelBuilder(**model_config)

    with open(run_path / "quant_sizes.json", "w") as f:
        json.dump(builder.get_b_io_quant_sizes(), f)

    train_model = builder.build()

    train_model.summary()
    with open(run_path / "qkeras_model.summary.txt", "w") as f:
        train_model.summary(print_fn=lambda line: f.write(line + "\n"))

    # optionally initialise from prior (float keras_v or qkeras_v) weights for fine-tuning
    init_weights_path = None
    if opts.init_weights is not None:
        if opts.init_weights.is_dir():
            init_weights_path = str(sorted(opts.init_weights.iterdir())[-1])
        else:
            init_weights_path = str(opts.init_weights)
        print("init weights from", init_weights_path)
        train_model.load_weights(init_weights_path)

    ramp_callback, beta_stft = setup_beta_stft_var_and_update_callback(
        opts.epochs, opts.beta_stft_warmup, opts.beta_stft_ramp, opts.beta_stft
    )

    class SaveQuantisedWeights(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            quantised_weights = model_save_quantized_weights(train_model)
            rff = train_model.get_layer("rff")
            quantised_weights["rff"] = {
                "B": rff.b_quantizer(rff.B).numpy(),
            }
            pkl_fname = qkeras_weights_dir / f"e{epoch:03d}.pkl"
            with open(pkl_fname, "wb") as f:
                pickle.dump(quantised_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
            # refresh a latest.pkl symlink
            latest_symlink = qkeras_weights_dir / "latest.pkl"
            try:
                latest_symlink.unlink()
            except FileNotFoundError:
                pass
            latest_symlink.symlink_to(pkl_fname.name)

    # TODO: bring across cosine schedule from CDCC

    callbacks = []
    # log beta_stft (and lr) into the logs before the TensorBoard callback
    callbacks.append(LogLrAndBetaStft(beta_stft_var=beta_stft))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(keras_weights_dir / "{epoch:03d}.weights.h5"),
            save_weights_only=True,
        )
    )
    callbacks.append(SaveQuantisedWeights())
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
    if opts.cosine_schedule:
        lr_warmup_epochs = opts.beta_stft_warmup + opts.beta_stft_ramp
        steps_per_epoch = max(1, opts.num_train_samples // opts.batch_size)
        total_steps = opts.epochs * steps_per_epoch
        warmup_steps = lr_warmup_epochs * steps_per_epoch
        if warmup_steps > 0:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.0,
                decay_steps=max(1, total_steps - warmup_steps),
                alpha=opts.lr_min_frac,
                warmup_target=opts.learning_rate,
                warmup_steps=warmup_steps,
            )
        else:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=opts.learning_rate,
                decay_steps=max(1, total_steps),
                alpha=opts.lr_min_frac,
            )
        print(
            "lr schedule: cosine decay"
            f" lr={opts.learning_rate} warmup_epochs={lr_warmup_epochs}"
            f" total_steps={total_steps} steps_per_epoch={steps_per_epoch}"
            f" min_frac={opts.lr_min_frac}"
        )
    else:
        lr_schedule = opts.learning_rate
        print(f"lr schedule: fixed lr={opts.learning_rate}")

    optimiser = AdamW(learning_rate=lr_schedule, weight_decay=opts.weight_decay)
    train_model.compile(
        optimiser,
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
    parser.add_argument(
        "--cosine-schedule",
        action="store_true",
        help="if set, use linear warmup + cosine decay; otherwise use a fixed learning rate",
    )
    parser.add_argument(
        "--lr-min-frac",
        type=float,
        default=0.01,
        help="cosine decay floor as a fraction of --learning-rate (0 => decay to 0)",
    )
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
        "--mlp-fp-int",
        type=int,
        default=3,
        help="MLP fixed-point integer bits (excluding sign)",
    )
    parser.add_argument(
        "--mlp-fp-frac",
        type=int,
        default=13,
        help="MLP fixed-point fractional bits",
    )
    parser.add_argument(
        "--io-fp-int",
        type=int,
        default=1,
        help="signal-path (phase/embed/rff-out/output) integer bits; these live in [-1, 1]",
    )
    parser.add_argument(
        "--io-fp-frac",
        type=int,
        default=15,
        help="signal-path (phase/embed/rff-out/output) fractional bits",
    )
    parser.add_argument(
        "--relu-upper-bound",
        type=float,
        default=6.0,
        help="upper bound for quantized_relu activations",
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="path (dir or file) to keras-format weights to initialise fine-tuning",
    )
    parser.add_argument(
        "--num-fourier-features",
        type=int,
        default=64,
        help="number of Random Fourier Features (output dim is 2x this)",
    )
    parser.add_argument(
        "--rff-scale",
        type=float,
        default=1.0,
        help="Gaussian std (sigma) for the fixed RFF frequency matrix B",
    )
    parser.add_argument(
        "--rff-lut-size",
        type=int,
        default=1024,
        help="target lut size for amaranth_v",
    )
    parser.add_argument("--rff-seed", type=int, default=0)
    parser.add_argument(
        "--mlp-dims",
        type=int,
        nargs="+",
        default=[16, 16],
        help="per-layer node counts, e.g. --mlp-dims 8 32 32 => 3 layers",
    )
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
