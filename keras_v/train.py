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
from tf_data.quadrature_data import Embed2DQuadratureData

# from tf_data.pcapture_static_data import ParametricCaptureStaticData
from tf_data.pcapture_inmem_data import ParametricCaptureStaticData
from common.losses import combined_loss_terms
from common.callbacks import (
    setup_beta_stft_var_and_update_callback,
    LogLrAndBetaStft,
    CheckYPred,
    PrintRffMlp0Weights,
)


def train(opts):

    if opts.mlp_activation != "relu":
        raise Exception("only relu supported in qkeras_v and amaranth_v")

    run_path = Path("runs") / opts.run

    tensorboard_dir = run_path / "tb"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_path / "weights" / "keras"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("opts", opts)
    with open(run_path / "opts.json", "w") as f:
        json.dump(vars(opts), f, default=str)

    in_d = 3  # phase, embed0, embed1
    out_d = 1  # output wave
    train_seq_len = int(opts.train_seq_mult * opts.base_stft_win_length)
    print("TRAIN_SEQ_LEN", train_seq_len)
    print("TEST_SEQ_LEN", opts.test_seq_len)

    if opts.dataset_type == "embed2d":
        data = Embed2DQuadratureData(
            min_note=opts.min_note,
            max_note=opts.max_note,
            sample_rate_khz=opts.sample_rate_khz,
            harsh=opts.harsh,
            seed=opts.seed,
        )
        train_ds = data.tf_dataset(
            batch_size=opts.batch_size,
            seq_len=train_seq_len,
            num_samples=opts.num_train_samples,
            emit_endpt_samples=True,
            emit_interpolated_samples=True,
        )
        validate_ds = data.tf_dataset(
            batch_size=opts.batch_size,
            seq_len=opts.test_seq_len,
            num_samples=opts.num_validate_samples,
            emit_endpt_samples=True,
            emit_interpolated_samples=True,
        )
    elif opts.dataset_type == "pcapture":
        data = ParametricCaptureStaticData(
            capture_run=opts.capture_run,
            keras_model=opts.keras_model,
            seed=123,
        )
        train_ds = data.tf_training_dataset(
            seq_len=train_seq_len,
            num_batches=opts.num_train_samples // opts.batch_size,
            batch_size=opts.batch_size,
            emit_weights=True,
            deterministic=False,
        )
        validate_ds = data.tf_training_dataset(
            seq_len=opts.test_seq_len,
            num_batches=opts.num_train_samples // opts.batch_size,
            batch_size=opts.batch_size,
            emit_weights=False,
            deterministic=True,
        )
    else:
        raise Exception("unknown --dataset-type")

    # make model
    model_config = {
        "in_d": data.in_d(),
        "rff": {
            "num_features": opts.num_fourier_features,
            "scale_min": opts.rff_scale_min,
            "scale_max": opts.rff_scale_max,
            "seed": opts.rff_seed,
            "basis": opts.rff_basis,
        },
        "mlp_dims": opts.mlp_dims,
        "mlp_activation": opts.mlp_activation,
        "out_d": data.out_d(),
        "rff_l1": opts.rff_l1,
        "film_layers": opts.film_layers,
    }
    print("model_config", model_config)
    with open(run_path / "model_config.json", "w") as f:
        json.dump(model_config, f)
    train_model = create_rff_inr_model(**model_config)

    if opts.lambda_morph_consistency > 0.0:
        train_model.enable_morph_consistency(opts.lambda_morph_consistency)

    train_model.summary()
    with open(run_path / "train_model_summary.txt", "w") as f:
        train_model.summary(print_fn=lambda line: f.write(line + "\n"))

    ramp_callback, beta_stft = setup_beta_stft_var_and_update_callback(
        opts.epochs, opts.beta_stft_warmup, opts.beta_stft_ramp, opts.beta_stft
    )

    callbacks = []

    # log beta_stft (and lr) into the logs before the TensorBoard callback
    callbacks.append(LogLrAndBetaStft(beta_stft_var=beta_stft))
    callbacks.append(
        PrintRffMlp0Weights(num_features=opts.num_fourier_features, freq=10)
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_dir / "{epoch:03d}.weights.h5"),
            save_weights_only=True,
        )
    )
    callbacks.append(CheckYPred(tb_dir=str(tensorboard_dir), dataset=validate_ds))
    if ramp_callback is not None:
        callbacks.append(ramp_callback)

    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)))

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

    combined_loss_fn, mse_metric, huber_metric, stft_metric, slope_metric, dc_metric = (
        combined_loss_terms(
            alpha_mse=opts.alpha_mse,
            alpha_huber=opts.alpha_huber,
            beta_stft=beta_stft,
            reduce_mean=False,
            seq_len=train_seq_len,
            stft_fft_sizes=halving_triple(opts.base_stft_fft_size),
            stft_win_lengths=halving_triple(opts.base_stft_win_length),
            stft_hop_sizes=stft_hop_sizes,
            gamma_slope=opts.gamma_slope,
            delta_dc=opts.delta_dc,
        )
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

    optimizer = AdamW(lr_schedule, weight_decay=opts.weight_decay)
    train_model.compile(
        optimizer,
        loss=combined_loss_fn,
        metrics=[mse_metric, huber_metric, stft_metric, slope_metric, dc_metric],
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
    parser.add_argument(
        "--dataset-type",
        choices=["embed2d", "pcapture"],
        help="dataset type",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-train-samples", type=int, default=10_000)
    parser.add_argument("--num-validate-samples", type=int, default=100)
    parser.add_argument(
        "--train-seq-mult",
        type=float,
        default=10,
        help="set training seqlen to base-stft-win-len * this",
    )
    parser.add_argument(
        "--test-seq-len",
        type=int,
        default=2048,
        help="for graphs etc",
    )
    parser.add_argument(
        "--num-fourier-features",
        type=int,
        default=64,
        help="number of Random Fourier Features (output dim is 2x this)",
    )
    parser.add_argument(
        "--rff-basis",
        choices=["gaussian", "harmonic"],
        default="gaussian",
        help="frequency basis for rff. 'gaussian' => random B and"
        " 'harmonic' => all int harmonics 1..num-fourier-features",
    )
    parser.add_argument(
        "--rff-l1",
        type=float,
        default=0.0,
        help="L1 weight on a per-frequency RFF gate for feature selection",
    )
    parser.add_argument(
        "--rff-scale-min",
        type=float,
        default=5.0,
        help="minimum gaussian std (sigma) for the fixed RFF frequency matrix B when basis is 'guassian'",
    )
    parser.add_argument(
        "--rff-scale-max",
        type=float,
        default=5.0,
        help="maximum gaussian std (sigma) for the fixed RFF frequency matrix B when basis is 'guassian'",
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
        "--mlp-activation",
        type=str,
        default="relu",
        help="activation function for mlp layer",
    )
    parser.add_argument(
        "--film-layers",
        type=int,
        default=1,
        help="set the first N MLP layers to use FiLM conditioning",
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
        "--gamma-slope",
        type=float,
        default=0.0,
        help="weight for first-difference (slope) L1 loss",
    )
    parser.add_argument(
        "--delta-dc",
        type=float,
        default=0.0,
        help="weight for dc/mean-offset loss",
    )
    parser.add_argument(
        "--lambda-morph-consistency",
        type=float,
        default=0.0,
        help="MSE weight for the morph-consistency ( a/b & -morph swapped should be equal )",
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

    embed_2d_data_args = parser.add_argument_group("Embed2DQuadratureData")
    Embed2DQuadratureData.add_args(embed_2d_data_args)

    pcapture_data_args = parser.add_argument_group("ParametricCaptureStaticData")
    ParametricCaptureStaticData.add_args(pcapture_data_args)

    return parser


if __name__ == "__main__":
    print("tf", tf.__version__)
    print("tf devices", tf.config.list_physical_devices())
    train(build_parser().parse_args())
