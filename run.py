from itertools import product

from tqdm import tqdm

from keras_v.train import build_parser, train

hyperparams = {
    "num_fourier_features": [32, 64],
    "rff_scale": [8.0, 4.0, 2.0],
    "mlp_layers": [2],
    "mlp_width": [16],
}
run_configs = [
    dict(zip(hyperparams.keys(), values)) for values in product(*hyperparams.values())
]


def ignore(run_config):
    # if run_config["alpha_mse"] == run_config["alpha_huber"] == 0:
    #     return True
    # if run_config["alpha_mse"] == run_config["alpha_huber"] == 1:
    #     return True
    return False


run_id = 77
for run_config in tqdm(run_configs):

    if ignore(run_config):
        continue

    print("run_id", run_id, "run_config", run_config)

    print("-" * 100)
    print("run", run_id)

    opts = build_parser().parse_args(["--run", f"{run_id:03d}"])

    for key, value in run_config.items():
        setattr(opts, key, value)

    opts.batch_size = 128
    opts.harsh = True
    opts.num_train_samples = 10_000

    opts.base_stft_fft_size = 2048
    opts.base_stft_win_length = 256

    opts.alpha_mse = 0.9
    opts.alpha_huber = 0.1
    opts.beta_stft = 0.001
    opts.beta_stft_warmup = 5
    opts.beta_stft_ramp = 5
    opts.epochs = 20

    train(opts)
    run_id += 1

print("final run_id", run_id)
