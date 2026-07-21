from itertools import product

from tqdm import tqdm

from qkeras_v.train import build_parser, train

# hyperparams = {
#     "num_fourier_features": [64, 128, 256],
#     "rff_scale": [2.0, 1.0],
#     "mlp_layers": [2, 3],
#     "mlp_dim": [16],
#     "beta_stft": [0.001, 0.0001],
# }
# run_configs = [
#     dict(zip(hyperparams.keys(), values)) for values in product(*hyperparams.values())
# ]


# def ignore(run_config):
#     # if run_config["alpha_mse"] == run_config["alpha_huber"] == 0:
#     #     return True
#     # if run_config["alpha_mse"] == run_config["alpha_huber"] == 1:
#     #     return True
#     return False

run_configs = [
    (129, {"beta_stft": 0.1, "rff_scale": 1.0}),
    (130, {"beta_stft": 0.1, "rff_scale": 0.1}),
    # baseline "beta_stft": 0.01
    (131, {"beta_stft": 0.001, "rff_scale": 1.0}),
    (132, {"beta_stft": 0.001, "rff_scale": 0.1}),
]

for run_id, run_config in run_configs:

    print("-" * 100)
    print("run_id", run_id, "run_config", run_config)

    opts = build_parser().parse_args(["--run", f"{run_id:03d}"])

    for key, value in run_config.items():
        setattr(opts, key, value)

    opts.dataset_type = "pcapture"
    opts.capture_run = "600"
    opts.keras_model = "232_keras/i9"
    opts.io_fp_int = 1
    opts.io_fp_frac = 15
    opts.num_fourier_features = 80
    # opts.rff_scale = 1.0
    opts.rff_lut_size = 4096
    opts.mlp_fp_int = 3
    opts.mlp_fp_frac = 13
    opts.relu_upper_bound = 8
    opts.mlp_dims = [24, 24, 24]
    opts.alpha_mse = 1  # try 0.001
    opts.alpha_huber = 0
    opts.beta_stft_warmup = opts.beta_stft_ramp = 5
    # opts.beta_stft = 0.01
    opts.epochs = 30
    opts.learning_rate = 1e-3
    opts.cosine_schedule = True
    opts.num_train_samples = 50_000
    opts.batch_size = 64

    train(opts)
