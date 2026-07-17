from itertools import product

from tqdm import tqdm

from keras_v.train import build_parser, train

hyperparams = {
    "batch_size": [128],
    "num_fourier_features": [32],
    "rff_scale": [3.0, 2.5, 2.0],
    "mlp_layers": [2, 3],
    "mlp_width": [16],
    "alpha_mse": [0.0, 1.0],
    "alpha_huber": [0.0, 1.0],
    "beta_stft": [0.001, 0.0001, 0.00001],
}
run_configs = [
    dict(zip(hyperparams.keys(), values)) for values in product(*hyperparams.values())
]


def ignore(run_config):
    if run_config["alpha_mse"] == run_config["alpha_huber"] == 0:
        return True
    if run_config["alpha_mse"] == run_config["alpha_huber"] == 1:
        return True
    return False


run_id = 63
for run_config in tqdm(run_configs):

    if ignore(run_config):
        continue

    print("run_id", run_id, "run_config", run_config)

    print("-" * 100)
    print("run", run_id)

    opts = build_parser().parse_args(["--run", f"{run_id:03d}"])
    opts.harsh = True
    for key, value in run_config.items():
        setattr(opts, key, value)
    opts.num_train_samples = 200_000
    opts.epochs = 1  # just interested in final result

    train(opts)
    run_id += 1

print("final run_id", run_id)
