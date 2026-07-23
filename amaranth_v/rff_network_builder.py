import pickle

from amaranth_v import rff_film_network, rff_concat_network

# TODO: the need for this class is clearly dumb :/


def is_film_network(weights_pkl):
    with open(weights_pkl, "rb") as f:
        weights = pickle.load(f)
    for k in weights.keys():
        if "film" in k:
            return True
    return False


def load_and_build_network(weights_pkl):
    if is_film_network(weights_pkl):
        weights, quant_sizes, model_config = rff_film_network.load_weights_and_config(
            weights_pkl
        )
        return rff_film_network.RffNetwork(weights, quant_sizes, model_config)
    else:
        weights, quant_sizes, model_config = rff_concat_network.load_weights_and_config(
            weights_pkl
        )
        return rff_concat_network.RffNetwork(weights, quant_sizes, model_config)


# WEIGHTS_PKL = os.getenv("WEIGHTS_PKL")
# if not WEIGHTS_PKL or not os.path.exists(WEIGHTS_PKL):
#     raise Exception(f"failed to load weights for WEIGHTS_PKL=[{WEIGHTS_PKL}]")
# print(f"loading weights from {WEIGHTS_PKL}")
# weights, quant_sizes, model_config = load_weights_and_config(WEIGHTS_PKL)
# self.net = RffNetwork(weights, quant_sizes, model_config)
