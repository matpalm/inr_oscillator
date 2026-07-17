from pathlib import Path
import pickle
import numpy as np

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--weights-pkl", type=Path, required=True)
opts = parser.parse_args()
print("opts", opts)


def ntiles(a, n=11, d=2):
    p = np.percentile(a, np.linspace(0, 100, n))
    return np.around(p, decimals=d)


with open(opts.weights_pkl, "rb") as f:
    weights = pickle.load(f)

ws = []
bs = []
for layer in weights.keys():
    if "weights" not in weights[layer]:
        # e.g. the "rff" entry, which stores the fixed frequency matrix, not (w, b)
        continue
    w, b = weights[layer]["weights"]
    ws.append(w.flatten())
    bs.append(b.flatten())
    print(f"{layer:10s} w {str(w.shape):10s} {ntiles(w)}")
    print(f"{layer:10s} b {str(b.shape):10s} {ntiles(b)}")
ws = np.concatenate(ws)
bs = np.concatenate(bs)

print(f"OVERALL w {ntiles(ws)} total {len(ws)} non_zero {np.count_nonzero(ws)}")
print(f"OVERALL b {ntiles(bs)} total {len(bs)} non_zero {np.count_nonzero(bs)}")
