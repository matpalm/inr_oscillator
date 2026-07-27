"""RFF LUT golden-model sanity checks.

Loads a pickled set of qkeras quantised weights (the quantised RFF ``B`` matrix)
along with the sibling ``qkeras_model.layer_info.json`` (which carries the RFF
fixed-point formats) via ``load_weights``, builds the shared cos/sin LUT, and
checks basic consistency of the integer-code golden model used by the build path.

Run from the repo root, e.g.::

    uv run -m unittest test_equivalences.test_rff_equivalence

By default it uses the newest ``runs/*/weights/qkeras/latest.pkl``; override with
the ``RFF_WEIGHTS_PKL`` environment variable.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_v.rff_film_network import load_weights_and_config
from qkeras_v.rff_lut import (
    build_io_luts,
    frac_bits,
    plan_shift,
    quantise_to_codes,
    rff_lut_features,
)


def _find_weights_pkl():
    """Resolve the qkeras weights pickle: RFF_WEIGHTS_PKL env, else newest run."""
    env = os.getenv("RFF_WEIGHTS_PKL")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("runs/*/weights/qkeras/latest.pkl"))
    return candidates[-1] if candidates else None


class TestRffEquivalence(unittest.TestCase):
    NUM_PHASES = 257

    def setUp(self):
        pkl = _find_weights_pkl()
        if pkl is None or not pkl.exists():
            self.skipTest("no qkeras weights pickle found (set RFF_WEIGHTS_PKL)")
        weights, self.quant_sizes, model_config = load_weights_and_config(pkl)
        if "rff" not in weights:
            self.skipTest(
                f"{pkl} has no 'rff' entry; retrain with the updated qkeras_v.train"
            )
        self.rff_w = weights["rff"]
        self.lut_size = model_config["rff"]["lut_size"]

    def test_rff_lut_golden_sanity(self):

        b_bits, b_integer = self.quant_sizes["b_bits"], self.quant_sizes["b_int"]
        io_bits, io_integer = self.quant_sizes["io_bits"], self.quant_sizes["io_int"]

        # RFF input is the scalar phase (in_dim == 1); B is (in_dim, num_features).
        B = np.asarray(self.rff_w["B"]).reshape(-1)
        b_f = frac_bits(b_bits, b_integer)
        b_codes = np.round(B * (2.0**b_f)).astype(np.int64)

        shift, lut_bits = plan_shift(
            io_bits, io_integer, b_bits, b_integer, self.lut_size
        )
        self.assertGreaterEqual(
            shift,
            0,
            f"lut_size {self.lut_size} too large for io_frac+b_frac="
            f"{frac_bits(io_bits, io_integer) + b_f}",
        )
        cos_lut, sin_lut = build_io_luts(self.lut_size, io_bits, io_integer)

        phases = np.linspace(-1.0, 1.0, self.NUM_PHASES, endpoint=False)
        phase_codes = quantise_to_codes(phases, io_bits, io_integer)

        ref = rff_lut_features(phase_codes, b_codes, cos_lut, sin_lut, shift, lut_bits)
        self.assertEqual(ref.shape, (self.NUM_PHASES, 2 * len(b_codes)))
        # Determinism guard: recomputing with identical inputs must match exactly.
        ref2 = rff_lut_features(phase_codes, b_codes, cos_lut, sin_lut, shift, lut_bits)
        self.assertTrue(np.array_equal(ref, ref2))


if __name__ == "__main__":
    unittest.main()
