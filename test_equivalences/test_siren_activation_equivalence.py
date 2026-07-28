"""qkeras siren activation vs amaranth CORDIC value equivalence.

Verifies that qkeras_v.NNQSineLUT emits the exact same NNQ-coded values as the
shared CORDIC reference used by amaranth_v.SirenCordic.

Run:
    uv run -m unittest test_equivalences.test_siren_activation_equivalence
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_v import NNQ
from amaranth_v.siren_cordic import siren_cordic_output_codes
from qkeras_v.qkeras_model import NNQSineLUT


class TestSirenActivationEquivalence(unittest.TestCase):
    OMEGA_0 = 30.0

    def test_nnq_sine_lut_matches_cordic_reference(self):
        width = NNQ.width
        frac = NNQ.f_bits
        scale = float(2**frac)
        lo = -(1 << (width - 1))
        hi = (1 << (width - 1)) - 1

        # Exhaustive by default for full confidence; set SIREN_EQ_FAST=1 to
        # sample a subset during quick local iteration.
        if os.getenv("SIREN_EQ_FAST") == "1":
            rng = np.random.default_rng(0)
            codes = rng.integers(lo, hi + 1, size=4096, dtype=np.int64)
            codes = np.unique(np.concatenate([codes, np.array([lo, -1, 0, 1, hi])]))
        else:
            codes = np.arange(lo, hi + 1, dtype=np.int64)

        x = (codes.astype(np.float32) / scale).reshape(1, -1, 1)
        layer = NNQSineLUT(
            n_word=width,
            n_int=NNQ.i_bits - 1,
            omega_0=self.OMEGA_0,
            name="qsiren_eq",
        )
        y_tf = layer(tf.constant(x, dtype=tf.float32)).numpy().reshape(-1)
        y_codes_qkeras = np.round(y_tf * scale).astype(np.int64)

        y_codes_ref = np.asarray(
            siren_cordic_output_codes(
                [int(c) for c in codes.tolist()],
                width=width,
                frac_bits=frac,
                omega_0=self.OMEGA_0,
            ),
            dtype=np.int64,
        )

        self.assertEqual(y_codes_qkeras.shape, y_codes_ref.shape)
        if not np.array_equal(y_codes_qkeras, y_codes_ref):
            diff = y_codes_qkeras != y_codes_ref
            idx = int(np.argwhere(diff)[0][0])
            self.fail(
                f"{int(diff.sum())} mismatched codes; first at input_code={int(codes[idx])}: "
                f"qkeras={int(y_codes_qkeras[idx])} ref={int(y_codes_ref[idx])}"
            )


if __name__ == "__main__":
    unittest.main()
