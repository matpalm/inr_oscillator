"""Bit-exact equivalence check: qkeras_v RFF golden model vs amaranth_v hardware.

Loads a pickled set of qkeras quantised weights (which now carries the fixed,
quantised RFF ``B`` matrix and its fixed-point formats), builds the shared cos/sin
LUT, then:

  * computes the golden RFF integer codes in numpy (``qkeras_v.rff_lut``), and
  * simulates the Amaranth ``RandomFourierFeaturesLUT`` fed the identical LUT/B,

and asserts the two match bit-for-bit over a sweep of phase inputs.

Run from the repo root, e.g.::

    uv run -m unittest test_equivalences.test_rff_equivalence

By default it uses the newest ``runs/*/weights/qkeras/latest.pkl``; override with
the ``RFF_WEIGHTS_PKL`` environment variable.
"""

import os
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_v.rff import RandomFourierFeaturesLUT
from qkeras_v.rff_lut import (
    build_io_luts,
    frac_bits,
    plan_shift,
    quantise_to_codes,
    rff_lut_features,
)


def simulate(dut, phase_codes):
    """Drive the Amaranth RFF over ``phase_codes`` and collect output integer codes."""
    n_out = 2 * dut._num_features
    results = []

    async def testbench(ctx):
        ctx.set(dut.o.ready, 1)
        for p in phase_codes:
            ctx.set(dut.i.payload, int(p))
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)
            while not ctx.get(dut.o.valid):
                await ctx.tick()
            row = [ctx.get(dut.o.payload[j]) for j in range(n_out)]
            results.append(row)
            await ctx.tick()  # o.ready is high -> handshake completes, back to IDLE

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
    return np.array(results, dtype=np.int64)


def _find_weights_pkl():
    """Resolve the qkeras weights pickle: RFF_WEIGHTS_PKL env, else newest run."""
    env = os.getenv("RFF_WEIGHTS_PKL")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("runs/*/weights/qkeras/latest.pkl"))
    return candidates[-1] if candidates else None


class TestRffEquivalence(unittest.TestCase):
    LUT_SIZE = 1024
    NUM_PHASES = 257

    def setUp(self):
        pkl = _find_weights_pkl()
        if pkl is None or not pkl.exists():
            self.skipTest("no qkeras weights pickle found (set RFF_WEIGHTS_PKL)")
        with open(pkl, "rb") as f:
            weights = pickle.load(f)
        if "rff" not in weights:
            self.skipTest(
                f"{pkl} has no 'rff' entry; retrain with the updated qkeras_v.train"
            )
        self.rff = weights["rff"]

    def test_rff_bit_exact(self):
        rff = self.rff
        b_bits, b_integer = int(rff["b_bits"]), int(rff["b_integer"])
        io_bits, io_integer = int(rff["io_bits"]), int(rff["io_integer"])

        # RFF input is the scalar phase (in_dim == 1); B is (in_dim, num_features).
        B = np.asarray(rff["B"]).reshape(-1)
        b_f = frac_bits(b_bits, b_integer)
        b_codes = np.round(B * (2.0**b_f)).astype(np.int64)

        shift, lut_bits = plan_shift(
            io_bits, io_integer, b_bits, b_integer, self.LUT_SIZE
        )
        self.assertGreaterEqual(
            shift,
            0,
            f"lut_size {self.LUT_SIZE} too large for io_frac+b_frac="
            f"{frac_bits(io_bits, io_integer) + b_f}",
        )
        cos_lut, sin_lut = build_io_luts(self.LUT_SIZE, io_bits, io_integer)

        phases = np.linspace(-1.0, 1.0, self.NUM_PHASES, endpoint=False)
        phase_codes = quantise_to_codes(phases, io_bits, io_integer)

        ref = rff_lut_features(phase_codes, b_codes, cos_lut, sin_lut, shift, lut_bits)

        dut = RandomFourierFeaturesLUT.from_rff(rff, lut_size=self.LUT_SIZE)
        hw = simulate(dut, phase_codes)

        self.assertEqual(ref.shape, hw.shape)
        if not np.array_equal(ref, hw):
            diff = ref != hw
            first = tuple(np.argwhere(diff)[0])
            self.fail(
                f"{int(diff.sum())} mismatched codes "
                f"(max abs {int(np.abs(ref - hw).max())}); "
                f"first at phase {first[0]} out {first[1]}: "
                f"golden={ref[first]} hw={hw[first]}"
            )


if __name__ == "__main__":
    unittest.main()
