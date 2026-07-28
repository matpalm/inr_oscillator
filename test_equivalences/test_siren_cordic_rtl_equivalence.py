"""Bit-exact check of amaranth_v.siren_cordic.SirenCordic RTL vs reference.

Run:
    uv run -m unittest test_equivalences.test_siren_cordic_rtl_equivalence

Set SIREN_EQ_FAST=1 for a faster sampled test.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from amaranth import Module
from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_v import NNQ
from amaranth_v.siren_cordic import SirenCordic, siren_cordic_output_codes


def simulate_siren_cordic(dut: SirenCordic, input_codes: np.ndarray) -> np.ndarray:
    out = []
    max_wait = 20000

    m = Module()
    m.submodules.dut = dut

    async def tb(ctx):
        ctx.set(dut.o.ready, 1)
        ctx.set(dut.i.valid, 0)

        for code in input_codes.tolist():
            code = int(code)
            ctx.set(dut.i.payload, code)
            ctx.set(dut.i.valid, 1)
            for _ in range(max_wait):
                if ctx.get(dut.i.ready):
                    await ctx.tick()
                    break
                await ctx.tick()
            else:
                raise AssertionError("timeout waiting for dut.i.ready")

            ctx.set(dut.i.valid, 0)
            for _ in range(max_wait):
                if ctx.get(dut.o.valid):
                    out.append(int(ctx.get(dut.o.payload)))
                    await ctx.tick()
                    break
                await ctx.tick()
            else:
                raise AssertionError("timeout waiting for dut.o.valid")

    sim = Simulator(m)
    sim.add_clock(1e-6)
    sim.add_testbench(tb)
    sim.run()
    return np.asarray(out, dtype=np.int64)


class TestSirenCordicRTLEquivalence(unittest.TestCase):
    OMEGA_0 = 30.0

    def test_rtl_matches_reference_codes(self):
        width = NNQ.width
        frac = NNQ.f_bits
        lo = -(1 << (width - 1))
        hi = (1 << (width - 1)) - 1

        if os.getenv("SIREN_EQ_FAST") == "1":
            rng = np.random.default_rng(0)
            inputs = rng.integers(lo, hi + 1, size=512, dtype=np.int64)
            edges = np.asarray([lo, lo + 1, -1, 0, 1, hi - 1, hi], dtype=np.int64)
            inputs = np.unique(np.concatenate([inputs, edges]))
        else:
            inputs = np.arange(lo, hi + 1, dtype=np.int64)

        ref = np.asarray(
            siren_cordic_output_codes(
                [int(c) for c in inputs.tolist()],
                width=width,
                frac_bits=frac,
                omega_0=self.OMEGA_0,
            ),
            dtype=np.int64,
        )

        rtl = simulate_siren_cordic(SirenCordic(self.OMEGA_0), inputs)

        self.assertEqual(ref.shape, rtl.shape)
        if not np.array_equal(ref, rtl):
            diff = ref != rtl
            i = int(np.argwhere(diff)[0][0])
            self.fail(
                f"{int(diff.sum())} mismatches; first at input_code={int(inputs[i])}: "
                f"ref={int(ref[i])} rtl={int(rtl[i])}"
            )


if __name__ == "__main__":
    unittest.main()
