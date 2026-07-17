"""Equivalence check: numpy golden vs amaranth_v QDenseLayer hardware.

Replicates the exact fixed-point datapath of ``amaranth_v.dense_layer.QDenseLayer``
(bias-seeded double-width MAC, then clamp -> optional relu -> re-clip -> truncate
toward zero while narrowing NNQ_DW -> out_shape) in numpy, then simulates the
Amaranth layer over random (exactly representable) inputs and asserts the two
match bit-for-bit. Exercises BOTH the relu MLP config (out_shape=NNQ) and the
no-relu regression config (out_shape=io).

Run from the repo root::

    uv run -m unittest test_equivalences.test_dense_equivalence
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_future import fixed
from amaranth_v import NNQ, NNQ_DW
from amaranth_v.dense_layer import QDenseLayer


def golden(x_codes, w_codes, b_codes, out_shape, apply_relu, relu_upper_bound):
    """Integer-exact reference for QDenseLayer over a batch of inputs.

    Args:
        x_codes  (N, IN_D)   NNQ raw integer codes
        w_codes  (IN_D, OUT_D) NNQ raw integer codes
        b_codes  (OUT_D,)    NNQ_DW raw integer codes
    Returns:
        (N, OUT_D) out_shape raw integer codes
    """
    W = NNQ_DW.width
    dw_mask = (1 << W) - 1
    frac_drop = NNQ_DW.f_bits - out_shape.f_bits
    out_width = out_shape.width

    lower = fixed.Const(out_shape.min().as_float(), shape=NNQ_DW)._value
    upper = fixed.Const(out_shape.max().as_float(), shape=NNQ_DW)._value
    if apply_relu:
        relu_ub = fixed.Const(relu_upper_bound, shape=NNQ_DW)._value

    n, in_d = x_codes.shape
    out_d = w_codes.shape[1]
    result = np.zeros((n, out_d), dtype=np.int64)

    for row in range(n):
        for o in range(out_d):
            acc = int(b_codes[o])
            for i in range(in_d):
                acc += int(x_codes[row, i]) * int(w_codes[i, o])

            # clamp
            acc = min(max(acc, lower), upper)

            # optional relu
            if apply_relu:
                if acc < 0:
                    acc = 0
                elif acc > relu_ub:
                    acc = relu_ub

            # re-clip
            acc = min(max(acc, lower), upper)

            # truncate toward zero while narrowing NNQ_DW -> out_shape
            u = acc & dw_mask
            sign_bit = (u >> (W - 1)) & 1
            frac_nonzero = (u & ((1 << frac_drop) - 1)) != 0
            if sign_bit and frac_nonzero:
                u = (u + (1 << frac_drop)) & dw_mask
            sliced = (u >> frac_drop) & ((1 << out_width) - 1)
            if sliced >= (1 << (out_width - 1)):
                sliced -= 1 << out_width
            result[row, o] = sliced

    return result


def simulate(dut, x_codes):
    """Drive the Amaranth QDenseLayer over ``x_codes`` and collect output codes."""
    in_d = dut.IN_D
    out_d = dut.OUT_D
    results = []

    async def testbench(ctx):
        ctx.set(dut.o.ready, 1)
        for row in x_codes:
            # set/get the raw underlying value so integer codes pass through
            # directly (a fixed payload interprets a bare int as a real value).
            for i in range(in_d):
                ctx.set(dut.i.payload[i].as_value(), int(row[i]))
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)
            while not ctx.get(dut.o.valid):
                await ctx.tick()
            results.append([ctx.get(dut.o.payload[j].as_value()) for j in range(out_d)])
            await ctx.tick()  # o.ready high -> handshake completes, back to IDLE

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
    return np.array(results, dtype=np.int64)


class TestDenseEquivalence(unittest.TestCase):
    IN_D = 6
    OUT_D = 4
    ROWS = 64
    SEED = 0

    def _check(self, apply_relu, relu_upper_bound, out_shape):
        rng = np.random.default_rng(self.SEED)
        n_frac = NNQ.f_bits
        dw_frac = NNQ_DW.f_bits

        # exactly representable NNQ kernel / inputs, NNQ_DW bias
        lo, hi = NNQ.min().as_float(), NNQ.max().as_float()
        kernel = (
            np.round(rng.uniform(lo, hi, (self.IN_D, self.OUT_D)) * 2**n_frac)
            / 2**n_frac
        )
        inputs = (
            np.round(rng.uniform(-1.0, 1.0, (self.ROWS, self.IN_D)) * 2**n_frac)
            / 2**n_frac
        )
        bias = np.round(rng.uniform(lo, hi, (self.OUT_D,)) * 2**dw_frac) / 2**dw_frac

        w_codes = np.round(kernel * 2**n_frac).astype(np.int64)
        x_codes = np.round(inputs * 2**n_frac).astype(np.int64)
        b_codes = np.round(bias * 2**dw_frac).astype(np.int64)

        dut = QDenseLayer(
            kernel.astype(np.float64),
            bias.astype(np.float64),
            apply_relu=apply_relu,
            relu_upper_bound=relu_upper_bound,
            out_shape=out_shape,
        )
        hw = simulate(dut, x_codes)
        ref = golden(x_codes, w_codes, b_codes, out_shape, apply_relu, relu_upper_bound)

        self.assertEqual(hw.shape, ref.shape)
        if not np.array_equal(hw, ref):
            diff = hw != ref
            first = tuple(np.argwhere(diff)[0])
            self.fail(
                f"{int(diff.sum())} mismatches; first at row {first[0]} out {first[1]}: "
                f"golden={ref[first]} hw={hw[first]}"
            )

    def test_mlp_relu(self):
        # relu MLP config: narrow to NNQ with relu applied
        self._check(apply_relu=True, relu_upper_bound=4.0, out_shape=NNQ)

    def test_regression_no_relu(self):
        # no-relu regression config: narrow to the io fixed-point shape
        self._check(apply_relu=False, relu_upper_bound=None, out_shape=fixed.SQ(1, 15))


if __name__ == "__main__":
    unittest.main()
