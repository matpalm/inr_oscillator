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
from amaranth_v import NNQ
from amaranth_v.dense_layer import QDenseLayer


def golden(
    x_codes,
    w_codes,
    b_codes,
    acc_shape,
    out_shape,
    apply_relu,
    relu_upper_bound,
    prod_shift=0,
):
    """Integer-exact reference for QDenseLayer over a batch of inputs.

    Args:
        x_codes  (N, IN_D)     in_shape raw integer codes
        w_codes  (IN_D, OUT_D) NNQ raw integer codes
        b_codes  (OUT_D,)      acc_shape raw integer codes
        prod_shift             left-shift applied to each a*b product to align
                               it with a wider accumulator (see QDenseLayer)
    Returns:
        (N, OUT_D) out_shape raw integer codes
    """

    frac_drop = acc_shape.f_bits - out_shape.f_bits
    out_width = out_shape.width

    # clamp=True: when out_shape's integer range is wider than the accumulator
    # (e.g. the io regression shape), saturate the bounds to acc_shape -- the
    # accumulator can never exceed its own range, so the clamp is a no-op.
    lower = fixed.Const(out_shape.min().as_float(), shape=acc_shape, clamp=True)._value
    upper = fixed.Const(out_shape.max().as_float(), shape=acc_shape, clamp=True)._value
    if apply_relu:
        relu_ub = fixed.Const(relu_upper_bound, shape=acc_shape)._value

    n, in_d = x_codes.shape
    out_d = w_codes.shape[1]
    result = np.zeros((n, out_d), dtype=np.int64)

    for row in range(n):
        for o in range(out_d):
            acc = int(b_codes[o])
            for i in range(in_d):
                acc += (int(x_codes[row, i]) * int(w_codes[i, o])) << prod_shift

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

            # truncate toward zero while narrowing acc_shape -> out_shape.
            # operate on the signed accumulator and arithmetic-shift so the sign
            # is extended across out_width even when out_shape is wider than the
            # accumulator (the io regression shape); this matches the hardware's
            # signed slice trunc_toward_zero[frac_drop:frac_drop+out_width].
            frac_nonzero = (acc & ((1 << frac_drop) - 1)) != 0
            if acc < 0 and frac_nonzero:
                trunc = acc + (1 << frac_drop)
            else:
                trunc = acc
            sliced = (trunc >> frac_drop) & ((1 << out_width) - 1)
            if sliced >= (1 << (out_width - 1)):
                sliced -= 1 << out_width
            result[row, o] = sliced

    return result


def simulate(dut, x_codes):
    """Drive the Amaranth QDenseLayer over ``x_codes`` and collect output codes."""
    in_d = dut.in_d
    out_d = dut.out_d
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
    in_d = 6
    out_d = 4
    rows = 64
    seed = 0

    def _check(self, apply_relu, relu_upper_bound, out_shape, in_shape=NNQ, n_lanes=1):
        rng = np.random.default_rng(self.seed)
        n_frac = NNQ.f_bits
        in_frac = in_shape.f_bits
        acc_f = in_frac + n_frac

        # exactly representable weights (NNQ), inputs (in_shape), bias (acc scale)
        w_lo, w_hi = NNQ.min().as_float(), NNQ.max().as_float()
        in_lo, in_hi = in_shape.min().as_float(), in_shape.max().as_float()
        kernel = (
            np.round(rng.uniform(w_lo, w_hi, (self.in_d, self.out_d)) * 2**n_frac)
            / 2**n_frac
        )
        inputs = (
            np.round(rng.uniform(in_lo, in_hi, (self.rows, self.in_d)) * 2**in_frac)
            / 2**in_frac
        )
        bias = np.round(rng.uniform(w_lo, w_hi, (self.out_d,)) * 2**acc_f) / 2**acc_f

        w_codes = np.round(kernel * 2**n_frac).astype(np.int64)
        x_codes = np.round(inputs * 2**in_frac).astype(np.int64)
        b_codes = np.round(bias * 2**acc_f).astype(np.int64)

        dut = QDenseLayer(
            kernel.astype(np.float64),
            bias.astype(np.float64),
            apply_relu=apply_relu,
            relu_upper_bound=relu_upper_bound,
            in_shape=in_shape,
            out_shape=out_shape,
            n_lanes=n_lanes,
        )
        hw = simulate(dut, x_codes)
        ref = golden(
            x_codes,
            w_codes,
            b_codes,
            dut.acc_shape,
            out_shape,
            apply_relu,
            relu_upper_bound,
        )

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

    def test_mlp0_io_input(self):
        # first MLP layer: io-format input (SQ(2,14)) -> NNQ output with relu
        self._check(
            apply_relu=True,
            relu_upper_bound=8.0,
            out_shape=NNQ,
            in_shape=fixed.SQ(2, 14),
        )

    def test_parallel_full(self):
        # fully parallel: one MAC lane per output column (out_d == n_lanes).
        self._check(apply_relu=True, relu_upper_bound=4.0, out_shape=NNQ, n_lanes=4)

    def test_parallel_partial(self):
        # partial parallelism: 2 lanes over out_d=4 -> 2 groups, exact divisor.
        self._check(apply_relu=True, relu_upper_bound=4.0, out_shape=NNQ, n_lanes=2)

    def test_parallel_nondivisor(self):
        # non-divisor lane count: 3 lanes over out_d=4 -> 2 groups with a padded
        # (discarded) column; exercises the padding / result-scatter guard.
        self._check(apply_relu=True, relu_upper_bound=4.0, out_shape=NNQ, n_lanes=3)

    def test_parallel_regression_nondivisor(self):
        # no-relu io-narrowing regression path with a non-divisor lane count.
        self._check(
            apply_relu=False,
            relu_upper_bound=None,
            out_shape=fixed.SQ(1, 15),
            n_lanes=3,
        )


if __name__ == "__main__":
    unittest.main()
