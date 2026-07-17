"""Bit-exact equivalence check: qkeras_v RFF golden model vs amaranth_v hardware.

Loads a pickled set of qkeras quantised weights (which now carries the fixed,
quantised RFF ``B`` matrix and its fixed-point formats), builds the shared cos/sin
LUT, then:

  * computes the golden RFF integer codes in numpy (``qkeras_v.rff_lut``), and
  * simulates the Amaranth ``RandomFourierFeaturesLUT`` fed the identical LUT/B,

and asserts the two match bit-for-bit over a sweep of phase inputs.

Run from the repo root, e.g.::

    uv run -m test_equivalence.rff_equivalence --weights-pkl runs/077/weights/qkeras/latest.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from amaranth.sim import Simulator

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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights-pkl", type=Path, required=True)
    parser.add_argument(
        "--lut-size", type=int, default=1024, help="cos/sin ROM depth (power of two)"
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=257,
        help="number of phase samples swept over [-1, 1)",
    )
    opts = parser.parse_args()
    print("opts", opts)

    with open(opts.weights_pkl, "rb") as f:
        weights = pickle.load(f)

    if "rff" not in weights:
        raise SystemExit(
            "pickle has no 'rff' entry; retrain with the updated qkeras_v.train "
            "so SaveQuantisedWeights stores the RFF B matrix"
        )
    rff = weights["rff"]
    b_bits, b_integer = int(rff["b_bits"]), int(rff["b_integer"])
    io_bits, io_integer = int(rff["io_bits"]), int(rff["io_integer"])

    # RFF input is the scalar phase (in_dim == 1); B is (in_dim, num_features).
    B = np.asarray(rff["B"]).reshape(-1)
    b_f = frac_bits(b_bits, b_integer)
    b_codes = np.round(B * (2.0**b_f)).astype(np.int64)

    shift, lut_bits = plan_shift(io_bits, io_integer, b_bits, b_integer, opts.lut_size)
    if shift < 0:
        raise SystemExit(
            f"--lut-size {opts.lut_size} too large for io_frac+b_frac="
            f"{frac_bits(io_bits, io_integer) + b_f}; reduce it"
        )
    cos_lut, sin_lut = build_io_luts(opts.lut_size, io_bits, io_integer)

    phases = np.linspace(-1.0, 1.0, opts.num_phases, endpoint=False)
    phase_codes = quantise_to_codes(phases, io_bits, io_integer)

    golden = rff_lut_features(phase_codes, b_codes, cos_lut, sin_lut, shift, lut_bits)

    dut = RandomFourierFeaturesLUT(
        b_codes=b_codes.tolist(),
        cos_lut=cos_lut.tolist(),
        sin_lut=sin_lut.tolist(),
        io_bits=io_bits,
        b_bits=b_bits,
        shift=shift,
    )
    hw = simulate(dut, phase_codes)

    print(
        f"features={len(b_codes)} io=Q{io_integer}.{frac_bits(io_bits, io_integer)} "
        f"b=Q{b_integer}.{b_f} lut_size={opts.lut_size} shift={shift} phases={len(phase_codes)}"
    )

    if golden.shape != hw.shape:
        raise SystemExit(f"shape mismatch: golden {golden.shape} vs hw {hw.shape}")

    if np.array_equal(golden, hw):
        print(
            f"PASS: bit-exact over {golden.shape[0]} phases x {golden.shape[1]} outputs"
        )
        return 0

    diff = golden != hw
    n_bad = int(diff.sum())
    max_abs = int(np.abs(golden - hw).max())
    first = np.argwhere(diff)[0]
    print(f"FAIL: {n_bad} mismatched codes (max abs code diff {max_abs})")
    print(
        f"  first mismatch at phase idx {first[0]}, output idx {first[1]}: "
        f"golden={golden[tuple(first)]} hw={hw[tuple(first)]}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
