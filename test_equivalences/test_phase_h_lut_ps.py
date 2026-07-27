"""Bit-exact equivalence check for the PSRAM-backed phase->h table
(``amaranth_v.phase_h_lut_ps.PhaseHLutPS``).

In the FiLM topology the first dense layer's pre-activation depends only on the
scalar phase::

    h = mlp0(RFF(phase))

``PhaseHLutPS`` materialises this whole table in PSRAM once at startup (BUILD),
then serves inference as a pure PSRAM read. This test:

  1. builds a ``PhaseHLutPS`` (with a *small* ``index_bits`` so the build sweep
     is quick) from a real trained run, backed by a ``FakePSRAM``;
  2. waits for the startup build to complete (``ready``);
  3. reads back every phase index and compares to an integer-exact numpy golden
     (``rff_lut_features`` -> ``dense_golden`` with ``apply_relu=False``).

Run from the repo root::

    uv run -m unittest test_equivalences.test_phase_h_lut_ps

Uses the newest ``runs/*/weights/qkeras/latest.pkl`` by default; override with
``RFF_WEIGHTS_PKL``.
"""

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from amaranth import Module
from amaranth.lib import wiring
from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_v import NNQ
from amaranth_v.dense_layer import QDenseLayer
from amaranth_v.rff_film_network import RffNetwork, load_weights_and_config
from qkeras_v.rff_lut import (
    build_io_luts,
    frac_bits,
    plan_shift,
    rff_lut_features,
)
from test_equivalences.test_dense_equivalence import golden as dense_golden
from test_equivalences.test_activation_cache_ps_helpers import FakePSRAM


def _is_pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def build_h_table(net, index_bits):
    """Integer-exact numpy phase->h table (num_entries, out_d) of NNQ codes."""
    quant_sizes = net.quant_sizes
    io_bits, io_integer = quant_sizes["io_bits"], quant_sizes["io_int"]
    b_bits, b_integer = quant_sizes["b_bits"], quant_sizes["b_int"]

    B = np.asarray(net.qkeras_weights["rff"]["B"]).reshape(-1)
    b_codes = np.round(B * (2.0 ** frac_bits(b_bits, b_integer))).astype(np.int64)
    rff_shift, lut_bits = plan_shift(
        io_bits, io_integer, b_bits, b_integer, net.lut_size
    )
    cos_lut, sin_lut = build_io_luts(net.lut_size, io_bits, io_integer)

    io_frac = net.io_shape.f_bits
    index_shift = io_frac - (index_bits - 1)
    num_entries = 1 << index_bits

    # phase io code for each table index, matching the hardware build sweep:
    #   phase_code = signed(idx, index_bits) << index_shift
    idxs = np.arange(num_entries, dtype=np.int64)
    idx_signed = np.where(idxs >= (num_entries >> 1), idxs - num_entries, idxs)
    phase_codes = (idx_signed << index_shift).astype(np.int64)

    rff_feats = rff_lut_features(
        phase_codes, b_codes, cos_lut, sin_lut, rff_shift, lut_bits
    ).astype(np.int64)

    # mlp0: io -> NNQ, no relu (pre-FiLM pre-activation).
    w0, b0 = net.dense_weights_biases_for(net.mlp_names[0])
    acc_shape, prod_shift = QDenseLayer.acc_shape_for(net.io_shape, w0.shape[0], b0)
    w0_codes = np.round(w0 * (2.0**NNQ.f_bits)).astype(np.int64)
    b0_codes = np.round(b0 * (2.0**acc_shape.f_bits)).astype(np.int64)
    h = dense_golden(
        rff_feats,
        w0_codes,
        b0_codes,
        acc_shape,
        NNQ,
        apply_relu=False,
        relu_upper_bound=None,
        prod_shift=prod_shift,
    )
    return h, phase_codes


def simulate(phlut, phase_codes, max_wait=2_000_000):
    """Build the table, then read h back for every phase code (io codes in)."""
    out_d = phlut.out_d

    m = Module()
    m.submodules.dut = phlut

    total_words = phlut.num_entries * phlut.dim_stride
    ext_words = math.ceil(total_words / 2)
    storage_words = 1 << math.ceil(math.log2(ext_words + 1))
    m.submodules.psram = psram = FakePSRAM(
        addr_width=22, data_width=32, storage_words=storage_words, latency_cycles=4
    )
    wiring.connect(m, phlut.bus, psram.bus)

    results = []

    async def testbench(ctx):
        ctx.set(phlut.o.ready, 1)

        # wait for the startup build to fill the whole table.
        built = False
        for _ in range(max_wait):
            if ctx.get(phlut.ready):
                built = True
                break
            await ctx.tick()
        assert built, "PhaseHLutPS build never completed"

        for code in phase_codes:
            ctx.set(phlut.i.payload, int(code))
            ctx.set(phlut.i.valid, 1)
            # hold valid until accepted
            for _ in range(max_wait):
                if ctx.get(phlut.i.ready):
                    await ctx.tick()
                    break
                await ctx.tick()
            ctx.set(phlut.i.valid, 0)
            for _ in range(max_wait):
                if ctx.get(phlut.o.valid):
                    break
                await ctx.tick()
            results.append(
                [ctx.get(phlut.o.payload[j].as_value()) for j in range(out_d)]
            )
            await ctx.tick()

    sim = Simulator(m)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
    return np.array(results, dtype=np.int64)


def _find_weights_pkl():
    env = os.getenv("RFF_WEIGHTS_PKL")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("runs/*/weights/qkeras/latest.pkl"))
    return candidates[-1] if candidates else None


class TestPhaseHLutPS(unittest.TestCase):
    # small index space keeps the pysim build sweep fast (2**INDEX_BITS entries).
    INDEX_BITS = 5

    def setUp(self):
        pkl = _find_weights_pkl()
        if pkl is None or not pkl.exists():
            self.skipTest("no qkeras weights pickle found (set RFF_WEIGHTS_PKL)")
        self.weights, self.quant_sizes, self.model_config = load_weights_and_config(pkl)
        if not {"rff", "y_pred"} <= set(self.weights):
            self.skipTest(f"{pkl} missing rff/y_pred entries; retrain")

    def test_phase_h_lut_bit_exact(self):
        # mlp0's hidden dim must be a power of two for the PSRAM entry layout.
        w0, _b0 = self.weights[
            sorted(k for k in self.weights if k.startswith("mlp"))[0]
        ]["weights"]
        mlp_dim = int(np.asarray(w0).shape[1])
        if not _is_pow2(mlp_dim):
            self.skipTest(f"mlp_dim={mlp_dim} is not a power of two")

        net = RffNetwork(
            self.weights,
            self.quant_sizes,
            self.model_config,
            index_bits=self.INDEX_BITS,
        )
        phlut = net.phlut

        golden, phase_codes = build_h_table(net, self.INDEX_BITS)
        hw = simulate(phlut, phase_codes)

        self.assertEqual(golden.shape, hw.shape)
        if not np.array_equal(golden, hw):
            diff = golden != hw
            first = tuple(np.argwhere(diff)[0])
            self.fail(
                f"{int(diff.sum())} mismatched codes "
                f"(max abs {int(np.abs(golden - hw).max())}); "
                f"first at idx {first[0]} out {first[1]}: "
                f"golden={golden[first]} hw={hw[first]}"
            )


if __name__ == "__main__":
    unittest.main()
