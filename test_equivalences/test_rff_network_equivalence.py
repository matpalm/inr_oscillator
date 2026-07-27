"""End-to-end bit-exact equivalence check: numpy golden vs the amaranth_v
``RffNetwork`` hardware, driven by a *real* trained run (its ``model_config``
and quantised weights pickle).

The golden reference composes the two building blocks the layer tests already
verify individually -- the RFF LUT feature map (``qkeras_v.rff_lut``) and the
QDenseLayer datapath (the ``golden`` helper from ``test_dense_equivalence``) --
in exactly the order ``RffNetwork`` wires them:

    phase -> RFF LUT ->\
                        concat -> mlp0 -> ... -> y_pred -> out
    embed ------------>/

Both the golden and the hardware are fed identical io-format input codes, so the
whole pipeline (RFF, the io->NNQ first layer, the NNQ MLP chain and the NNQ->io
regressor, including the embedding concat) must agree bit-for-bit.

Run from the repo root::

    uv run -m unittest test_equivalences.test_rff_network_equivalence

By default it uses the newest ``runs/*/weights/qkeras/latest.pkl``; override with
the ``RFF_WEIGHTS_PKL`` environment variable.
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

from amaranth_future import fixed
from amaranth_v import NNQ
from amaranth_v.dense_layer import QDenseLayer
from amaranth_v.rff_film_network import RffNetwork, load_weights_and_config
from qkeras_v.rff_lut import (
    build_io_luts,
    frac_bits,
    plan_shift,
    quantise_to_codes,
    rff_lut_features,
)
from test_equivalences.test_dense_equivalence import golden as dense_golden
from test_equivalences.test_activation_cache_ps_helpers import FakePSRAM

def simulate(net, sample_codes):
    """Drive the RffNetwork over ``sample_codes`` (N, IN_D) io codes.

    The network is backed by a FakePSRAM (phase->h table). The startup build
    must complete (net.ready) before any sample is accepted.
    """

    in_d = net.in_d
    out_d = net.out_d
    results = []

    m = Module()
    m.submodules.net = net
    phlut = net.phlut
    total_words = phlut.num_entries * phlut.dim_stride
    ext_words = math.ceil(total_words / 2)
    storage_words = 1 << math.ceil(math.log2(ext_words + 1))
    m.submodules.psram = psram = FakePSRAM(
        addr_width=22, data_width=32, storage_words=storage_words, latency_cycles=4
    )
    wiring.connect(m, net.bus_h, psram.bus)

    max_wait = 2_000_000

    async def testbench(ctx):
        ctx.set(net.o.ready, 1)

        # wait for the startup phase->h build to complete.
        built = False
        for _ in range(max_wait):
            if ctx.get(net.ready):
                built = True
                break
            await ctx.tick()
        assert built, "network phase->h build never completed"

        for row in sample_codes:
            for k in range(in_d):
                ctx.set(net.i.payload[k].as_value(), int(row[k]))
            ctx.set(net.i.valid, 1)
            for _ in range(max_wait):
                if ctx.get(net.i.ready):
                    await ctx.tick()
                    break
                await ctx.tick()
            ctx.set(net.i.valid, 0)
            for _ in range(max_wait):
                if ctx.get(net.o.valid):
                    break
                await ctx.tick()
            results.append([ctx.get(net.o.payload[j].as_value()) for j in range(out_d)])
            await ctx.tick()  # o.ready high -> handshake completes

    sim = Simulator(m)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
    return np.array(results, dtype=np.int64)


def _film_combine_golden(h, gamma, beta, relu_upper_bound):
    """Integer-exact reference for FiLMCombine: post((1+gamma)*h + beta).

    h / gamma / beta are NNQ raw codes; returns NNQ raw codes. Mirrors
    ``amaranth_v.film.FiLMCombine`` (clamp -> relu -> re-clip -> trunc-to-zero).
    """
    acc_shape = fixed.SQ(2 * NNQ.i_bits + 2, 2 * NNQ.f_bits)
    beta_shift = NNQ.f_bits
    nnq_one = 1 << NNQ.f_bits
    frac_drop = acc_shape.f_bits - NNQ.f_bits
    out_width = NNQ.width

    lower = fixed.Const(NNQ.min().as_float(), shape=acc_shape, clamp=True)._value
    upper = fixed.Const(NNQ.max().as_float(), shape=acc_shape, clamp=True)._value
    relu_ub = fixed.Const(relu_upper_bound, shape=acc_shape)._value

    out = np.zeros_like(h)
    n, dim = h.shape
    for r in range(n):
        for c in range(dim):
            acc = (nnq_one + int(gamma[r, c])) * int(h[r, c]) + (
                int(beta[r, c]) << beta_shift
            )
            acc = min(max(acc, lower), upper)
            if acc < 0:
                acc = 0
            elif acc > relu_ub:
                acc = relu_ub
            acc = min(max(acc, lower), upper)
            frac_nonzero = (acc & ((1 << frac_drop) - 1)) != 0
            trunc = acc + (1 << frac_drop) if (acc < 0 and frac_nonzero) else acc
            sliced = (trunc >> frac_drop) & ((1 << out_width) - 1)
            if sliced >= (1 << (out_width - 1)):
                sliced -= 1 << out_width
            out[r, c] = sliced
    return out


def _dense_codes(net, name, x, in_shape, out_shape, apply_relu):
    """Run a single QDenseLayer golden from raw ``x`` codes -> raw out codes."""
    w, b = net.dense_weights_biases_for(name)
    relu_ub = net.relu_upper_bound if apply_relu else None
    acc_shape, prod_shift = QDenseLayer.acc_shape_for(in_shape, w.shape[0], b)
    w_codes = np.round(w * (2.0**NNQ.f_bits)).astype(np.int64)
    b_codes = np.round(b * (2.0**acc_shape.f_bits)).astype(np.int64)
    return dense_golden(
        x, w_codes, b_codes, acc_shape, out_shape, apply_relu, relu_ub, prod_shift
    )


def build_reference(net, phase_codes, embed_codes, quant_sizes):
    """Integer-exact numpy reference for the whole FiLM network (io input codes).

    Mirrors ``RffNetwork``: h = mlp0(RFF(phase)) (no relu); each filmed layer
    applies post((1+gamma)*h + beta); plain layers apply relu; y_pred maps NNQ
    back to io. The phase feeding the RFF is expected to already sit on the
    PSRAM table's index grid (so the cached h matches this reference exactly).
    """

    io_bits, io_integer = quant_sizes["io_bits"], quant_sizes["io_int"]
    b_bits, b_integer = quant_sizes["b_bits"], quant_sizes["b_int"]

    B = np.asarray(net.qkeras_weights["rff"]["B"]).reshape(-1)
    b_codes = np.round(B * (2.0 ** frac_bits(b_bits, b_integer))).astype(np.int64)
    rff_shift, lut_bits = plan_shift(
        io_bits, io_integer, b_bits, b_integer, net.lut_size
    )
    cos_lut, sin_lut = build_io_luts(net.lut_size, io_bits, io_integer)
    rff_feats = rff_lut_features(
        phase_codes, b_codes, cos_lut, sin_lut, rff_shift, lut_bits
    ).astype(np.int64)

    film_set = set(net.film_layer_idxs)

    # mlp0 (io -> NNQ, no relu) -> film0.
    h = _dense_codes(net, net.mlp_names[0], rff_feats, net.io_shape, NNQ, False)

    prev = None
    for pos, idx in enumerate(net.mlp_idxs):
        name = net.mlp_names[pos]
        if pos == 0:
            h_layer = h  # from the RFF path above
        else:
            filmed = idx in film_set
            h_layer = _dense_codes(net, name, prev, NNQ, NNQ, apply_relu=not filmed)
        if idx in film_set:
            gamma = _dense_codes(
                net, f"film{idx}_gamma", embed_codes, net.io_shape, NNQ, False
            )
            beta = _dense_codes(
                net, f"film{idx}_beta", embed_codes, net.io_shape, NNQ, False
            )
            prev = _film_combine_golden(h_layer, gamma, beta, net.relu_upper_bound)
        else:
            prev = h_layer

    return _dense_codes(net, "y_pred", prev, NNQ, net.io_shape, False)


def _find_weights_pkl():
    """Resolve the qkeras weights pickle: RFF_WEIGHTS_PKL env, else newest run."""
    env = os.getenv("RFF_WEIGHTS_PKL")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("runs/*/weights/qkeras/latest.pkl"))
    return candidates[-1] if candidates else None


class TestRffNetworkEquivalence(unittest.TestCase):
    # small index space keeps the pysim startup build fast (2**INDEX_BITS entries).
    INDEX_BITS = 6
    SEED = 0
    SEED = 0

    def setUp(self):
        pkl = _find_weights_pkl()
        if pkl is None or not pkl.exists():
            self.skipTest("no qkeras weights pickle found (set RFF_WEIGHTS_PKL)")
        self.weights, self.quant_sizes, self.model_config = load_weights_and_config(pkl)
        if not {"rff", "y_pred"} <= set(self.weights):
            self.skipTest(f"{pkl} missing rff/y_pred entries; retrain")
        self.lut_size = self.model_config["rff"]["lut_size"]

    def test_network_bit_exact(self):

        # mlp0's hidden dim must be a power of two for the PSRAM table layout.
        w0 = np.asarray(
            self.weights[sorted(k for k in self.weights if k.startswith("mlp"))[0]][
                "weights"
            ][0]
        )
        mlp_dim = int(w0.shape[1])
        if mlp_dim < 1 or (mlp_dim & (mlp_dim - 1)) != 0:
            self.skipTest(f"mlp_dim={mlp_dim} is not a power of two")

        net = RffNetwork(
            self.weights,
            self.quant_sizes,
            self.model_config,
            index_bits=self.INDEX_BITS,
        )

        io_bits, io_integer = self.quant_sizes["io_bits"], self.quant_sizes["io_int"]

        # phases must sit exactly on the PSRAM table's index grid so the cached
        # h (built from the quantised phase) matches the golden bit-for-bit.
        num_entries = 1 << self.INDEX_BITS
        index_shift = net.io_shape.f_bits - (self.INDEX_BITS - 1)
        idxs = np.arange(num_entries, dtype=np.int64)
        idx_signed = np.where(idxs >= (num_entries >> 1), idxs - num_entries, idxs)
        phase_codes = (idx_signed << index_shift).astype(np.int64)
        num_samples = num_entries

        rng = np.random.default_rng(self.SEED)
        embed = rng.uniform(-1.0, 1.0, (num_samples, net.embed_dim))
        embed_codes = (
            quantise_to_codes(embed, io_bits, io_integer)
            if net.embed_dim
            else np.zeros((num_samples, 0), dtype=np.int64)
        )
        sample_codes = np.concatenate([phase_codes.reshape(-1, 1), embed_codes], axis=1)

        ref = build_reference(net, phase_codes, embed_codes, self.quant_sizes)
        hw = simulate(net, sample_codes)

        self.assertEqual(ref.shape, hw.shape)
        if not np.array_equal(ref, hw):
            diff = ref != hw
            first = tuple(np.argwhere(diff)[0])
            self.fail(
                f"{int(diff.sum())} mismatched codes "
                f"(max abs {int(np.abs(ref - hw).max())}); "
                f"first at sample {first[0]} out {first[1]}: "
                f"golden={ref[first]} hw={hw[first]}"
            )


if __name__ == "__main__":
    unittest.main()
