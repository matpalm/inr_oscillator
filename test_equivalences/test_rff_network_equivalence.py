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

import json
import os
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_v import NNQ
from amaranth_v.dense_layer import QDenseLayer
from amaranth_v.rff_network import RffNetwork
from qkeras_v.rff_lut import (
    build_io_luts,
    frac_bits,
    plan_shift,
    quantise_to_codes,
    rff_lut_features,
)
from test_equivalences.test_dense_equivalence import golden as dense_golden


def simulate(net, sample_codes):
    """Drive the Amaranth RffNetwork over ``sample_codes`` (N, IN_D) io codes."""
    in_d = net.IN_D
    out_d = net.OUT_D
    results = []

    async def testbench(ctx):
        ctx.set(net.o.ready, 1)
        for row in sample_codes:
            for k in range(in_d):
                ctx.set(net.i.payload[k].as_value(), int(row[k]))
            ctx.set(net.i.valid, 1)
            await ctx.tick()
            ctx.set(net.i.valid, 0)
            while not ctx.get(net.o.valid):
                await ctx.tick()
            results.append([ctx.get(net.o.payload[j].as_value()) for j in range(out_d)])
            await ctx.tick()  # o.ready high -> handshake completes, back to IDLE

    sim = Simulator(net)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
    return np.array(results, dtype=np.int64)


def build_reference(net, phase_codes, embed_codes):
    """Integer-exact numpy reference for the whole network (io input codes)."""
    rff = net.qkeras_weights["rff"]
    io_bits, io_integer = int(rff["io_bits"]), int(rff["io_integer"])
    b_bits, b_integer = int(rff["b_bits"]), int(rff["b_integer"])

    B = np.asarray(rff["B"]).reshape(-1)
    b_codes = np.round(B * (2.0 ** frac_bits(b_bits, b_integer))).astype(np.int64)
    shift, lut_bits = plan_shift(io_bits, io_integer, b_bits, b_integer, net.lut_size)
    cos_lut, sin_lut = build_io_luts(net.lut_size, io_bits, io_integer)

    # RFF features (io codes), then concat the embedding channels (io codes).
    rff_feats = rff_lut_features(
        phase_codes, b_codes, cos_lut, sin_lut, shift, lut_bits
    )
    x = np.concatenate([rff_feats, embed_codes], axis=1).astype(np.int64)

    # MLP chain (io -> NNQ for the first layer, NNQ -> NNQ afterwards) + y_pred.
    in_shape = net.io_shape
    layer_specs = [(name, True, NNQ) for name in net.mlp_names]
    layer_specs.append(("y_pred", False, net.io_shape))
    for name, apply_relu, out_shape in layer_specs:
        w, b = net.dense_weights_biases_for(name)
        relu_ub = net.relu_upper_bound if apply_relu else None
        acc_shape, prod_shift = QDenseLayer.acc_shape_for(in_shape, w.shape[0], b)
        w_codes = np.round(w * (2.0**NNQ.f_bits)).astype(np.int64)
        b_codes_l = np.round(b * (2.0**acc_shape.f_bits)).astype(np.int64)
        x = dense_golden(
            x,
            w_codes,
            b_codes_l,
            acc_shape,
            out_shape,
            apply_relu,
            relu_ub,
            prod_shift,
        )
        in_shape = out_shape

    return x


def _find_weights_pkl():
    """Resolve the qkeras weights pickle: RFF_WEIGHTS_PKL env, else newest run."""
    env = os.getenv("RFF_WEIGHTS_PKL")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("runs/*/weights/qkeras/latest.pkl"))
    return candidates[-1] if candidates else None


class TestRffNetworkEquivalence(unittest.TestCase):
    LUT_SIZE = 1024
    NUM_SAMPLES = 32
    SEED = 0

    def setUp(self):
        pkl = _find_weights_pkl()
        if pkl is None or not pkl.exists():
            self.skipTest("no qkeras weights pickle found (set RFF_WEIGHTS_PKL)")
        with open(pkl, "rb") as f:
            self.weights = pickle.load(f)
        if not {"rff", "y_pred"} <= set(self.weights):
            self.skipTest(f"{pkl} missing rff/y_pred entries; retrain")

        # real model_config (for the relu upper bound); fall back to the default.
        self.relu_upper_bound = 8.0
        cfg = pkl.parents[2] / "model_config.json"
        if cfg.exists():
            with open(cfg, "r") as f:
                self.relu_upper_bound = float(json.load(f).get("relu_upper_bound", 8.0))

    def test_network_bit_exact(self):
        net = RffNetwork(
            self.weights,
            lut_size=self.LUT_SIZE,
            relu_upper_bound=self.relu_upper_bound,
        )

        rff = self.weights["rff"]
        io_bits, io_integer = int(rff["io_bits"]), int(rff["io_integer"])

        rng = np.random.default_rng(self.SEED)
        phases = np.linspace(-1.0, 1.0, self.NUM_SAMPLES, endpoint=False)
        embed = rng.uniform(-1.0, 1.0, (self.NUM_SAMPLES, net.EMBED_D))

        phase_codes = quantise_to_codes(phases, io_bits, io_integer)
        embed_codes = (
            quantise_to_codes(embed, io_bits, io_integer)
            if net.EMBED_D
            else np.zeros((self.NUM_SAMPLES, 0), dtype=np.int64)
        )
        sample_codes = np.concatenate([phase_codes.reshape(-1, 1), embed_codes], axis=1)

        ref = build_reference(net, phase_codes, embed_codes)
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
