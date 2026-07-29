"""Bit-exact equivalence test for a packed phase_h_lut.bin image.

This test reconstructs the phase->h table with integer math equivalent to the
PhaseHLutPS BUILD path, then compares it against a packed 32-bit PSRAM image
where each word stores two signed 16-bit NNQ entries:

    low16  = internal address 2*n
    high16 = internal address 2*n + 1

Run from repo root:

    uv run -m unittest test_equivalences.test_phase_h_lut_bin_equivalence

Environment overrides:

- RFF_WEIGHTS_PKL   : path to qkeras weights pickle
- PHASE_H_BIN       : path to packed .bin image
- PHASE_H_INDEX_BITS: phase index_bits used to generate the table (default: 13)
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_v import NNQ
from amaranth_v.dense_layer import QDenseLayer
from amaranth_v.rff_film_network import load_weights_and_config
from qkeras_v.rff_lut import build_io_luts, frac_bits, plan_shift, rff_lut_features
from test_equivalences.test_dense_equivalence import golden as dense_golden


def _find_weights_pkl() -> Path | None:
    env = os.getenv("RFF_WEIGHTS_PKL")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    candidates = sorted(root.glob("runs/*/weights/qkeras/latest.pkl"))
    return candidates[-1] if candidates else None


def _resolve_bin_path(weights_pkl: Path) -> Path:
    env = os.getenv("PHASE_H_BIN")
    if env:
        return Path(env)
    return weights_pkl.parent / "phase_h_lut.bin"


def _first_mlp_name(weights: dict) -> str:
    names = sorted(
        (k for k in weights if k.startswith("mlp")), key=lambda k: int(k[len("mlp") :])
    )
    if not names:
        raise ValueError("expected at least one mlp* layer in weights")
    return names[0]


def _phase_codes(index_bits: int, io_frac: int) -> np.ndarray:
    num_entries = 1 << index_bits
    shift = io_frac - (index_bits - 1)
    if shift < 0:
        raise ValueError(
            f"index_bits={index_bits} exceeds io_frac+1={io_frac + 1}; adjust PHASE_H_INDEX_BITS"
        )
    idxs = np.arange(num_entries, dtype=np.int64)
    idx_signed = np.where(idxs >= (num_entries >> 1), idxs - num_entries, idxs)
    return (idx_signed << shift).astype(np.int64)


def _build_equivalent_h_table(
    weights: dict, quant_sizes: dict, model_config: dict, index_bits: int
) -> np.ndarray:
    io_bits = int(quant_sizes["io_bits"])
    io_int = int(quant_sizes["io_int"])
    b_bits = int(quant_sizes["b_bits"])
    b_int = int(quant_sizes["b_int"])

    io_frac = io_bits - io_int - 1

    B = np.asarray(weights["rff"]["B"]).reshape(-1)
    b_f = frac_bits(b_bits, b_int)
    b_codes = np.round(B * (2.0**b_f)).astype(np.int64)

    lut_size = int(model_config["rff"]["lut_size"])
    rff_shift, lut_bits = plan_shift(io_bits, io_int, b_bits, b_int, lut_size)
    if rff_shift < 0:
        raise ValueError(
            f"negative rff_shift={rff_shift}; lut_size={lut_size} incompatible with quant sizes"
        )

    cos_lut, sin_lut = build_io_luts(lut_size, io_bits, io_int)
    phase_codes = _phase_codes(index_bits=index_bits, io_frac=io_frac)

    rff_feats = rff_lut_features(
        phase_codes, b_codes, cos_lut, sin_lut, rff_shift, lut_bits
    ).astype(np.int64)

    mlp0_name = _first_mlp_name(weights)
    w0, b0 = weights[mlp0_name]["weights"]
    w0 = np.asarray(w0)
    b0 = np.asarray(b0)

    # Equivalent to BUILD path quantisation: same acc shape and product alignment.
    io_shape = type("IoShape", (), {"i_bits": io_int + 1, "f_bits": io_frac})
    acc_shape, prod_shift = QDenseLayer.acc_shape_for(io_shape, int(w0.shape[0]), b0)

    w0_codes = np.round(w0 * (2.0**NNQ.f_bits)).astype(np.int64)
    b0_codes = np.round(b0 * (2.0**acc_shape.f_bits)).astype(np.int64)

    return dense_golden(
        rff_feats,
        w0_codes,
        b0_codes,
        acc_shape,
        NNQ,
        apply_relu=False,
        relu_upper_bound=None,
        prod_shift=prod_shift,
    ).astype(np.int64)


def _decode_bin_to_table(bin_path: Path, num_entries: int, out_d: int) -> np.ndarray:
    words32 = np.fromfile(bin_path, dtype="<u4")
    if words32.size == 0:
        raise ValueError(f"empty bin file: {bin_path}")

    low = (words32 & 0xFFFF).astype(np.uint16)
    high = ((words32 >> 16) & 0xFFFF).astype(np.uint16)

    halfwords_u16 = np.empty(words32.size * 2, dtype=np.uint16)
    halfwords_u16[0::2] = low
    halfwords_u16[1::2] = high

    expected_halfwords = num_entries * out_d
    if halfwords_u16.size < expected_halfwords:
        raise ValueError(
            f"bin too small: has {halfwords_u16.size} halfwords, expected {expected_halfwords}"
        )

    if halfwords_u16.size > expected_halfwords:
        tail = halfwords_u16[expected_halfwords:]
        if np.any(tail != 0):
            raise ValueError(
                f"bin has {tail.size} extra non-zero padded halfwords after expected table"
            )

    table_u16 = halfwords_u16[:expected_halfwords].reshape(num_entries, out_d)
    return table_u16.view(np.int16).astype(np.int64)


class TestPhaseHLutBinEquivalence(unittest.TestCase):
    INDEX_BITS = int(os.getenv("PHASE_H_INDEX_BITS", "13"))

    def setUp(self):
        self.weights_pkl = _find_weights_pkl()
        if self.weights_pkl is None or not self.weights_pkl.exists():
            self.skipTest("no qkeras weights pickle found (set RFF_WEIGHTS_PKL)")

        self.weights, self.quant_sizes, self.model_config = load_weights_and_config(
            self.weights_pkl
        )
        if "rff" not in self.weights:
            self.skipTest(f"{self.weights_pkl} missing rff entry")

        self.bin_path = _resolve_bin_path(self.weights_pkl)
        if not self.bin_path.exists():
            self.skipTest(
                f"phase_h bin not found at {self.bin_path} (set PHASE_H_BIN or generate it first)"
            )

    def test_phase_h_bin_matches_build_equivalent(self):
        golden = _build_equivalent_h_table(
            self.weights, self.quant_sizes, self.model_config, self.INDEX_BITS
        )

        num_entries, out_d = golden.shape
        got = _decode_bin_to_table(self.bin_path, num_entries, out_d)

        self.assertEqual(golden.shape, got.shape)
        if not np.array_equal(golden, got):
            diff = golden != got
            first = tuple(np.argwhere(diff)[0])
            self.fail(
                f"{int(diff.sum())} mismatched codes "
                f"(max abs {int(np.abs(golden - got).max())}); "
                f"first at idx {first[0]} out {first[1]}: "
                f"golden={golden[first]} bin={got[first]}"
            )


if __name__ == "__main__":
    unittest.main()
