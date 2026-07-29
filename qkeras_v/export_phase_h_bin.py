import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from amaranth_future import fixed

from amaranth_v import NNQ
from amaranth_v.dense_layer import QDenseLayer
from qkeras_v.rff_lut import build_io_luts, frac_bits, plan_shift, rff_lut_features


def _load_weights_and_config(weights_pkl: Path):
    with weights_pkl.open("rb") as f:
        weights = pickle.load(f)

    model_root = weights_pkl.parents[2]
    with (model_root / "model_config.json").open("r") as f:
        model_config = json.load(f)
    with (model_root / "quant_sizes.json").open("r") as f:
        quant_sizes = json.load(f)

    return weights, quant_sizes, model_config


def _first_mlp_name(weights: dict) -> str:
    mlp_names = sorted(
        (k for k in weights if k.startswith("mlp")), key=lambda k: int(k[len("mlp") :])
    )
    if not mlp_names:
        raise ValueError("expected at least one mlp* layer in the weights pickle")
    return mlp_names[0]


def _phase_codes(index_bits: int, io_frac: int) -> np.ndarray:
    num_entries = 1 << index_bits
    idx_shift = io_frac - (index_bits - 1)
    if idx_shift < 0:
        raise ValueError(
            f"index_bits={index_bits} requires io_frac>={index_bits - 1}, got io_frac={io_frac}"
        )

    idxs = np.arange(num_entries, dtype=np.int64)
    idx_signed = np.where(idxs >= (num_entries >> 1), idxs - num_entries, idxs)
    return (idx_signed << idx_shift).astype(np.int64)


def _build_h_table(
    weights: dict, quant_sizes: dict, model_config: dict, index_bits: int
) -> np.ndarray:
    io_bits = int(quant_sizes["io_bits"])
    io_int = int(quant_sizes["io_int"])
    b_bits = int(quant_sizes["b_bits"])
    b_int = int(quant_sizes["b_int"])

    io_shape = fixed.SQ(io_int + 1, io_bits - io_int - 1)

    B = np.asarray(weights["rff"]["B"]).reshape(-1)
    b_f = frac_bits(b_bits, b_int)
    b_codes = np.round(B * (2.0**b_f)).astype(np.int64)

    lut_size = int(model_config["rff"]["lut_size"])
    rff_shift, lut_bits = plan_shift(io_bits, io_int, b_bits, b_int, lut_size)
    if rff_shift < 0:
        raise ValueError(
            f"lut_size={lut_size} too large for io/b quant setup; computed negative shift {rff_shift}"
        )

    cos_lut, sin_lut = build_io_luts(lut_size, io_bits, io_int)
    phase_codes = _phase_codes(index_bits=index_bits, io_frac=io_shape.f_bits)
    rff_feats = rff_lut_features(
        phase_codes, b_codes, cos_lut, sin_lut, rff_shift, lut_bits
    ).astype(np.int64)

    mlp0_name = _first_mlp_name(weights)
    w0, b0 = weights[mlp0_name]["weights"]
    w0 = np.asarray(w0)
    b0 = np.asarray(b0)

    if w0.ndim != 2 or b0.ndim != 1:
        raise ValueError(
            f"expected {mlp0_name} to have 2D weights and 1D bias, got {w0.shape=} {b0.shape=}"
        )
    if int(w0.shape[0]) != int(rff_feats.shape[1]):
        raise ValueError(
            f"{mlp0_name} input dim {w0.shape[0]} does not match RFF dim {rff_feats.shape[1]}"
        )

    acc_shape, prod_shift = QDenseLayer.acc_shape_for(io_shape, int(w0.shape[0]), b0)
    w0_codes = np.round(w0 * (2.0**NNQ.f_bits)).astype(np.int64)
    b0_codes = np.round(b0 * (2.0**acc_shape.f_bits)).astype(np.int64)

    mac = (rff_feats @ w0_codes).astype(np.int64)
    acc = b0_codes + (mac << prod_shift)

    lower_bound = int(
        fixed.Const(NNQ.min().as_float(), shape=acc_shape, clamp=True)._value
    )
    upper_bound = int(
        fixed.Const(NNQ.max().as_float(), shape=acc_shape, clamp=True)._value
    )
    clamped = np.clip(acc, lower_bound, upper_bound)

    frac_drop = int(acc_shape.f_bits - NNQ.f_bits)
    frac_mask = (1 << frac_drop) - 1

    # Match hardware truncation toward zero when reducing accumulator frac bits.
    needs_adjust = (clamped < 0) & ((clamped & frac_mask) != 0)
    trunc = clamped + np.where(needs_adjust, 1 << frac_drop, 0)
    h = (trunc >> frac_drop).astype(np.int64)

    return h


def _pack_psram_words(h_table: np.ndarray) -> np.ndarray:
    if h_table.ndim != 2:
        raise ValueError(f"expected h_table to be rank-2, got shape {h_table.shape}")

    flat = h_table.reshape(-1)

    # Internal 16-bit words are mapped linearly by address. The adapter maps
    # even addresses to low 16b and odd addresses to high 16b of each 32b word.
    halfwords = (flat & 0xFFFF).astype(np.uint16)
    if (halfwords.size & 1) != 0:
        halfwords = np.append(halfwords, np.uint16(0))

    lo = halfwords[0::2].astype(np.uint32)
    hi = halfwords[1::2].astype(np.uint32)
    return lo | (hi << 16)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Export the PhaseHLutPS phase->h table to a packed 32-bit PSRAM binary."
        ),
    )
    parser.add_argument("--weights-pkl", type=Path, required=True)
    parser.add_argument(
        "--out-bin",
        type=Path,
        default=None,
        help="Output path for packed PSRAM image (.bin)",
    )
    parser.add_argument(
        "--index-bits",
        type=int,
        default=13,
        help="PhaseHLutPS index_bits used to enumerate the phase table",
    )

    args = parser.parse_args()

    weights_pkl = args.weights_pkl.resolve()
    if not weights_pkl.exists():
        raise FileNotFoundError(f"weights pickle not found: {weights_pkl}")

    out_bin = (
        args.out_bin.resolve()
        if args.out_bin is not None
        else (weights_pkl.parent / "phase_h_lut.bin").resolve()
    )

    weights, quant_sizes, model_config = _load_weights_and_config(weights_pkl)
    h_table = _build_h_table(
        weights, quant_sizes, model_config, index_bits=args.index_bits
    )
    psram_words = _pack_psram_words(h_table)

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(psram_words.astype("<u4").tobytes())

    num_entries, out_d = h_table.shape
    print(f"wrote {out_bin}")
    print(
        f"entries={num_entries} out_d={out_d} halfwords={h_table.size} words32={psram_words.size} bytes={out_bin.stat().st_size}"
    )


if __name__ == "__main__":
    main()
