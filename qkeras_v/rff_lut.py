"""Bit-exact fixed-point golden model of the RFF datapath.

This module produces the *integer codes* that the hardware
(``amaranth_v.rff.RandomFourierFeaturesLUT``) must reproduce exactly.  Both the
golden model here and the Amaranth ROM are fed the identical LUT / B integer
codes, so the equivalence test can assert a bit-for-bit match.

Fixed-point convention follows qkeras ``quantized_bits(bits, integer)``:
one sign bit, ``integer`` integer bits, and ``bits - integer - 1`` fractional
bits.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from qkeras import quantized_bits


def frac_bits(bits, integer, keep_negative=1):
    """Number of fractional bits for a qkeras ``quantized_bits(bits, integer)``."""
    return bits - integer - keep_negative


def quantise_to_codes(values, bits, integer):
    """Quantise ``values`` with qkeras and return the underlying integer codes."""
    q = quantized_bits(bits=bits, integer=integer, alpha=1)
    qv = np.asarray(q(np.asarray(values, dtype=np.float32)))
    f = frac_bits(bits, integer)
    return np.round(qv * (2.0**f)).astype(np.int64)


def build_io_luts(lut_size, io_bits, io_integer):
    """Build cos/sin ROM codes indexed by a full turn split into ``lut_size`` steps."""
    angles = 2.0 * np.pi * np.arange(lut_size) / lut_size
    cos_codes = quantise_to_codes(np.cos(angles), io_bits, io_integer)
    sin_codes = quantise_to_codes(np.sin(angles), io_bits, io_integer)
    return cos_codes, sin_codes


def plan_shift(io_bits, io_integer, b_bits, b_integer, lut_size):
    """Return (shift, lut_bits): how far to shift ``phase*B`` down to the LUT index."""
    lut_bits = (lut_size - 1).bit_length()
    shift = frac_bits(io_bits, io_integer) + frac_bits(b_bits, b_integer) - lut_bits
    return shift, lut_bits


def rff_lut_features(phase_codes, b_codes, cos_lut, sin_lut, shift, lut_bits):
    """Golden RFF: integer codes for every phase, ordered [cos_0..cos_K, sin_0..sin_K].

    This mirrors the hardware exactly: an integer multiply, an arithmetic right
    shift by ``shift`` (floor toward -inf, matching a signed HW shift), and the
    low ``lut_bits`` taken as the ROM index (== ``mod 2**lut_bits``).
    """
    mask = (1 << lut_bits) - 1
    K = len(b_codes)
    out = np.zeros((len(phase_codes), 2 * K), dtype=np.int64)
    for i, p in enumerate(phase_codes):
        for k, bk in enumerate(b_codes):
            prod = int(p) * int(bk)
            idx = (prod >> shift) & mask
            out[i, k] = cos_lut[idx]
            out[i, K + k] = sin_lut[idx]
    return out
