import os

from amaranth_future import fixed

# Fixed-point config for the quantised MLP path, mirroring the cdcc network.
# NNQ matches the qkeras MLP weight/activation format (FP N_INT.N_FRAC, signed).
# Defaults SQ(4, 12) == qkeras bits=16, integer=3 (+sign), frac=12.
N_INT = int(os.getenv("N_INT", "4"))
N_FRAC = int(os.getenv("N_FRAC", "12"))
NNQ = fixed.SQ(N_INT, N_FRAC)

# Double-width config for dot-product accumulation and (double-width) biases.
NNQ_DW = fixed.SQ(N_INT * 2, N_FRAC * 2)


def parse_nnq(v, assert_exact: bool = True, shape=NNQ):
    """Recursively convert python/numpy scalars (or nested lists) into
    fixed.Const of the given shape. When assert_exact is set, raises if a value
    is not exactly representable in the target fixed-point shape."""
    try:
        iterator = iter(v)
    except TypeError:
        v = float(v)
        fpv = fixed.Const(v, shape=shape)
        if assert_exact and fpv.as_float() != v:
            raise ValueError(
                f"value {v} parsed to NNQ {fpv.as_float()} which isn't exact"
            )
        return fpv
    else:
        return [parse_nnq(x, assert_exact=assert_exact, shape=shape) for x in iterator]
