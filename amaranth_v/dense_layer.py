from numpy.typing import NDArray
import numpy as np

from amaranth import Array, Module, Mux, Signal, signed
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory

from amaranth_future import fixed

from . import NNQ, parse_nnq


def _min_frac_bits(values, max_bits=48):
    """Smallest number of fractional bits on which every value lands exactly."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    for f in range(max_bits + 1):
        scaled = arr * (2.0**f)
        if np.all(scaled == np.round(scaled)):
            return f
    raise ValueError(f"values not on a 2^-{max_bits} grid: {arr}")


class QDenseLayer(wiring.Component):
    """Quantised dense (fully-connected) layer: out = post(x @ W + b).

    Mirrors the fundamental approach of cdcc's RowByMatrixMultiply (sequential,
    single-multiplier MAC computing one output column at a time) and folds in
    the conv1d POST_PROCESS tail (clamp -> optional relu -> re-clip -> truncate
    toward zero while narrowing NNQ_DW -> out_shape).

    A single instance supports both the qkeras model's ReLU MLP layers
    (apply_relu=True, out_shape=NNQ) and the final regression layer
    (apply_relu=False, out_shape=<io fixed-point shape>).

    Weights are stored in NNQ. The input may be in a different fixed-point shape
    (``in_shape``) than the weights -- e.g. the first MLP layer consumes the io
    format while later layers consume NNQ -- so the running accumulator and the
    (double-width) bias are held at the full product scale
    ``acc_shape = SQ(in_i + NNQ_i + headroom, in_f + NNQ_f)``.
    """

    def __init__(
        self,
        np_kernel: NDArray,
        np_bias: NDArray | None,
        apply_relu: bool,
        relu_upper_bound: float | None = None,
        in_shape=NNQ,
        out_shape=NNQ,
    ):
        """
        Args:
            np_kernel   (IN_D, OUT_D)   qkeras QDense kernel
            np_bias     (OUT_D,)|None   qkeras QDense bias (double-width), or None
            apply_relu                  apply the relu clamp in post-process
            relu_upper_bound            upper bound for the relu (required if apply_relu)
            in_shape                    fixed-point shape of the input activations
                                        (NNQ for later layers, io shape for the first)
            out_shape                   fixed-point shape the output is narrowed to
                                        (NNQ for the relu MLP, the io shape for y_pred)
        """

        if len(np_kernel.shape) != 2:
            raise Exception(
                "Expect QDenseLayer kernel with shape (IN_D, OUT_D) "
                f"but received {np_kernel.shape}"
            )

        self.IN_D, self.OUT_D = np_kernel.shape

        if np_bias is not None:
            if len(np_bias.shape) != 1 or np_bias.shape[0] != self.OUT_D:
                raise Exception(
                    f"Expect QDenseLayer bias with shape ({self.OUT_D},) "
                    f"but received {np_bias.shape}"
                )

        if apply_relu and relu_upper_bound is None:
            raise Exception("relu_upper_bound is required when apply_relu is set")

        print(
            f">QDenseLayer IN_D={self.IN_D} OUT_D={self.OUT_D} apply_relu={apply_relu}"
            f" ( relu_upper_bound={relu_upper_bound} )"
            f" in_shape={in_shape!r} out_shape={out_shape!r}"
        )

        self.apply_relu = apply_relu
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.acc_shape, self.prod_shift = self.acc_shape_for(
            in_shape, self.IN_D, np_bias
        )

        self.num_weights = self.IN_D * self.OUT_D

        # Flattened as [out_d][in_d], row-major.
        weight_rows = np_kernel.T
        weight_init = []
        for o in range(self.OUT_D):
            for i in range(self.IN_D):
                try:
                    weight_init.append(parse_nnq(weight_rows[o][i], shape=NNQ))
                except ValueError as e:
                    raise Exception(
                        f"!!!!!!!! weight_init o={o} i={i} {weight_rows[o][i]}", e
                    )

        self.weight_mem = Memory(
            shape=NNQ,
            depth=self.num_weights,
            init=weight_init,
            attrs={"ram_style": "block"},
        )

        # biases: qkeras uses a double-width bias quantiser; expressed at the
        # accumulator scale (in_frac + weight_frac >= 2*weight_frac) this is an
        # exact left shift, so parse_nnq(assert_exact) stays satisfied.
        if np_bias is not None:
            bias_init = [parse_nnq(b, shape=self.acc_shape) for b in np_bias]
        else:
            bias_init = [
                parse_nnq(0.0, shape=self.acc_shape) for _ in range(self.OUT_D)
            ]

        self.bias_mem = Memory(
            shape=self.acc_shape,
            depth=self.OUT_D,
            init=bias_init,
            attrs={"ram_style": "block"},
        )

        self.accumulator = Signal(self.acc_shape, name="qdense_running_accum", init=0)

        self.input = Array(
            Signal(in_shape, name=f"qdense_in_{i}", init=0) for i in range(self.IN_D)
        )
        self.i_idx = Signal(range(self.IN_D), init=0)

        self.result = Array(
            Signal(out_shape, name=f"qdense_result_{j}", init=0)
            for j in range(self.OUT_D)
        )
        self.o_idx = Signal(range(self.OUT_D), init=0)

        if apply_relu:
            self.relu_upper_bound = fixed.Const(relu_upper_bound, shape=self.acc_shape)

        # clip to representable out_shape bounds (expressed at the accumulator
        # scale) before narrowing acc_shape -> out_shape. note: can't use fixed
        # Value utils (clamp) directly since we're matching FXPmath/qkeras which
        # does things slightly differently.
        self.lower_bound = fixed.Const(
            out_shape.min().as_float(), shape=self.acc_shape
        ).as_value()
        self.upper_bound = fixed.Const(
            out_shape.max().as_float(), shape=self.acc_shape
        ).as_value()

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(in_shape, self.IN_D))),
                "o": wiring.Out(
                    stream.Signature(data.ArrayLayout(out_shape, self.OUT_D))
                ),
            }
        )

    @staticmethod
    def acc_shape_for(in_shape, in_d, np_bias):
        """Full-precision accumulator shape and the product left-shift.

        The accumulator's fractional field must hold BOTH the a*b products
        (in_frac + weight(NNQ) frac) and the bias grid exactly (the qkeras
        double-width bias quantiser carries 2*NNQ_frac + 1 frac bits, one more
        than the product scale of the NNQ-input layers). Integer bits are sized
        for the worst-case in_d-term sum (plus bias headroom). ``prod_shift``
        aligns the products up to the (possibly wider) accumulator scale.
        """
        acc_f_bits = in_shape.f_bits + NNQ.f_bits
        if np_bias is not None:
            acc_f_bits = max(acc_f_bits, _min_frac_bits(np_bias))
        acc_i_bits = in_shape.i_bits + NNQ.i_bits + in_d.bit_length()
        acc_shape = fixed.SQ(acc_i_bits, acc_f_bits)
        prod_shift = acc_f_bits - (in_shape.f_bits + NNQ.f_bits)
        return acc_shape, prod_shift

    def elaborate(self, platform):
        m = Module()
        m.submodules["weight_mem"] = self.weight_mem
        m.submodules["bias_mem"] = self.bias_mem

        rd_w = self.weight_mem.read_port(domain="sync")
        rd_b = self.bias_mem.read_port(domain="sync")

        mul_a = Signal(signed(self.in_shape.width), name="qdense_mul_a")
        mul_b = Signal(signed(NNQ.width), name="qdense_mul_b")

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            rd_w.en.eq(0),
            rd_w.addr.eq(0),
            rd_b.en.eq(0),
            rd_b.addr.eq(0),
        ]

        for j in range(self.OUT_D):
            m.d.comb += self.o.payload[j].eq(self.result[j])

        frac_drop = self.acc_shape.f_bits - self.out_shape.f_bits
        out_width = self.out_shape.width
        lower = self.lower_bound
        upper = self.upper_bound

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid & self.i.ready):
                    for i in range(self.IN_D):
                        m.d.sync += self.input[i].eq(self.i.payload[i])
                    m.d.sync += [
                        self.i_idx.eq(0),
                        self.o_idx.eq(0),
                    ]
                    m.next = "PREFETCH_BIAS"

            with m.State("PREFETCH_BIAS"):
                # prep read of this output column's bias ( ready next cycle )
                m.d.comb += [
                    rd_b.en.eq(1),
                    rd_b.addr.eq(self.o_idx),
                ]
                m.next = "LOAD_BIAS"

            with m.State("LOAD_BIAS"):
                # seed accumulator with the ( double-width ) bias
                m.d.sync += self.accumulator.eq(rd_b.data.as_value().as_signed())
                # prep read of first weight for this column
                m.d.comb += [
                    rd_w.en.eq(1),
                    rd_w.addr.eq(self.o_idx * self.IN_D + self.i_idx),
                ]
                m.next = "LOAD_MUL_INPUTS"

            with m.State("LOAD_MUL_INPUTS"):
                # registering mul_a and mul_b in a state before the MAC greatly
                # helps routing / comb depth ( per cdcc RowByMatrixMultiply ).
                m.d.sync += [
                    mul_a.eq(self.input[self.i_idx].as_value().as_signed()),
                    mul_b.eq(rd_w.data.as_value().as_signed()),
                ]
                m.next = "MAC"

            with m.State("MAC"):
                m.d.sync += self.accumulator.eq(
                    self.accumulator.as_value().as_signed()
                    + ((mul_a * mul_b) << self.prod_shift)
                )
                with m.If(self.i_idx == self.IN_D - 1):
                    m.next = "POST_PROCESS"
                with m.Else():
                    m.d.sync += self.i_idx.eq(self.i_idx + 1)
                    m.d.comb += [
                        rd_w.en.eq(1),
                        rd_w.addr.eq(self.o_idx * self.IN_D + self.i_idx + 1),
                    ]
                    m.next = "LOAD_MUL_INPUTS"

            with m.State("POST_PROCESS"):
                # clamp -> optional relu -> re-clip -> truncate toward zero while
                # narrowing NNQ_DW -> out_shape. matches cdcc/conv1d POST_PROCESS.
                acc = self.accumulator.as_value()

                # clamp to [lower, upper]
                clipped = Mux(
                    acc < lower,
                    lower,
                    Mux(acc > upper, upper, acc),
                )

                # optional relu: negatives -> 0, cap at relu_upper_bound
                if self.apply_relu:
                    relu_ub = self.relu_upper_bound.as_value()
                    post = Mux(
                        clipped < 0,
                        0,
                        Mux(clipped > relu_ub, relu_ub, clipped),
                    )
                else:
                    post = clipped

                # re-clip ( matches fxpmath/qkeras ) then truncate toward zero
                # while narrowing NNQ_DW -> out_shape.
                acc_clipped = Mux(
                    post < lower,
                    lower,
                    Mux(post > upper, upper, post),
                )
                frac_nonzero = acc_clipped[:frac_drop].any()
                trunc_toward_zero = Mux(
                    acc_clipped[-1] & frac_nonzero,
                    acc_clipped + (1 << frac_drop),
                    acc_clipped,
                )
                m.d.sync += self.result[self.o_idx].eq(
                    trunc_toward_zero[frac_drop : frac_drop + out_width].as_signed()
                )

                m.d.sync += self.i_idx.eq(0)
                with m.If(self.o_idx == self.OUT_D - 1):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += self.o_idx.eq(self.o_idx + 1)
                    m.next = "PREFETCH_BIAS"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
