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
            apply_relu                  whether to run relu
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

        self.in_d, self.out_d = np_kernel.shape

        if np_bias is not None:
            if len(np_bias.shape) != 1 or np_bias.shape[0] != self.out_d:
                raise Exception(
                    f"Expect QDenseLayer bias with shape ({self.out_d},) "
                    f"but received {np_bias.shape}"
                )

        if apply_relu and relu_upper_bound is None:
            raise Exception("relu_upper_bound is required when apply_relu is set")

        print(
            f">QDenseLayer IN_D={self.in_d} OUT_D={self.out_d} apply_relu={apply_relu}"
            f" ( relu_upper_bound={relu_upper_bound} )"
            f" in_shape={in_shape!r} out_shape={out_shape!r}"
        )

        self.apply_relu = apply_relu
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.acc_shape, self.prod_shift = self.acc_shape_for(
            in_shape, self.in_d, np_bias
        )

        self.num_weights = self.in_d * self.out_d

        # Flattened as [out_d][in_d], row-major.
        weight_rows = np_kernel.T
        weight_init = []
        for o in range(self.out_d):
            for i in range(self.in_d):
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
                parse_nnq(0.0, shape=self.acc_shape) for _ in range(self.out_d)
            ]

        self.bias_mem = Memory(
            shape=self.acc_shape,
            depth=self.out_d,
            init=bias_init,
            attrs={"ram_style": "block"},
        )

        self.accumulator = Signal(self.acc_shape, name="qdense_running_accum", init=0)

        self.input = Array(
            Signal(in_shape, name=f"qdense_in_{i}", init=0) for i in range(self.in_d)
        )

        self.result = Array(
            Signal(out_shape, name=f"qdense_result_{j}", init=0)
            for j in range(self.out_d)
        )
        self.o_idx = Signal(range(self.out_d), init=0)

        if apply_relu:
            self.relu_upper_bound = fixed.Const(relu_upper_bound, shape=self.acc_shape)

        self.lower_bound = fixed.Const(
            out_shape.min().as_float(), shape=self.acc_shape, clamp=True
        ).as_value()
        self.upper_bound = fixed.Const(
            out_shape.max().as_float(), shape=self.acc_shape, clamp=True
        ).as_value()

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(in_shape, self.in_d))),
                "o": wiring.Out(
                    stream.Signature(data.ArrayLayout(out_shape, self.out_d))
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

        post_clamped = Signal(signed(self.acc_shape.width), name="qdense_post_clamped")
        post_reclipped = Signal(
            signed(self.acc_shape.width), name="qdense_post_reclipped"
        )

        w_addr = Signal(range(self.num_weights + 1), name="qdense_w_addr")

        # single-multiplier MAC pipeline (3 stages: issue weight read /
        # register operands / accumulate) so we sustain one multiply-accumulate
        # per clock instead of alternating separate load and mac states.
        a_reg = Signal(signed(self.in_shape.width), name="qdense_a_reg")
        n_issued = Signal(range(self.in_d + 1), name="qdense_n_issued")
        mac_cnt = Signal(range(self.in_d + 1), name="qdense_mac_cnt")
        s1_valid = Signal(name="qdense_s1_valid")
        s2_valid = Signal(name="qdense_s2_valid")

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            rd_w.en.eq(0),
            rd_w.addr.eq(0),
            rd_b.en.eq(0),
            rd_b.addr.eq(0),
        ]

        for j in range(self.out_d):
            m.d.comb += self.o.payload[j].eq(self.result[j])

        frac_drop = self.acc_shape.f_bits - self.out_shape.f_bits
        out_width = self.out_shape.width
        lower = self.lower_bound
        upper = self.upper_bound

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid & self.i.ready):
                    for i in range(self.in_d):
                        m.d.sync += self.input[i].eq(self.i.payload[i])
                    m.d.sync += [
                        self.o_idx.eq(0),
                        w_addr.eq(0),
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
                # seed accumulator with the ( double-width ) bias and reset the
                # MAC pipeline; STREAM issues the first weight read itself.
                m.d.sync += [
                    self.accumulator.eq(rd_b.data.as_value().as_signed()),
                    n_issued.eq(0),
                    mac_cnt.eq(0),
                    s1_valid.eq(0),
                    s2_valid.eq(0),
                ]
                m.next = "STREAM"

            with m.State("STREAM"):
                # Pipelined MAC: one multiply-accumulate per clock.
                #
                # stage 1 -- issue the next weight read, capture the matching
                # input, and rotate the circular input buffer by one. The weight
                # RAM has 1-cycle read latency, so its data lands in stage 2;
                # a_reg carries the paired input forward to stay aligned. Reading
                # the head of a rotating buffer avoids a wide input multiplexer.
                issue = Signal(name="qdense_issue")
                m.d.comb += issue.eq(n_issued < self.in_d)
                with m.If(issue):
                    m.d.comb += [
                        rd_w.en.eq(1),
                        rd_w.addr.eq(w_addr),
                    ]
                    m.d.sync += [
                        w_addr.eq(w_addr + 1),
                        a_reg.eq(self.input[0].as_value().as_signed()),
                        n_issued.eq(n_issued + 1),
                    ]
                    for i in range(self.in_d - 1):
                        m.d.sync += self.input[i].eq(self.input[i + 1])
                    m.d.sync += self.input[self.in_d - 1].eq(self.input[0])
                m.d.sync += s1_valid.eq(issue)

                # stage 2 -- register the multiplier operands ( registering
                # before the MAC keeps comb depth / routing in check ).
                with m.If(s1_valid):
                    m.d.sync += [
                        mul_a.eq(a_reg),
                        mul_b.eq(rd_w.data.as_value().as_signed()),
                    ]
                m.d.sync += s2_valid.eq(s1_valid)

                # stage 3 -- multiply-accumulate one tap per clock. After the
                # last tap ( mac_cnt == in_d - 1 ) the accumulator is complete.
                with m.If(s2_valid):
                    m.d.sync += self.accumulator.eq(
                        self.accumulator.as_value().as_signed()
                        + ((mul_a * mul_b) << self.prod_shift)
                    )
                    with m.If(mac_cnt == self.in_d - 1):
                        m.next = "CLAMP"
                    with m.Else():
                        m.d.sync += mac_cnt.eq(mac_cnt + 1)

            with m.State("CLAMP"):
                # clamp accumulator to [lower, upper].
                acc = self.accumulator.as_value()
                m.d.sync += post_clamped.eq(
                    Mux(acc < lower, lower, Mux(acc > upper, upper, acc))
                )
                m.next = "RELU"

            with m.State("RELU"):
                # optional relu, then re-clip to
                # [lower, upper] ( matches fxpmath/qkeras ).
                clipped = post_clamped
                if self.apply_relu:
                    relu_ub = self.relu_upper_bound.as_value()
                    post = Mux(
                        clipped < 0,
                        0,
                        Mux(clipped > relu_ub, relu_ub, clipped),
                    )
                else:
                    post = clipped
                m.d.sync += post_reclipped.eq(
                    Mux(post < lower, lower, Mux(post > upper, upper, post))
                )
                m.next = "TRUNCATE"

            with m.State("TRUNCATE"):
                # truncate toward zero while narrowing NNQ_DW -> out_shape.
                acc_clipped = post_reclipped
                frac_nonzero = acc_clipped[:frac_drop].any()
                trunc_toward_zero = Mux(
                    acc_clipped[-1] & frac_nonzero,
                    acc_clipped + (1 << frac_drop),
                    acc_clipped,
                )
                m.d.sync += self.result[self.o_idx].eq(
                    trunc_toward_zero[frac_drop : frac_drop + out_width].as_signed()
                )

                with m.If(self.o_idx == self.out_d - 1):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += self.o_idx.eq(self.o_idx + 1)
                    m.next = "PREFETCH_BIAS"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
