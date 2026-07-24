from numpy.typing import NDArray
import numpy as np

from amaranth import Array, Module, Mux, Signal, signed
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory

from amaranth_future import fixed

from . import NNQ, parse_nnq


# per-output-column fixed cost of a QDenseLayer eval, in sync cycles: the
# PREFETCH_BIAS / LOAD_BIAS / CLAMP / RELU / TRUNCATE states plus the MAC
# pipeline drain. Used only to rank layers when allocating parallel MAC lanes.
_DENSE_GROUP_OVERHEAD = 6


def allocate_mlp_lanes(layer_dims, budget, overhead=_DENSE_GROUP_OVERHEAD):
    """Greedily split a MAC-lane (=multiplier) budget across dense layers.

    Each layer computes ``ceil(out_d / lanes)`` groups of ``in_d`` taps, so a
    layer's eval cost is ``groups * (in_d + overhead)`` cycles. Starting from one
    lane per layer, repeatedly hand the next lane increment (the smallest that
    actually removes a group -- skipping plateaus where extra lanes don't change
    the group count) to whichever layer saves the most cycles per extra lane,
    until the budget is spent or no layer benefits.

    Args:
        layer_dims  list of (in_d, out_d) for the layers sharing the budget
        budget      total number of MAC lanes (multipliers) to distribute
    Returns:
        list of lane counts, one per layer (each in 1..out_d)
    """

    def groups(out_d, lanes):
        return (out_d + lanes - 1) // lanes

    def cost(in_d, out_d, lanes):
        return groups(out_d, lanes) * (in_d + overhead)

    lanes = [1] * len(layer_dims)
    remaining = budget - sum(lanes)
    while remaining > 0:
        best = None  # (ratio, extra, idx, new_lanes)
        for idx, (in_d, out_d) in enumerate(layer_dims):
            cur = lanes[idx]
            if cur >= out_d:
                continue
            cur_g = groups(out_d, cur)
            # smallest lane count > cur that actually reduces the group count.
            new_lanes = cur + 1
            while new_lanes < out_d and groups(out_d, new_lanes) == cur_g:
                new_lanes += 1
            extra = new_lanes - cur
            if extra > remaining:
                continue
            saving = cost(in_d, out_d, cur) - cost(in_d, out_d, new_lanes)
            if saving <= 0:
                continue
            ratio = saving / extra
            if best is None or ratio > best[0]:
                best = (ratio, extra, idx, new_lanes)
        if best is None:
            break
        _, extra, idx, new_lanes = best
        lanes[idx] = new_lanes
        remaining -= extra
    return lanes


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
        n_lanes: int = 1,
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
            n_lanes                     number of parallel MAC lanes (output columns
                                        computed simultaneously from one shared input
                                        stream). Clamped to out_d. 1 == fully
                                        sequential (one multiplier).
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

        # output-column parallelism: split the out_d columns across n_lanes MAC
        # lanes so P columns are computed simultaneously from a single shared
        # input stream. Lane p owns weight/bias bank p, which holds the output
        # columns o with o % P == p (i.e. column g*P + p is the g-th entry of
        # bank p). Every bank is laid out group-major so ONE running address
        # walks all P banks in lock-step -- P weight reads per clock without a
        # multi-port RAM. Cycles/layer drop from out_d*in_d to ceil(out_d/P)*in_d.
        self.n_lanes = max(1, min(n_lanes, self.out_d))
        P = self.n_lanes
        self.num_groups = (self.out_d + P - 1) // P
        self.bank_depth = self.num_groups * self.in_d
        self.num_weights = self.in_d * self.out_d

        # kernel as [out_d][in_d], row-major.
        weight_rows = np_kernel.T

        if np_bias is not None:
            bias_vals = list(np_bias)
        else:
            bias_vals = [0.0] * self.out_d

        self.weight_banks = []
        self.bias_banks = []
        for p in range(P):
            w_bank_init = []
            b_bank_init = []
            for g in range(self.num_groups):
                o = g * P + p
                for i in range(self.in_d):
                    if o < self.out_d:
                        try:
                            w_bank_init.append(parse_nnq(weight_rows[o][i], shape=NNQ))
                        except ValueError as e:
                            raise Exception(
                                f"!!!!!!!! weight_init o={o} i={i} "
                                f"{weight_rows[o][i]}",
                                e,
                            )
                    else:
                        # padding column (out_d not divisible by P): zero weights
                        # produce a discarded accumulator, never written to result.
                        w_bank_init.append(parse_nnq(0.0, shape=NNQ))
                if o < self.out_d:
                    b_bank_init.append(parse_nnq(bias_vals[o], shape=self.acc_shape))
                else:
                    b_bank_init.append(parse_nnq(0.0, shape=self.acc_shape))

            self.weight_banks.append(
                Memory(
                    shape=NNQ,
                    depth=self.bank_depth,
                    init=w_bank_init,
                    attrs={"ram_style": "block"},
                )
            )
            self.bias_banks.append(
                Memory(
                    shape=self.acc_shape,
                    depth=self.num_groups,
                    init=b_bank_init,
                    attrs={"ram_style": "block"},
                )
            )

        # one accumulator per lane.
        self.accumulator = Array(
            Signal(self.acc_shape, name=f"qdense_running_accum_{p}", init=0)
            for p in range(P)
        )

        self.input = Array(
            Signal(in_shape, name=f"qdense_in_{i}", init=0) for i in range(self.in_d)
        )

        self.result = Array(
            Signal(out_shape, name=f"qdense_result_{j}", init=0)
            for j in range(self.out_d)
        )
        self.g_idx = Signal(range(self.num_groups), init=0)

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
        P = self.n_lanes

        # per-lane weight/bias banks, each with its own 1-cycle-latency read port.
        rd_w = []
        rd_b = []
        for p in range(P):
            m.submodules[f"weight_bank_{p}"] = self.weight_banks[p]
            m.submodules[f"bias_bank_{p}"] = self.bias_banks[p]
            rd_w.append(self.weight_banks[p].read_port(domain="sync"))
            rd_b.append(self.bias_banks[p].read_port(domain="sync"))

        # one shared input operand (broadcast to every lane) and one weight
        # operand per lane.
        mul_a = Signal(signed(self.in_shape.width), name="qdense_mul_a")
        mul_b = [Signal(signed(NNQ.width), name=f"qdense_mul_b_{p}") for p in range(P)]

        post_clamped = Array(
            Signal(signed(self.acc_shape.width), name=f"qdense_post_clamped_{p}")
            for p in range(P)
        )
        post_reclipped = Array(
            Signal(signed(self.acc_shape.width), name=f"qdense_post_reclipped_{p}")
            for p in range(P)
        )

        w_addr = Signal(range(self.bank_depth + 1), name="qdense_w_addr")

        # single-address MAC pipeline (3 stages: issue weight read / register
        # operands / accumulate) so we sustain one multiply-accumulate per clock
        # per lane. All P lanes share the address stream, input register and
        # counters; only the multiplier and accumulator are replicated.
        a_reg = Signal(signed(self.in_shape.width), name="qdense_a_reg")
        n_issued = Signal(range(self.in_d + 1), name="qdense_n_issued")
        mac_cnt = Signal(range(self.in_d + 1), name="qdense_mac_cnt")
        s1_valid = Signal(name="qdense_s1_valid")
        s2_valid = Signal(name="qdense_s2_valid")

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
        ]
        for p in range(P):
            m.d.comb += [
                rd_w[p].en.eq(0),
                rd_w[p].addr.eq(0),
                rd_b[p].en.eq(0),
                rd_b[p].addr.eq(0),
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
                        self.g_idx.eq(0),
                        w_addr.eq(0),
                    ]
                    m.next = "PREFETCH_BIAS"

            with m.State("PREFETCH_BIAS"):
                # prep read of this group's per-lane biases ( ready next cycle )
                for p in range(P):
                    m.d.comb += [
                        rd_b[p].en.eq(1),
                        rd_b[p].addr.eq(self.g_idx),
                    ]
                m.next = "LOAD_BIAS"

            with m.State("LOAD_BIAS"):
                # seed each lane's accumulator with its ( double-width ) bias and
                # reset the shared MAC pipeline; STREAM issues the first read.
                for p in range(P):
                    m.d.sync += self.accumulator[p].eq(
                        rd_b[p].data.as_value().as_signed()
                    )
                m.d.sync += [
                    n_issued.eq(0),
                    mac_cnt.eq(0),
                    s1_valid.eq(0),
                    s2_valid.eq(0),
                ]
                m.next = "STREAM"

            with m.State("STREAM"):
                # Pipelined MAC: one multiply-accumulate per clock per lane.
                #
                # stage 1 -- issue the next weight read (same address into every
                # bank), capture the matching input, and rotate the circular
                # input buffer by one. The weight RAMs have 1-cycle read latency,
                # so their data lands in stage 2; a_reg carries the paired input
                # forward to stay aligned. Reading the head of a rotating buffer
                # avoids a wide input multiplexer.
                issue = Signal(name="qdense_issue")
                m.d.comb += issue.eq(n_issued < self.in_d)
                with m.If(issue):
                    for p in range(P):
                        m.d.comb += [
                            rd_w[p].en.eq(1),
                            rd_w[p].addr.eq(w_addr),
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
                    m.d.sync += mul_a.eq(a_reg)
                    for p in range(P):
                        m.d.sync += mul_b[p].eq(rd_w[p].data.as_value().as_signed())
                m.d.sync += s2_valid.eq(s1_valid)

                # stage 3 -- multiply-accumulate one tap per clock in each lane.
                # After the last tap ( mac_cnt == in_d - 1 ) every accumulator is
                # complete.
                with m.If(s2_valid):
                    for p in range(P):
                        m.d.sync += self.accumulator[p].eq(
                            self.accumulator[p].as_value().as_signed()
                            + ((mul_a * mul_b[p]) << self.prod_shift)
                        )
                    with m.If(mac_cnt == self.in_d - 1):
                        m.next = "CLAMP"
                    with m.Else():
                        m.d.sync += mac_cnt.eq(mac_cnt + 1)

            with m.State("CLAMP"):
                # clamp each lane's accumulator to [lower, upper].
                for p in range(P):
                    acc = self.accumulator[p].as_value()
                    m.d.sync += post_clamped[p].eq(
                        Mux(acc < lower, lower, Mux(acc > upper, upper, acc))
                    )
                m.next = "RELU"

            with m.State("RELU"):
                # optional relu, then re-clip to
                # [lower, upper] ( matches fxpmath/qkeras ), per lane.
                for p in range(P):
                    clipped = post_clamped[p]
                    if self.apply_relu:
                        relu_ub = self.relu_upper_bound.as_value()
                        post = Mux(
                            clipped < 0,
                            0,
                            Mux(clipped > relu_ub, relu_ub, clipped),
                        )
                    else:
                        post = clipped
                    m.d.sync += post_reclipped[p].eq(
                        Mux(post < lower, lower, Mux(post > upper, upper, post))
                    )
                m.next = "TRUNCATE"

            with m.State("TRUNCATE"):
                # truncate toward zero while narrowing NNQ_DW -> out_shape, per
                # lane, then scatter the P lane results to their output columns
                # ( column g*P + p ). Static per-group indices avoid a dynamic
                # Array write; padding columns ( o >= out_d ) are dropped.
                trunc = []
                for p in range(P):
                    acc_clipped = post_reclipped[p]
                    frac_nonzero = acc_clipped[:frac_drop].any()
                    trunc_toward_zero = Mux(
                        acc_clipped[-1] & frac_nonzero,
                        acc_clipped + (1 << frac_drop),
                        acc_clipped,
                    )
                    trunc.append(
                        trunc_toward_zero[frac_drop : frac_drop + out_width].as_signed()
                    )

                for g in range(self.num_groups):
                    with m.If(self.g_idx == g):
                        for p in range(P):
                            o = g * P + p
                            if o < self.out_d:
                                m.d.sync += self.result[o].eq(trunc[p])

                with m.If(self.g_idx == self.num_groups - 1):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += self.g_idx.eq(self.g_idx + 1)
                    m.next = "PREFETCH_BIAS"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
