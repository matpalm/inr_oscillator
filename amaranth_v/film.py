"""amaranth_v version of qkeras FiLM

out_c = post( (1 + gamma_c) * h_c + beta_c )

where 'h' is the (NNQ) pre-activation from an 'mlp{i}' QDense layer and
gamma/beta are the (NNQ) modulation vectors produced by the
film{i}_gamma / film{i}_beta QDense layers

'post' is the same statemachine QDenseLayer uses
clamp -> activation -> re-clip -> truncate-toward-zero tail
narrowing the double-width accumulator back to NNQ.

"""

from amaranth import Array, Module, Mux, Signal, signed
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from amaranth_future import fixed

from . import NNQ
from .siren_cordic import SirenCordic

class FiLMCombine(wiring.Component):

    def __init__(
        self,
        dim: int,
        relu_upper_bound: float | None,
        activation: str = "relu",
        siren_omega_0: float = 30.0,
    ):
        self.dim = dim
        self.relu_upper_bound = relu_upper_bound
        self.activation = str(activation)
        self.siren_omega_0 = float(siren_omega_0)
        if self.activation not in {"relu", "siren", "none"}:
            raise ValueError(
                f"unsupported activation={self.activation}; expected relu/siren/none"
            )
        if self.activation == "relu" and self.relu_upper_bound is None:
            raise ValueError("relu_upper_bound is required when activation='relu'")
        if self.activation == "siren" and self.siren_omega_0 <= 0.0:
            raise ValueError(f"siren_omega_0 must be > 0, got {self.siren_omega_0}")

        # products of two NNQ values live at 2*NNQ.f_bits; size the integer part
        # for the worst-case (1+gamma)*h plus a couple of hacky extra bits (?)
        acc_i_bits = 2 * NNQ.i_bits + 2
        acc_f_bits = 2 * NNQ.f_bits
        self.acc_shape = fixed.SQ(acc_i_bits, acc_f_bits)
        self.frac_drop = acc_f_bits - NNQ.f_bits
        self.beta_shift = NNQ.f_bits  # NNQ (2^-f) -> acc (2^-2f)

        if self.activation == "relu":
            self.relu_bound = fixed.Const(relu_upper_bound, shape=self.acc_shape)
        self.lower_bound = fixed.Const(
            NNQ.min().as_float(), shape=self.acc_shape, clamp=True
        ).as_value()
        self.upper_bound = fixed.Const(
            NNQ.max().as_float(), shape=self.acc_shape, clamp=True
        ).as_value()

        super().__init__(
            {
                "i_h": In(stream.Signature(data.ArrayLayout(NNQ, dim))),
                "i_gamma": In(stream.Signature(data.ArrayLayout(NNQ, dim))),
                "i_beta": In(stream.Signature(data.ArrayLayout(NNQ, dim))),
                "o": Out(stream.Signature(data.ArrayLayout(NNQ, dim))),
            }
        )

    def elaborate(self, platform):
        m = Module()

        h_reg = Array(Signal(NNQ, name=f"film_h_{c}") for c in range(self.dim))
        g_reg = Array(Signal(NNQ, name=f"film_g_{c}") for c in range(self.dim))
        b_reg = Array(Signal(NNQ, name=f"film_b_{c}") for c in range(self.dim))
        result = Array(Signal(NNQ, name=f"film_res_{c}") for c in range(self.dim))

        c_idx = Signal(range(self.dim), init=0)

        # (1 + gamma) carries one extra integer bit over NNQ; +1 guard for safety.
        mul_a = Signal(signed(NNQ.width + 2), name="film_mul_a")
        mul_b = Signal(signed(NNQ.width), name="film_mul_b")

        accumulator = Signal(self.acc_shape, name="film_acc", init=0)
        post_clamped = Signal(signed(self.acc_shape.width), name="film_post_clamped")
        post_reclipped = Signal(
            signed(self.acc_shape.width), name="film_post_reclipped"
        )
        trunc_reg = Signal(signed(NNQ.width), name="film_trunc_reg")

        if self.activation == "siren":
            m.submodules.siren = siren = SirenCordic(self.siren_omega_0)

        nnq_one = 1 << NNQ.f_bits  # representation of 1.0 for gamma add

        for c in range(self.dim):
            m.d.comb += self.o.payload[c].eq(result[c])

        m.d.comb += [
            self.i_h.ready.eq(0),
            self.i_gamma.ready.eq(0),
            self.i_beta.ready.eq(0),
            self.o.valid.eq(0),
        ]
        if self.activation == "siren":
            m.d.comb += [
                siren.i.valid.eq(0),
                siren.i.payload.eq(0),
                siren.o.ready.eq(0),
            ]

        frac_drop = self.frac_drop
        out_width = NNQ.width
        lower = self.lower_bound
        upper = self.upper_bound

        with m.FSM():
            with m.State("IDLE"):
                # atomic 3-way join: only accept when all inputs are valid, so
                # an early producer (gamma/beta depend only on the embedding)
                # never completes its handshake before h has arrived.
                all_valid = self.i_h.valid & self.i_gamma.valid & self.i_beta.valid
                m.d.comb += [
                    self.i_h.ready.eq(all_valid),
                    self.i_gamma.ready.eq(all_valid),
                    self.i_beta.ready.eq(all_valid),
                ]
                with m.If(all_valid):
                    for c in range(self.dim):
                        m.d.sync += [
                            h_reg[c].eq(self.i_h.payload[c]),
                            g_reg[c].eq(self.i_gamma.payload[c]),
                            b_reg[c].eq(self.i_beta.payload[c]),
                        ]
                    m.d.sync += c_idx.eq(0)
                    m.next = "LOAD_MUL_INPUTS"

            with m.State("LOAD_MUL_INPUTS"):
                # register the operands before the multiply (helps routing /
                # comb depth, per QDenseLayer).
                m.d.sync += [
                    mul_a.eq(nnq_one + g_reg[c_idx].as_value().as_signed()),
                    mul_b.eq(h_reg[c_idx].as_value().as_signed()),
                ]
                m.next = "MAC"

            with m.State("MAC"):
                # (1+gamma)*h at 2*frac, plus beta lifted from frac to 2*frac.
                m.d.sync += accumulator.eq(
                    (mul_a * mul_b)
                    + (b_reg[c_idx].as_value().as_signed() << self.beta_shift)
                )
                m.next = "CLAMP"

            with m.State("CLAMP"):
                acc = accumulator.as_value()
                m.d.sync += post_clamped.eq(
                    Mux(acc < lower, lower, Mux(acc > upper, upper, acc))
                )
                m.next = "RELU"

            with m.State("RELU"):
                if self.activation == "relu":
                    relu_ub = self.relu_bound.as_value()
                    post = Mux(
                        post_clamped < 0,
                        0,
                        Mux(post_clamped > relu_ub, relu_ub, post_clamped),
                    )
                else:
                    post = post_clamped
                m.d.sync += post_reclipped.eq(
                    Mux(post < lower, lower, Mux(post > upper, upper, post))
                )
                m.next = "TRUNCATE"

            with m.State("TRUNCATE"):
                acc_clipped = post_reclipped
                frac_nonzero = acc_clipped[:frac_drop].any()
                trunc_toward_zero = Mux(
                    acc_clipped[-1] & frac_nonzero,
                    acc_clipped + (1 << frac_drop),
                    acc_clipped,
                )
                trunc = trunc_toward_zero[frac_drop : frac_drop + out_width].as_signed()
                if self.activation == "siren":
                    m.d.sync += trunc_reg.eq(trunc)
                    m.next = "SIREN_SEND"
                else:
                    m.d.sync += result[c_idx].eq(trunc)
                    with m.If(c_idx == self.dim - 1):
                        m.next = "DONE"
                    with m.Else():
                        m.d.sync += c_idx.eq(c_idx + 1)
                        m.next = "LOAD_MUL_INPUTS"

            if self.activation == "siren":
                with m.State("SIREN_SEND"):
                    m.d.comb += [
                        siren.i.valid.eq(1),
                        siren.i.payload.eq(trunc_reg),
                    ]
                    with m.If(siren.i.ready):
                        m.next = "SIREN_RECV"

                with m.State("SIREN_RECV"):
                    m.d.comb += siren.o.ready.eq(1)
                    with m.If(siren.o.valid):
                        m.d.sync += result[c_idx].eq(siren.o.payload)
                        with m.If(c_idx == self.dim - 1):
                            m.next = "DONE"
                        with m.Else():
                            m.d.sync += c_idx.eq(c_idx + 1)
                            m.next = "LOAD_MUL_INPUTS"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
