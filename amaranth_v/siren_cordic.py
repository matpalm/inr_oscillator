import math

from amaranth import Array, Const, Module, Mux, Signal, signed
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out

from . import NNQ


def siren_cordic_output_codes(
    input_codes: list[int] | tuple[int, ...],
    *,
    width: int,
    frac_bits: int,
    omega_0: float,
    iters: int | None = None,
    guard: int = 10,
) -> list[int]:
    """Reference CORDIC outputs matching SirenCordic's fixed-point math.

    Args:
        input_codes: signed fixed-point input codes.
        width: signed output/input bit width.
        frac_bits: fractional bit count.
        omega_0: siren frequency scale.
        iters: CORDIC iterations; defaults to frac_bits + 2.
        guard: extra working integer guard bits.
    """
    width = int(width)
    frac_bits = int(frac_bits)
    omega_0 = float(omega_0)
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width}")
    if frac_bits < 0:
        raise ValueError(f"frac_bits must be >= 0, got {frac_bits}")
    if omega_0 <= 0.0:
        raise ValueError(f"omega_0 must be > 0, got {omega_0}")

    F = frac_bits
    SCALE = 1 << F
    ITERS = int(F + 2 if iters is None else iters)

    # Keep identical formulas to the hardware module.
    omega_code = int(round(omega_0 * SCALE))
    K = 1.0
    for i in range(ITERS):
        K *= math.sqrt(1 + 2.0 ** (-2 * i))
    x0 = int(round((1.0 / K) * SCALE))

    atan_table = [int(round(math.atan(2.0**-i) * SCALE)) for i in range(ITERS)]
    half_pi = int(round((math.pi / 2) * SCALE))
    pi_c = int(round(math.pi * SCALE))
    two_pi = int(round(2 * math.pi * SCALE))

    lo = -(1 << (width - 1))
    hi = (1 << (width - 1)) - 1
    _working_width = int(width + guard)  # documented shape parity only
    del _working_width

    out = []
    for c in input_codes:
        c = int(c)
        angle_raw = (c * omega_code) >> F

        # Wrap to [-pi, pi) exactly (equivalent to repeated +/-2pi in HDL).
        z_in = angle_raw
        while z_in >= pi_c:
            z_in -= two_pi
        while z_in < -pi_c:
            z_in += two_pi

        if z_in > half_pi:
            z = z_in - pi_c
            neg = 1
        elif z_in < -half_pi:
            z = z_in + pi_c
            neg = 1
        else:
            z = z_in
            neg = 0

        x = x0
        y = 0
        for i in range(ITERS):
            shift_x = x >> i
            shift_y = y >> i
            a = atan_table[i]
            if z >= 0:
                x = x - shift_y
                y = y + shift_x
                z = z - a
            else:
                x = x + shift_y
                y = y - shift_x
                z = z + a

        y_final = -y if neg else y
        y_sat = lo if y_final < lo else hi if y_final > hi else y_final
        out.append(int(y_sat))
    return out


class SirenCordic(wiring.Component):
    """Sequential CORDIC sine for y = sin(omega_0 * x) on NNQ-coded inputs."""

    def __init__(self, omega_0: float = 30.0):
        self.omega_0 = float(omega_0)
        if self.omega_0 <= 0.0:
            raise ValueError(f"omega_0 must be > 0, got {self.omega_0}")

        super().__init__(
            {
                "i": In(stream.Signature(signed(NNQ.width))),
                "o": Out(stream.Signature(signed(NNQ.width))),
            }
        )

    def elaborate(self, platform):
        m = Module()

        F = NNQ.f_bits
        SCALE = 1 << F
        ITERS = F + 2
        W = NNQ.width + 10

        # Prescale by 1/K so CORDIC output lands at unit amplitude.
        K = 1.0
        for i in range(ITERS):
            K *= math.sqrt(1 + 2.0 ** (-2 * i))
        x0 = round((1.0 / K) * SCALE)

        atan_table = Array(
            Const(round(math.atan(2.0**-i) * SCALE), signed(W)) for i in range(ITERS)
        )

        half_pi = round((math.pi / 2) * SCALE)
        pi_c = round(math.pi * SCALE)
        two_pi = round(2 * math.pi * SCALE)
        omega_code = round(self.omega_0 * SCALE)

        in_code = Signal(signed(NNQ.width), init=0)
        mul_w = NNQ.width + W
        prod = Signal(signed(mul_w), init=0)
        z_in = Signal(signed(W), init=0)

        z = Signal(signed(W), init=0)
        x = Signal(signed(W), init=0)
        y = Signal(signed(W), init=0)
        neg = Signal(init=0)
        i_cnt = Signal(range(ITERS), init=0)

        y_sat = Signal(signed(NNQ.width), init=0)
        lo = -(1 << (NNQ.width - 1))
        hi = (1 << (NNQ.width - 1)) - 1

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            self.o.payload.eq(y_sat),
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid & self.i.ready):
                    m.d.sync += in_code.eq(self.i.payload)
                    m.next = "MUL"

            with m.State("MUL"):
                m.d.sync += prod.eq(in_code * omega_code)
                m.next = "SCALE"

            with m.State("SCALE"):
                m.d.sync += z_in.eq((prod >> F).as_signed())
                m.next = "NORM"

            with m.State("NORM"):
                with m.If(z_in > pi_c):
                    m.d.sync += z_in.eq(z_in - two_pi)
                with m.Elif(z_in < -pi_c):
                    m.d.sync += z_in.eq(z_in + two_pi)
                with m.Else():
                    m.next = "FOLD"

            with m.State("FOLD"):
                # Fold to CORDIC convergence interval [-pi/2, pi/2].
                with m.If(z_in < -half_pi):
                    m.d.sync += [z.eq(z_in + pi_c), neg.eq(1)]
                with m.Elif(z_in > half_pi):
                    m.d.sync += [z.eq(z_in - pi_c), neg.eq(1)]
                with m.Else():
                    m.d.sync += [z.eq(z_in), neg.eq(0)]
                m.d.sync += [x.eq(x0), y.eq(0), i_cnt.eq(0)]
                m.next = "ROTATE"

            with m.State("ROTATE"):
                shift_x = x >> i_cnt
                shift_y = y >> i_cnt
                a = atan_table[i_cnt]
                with m.If(z >= 0):
                    m.d.sync += [
                        x.eq(x - shift_y),
                        y.eq(y + shift_x),
                        z.eq(z - a),
                    ]
                with m.Else():
                    m.d.sync += [
                        x.eq(x + shift_y),
                        y.eq(y - shift_x),
                        z.eq(z + a),
                    ]
                with m.If(i_cnt == ITERS - 1):
                    m.next = "PACK"
                with m.Else():
                    m.d.sync += i_cnt.eq(i_cnt + 1)

            with m.State("PACK"):
                y_final = Mux(neg, -y, y)
                sat = Mux(y_final < lo, lo, Mux(y_final > hi, hi, y_final))
                m.d.sync += y_sat.eq(sat)
                m.next = "OUT"

            with m.State("OUT"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
