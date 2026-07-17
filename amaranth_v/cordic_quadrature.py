import math

from amaranth.lib import stream, wiring, data
from amaranth import Array, Const, Module, Signal, signed, unsigned

raise Exception("just use LUTs?")

class CordicQuadrature(wiring.Component):
    # drop-in replacement for QuadratureGenerator: same constructor args and
    # same interface ( zero-width handshake in, (sin, cos) pair out ), but the
    # pair is computed with an iterative CORDIC rotation instead of a lookup
    # table. one rotation step per cycle keeps the combinational path short;
    # o.valid is only asserted once the rotation has converged.

    def __init__(self, sample_rate: int, freq_hz: float, amplitude: float):
        self.sample_rate = sample_rate
        self.freq_hz = freq_hz
        self.amplitude = amplitude
        self.lut_size = max(1, round(sample_rate / freq_hz))
        just_handshake_input = unsigned(0)
        super().__init__(
            {
                "i": wiring.In(stream.Signature(just_handshake_input)),
                "o": wiring.Out(stream.Signature(data.ArrayLayout(NNQ, 2))),
            }
        )

    def elaborate(self, platform):
        m = Module()

        F = NNQ.f_bits  # fractional bits of the fixed point format
        SCALE = 1 << F  # real -> scaled-integer factor
        ITERS = F + 2  # cordic iterations ( ~one bit per step )

        # working width: NNQ storage plus a few guard bits for the rotation
        # accumulators. the phase accumulator reaches ~2*pi (< 8) which already
        # fits in NNQ's 4 integer bits.
        GUARD = 4
        W = NNQ.width + GUARD

        # cordic gain K = prod sqrt(1 + 2^-2i). prescale the start vector by 1/K
        # ( x0 = amplitude/K, y0 = 0 ) so the converged result is at the
        # requested amplitude.
        K = 1.0
        for i in range(ITERS):
            K *= math.sqrt(1 + 2.0 ** (-2 * i))
        x0 = round((self.amplitude / K) * SCALE)

        # per-iteration rotation angles atan(2^-i), scaled.
        atan_table = Array(
            Const(round(math.atan(2.0**-i) * SCALE), signed(W)) for i in range(ITERS)
        )

        half_pi = round((math.pi / 2) * SCALE)
        pi_c = round(math.pi * SCALE)
        three_half_pi = round((3 * math.pi / 2) * SCALE)
        two_pi = round(2 * math.pi * SCALE)
        delta = round((2 * math.pi / self.lut_size) * SCALE)

        # phase accumulator ( radians, scaled ), kept in [0, 2*pi). idx tracks
        # the step within a period so we can reset the accumulator exactly on
        # wrap and avoid fixed-point drift.
        z_acc = Signal(signed(W), init=0)
        idx = Signal(range(self.lut_size), init=0)

        # cordic working registers
        x = Signal(signed(W))
        y = Signal(signed(W))
        z = Signal(signed(W))
        neg = Signal()
        i_cnt = Signal(range(ITERS))

        # outputs truncated back to NNQ storage ( magnitudes stay < amplitude
        # < 1, so the low NNQ.width bits are exact ).
        sin_out = Signal(signed(NNQ.width))
        cos_out = Signal(signed(NNQ.width))
        m.d.comb += [
            self.o.payload[0].eq(NNQ(sin_out)),
            self.o.payload[1].eq(NNQ(cos_out)),
        ]

        with m.FSM():
            with m.State("FOLD"):
                # fold z_acc from [0, 2*pi) into cordic's convergence range
                # [-pi/2, pi/2], remembering whether the result must be negated
                # ( angles in the left half plane, i.e. shifted by pi ).
                with m.If(z_acc < half_pi):
                    m.d.sync += [z.eq(z_acc), neg.eq(0)]
                with m.Elif(z_acc < three_half_pi):
                    m.d.sync += [z.eq(z_acc - pi_c), neg.eq(1)]
                with m.Else():
                    m.d.sync += [z.eq(z_acc - two_pi), neg.eq(0)]
                m.d.sync += [
                    x.eq(x0),
                    y.eq(0),
                    i_cnt.eq(0),
                ]
                m.next = "ROTATE"

            with m.State("ROTATE"):
                # one cordic rotation step: drive z towards zero, rotating
                # (x, y) by +/- atan(2^-i) accordingly.
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
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += i_cnt.eq(i_cnt + 1)

            with m.State("DONE"):
                # apply the quadrant sign flip; x -> cos, y -> sin.
                with m.If(neg):
                    m.d.comb += [sin_out.eq(-y), cos_out.eq(-x)]
                with m.Else():
                    m.d.comb += [sin_out.eq(y), cos_out.eq(x)]
                m.d.comb += [
                    self.o.valid.eq(self.i.valid),
                    self.i.ready.eq(self.o.ready),
                ]
                with m.If(self.i.valid & self.o.ready):
                    # advance phase; reset the accumulator exactly at the end of
                    # a period so accumulated rounding does not drift.
                    with m.If(idx == self.lut_size - 1):
                        m.d.sync += [idx.eq(0), z_acc.eq(0)]
                    with m.Else():
                        m.d.sync += [idx.eq(idx + 1), z_acc.eq(z_acc + delta)]
                    m.next = "FOLD"

        return m
