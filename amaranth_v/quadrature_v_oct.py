"""V/oct quadrature oscillator

takes 5 -> 7V (clipped) and maps to C3 -> c5 sin/cos waves
( with A4=440Hz as reference )

assumes 48kHz sample rate.
"""

import math

from amaranth import Array, Const, Module, Signal, signed, unsigned
from amaranth.lib import stream, wiring, data
from amaranth.lib.wiring import In, Out

from amaranth_future import fixed

from amaranth_v import NNQ


class QuadratureVOct(wiring.Component):

    i: In(stream.Signature(NNQ))
    o: Out(stream.Signature(data.ArrayLayout(NNQ, 2)))

    FS_HZ = 48000  # sample rate
    A4_HZ = 440.0  # tuning reference
    V_MIN = 5.0  # lower bound input volts -> C3
    V_MAX = 7.0  # upper bound input volts -> C5
    F0_HZ = A4_HZ * 2 ** ((48 - 69) / 12)  # C3 (MIDI 48), pitch at V_MIN
    OCTAVES = 2  # octaves (== volts) spanned by V_MIN -> V_MAX
    PITCH_BITS = 10  # log2 of pitch-rom resolution (2**PITCH_BITS + 1 entries)
    AMPLITUDE = 0.99  # output magnitude ( some noisy clipping at exactly 1.0? )

    def elaborate(self, platform):
        m = Module()

        F = NNQ.f_bits  # fractional bits of the fixed point format
        SCALE = 1 << F  # real -> scaled-integer factor
        ITERS = F + 2  # cordic iterations ( ~one bit per step )

        # working width: NNQ storage plus a few guard bits for the rotation
        # accumulators and the phase accumulator ( which reaches ~2*pi < 8,
        # already within NNQ's integer range ).
        GUARD = 4
        W = NNQ.width + GUARD

        # cordic gain K = prod sqrt(1 + 2^-2i). prescale the start vector by
        # 1/K ( x0 = amplitude/K, y0 = 0 ) so the converged result lands at the
        # requested amplitude.
        K = 1.0
        for i in range(ITERS):
            K *= math.sqrt(1 + 2.0 ** (-2 * i))
        x0 = round((self.AMPLITUDE / K) * SCALE)

        # per-iteration rotation angles atan(2^-i), scaled.
        atan_table = Array(
            Const(round(math.atan(2.0**-i) * SCALE), signed(W)) for i in range(ITERS)
        )

        half_pi = round((math.pi / 2) * SCALE)
        pi_c = round(math.pi * SCALE)
        three_half_pi = round((3 * math.pi / 2) * SCALE)
        two_pi = round(2 * math.pi * SCALE)

        # --- pitch ( control voltage ) -> phase increment ---------------------
        # A combinational ROM maps the normalised input to the per-sample phase
        # increment delta = 2*pi*freq/fs ( scaled ). The 2**V exponential is
        # baked into the constants, so no runtime multiply ( and no DSP ) is
        # needed. Depth is PITCH_SIZE + 1 so the full-scale input (V_MAX) lands
        # exactly on C5.
        PITCH_SIZE = 1 << self.PITCH_BITS

        def _delta_code(addr):
            frac = addr / PITCH_SIZE  # 0 .. 1  (fraction of the V_MIN..V_MAX range)
            volts_above = self.OCTAVES * frac  # 0 .. 2  (volts above V_MIN)
            freq = self.F0_HZ * (2.0**volts_above)  # C3 * 2**(V - V_MIN)
            return round((2 * math.pi * freq / self.FS_HZ) * SCALE)

        delta_rom = Array(
            Const(_delta_code(addr), signed(W)) for addr in range(PITCH_SIZE + 1)
        )

        # clip the input to [V_MIN, V_MAX] volts, then map the volts above
        # V_MIN onto the ROM address. The lookup is pipelined across two
        # registers so the (large) ROM mux is isolated from the phase
        # accumulator adder, keeping the combinational path short ( important
        # for timing closure in a full SoC build ). The 2 cycle latency is
        # irrelevant: the input is quasi-static per sample and delta is only
        # consumed ~ITERS cycles later, in DONE.
        code_min = round(self.V_MIN * SCALE)  # V_MIN in NNQ codes
        # fold the (V_MAX - V_MIN) volt window down onto PITCH_SIZE addresses.
        span_shift = round(math.log2((self.V_MAX - self.V_MIN) * SCALE / PITCH_SIZE))
        x_clipped = self.i.payload.clamp(
            fixed.Const(self.V_MIN, NNQ), fixed.Const(self.V_MAX, NNQ)
        )
        x_code = Signal(unsigned(NNQ.width))
        m.d.comb += x_code.eq(x_clipped.as_value())
        # stage 1: (clipped) codes above V_MIN -> registered ROM address.
        pitch_addr = Signal(range(PITCH_SIZE + 1))
        m.d.sync += pitch_addr.eq((x_code - code_min) >> span_shift)
        # stage 2: ROM address -> registered phase increment.
        delta = Signal(signed(W))
        m.d.sync += delta.eq(delta_rom[pitch_addr])

        # --- phase accumulator + cordic working registers ---------------------
        z_acc = Signal(signed(W), init=0)  # phase in [0, 2*pi), scaled
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
                m.d.sync += [x.eq(x0), y.eq(0), i_cnt.eq(0)]
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
                    # advance the phase accumulator, wrapping modulo 2*pi.
                    with m.If(z_acc + delta >= two_pi):
                        m.d.sync += z_acc.eq(z_acc + delta - two_pi)
                    with m.Else():
                        m.d.sync += z_acc.eq(z_acc + delta)
                    m.next = "FOLD"

        return m
