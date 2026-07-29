"""V/oct ramp (sawtooth) oscillator

takes 5 -> 7V (clipped) and maps to a C3 -> C5 sawtooth ramp
( with A4=440Hz as reference ).

assumes 48kHz sample rate.

see also quadrature_v_oct for a version that output sin/cos pair
"""

import math

from amaranth import Array, Const, Module, Signal, signed, unsigned
from amaranth.lib import stream, wiring
from amaranth.lib.wiring import In, Out

from amaranth_future import fixed

from amaranth_v import NNQ


class RampVOct(wiring.Component):

    FS_HZ = 192_000  # sample rate  TODO! configure this with --fs-192
    A4_HZ = 440.0  # tuning reference
    V_MIN = 4.0  # lower bound input volts -> C2
    V_MAX = 8.0  # upper bound input volts -> C6
    # F0_HZ and octave span derive from V_MIN/V_MAX.
    F0_HZ = A4_HZ * 2 ** (((12.0 * (V_MIN - 1.0)) - 69.0) / 12.0)
    OCTAVES = V_MAX - V_MIN
    PITCH_BITS = 10  # log2 of pitch-rom resolution (2**PITCH_BITS + 1 entries)

    # output sawtooth spans [-RAMP_V, +RAMP_V) volts, one ramp per cycle.
    # note: needs to match the phase convention / range the model is trained on.
    RAMP_V = 5.0

    def __init__(self, i_shape=NNQ, o_shape=None):
        # ``i_shape`` is the fixed-point shape of the incoming v/oct control
        # value; ``o_shape`` is the shape of the emitted phase ramp (defaults to
        # ``i_shape``). They may differ so a caller can feed audio (e.g. ASQ)
        # straight in and read a network-io-shaped ramp straight out, with no
        # intermediate re-quantisation.
        self.i_shape = i_shape
        self.o_shape = i_shape if o_shape is None else o_shape
        super().__init__(
            {
                "i": In(stream.Signature(self.i_shape)),
                "o": Out(stream.Signature(self.o_shape)),
            }
        )

    def elaborate(self, platform):
        m = Module()

        IN_F = self.i_shape.f_bits  # input (v/oct) fractional bits
        IN_SCALE = 1 << IN_F  # real -> input scaled-integer factor
        OUT_F = self.o_shape.f_bits  # output (phase ramp) fractional bits
        OUT_SCALE = 1 << OUT_F  # real -> output scaled-integer factor

        # working width: output storage plus a few guard bits for the phase
        # accumulator ( which spans OUT_RANGE ~ 10, i.e. 4 integer bits ).
        GUARD = 4
        W = self.o_shape.width + GUARD

        # peak-to-peak span of one ramp ( -RAMP_V .. +RAMP_V ), in output units.
        OUT_RANGE = 2.0 * self.RAMP_V
        range_code = round(OUT_RANGE * OUT_SCALE)  # accumulator wrap point
        offset_code = round(
            -self.RAMP_V * OUT_SCALE
        )  # shift [0, range) -> [-RAMP_V, +RAMP_V)

        # --- pitch ( control voltage ) -> ramp phase increment ----------------
        # A combinational ROM maps the input to the per-sample ramp increment
        # delta = OUT_RANGE * freq / fs ( scaled ), so one full ramp takes
        # exactly fs / freq samples. The 2**V exponential is baked into the
        # constants, so no runtime multiply ( and no DSP ) is needed. Depth is
        # PITCH_SIZE + 1 so the full-scale input (V_MAX) lands exactly on the
        # configured top pitch.
        PITCH_SIZE = 1 << self.PITCH_BITS

        def _delta_code(addr):
            frac = addr / PITCH_SIZE  # 0 .. 1  (fraction of the V_MIN..V_MAX range)
            volts_above = self.OCTAVES * frac  # 0 .. (V_MAX - V_MIN)
            freq = self.F0_HZ * (2.0**volts_above)  # 1V/oct above V_MIN
            return round((OUT_RANGE * freq / self.FS_HZ) * OUT_SCALE)

        delta_rom = Array(
            Const(_delta_code(addr), signed(W)) for addr in range(PITCH_SIZE + 1)
        )

        # clip the input to [V_MIN, V_MAX] volts, then map the volts above
        # V_MIN onto the ROM address. The lookup is pipelined across two
        # registers so the (large) ROM mux is isolated from the accumulator
        # adder, keeping the combinational path short ( important for timing
        # closure in a full SoC build ). The 2 cycle latency is irrelevant: the
        # input is quasi-static per sample.
        code_min = round(self.V_MIN * IN_SCALE)  # V_MIN in input codes
        # fold the (V_MAX - V_MIN) volt window down onto PITCH_SIZE addresses.
        span_shift = round(math.log2((self.V_MAX - self.V_MIN) * IN_SCALE / PITCH_SIZE))
        x_clipped = self.i.payload.clamp(
            fixed.Const(self.V_MIN, self.i_shape, clamp=True),
            fixed.Const(self.V_MAX, self.i_shape, clamp=True),
        )
        x_code = Signal(unsigned(self.i_shape.width))
        m.d.comb += x_code.eq(x_clipped.as_value())
        # stage 1: (clipped) codes above V_MIN -> registered ROM address.
        pitch_addr = Signal(range(PITCH_SIZE + 1))
        m.d.sync += pitch_addr.eq((x_code - code_min) >> span_shift)
        # stage 2: ROM address -> registered ramp increment.
        delta = Signal(signed(W))
        m.d.sync += delta.eq(delta_rom[pitch_addr])

        # --- phase accumulator ------------------------------------------------
        # acc is the rising phase in [0, range_code); the output is that phase
        # offset down to [-RAMP_V, +RAMP_V).
        acc = Signal(signed(W), init=0)

        ramp_out = Signal(signed(self.o_shape.width))
        m.d.comb += ramp_out.eq(acc + offset_code)
        m.d.comb += self.o.payload.eq(self.o_shape(ramp_out))

        # one ramp sample per input handshake; advance ( and wrap ) the phase.
        m.d.comb += [
            self.o.valid.eq(self.i.valid),
            self.i.ready.eq(self.o.ready),
        ]
        with m.If(self.i.valid & self.o.ready):
            with m.If(acc + delta >= range_code):
                m.d.sync += acc.eq(acc + delta - range_code)
            with m.Else():
                m.d.sync += acc.eq(acc + delta)

        return m
