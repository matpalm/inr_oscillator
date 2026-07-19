"""Fixed-point, LUT-based Random Fourier Features (hardware mirror of the
qkeras_v ``QRandomFourierFeatures``).

For each feature ``k`` the layer computes::

    out_k = [cos(2*pi * phase * B_k), sin(2*pi * phase * B_k)]

The product ``phase * B_k`` is formed in integer (fixed-point) arithmetic.
Because cos/sin are 1-periodic in the *turn* ``phase * B_k``, the ``mod 1``
reduction is free: it is simply the fractional bits of the product.  The top
``log2(lut_size)`` bits of that fractional turn index a shared cos/sin ROM.

Output order matches the qkeras concat: ``[cos_0 .. cos_{K-1}, sin_0 .. sin_{K-1}]``.

All tables (``b_codes``, ``cos_lut``, ``sin_lut``) are supplied as integer codes
so this module is bit-exact with the numpy golden model in
``qkeras_v.rff_lut`` that generates the very same integers.

One feature is processed per clock; a stream handshake gates input/output.
This is a verification-oriented implementation, not an area-optimised one.
"""

from amaranth import Module, Signal, signed, Array, Const
from amaranth.lib import wiring, stream, data
from amaranth.lib.wiring import In, Out

import numpy as np
from qkeras_v.rff_lut import build_io_luts, frac_bits, plan_shift

class RandomFourierFeaturesLUT(wiring.Component):

    def __init__(self, b_codes, cos_lut, sin_lut, io_bits, b_bits, shift):
        self._b_codes = [int(c) for c in b_codes]
        self._cos_lut = [int(c) for c in cos_lut]
        self._sin_lut = [int(c) for c in sin_lut]
        self._io_bits = int(io_bits)
        self._b_bits = int(b_bits)
        self._shift = int(shift)
        self._num_features = len(self._b_codes)
        self._lut_size = len(self._cos_lut)

        assert len(self._sin_lut) == self._lut_size, "cos/sin LUTs must match"
        assert (
            self._lut_size >= 2 and (self._lut_size & (self._lut_size - 1)) == 0
        ), "lut_size must be a power of two"
        assert (
            self._shift >= 0
        ), "shift must be non-negative (io_frac + b_frac >= log2(lut_size))"
        self._lut_bits = (self._lut_size - 1).bit_length()

        super().__init__(
            {
                "i": In(stream.Signature(signed(self._io_bits))),
                "o": Out(
                    stream.Signature(
                        data.ArrayLayout(signed(self._io_bits), 2 * self._num_features)
                    )
                ),
            }
        )

    @classmethod
    def from_rff(cls, B, quant_sizes, lut_size):
        """Build the component from a pickled ``rff`` entry (see ``qkeras_v.train``).
        ``{"B", "b_bits", "b_integer", "io_bits", "io_integer"}``.  The B codes and
        cos/sin LUT are derived with ``qkeras_v.rff_lut`` (the same helpers the
        numpy golden model uses), so the resulting hardware stays bit-exact.
        """

        b_bits, b_integer = quant_sizes["b_bits"], quant_sizes["b_int"]
        io_bits, io_integer = quant_sizes["io_bits"], quant_sizes["io_int"]

        # RFF input is the scalar phase (in_dim == 1); B is (in_dim, num_features).
        b_f = frac_bits(b_bits, b_integer)
        B = np.asarray(B).reshape(-1)
        b_codes = np.round(B * (2.0**b_f)).astype(np.int64).tolist()

        shift, _ = plan_shift(io_bits, io_integer, b_bits, b_integer, lut_size)
        if shift < 0:
            raise ValueError(
                f"lut_size={lut_size} too large for io_frac+b_frac="
                f"{frac_bits(io_bits, io_integer) + b_f}"
            )
        cos_lut, sin_lut = build_io_luts(lut_size, io_bits, io_integer)

        return cls(
            b_codes=b_codes,
            cos_lut=cos_lut.tolist(),
            sin_lut=sin_lut.tolist(),
            io_bits=io_bits,
            b_bits=b_bits,
            shift=shift,
        )

    def elaborate(self, platform):
        m = Module()

        K = self._num_features
        io_bits = self._io_bits

        b_rom = Array(Const(c, signed(self._b_bits)) for c in self._b_codes)
        cos_rom = Array(Const(c, signed(io_bits)) for c in self._cos_lut)
        sin_rom = Array(Const(c, signed(io_bits)) for c in self._sin_lut)

        phase = Signal(signed(io_bits))
        k = Signal(range(K + 1))
        cos_arr = Array(Signal(signed(io_bits), name=f"cos_{j}") for j in range(K))
        sin_arr = Array(Signal(signed(io_bits), name=f"sin_{j}") for j in range(K))

        for j in range(K):
            m.d.comb += self.o.payload[j].eq(cos_arr[j])
            m.d.comb += self.o.payload[K + j].eq(sin_arr[j])

        # fixed-point product and fractional-turn LUT index for the current feature
        prod = Signal(signed(io_bits + self._b_bits))
        m.d.comb += prod.eq(phase * b_rom[k])
        idx = Signal(range(self._lut_size))
        m.d.comb += idx.eq((prod >> self._shift)[: self._lut_bits])

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += phase.eq(self.i.payload)
                    m.d.sync += k.eq(0)
                    m.next = "COMPUTE"

            with m.State("COMPUTE"):
                with m.If(k == K):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += cos_arr[k].eq(cos_rom[idx])
                    m.d.sync += sin_arr[k].eq(sin_rom[idx])
                    m.d.sync += k.eq(k + 1)

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
