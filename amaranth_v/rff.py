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

from amaranth import Module, Signal, signed, Array
from amaranth.lib import wiring, stream, data
from amaranth.lib.memory import Memory
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

        # cos(x) == sin(x + pi/2): so can use same ROM for both
        _q = self._lut_size // 4
        _mask = self._lut_size - 1
        for _i in range(self._lut_size):
            assert self._cos_lut[_i] == self._sin_lut[(_i + _q) & _mask], (
                f"cos/sin quarter-turn identity failed at index {_i}; "
                "cannot share a single ROM"
            )

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
        lut_size = self._lut_size
        q = lut_size // 4  # quarter turn: cos(x) = sin(x + pi/2)
        mask = lut_size - 1

        # store B as memory
        b_mem = Memory(
            shape=signed(self._b_bits),
            depth=self._num_features,
            init=self._b_codes,
            attrs={"ram_style": "block"},
        )
        m.submodules["b_mem"] = b_mem
        rd_b = b_mem.read_port(domain="sync")

        # shared memory for both sin ane cos
        # cos = trig[(idx + lut_size/4) & mask].
        trig_mem = Memory(
            shape=signed(io_bits),
            depth=lut_size,
            init=self._sin_lut,
            attrs={"ram_style": "block"},
        )
        m.submodules["trig_mem"] = trig_mem
        rd_sin = trig_mem.read_port(domain="sync")
        rd_cos = trig_mem.read_port(domain="sync")

        phase = Signal(signed(io_bits))
        k = Signal(range(K + 1))
        prod = Signal(signed(io_bits + self._b_bits))
        cos_arr = Array(Signal(signed(io_bits), name=f"cos_{j}") for j in range(K))
        sin_arr = Array(Signal(signed(io_bits), name=f"sin_{j}") for j in range(K))

        for j in range(K):
            m.d.comb += self.o.payload[j].eq(cos_arr[j])
            m.d.comb += self.o.payload[K + j].eq(sin_arr[j])

        # fractional-turn LUT index for the (registered) product of this feature
        idx = Signal(range(lut_size))
        m.d.comb += idx.eq((prod >> self._shift)[: self._lut_bits])

        m.d.comb += [
            rd_sin.en.eq(0),
            rd_sin.addr.eq(0),
            rd_cos.en.eq(0),
            rd_cos.addr.eq(0),
            rd_b.en.eq(0),
            rd_b.addr.eq(0),
        ]

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += phase.eq(self.i.payload)
                    m.d.sync += k.eq(0)
                    # prefetch B[0] so it is ready in the MUL state.
                    m.d.comb += [rd_b.en.eq(1), rd_b.addr.eq(0)]
                    m.next = "MUL"

            with m.State("MUL"):
                with m.If(k == K):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += prod.eq(phase * rd_b.data)
                    m.next = "ADDR"

            with m.State("ADDR"):
                # drive both read ports -- sin at idx, cos a quarter
                # turn ahead. data is valid next cycle (synchronous EBR read).
                m.d.comb += [
                    rd_sin.en.eq(1),
                    rd_sin.addr.eq(idx),
                    rd_cos.en.eq(1),
                    rd_cos.addr.eq((idx + q) & mask),
                ]
                m.next = "READ"

            with m.State("READ"):
                # read sin and cos
                m.d.sync += sin_arr[k].eq(rd_sin.data)
                m.d.sync += cos_arr[k].eq(rd_cos.data)
                m.d.sync += k.eq(k + 1)
                # prefetch B[k+1] for the next MUL
                m.d.comb += [rd_b.en.eq(1), rd_b.addr.eq(k + 1)]
                m.next = "MUL"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
