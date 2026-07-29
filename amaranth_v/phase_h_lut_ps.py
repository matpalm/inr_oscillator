"""PSRAM-backed cache of the FiLM ``mlp0`` pre-activation ``h`` as a function
of phase.

In the FiLM topology the first dense layer consumes ONLY the RFF features of
the scalar phase (there is no embedding concat into the main path), so::

    h = mlp0(RFF(phase))

depends solely on phase. Quantising the phase to ``index_bits`` (default 13 =>
8192 distinct codes) means there are only ``2**index_bits`` distinct ``h``
vectors, each ``out_d`` NNQ values. That whole table is too large for EBR, so
we materialise it in PSRAM once at startup and then serve inference as a pure
PSRAM read.

The table is preloaded into PSRAM (from ``phase_h_lut.bin``, e.g. by the
bootloader as a RamLoad region);  on each input phase, derive ``idx`` from its top bits, read
back the ``out_d`` words from ``(idx << dim_bits) | dim_idx`` and present them
on ``o`` -- ``ready`` is asserted from reset.

The ``i`` / ``o`` ports mirror scalar phase input (``signed(io_bits)``) and the
cached ``h`` vector (``ArrayLayout(NNQ, out_d)``), so this is a drop-in
replacement for the ``rff -> mlp0`` sub-chain in
``rff_film_network.RffNetwork``.

The 16-bit NNQ words are packed x2 into 32-bit PSRAM words by ``_WishboneAdapter``
and fronted by ``WishboneL2Cache`` (both vendored in ``wishbone_cache``).
"""

import math

import numpy as np
from amaranth import Module, Signal, signed
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import exact_log2
from amaranth_soc import wishbone

from . import NNQ
from .wishbone_cache import WishboneL2Cache, _WishboneAdapter


class PhaseHLutPS(wiring.Component):

    def __init__(
        self,
        mlp0_kernel,
        io_shape,
        index_bits: int = 13,
        addr_width_o: int = 22,
        base: int = 0,
        cache_kwargs=None,
    ):
        """
        The phase->h table is always preloaded into PSRAM (from
        ``phase_h_lut.bin``, e.g. by the bootloader as a RamLoad region).

        Args:
            mlp0_kernel   first dense kernel; only its shape (out_d) is used to
                          size the table.
            io_shape      fixed-point shape of the phase / rff features
            index_bits    number of phase bits to enumerate; the table holds
                          ``2**index_bits`` entries (default 13 => 8192)
            addr_width_o  external (32-bit) PSRAM wishbone address width
            base          byte offset of this table's region in PSRAM (4-aligned)
            cache_kwargs  extra kwargs forwarded to ``WishboneL2Cache``
        """

        self.io_bits = int(io_shape.width)
        self._io_frac = int(io_shape.f_bits)
        self.index_bits = int(index_bits)

        mlp0_kernel = np.asarray(mlp0_kernel)
        if len(mlp0_kernel.shape) != 2:
            raise ValueError(
                f"expected mlp0_kernel shape (in_d, out_d), got {mlp0_kernel.shape}"
            )
        # only the output dimension (table width) is needed for preload/INFER.
        self.out_d = int(mlp0_kernel.shape[1])

        NNQ_BITS = NNQ.width
        assert NNQ_BITS <= 16, f"psram packing assumes NNQ <= 16bits but was {NNQ_BITS}"

        # out_d must be a power of two so the per-entry dim stride == out_d and
        # the entry index is simply the high address bits (addr = idx<<dim_bits).
        assert (
            self.out_d >= 1 and (self.out_d & (self.out_d - 1)) == 0
        ), f"PhaseHLutPS requires out_d to be a power of two, got {self.out_d}"
        self.dim_bits = exact_log2(self.out_d)
        self.dim_stride = self.out_d

        # phase (io fixed point, io_frac fractional bits) -> index (SQ(1, index_frac)).
        # index_frac = index_bits - 1 (one sign bit). shift aligns io fractional
        # bits down to the index fractional grid.
        self.index_frac = self.index_bits - 1
        self.shift = self._io_frac - self.index_frac
        assert self.shift >= 0, (
            f"index_frac={self.index_frac} exceeds io_frac={self._io_frac};"
            " reduce index_bits"
        )
        assert self.index_bits + self.shift <= self.io_bits, (
            f"index_bits({self.index_bits})+shift({self.shift})"
            f" exceeds io_bits({self.io_bits})"
        )

        self.num_entries = 1 << self.index_bits
        total_words = self.num_entries * self.dim_stride
        self.internal_addr_width = (
            int(math.ceil(math.log2(total_words))) if total_words > 1 else 1
        )

        # internal 16-bit wishbone master, packed to 32-bit PSRAM via adapter.
        self._bus = wishbone.Signature(
            addr_width=self.internal_addr_width,
            data_width=16,
            granularity=8,
        ).create()
        self._adapter = _WishboneAdapter(
            addr_width_i=self.internal_addr_width,
            addr_width_o=addr_width_o,
            base=base,
        )
        _ck = cache_kwargs if cache_kwargs is not None else {}
        self._cache = WishboneL2Cache(addr_width=addr_width_o, **_ck)

        self.bus_signature = wishbone.Signature(
            addr_width=addr_width_o,
            data_width=32,
            granularity=8,
            features={"bte", "cti"},
        )

        super().__init__(
            {
                "i": In(stream.Signature(signed(self.io_bits))),
                "o": Out(stream.Signature(data.ArrayLayout(NNQ, self.out_d))),
                "bus": Out(self.bus_signature),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.submodules.adapter = self._adapter
        m.submodules.cache = self._cache

        # internal 16b master -> adapter.i (manual) ; adapter.o -> cache.master ;
        # cache.slave -> self.bus (external 32b PSRAM port).
        bus = self._bus
        m.d.comb += [
            self._adapter.i.stb.eq(bus.stb),
            self._adapter.i.cyc.eq(bus.cyc),
            self._adapter.i.we.eq(bus.we),
            self._adapter.i.adr.eq(bus.adr),
            self._adapter.i.dat_w.eq(bus.dat_w),
            self._adapter.i.sel.eq(bus.sel),
            bus.dat_r.eq(self._adapter.i.dat_r),
            bus.ack.eq(self._adapter.i.ack),
        ]
        wiring.connect(m, self._adapter.o, self._cache.master)
        wiring.connect(m, self._cache.slave, wiring.flipped(self.bus))

        NNQ_BITS = NNQ.width
        dim_bits = self.dim_bits
        index_bits = self.index_bits
        shift = self.shift

        # state
        infer_idx = Signal(index_bits)
        dim_idx = Signal(range(self.out_d))
        h_latch = Signal(data.ArrayLayout(NNQ, self.out_d))
        m.d.comb += self.ready.eq(1)

        # defaults each cycle
        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            bus.stb.eq(0),
            bus.cyc.eq(0),
            bus.we.eq(0),
            bus.sel.eq(0),
            bus.adr.eq(0),
            bus.dat_w.eq(0),
        ]
        for j in range(self.out_d):
            m.d.comb += self.o.payload[j].eq(h_latch[j])

        with m.FSM():

            # ---- INFER ----------------------------------------------------
            with m.State("INFER_WAIT"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += [
                        infer_idx.eq(self.i.payload[shift : shift + index_bits]),
                        dim_idx.eq(0),
                    ]
                    m.next = "INFER_READ"

            with m.State("INFER_READ"):
                m.d.comb += [
                    bus.stb.eq(1),
                    bus.cyc.eq(1),
                    bus.we.eq(0),
                    bus.sel.eq(-1),
                    bus.adr.eq((infer_idx << dim_bits) | dim_idx),
                ]
                with m.If(bus.ack):
                    m.d.sync += (
                        h_latch.as_value().word_select(dim_idx, NNQ_BITS).eq(bus.dat_r)
                    )
                    with m.If(dim_idx == self.out_d - 1):
                        m.next = "INFER_OUTPUT"
                    with m.Else():
                        m.d.sync += dim_idx.eq(dim_idx + 1)

            with m.State("INFER_OUTPUT"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "INFER_WAIT"

        return m
