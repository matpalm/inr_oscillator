"""Wishbone L2 cache + 16b->32b adapter, vendored from the cdcc project
(``cached_dilated_causal_convolutions``), which in turn adapted them from
tiliqua.

  - ``WishboneL2Cache``  : write-back burst cache in front of PSRAM.
  - ``_WishboneAdapter`` : packs x2 16-bit samples into one 32-bit PSRAM word.

Both are generic (no NNQ / K dependency) and are used by
``phase_h_lut_ps.PhaseHLutPS`` to back the phase->h table in PSRAM.

Original copyright: (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
Original SPDX-License-Identifier: CERN-OHL-S-2.0
"""

from amaranth import *
from amaranth.lib import data, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out
from amaranth.utils import exact_log2
from amaranth_soc import wishbone


class WishboneL2Cache(wiring.Component):
    """
    adapted from tiliqua/gateware/src/tiliqua/cache.py
    Original copyright: (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
    Original SPDX-License-Identifier: CERN-OHL-S-2.0
    """

    def __init__(
        self,
        cachesize_words=64,
        addr_width=22,
        data_width=32,
        granularity=8,
        burst_len=4,
        autoflush=False,
    ):
        assert burst_len > 1
        self.cachesize_words = cachesize_words
        self.data_width = data_width
        self.burst_len = burst_len
        self.granularity = granularity
        self.autoflush = autoflush
        super().__init__(
            {
                "master": In(
                    wishbone.Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        granularity=granularity,
                    )
                ),
                "slave": Out(
                    wishbone.Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        granularity=granularity,
                        features={"cti", "bte"},
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()

        master = self.master
        slave = self.slave

        dw_from = dw_to = self.data_width

        addressbits = len(slave.adr)
        offsetbits = exact_log2(self.burst_len)
        linebits = exact_log2(self.cachesize_words // self.burst_len)
        tagbits = addressbits - linebits - offsetbits
        adr_offset = master.adr.bit_select(0, offsetbits)
        adr_line = Signal(linebits)
        adr_tag = master.adr.bit_select(offsetbits + linebits, tagbits)

        m.d.comb += adr_line.eq(master.adr.bit_select(offsetbits, linebits))

        burst_offset = Signal.like(adr_offset)
        burst_offset_lookahead = Signal.like(burst_offset)

        m.submodules.data_mem = data_mem = Memory(
            shape=unsigned(self.data_width), depth=2**linebits * self.burst_len, init=[]
        )
        wr_port = data_mem.write_port(granularity=self.granularity)
        rd_port = data_mem.read_port()

        write_from_slave = Signal()
        word_select = Const(1).replicate(dw_to // self.granularity)

        m.d.comb += [
            rd_port.addr.eq(Cat(adr_offset, adr_line)),
            slave.sel.eq(word_select),
            master.dat_r.eq(rd_port.data),
            slave.dat_w.eq(rd_port.data),
        ]

        with m.If(write_from_slave):
            m.d.comb += [
                wr_port.addr.eq(Cat(burst_offset, adr_line)),
                wr_port.data.eq(slave.dat_r),
                wr_port.en.eq(word_select),
            ]
        with m.Else():
            m.d.comb += wr_port.addr.eq(Cat(adr_offset, adr_line))
            m.d.comb += wr_port.data.eq(master.dat_w)
            with m.If(master.cyc & master.stb & master.we & master.ack):
                m.d.comb += wr_port.en.eq(master.sel)

        tag_layout = data.StructLayout(
            {
                "tag": unsigned(tagbits),
                "dirty": unsigned(1),
                "valid": unsigned(1),
            }
        )
        m.submodules.tag_mem = tag_mem = Memory(
            shape=tag_layout, depth=2**linebits, init=[]
        )
        tag_wr_port = tag_mem.write_port()
        tag_rd_port = tag_mem.read_port(domain="comb")
        tag_do = Signal(shape=tag_layout)
        tag_di = Signal(shape=tag_layout)
        m.d.comb += [
            tag_do.eq(tag_rd_port.data),
            tag_wr_port.data.eq(tag_di),
        ]
        m.d.comb += [
            tag_wr_port.addr.eq(adr_line),
            tag_rd_port.addr.eq(adr_line),
            tag_di.tag.eq(adr_tag),
        ]
        m.d.comb += slave.adr.eq(Cat(burst_offset, adr_line, tag_do.tag))
        m.d.sync += master.ack.eq(0)

        if self.autoflush:
            flush_wait = Signal(10, init=1)
            adr_line_flush = Signal.like(adr_line)

        with m.FSM() as fsm:

            with m.State("IDLE"):
                with m.If(master.cyc & master.stb):
                    m.next = "TEST_HIT"
                if self.autoflush:
                    m.d.sync += flush_wait.eq(flush_wait + 1)
                    with m.If(flush_wait == 0):
                        m.d.comb += adr_line.eq(adr_line_flush)
                        m.next = "TEST_FLUSH"

            with m.State("WAIT"):
                m.next = "IDLE"

            with m.State("TEST_HIT"):
                with m.If((tag_do.tag == adr_tag) & tag_do.valid):
                    m.d.sync += master.ack.eq(1)
                    with m.If(master.we):
                        m.d.comb += [
                            tag_di.valid.eq(1),
                            tag_di.dirty.eq(1),
                            tag_wr_port.en.eq(1),
                        ]
                    m.next = "WAIT"
                with m.Else():
                    with m.If(tag_do.dirty):
                        m.d.comb += rd_port.addr.eq(
                            Cat(burst_offset_lookahead, adr_line)
                        )
                        m.next = "EVICT"
                    with m.Else():
                        m.d.comb += [
                            tag_di.valid.eq(1),
                            tag_wr_port.en.eq(1),
                        ]
                        m.next = "REFILL"

            with m.State("EVICT"):
                m.d.comb += [
                    slave.stb.eq(1),
                    slave.cyc.eq(1),
                    slave.we.eq(1),
                    slave.cti.eq(wishbone.CycleType.INCR_BURST),
                    rd_port.addr.eq(Cat(burst_offset_lookahead, adr_line)),
                ]
                with m.If(slave.ack):
                    m.d.comb += burst_offset_lookahead.eq(burst_offset + 1)
                    m.d.sync += burst_offset.eq(burst_offset + 1)
                    with m.If(burst_offset == (self.burst_len - 1)):
                        m.d.comb += slave.cti.eq(wishbone.CycleType.END_OF_BURST)
                        m.next = "WAIT-REFILL"

            with m.State("WAIT-REFILL"):
                m.d.comb += [
                    tag_di.valid.eq(1),
                    tag_wr_port.en.eq(1),
                ]
                m.next = "REFILL"

            with m.State("REFILL"):
                m.d.comb += [
                    slave.stb.eq(1),
                    slave.cyc.eq(1),
                    slave.we.eq(0),
                    slave.cti.eq(wishbone.CycleType.INCR_BURST),
                ]
                with m.If(slave.ack):
                    m.d.comb += write_from_slave.eq(1)
                    m.d.sync += burst_offset.eq(burst_offset + 1)
                    with m.If(burst_offset == (self.burst_len - 1)):
                        m.d.comb += slave.cti.eq(wishbone.CycleType.END_OF_BURST)
                        m.next = "TEST_HIT"

            if self.autoflush:
                with m.State("TEST_FLUSH"):
                    m.d.comb += adr_line.eq(adr_line_flush)
                    with m.If(tag_do.valid & tag_do.dirty):
                        m.next = "FLUSH_LINE"
                    with m.Else():
                        m.d.sync += adr_line_flush.eq(adr_line_flush + 1)
                        m.next = "IDLE"

                with m.State("FLUSH_LINE"):
                    m.d.comb += [
                        adr_line.eq(adr_line_flush),
                        slave.stb.eq(1),
                        slave.cyc.eq(1),
                        slave.we.eq(1),
                        slave.cti.eq(wishbone.CycleType.INCR_BURST),
                        rd_port.addr.eq(Cat(burst_offset_lookahead, adr_line)),
                    ]
                    with m.If(slave.ack):
                        m.d.comb += burst_offset_lookahead.eq(burst_offset + 1)
                        m.d.sync += burst_offset.eq(burst_offset + 1)
                        with m.If(burst_offset == (self.burst_len - 1)):
                            m.d.comb += [
                                slave.cti.eq(wishbone.CycleType.END_OF_BURST),
                                tag_di.valid.eq(0),
                                tag_wr_port.en.eq(1),
                            ]
                            m.d.sync += adr_line_flush.eq(adr_line_flush + 1)
                            m.next = "IDLE"

        return m


class _WishboneAdapter(wiring.Component):
    """
    adapted from tiliqua/gateware/src/tiliqua/dsp/delay_line.py
    Original copyright: (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
    Original SPDX-License-Identifier: CERN-OHL-S-2.0

    note: x2 16bit samples share one 32b wishbone word
    """

    def __init__(self, addr_width_i, addr_width_o, base):
        self.base = base
        assert (base & 0x3) == 0, f"base addr must be 4byte aligned base={base}"
        super().__init__(
            {
                "i": In(
                    wishbone.Signature(
                        addr_width=addr_width_i, data_width=16, granularity=8
                    )
                ),
                "o": Out(
                    wishbone.Signature(
                        addr_width=addr_width_o, data_width=32, granularity=8
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()
        m.d.comb += [
            self.i.ack.eq(self.o.ack),
            self.o.adr.eq((self.base >> 2) + (self.i.adr >> 1)),
            self.o.we.eq(self.i.we),
            self.o.cyc.eq(self.i.cyc),
            self.o.stb.eq(self.i.stb),
        ]
        with m.If(self.i.adr[0]):
            m.d.comb += [
                self.i.dat_r.eq(self.o.dat_r >> 16),
                self.o.sel.eq(self.i.sel << 2),
                self.o.dat_w.eq(self.i.dat_w << 16),
            ]
        with m.Else():
            m.d.comb += [
                self.i.dat_r.eq(self.o.dat_r),
                self.o.sel.eq(self.i.sel),
                self.o.dat_w.eq(self.i.dat_w),
            ]
        return m
