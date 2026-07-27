"""FakePSRAM testbench helper, vendored from the cdcc project's
``tests/test_activation_cache_ps.py`` (itself inlined from tiliqua).

A minimal wishbone-backed fake PSRAM with configurable access latency, used to
back the PSRAM phase->h table in ``test_phase_h_lut_ps``.

Original tiliqua copyright: (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
Original SPDX-License-Identifier: CERN-OHL-S-2.0
"""

from amaranth import Module, Signal, unsigned
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In
from amaranth_soc import wishbone


class FakePSRAM(wiring.Component):
    """Fake PSRAM for testbenches with configurable access latency."""

    def __init__(
        self, *, addr_width=22, data_width=32, storage_words=512, latency_cycles=4
    ):
        self.latency_cycles = latency_cycles
        self.storage_words = storage_words
        super().__init__(
            {
                "bus": In(
                    wishbone.Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        granularity=8,
                        features={"cti", "bte"},
                    )
                )
            }
        )

    def elaborate(self, platform):
        m = Module()
        bus = self.bus

        memory = Memory(
            shape=unsigned(self.bus.signature.data_width),
            depth=self.storage_words,
            init=[],
        )
        m.submodules.memory = memory
        mem_wr_port = memory.write_port(granularity=8)
        mem_rd_port = memory.read_port()

        latency_counter = Signal(range(self.latency_cycles + 1))
        in_burst = Signal()
        burst_counter = Signal(8)

        m.d.comb += [
            mem_rd_port.addr.eq(bus.adr),
            mem_wr_port.addr.eq(bus.adr),
            mem_wr_port.data.eq(bus.dat_w),
            bus.dat_r.eq(mem_rd_port.data),
        ]

        m.d.sync += mem_wr_port.en.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                m.d.sync += [
                    bus.ack.eq(0),
                    latency_counter.eq(0),
                    in_burst.eq(0),
                    burst_counter.eq(0),
                ]
                with m.If(bus.cyc & bus.stb):
                    m.d.sync += in_burst.eq(bus.cti != wishbone.CycleType.CLASSIC)
                    m.next = "LATENCY"

            with m.State("LATENCY"):
                m.d.sync += latency_counter.eq(latency_counter + 1)
                with m.If(latency_counter == (self.latency_cycles - 1)):
                    m.next = "RESPOND"

            with m.State("RESPOND"):
                m.d.sync += bus.ack.eq(1)
                with m.If(bus.we):
                    m.d.sync += mem_wr_port.en.eq(bus.sel)
                with m.If(bus.ack):
                    m.d.comb += mem_rd_port.addr.eq(bus.adr + 1)
                with m.If(in_burst):
                    m.d.sync += burst_counter.eq(burst_counter + 1)
                    with m.If(bus.cti == wishbone.CycleType.END_OF_BURST):
                        m.d.sync += bus.ack.eq(0)
                        m.next = "IDLE"
                with m.Else():
                    m.next = "IDLE"

        return m
