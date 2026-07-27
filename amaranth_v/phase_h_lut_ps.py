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

Two phases of operation (single FSM, one-way ``build -> infer`` transition):

    * BUILD  : sweep every phase index ``0 .. 2**index_bits - 1``, run a local
                         streaming RFF->mlp0 accumulator pipeline, and write ``out_d`` NNQ
                         results to
             PSRAM at ``(idx << dim_bits) | dim_idx``. When the last index is
             written ``ready`` is asserted (permanently).
  * INFER  : on each input phase, derive ``idx`` from its top bits, read back the
             ``out_d`` words and present them on ``o``. the build datapath is
             idle -- every access is a guaranteed hit.

The ``i`` / ``o`` ports mirror scalar phase input (``signed(io_bits)``) and the
cached ``h`` vector (``ArrayLayout(NNQ, out_d)``), so this is a drop-in
replacement for the ``rff -> mlp0`` sub-chain in
``rff_film_network.RffNetwork``.

The 16-bit NNQ words are packed x2 into 32-bit PSRAM words by ``_WishboneAdapter``
and fronted by ``WishboneL2Cache`` (both vendored in ``wishbone_cache``).
"""

import math

import numpy as np
from amaranth import Array, Const, Module, Mux, Signal, signed
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out
from amaranth.utils import exact_log2
from amaranth_soc import wishbone

from amaranth_future import fixed

from . import NNQ
from .dense_layer import QDenseLayer
from .wishbone_cache import WishboneL2Cache, _WishboneAdapter


class PhaseHLutPS(wiring.Component):

    def __init__(
        self,
        b_codes,
        sin_lut,
        rff_shift,
        b_bits,
        mlp0_kernel,
        mlp0_bias,
        io_shape,
        index_bits: int = 13,
        addr_width_o: int = 22,
        base: int = 0,
        cache_kwargs=None,
    ):
        """
        Args:
            b_codes       quantised RFF basis coefficients (signed integer codes)
            sin_lut       shared trig LUT storing sin on the io fixed-point grid
            rff_shift     right-shift applied to phase*b to form LUT index
            b_bits        width of quantised ``b`` coefficients
            mlp0_kernel   first dense kernel (in_d=2*num_features, out_d=mlp_dim)
            mlp0_bias     first dense bias (double-width quant in training)
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
        self._rff_shift = int(rff_shift)
        self._b_bits = int(b_bits)
        self._b_codes = [int(c) for c in b_codes]
        self._sin_lut = [int(c) for c in sin_lut]
        self._lut_size = len(self._sin_lut)
        assert (
            self._lut_size >= 2 and (self._lut_size & (self._lut_size - 1)) == 0
        ), f"sin_lut size must be a power of two, got {self._lut_size}"
        self._lut_bits = (self._lut_size - 1).bit_length()
        self._quarter_turn = self._lut_size // 4
        self._lut_mask = self._lut_size - 1

        mlp0_kernel = np.asarray(mlp0_kernel)
        mlp0_bias = np.asarray(mlp0_bias)
        if len(mlp0_kernel.shape) != 2:
            raise ValueError(
                f"expected mlp0_kernel shape (in_d, out_d), got {mlp0_kernel.shape}"
            )
        self.num_rff, self.out_d = [int(x) for x in mlp0_kernel.shape]
        if mlp0_bias.shape != (self.out_d,):
            raise ValueError(
                f"expected mlp0_bias shape ({self.out_d},), got {mlp0_bias.shape}"
            )

        self._num_features = len(self._b_codes)
        if self.num_rff != 2 * self._num_features:
            raise ValueError(
                f"mlp0 in_d={self.num_rff} must equal 2*num_features={2 * self._num_features}"
            )

        # Quantised storage for the build-only streaming MAC path.
        self.acc_shape, self.prod_shift = QDenseLayer.acc_shape_for(
            io_shape, self.num_rff, mlp0_bias
        )
        self._w_codes = np.round(mlp0_kernel * (2.0**NNQ.f_bits)).astype(np.int64)
        self._b_codes_acc = np.round(mlp0_bias * (2.0**self.acc_shape.f_bits)).astype(
            np.int64
        )
        self._w_init = self._w_codes.reshape(-1).tolist()
        self._b_init = self._b_codes_acc.tolist()

        self.lower_bound = fixed.Const(
            NNQ.min().as_float(), shape=self.acc_shape, clamp=True
        )._value
        self.upper_bound = fixed.Const(
            NNQ.max().as_float(), shape=self.acc_shape, clamp=True
        )._value
        self.frac_drop = self.acc_shape.f_bits - NNQ.f_bits

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

        m.submodules.b_mem = b_mem = Memory(
            shape=signed(self._b_bits),
            depth=self._num_features,
            init=self._b_codes,
            attrs={"ram_style": "block"},
        )
        rd_b = b_mem.read_port(domain="sync")

        m.submodules.trig_mem = trig_mem = Memory(
            shape=signed(self.io_bits),
            depth=self._lut_size,
            init=self._sin_lut,
            attrs={"ram_style": "block"},
        )
        rd_trig = trig_mem.read_port(domain="sync")

        m.submodules.w_mem = w_mem = Memory(
            shape=signed(NNQ.width),
            depth=self.num_rff * self.out_d,
            init=self._w_init,
            attrs={"ram_style": "block"},
        )
        rd_w = w_mem.read_port(domain="sync")

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
        num_entries = self.num_entries
        shift = self.shift

        # state
        build_idx = Signal(index_bits)
        infer_idx = Signal(index_bits)
        dim_idx = Signal(range(self.out_d))
        feat_idx = Signal(range(self.num_rff))
        h_latch = Signal(data.ArrayLayout(NNQ, self.out_d))
        built = Signal()

        accum = Array(
            Signal(self.acc_shape, name=f"phlut_acc_{o}", init=0)
            for o in range(self.out_d)
        )
        phase_code = Signal(signed(self.io_bits))
        prod = Signal(signed(self.io_bits + self._b_bits))
        feat_val = Signal(signed(self.io_bits))
        is_sin = Signal()
        m.d.comb += self.ready.eq(built)

        # defaults each cycle
        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            rd_b.en.eq(0),
            rd_b.addr.eq(0),
            rd_trig.en.eq(0),
            rd_trig.addr.eq(0),
            rd_w.en.eq(0),
            rd_w.addr.eq(0),
            bus.stb.eq(0),
            bus.cyc.eq(0),
            bus.we.eq(0),
            bus.sel.eq(0),
            bus.adr.eq(0),
            bus.dat_w.eq(0),
        ]
        for j in range(self.out_d):
            m.d.comb += self.o.payload[j].eq(h_latch[j])

        lower = Const(self.lower_bound, signed(self.acc_shape.width))
        upper = Const(self.upper_bound, signed(self.acc_shape.width))
        acc_val = accum[dim_idx].as_value().as_signed()
        clamped = Signal(signed(self.acc_shape.width))
        trunc_toward_zero = Signal(signed(self.acc_shape.width))

        m.d.comb += [
            clamped.eq(
                Mux(acc_val < lower, lower, Mux(acc_val > upper, upper, acc_val))
            ),
            trunc_toward_zero.eq(
                Mux(
                    clamped[-1] & clamped[: self.frac_drop].any(),
                    clamped + (1 << self.frac_drop),
                    clamped,
                )
            ),
        ]

        with m.FSM():

            # ---- BUILD ----------------------------------------------------
            with m.State("BUILD_INIT"):
                for o in range(self.out_d):
                    m.d.sync += accum[o].as_value().eq(self._b_init[o])
                m.d.sync += [
                    phase_code.eq(build_idx.as_signed() << shift),
                    feat_idx.eq(0),
                    dim_idx.eq(0),
                ]
                m.next = "BUILD_RFF_LOAD_B"

            with m.State("BUILD_RFF_LOAD_B"):
                m.d.comb += [
                    rd_b.en.eq(1),
                    rd_b.addr.eq(
                        Mux(
                            feat_idx < self._num_features,
                            feat_idx,
                            feat_idx - self._num_features,
                        )
                    ),
                ]
                m.d.sync += [
                    is_sin.eq(feat_idx >= self._num_features),
                ]
                m.next = "BUILD_RFF_MUL"

            with m.State("BUILD_RFF_MUL"):
                m.d.sync += [
                    prod.eq(phase_code * rd_b.data),
                ]
                m.next = "BUILD_RFF_ADDR"

            with m.State("BUILD_RFF_ADDR"):
                # shared trig rom stores sin; cos(x)=sin(x+quarter_turn)
                lut_idx = (prod >> self._rff_shift)[: self._lut_bits]
                m.d.comb += [
                    rd_trig.en.eq(1),
                    rd_trig.addr.eq(
                        Mux(
                            is_sin,
                            lut_idx,
                            (lut_idx + self._quarter_turn) & self._lut_mask,
                        )
                    ),
                ]
                m.next = "BUILD_RFF_READ"

            with m.State("BUILD_RFF_READ"):
                m.d.sync += [
                    feat_val.eq(rd_trig.data),
                    dim_idx.eq(0),
                ]
                m.next = "BUILD_W_READ"

            with m.State("BUILD_W_READ"):
                m.d.comb += [
                    rd_w.en.eq(1),
                    rd_w.addr.eq(feat_idx * self.out_d + dim_idx),
                ]
                m.next = "BUILD_MAC"

            with m.State("BUILD_MAC"):
                m.d.sync += (
                    accum[dim_idx]
                    .as_value()
                    .eq(
                        accum[dim_idx].as_value().as_signed()
                        + ((feat_val * rd_w.data.as_signed()) << self.prod_shift)
                    )
                )
                with m.If(dim_idx == self.out_d - 1):
                    with m.If(feat_idx == self.num_rff - 1):
                        m.d.sync += dim_idx.eq(0)
                        m.next = "BUILD_QUANT"
                    with m.Else():
                        m.d.sync += [
                            feat_idx.eq(feat_idx + 1),
                            dim_idx.eq(0),
                        ]
                        m.next = "BUILD_RFF_LOAD_B"
                with m.Else():
                    m.d.sync += dim_idx.eq(dim_idx + 1)
                    m.next = "BUILD_W_READ"

            with m.State("BUILD_QUANT"):
                m.d.sync += (
                    h_latch.as_value()
                    .word_select(dim_idx, NNQ_BITS)
                    .eq(
                        trunc_toward_zero[
                            self.frac_drop : self.frac_drop + NNQ_BITS
                        ].as_signed()
                    )
                )
                with m.If(dim_idx == self.out_d - 1):
                    m.d.sync += dim_idx.eq(0)
                    m.next = "BUILD_WRITE"
                with m.Else():
                    m.d.sync += dim_idx.eq(dim_idx + 1)

            with m.State("BUILD_WRITE"):
                m.d.comb += [
                    bus.stb.eq(1),
                    bus.cyc.eq(1),
                    bus.we.eq(1),
                    bus.sel.eq(-1),
                    bus.adr.eq((build_idx << dim_bits) | dim_idx),
                    bus.dat_w.eq(h_latch.as_value().word_select(dim_idx, NNQ_BITS)),
                ]
                with m.If(bus.ack):
                    with m.If(dim_idx == self.out_d - 1):
                        with m.If(build_idx == num_entries - 1):
                            m.d.sync += built.eq(1)
                            m.next = "INFER_WAIT"
                        with m.Else():
                            m.d.sync += build_idx.eq(build_idx + 1)
                            m.next = "BUILD_INIT"
                    with m.Else():
                        m.d.sync += dim_idx.eq(dim_idx + 1)

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
