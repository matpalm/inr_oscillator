"""sanity test for amaranth_v.quadrature_v_oct.QuadratureVOct.
uv run -m unittest test_equivalences.test_quadrature_v_oct

"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt

from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amaranth_future import fixed
from amaranth_v import NNQ
from amaranth_v.quadrature_v_oct import QuadratureVOct


def simulate(x_value, n_samples):
    dut = QuadratureVOct()
    scale = float(1 << NNQ.f_bits)
    sin_f, cos_f = [], []

    async def testbench(ctx):
        ctx.set(dut.o.ready, 1)
        ctx.set(dut.i.payload.as_value(), fixed.Const(x_value, NNQ).as_value())
        ctx.set(dut.i.valid, 1)
        for _ in range(n_samples):
            while not ctx.get(dut.o.valid):
                await ctx.tick()
            sin_f.append(ctx.get(dut.o.payload[0].as_value()) / scale)
            cos_f.append(ctx.get(dut.o.payload[1].as_value()) / scale)
            await ctx.tick()  # o.ready high -> handshake completes, back to FOLD

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()
    return np.array(sin_f), np.array(cos_f)


def _dominant_freq(samples, fs):
    """Estimate the fundamental frequency (Hz) via the FFT peak."""
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / fs)
    return freqs[np.argmax(spectrum[1:]) + 1]  # skip DC


class TestQuadratureVOct(unittest.TestCase):
    FS = QuadratureVOct.FS_HZ
    F0 = QuadratureVOct.F0_HZ
    V_MIN = QuadratureVOct.V_MIN
    VOLTS = [5.0, 5.5, 6.0, 6.5, 7.0]  # 5V..7V in 0.5V increments
    N_FFT = 4096  # samples for the frequency estimate
    N_PLOT = 512  # samples to draw per voltage

    def test_plot_and_frequency(self):
        fig, axes = plt.subplots(
            len(self.VOLTS), 1, figsize=(10, 2.2 * len(self.VOLTS)), sharex=True
        )
        t_ms = np.arange(self.N_PLOT) / self.FS * 1e3

        for ax, volts in zip(axes, self.VOLTS):
            x = volts  # input is a control voltage (volts)
            expected = self.F0 * (2.0 ** (volts - self.V_MIN))  # one octave per volt
            sin_s, cos_s = simulate(x, self.N_FFT)

            measured = _dominant_freq(sin_s, self.FS)
            self.assertAlmostEqual(
                measured,
                expected,
                delta=20.0,
                msg=f"{volts}V: measured {measured:.1f}Hz vs expected {expected:.1f}Hz",
            )
            # quadrature magnitude should sit near the requested amplitude
            mag = np.sqrt(sin_s**2 + cos_s**2)
            self.assertAlmostEqual(mag.mean(), QuadratureVOct.AMPLITUDE, delta=0.02)

            ax.plot(t_ms, sin_s[: self.N_PLOT], label="sin")
            ax.plot(t_ms, cos_s[: self.N_PLOT], label="cos")
            ax.set_title(
                f"{volts:.1f}V  ->  {expected:.1f} Hz " f"[measured {measured:.1f} Hz]"
            )
            ax.set_ylabel("amplitude")
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize="small")

        axes[-1].set_xlabel("time (ms)")
        fig.suptitle("QuadratureVOct: sin/cos vs V/oct input (5V-7V)")
        fig.tight_layout()

        out = Path(
            os.getenv(
                "QVO_PLOT_PATH",
                str(Path(__file__).resolve().parent / "quadrature_v_oct.png"),
            )
        )
        fig.savefig(out, dpi=120)
        plt.close(fig)
        self.assertTrue(out.exists(), f"plot not written to {out}")
        print(f"wrote {out}")


if __name__ == "__main__":
    unittest.main()
