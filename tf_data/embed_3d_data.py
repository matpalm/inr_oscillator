import numpy as np
from enum import Enum
import random
import tensorflow as tf

# base everything off C3 -> C5 to easier match BSP
A4 = 440
FREQS = {
    "A4": A4,
    "C4": A4 * (2 ** (-9 / 12)),
}
FREQS["C3"] = FREQS["C4"] / 2
FREQS["C5"] = FREQS["C4"] * 2

# must match QuadratureVOct
# QUADRATURE_AMPLITUDE = 0.99


class Waveform(Enum):
    ZIGZAG = "triangle"
    SQUARE = "square"
    SINE = "sine"
    SAW = "saw"

    @staticmethod
    def pairs_from_pt(pt: float):
        assert -1 <= pt <= 1
        if pt <= -0.5:
            # rescale (-1, -0.5) -> (0, 1)
            pt = (pt + 1) * 2
            return Waveform.ZIGZAG, Waveform.SQUARE, pt
        elif pt <= 0:
            # rescale (-0.5, 0) -> (0, 1)
            pt = (pt + 0.5) * 2
            return Waveform.SQUARE, Waveform.SINE, pt
        elif pt <= 0.5:
            # rescale (0, 0.5) -> (0, 1)
            pt *= 2
            return Waveform.SINE, Waveform.SAW, pt
        else:  # <= 1
            # rescale (0.5, 1.0) -> (0, 1)
            pt = (pt - 0.5) * 2
            return Waveform.SAW, Waveform.ZIGZAG, pt


def sample_freq(min_freq, max_freq, alpha):
    if min_freq == max_freq:
        return min_freq
    assert min_freq < max_freq
    assert 0 <= alpha <= 1
    min_freq_2 = np.log2(min_freq)
    max_freq_2 = np.log2(max_freq)
    diff = max_freq_2 - min_freq_2
    sample_freq_2 = min_freq_2 + alpha * diff
    return 2**sample_freq_2


class Embed3DData(object):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--min-note", type=str, default="C3")
        parser.add_argument("--max-note", type=str, default="C5")
        parser.add_argument("--harsh", action="store_true")
        parser.add_argument("--sample-rate-khz", type=float, default=192)

    def __init__(
        self,
        min_note: str,
        max_note: str,
        sample_rate_khz: float,
        harsh: bool = False,
        seed: int = 123,
    ):
        self.min_note = min_note
        self.max_note = max_note
        if sample_rate_khz > 1_000:
            print("WARNING sample_rate_khz! not sample_rate_hz")
        self.sample_rate_hz = sample_rate_khz * 1000
        self.harsh = harsh
        self.rng = random.Random(seed)

    def in_d(self):
        # input features: wrapped phase + 2D embedding point
        return 3

    def out_d(self):
        return 1

    def calculate_wave(
        self,
        frequency_hz: float,
        seq_len: int,
        starting_phase: float,
        waveform1: Waveform,
        waveform2: Waveform = None,
        interp: float = 0,
        scale: float = 0.8,
    ):

        if frequency_hz > (self.sample_rate_hz / 2.0):
            raise ValueError("faildog! nyquist limit")

        phase_step = 2.0 * np.pi * (frequency_hz / self.sample_rate_hz)
        phase = starting_phase + (phase_step * np.arange(seq_len))
        phase_sin = np.sin(phase)
        phase_cos = np.cos(phase)
        cycle = np.mod(phase / (2.0 * np.pi), 1.0)  # [0, 1)

        saw_rising = False  # vs falling

        if self.harsh:
            # harsh waves
            inverted_zigzag = True
            inverted_sine = True
        else:
            # cleaner waves
            inverted_zigzag = False
            inverted_sine = False

        def wave(w):
            match w:
                case Waveform.ZIGZAG:
                    zigzag = np.where(cycle < 0.5, 2.0 * cycle, -(2.0 * (cycle - 0.5)))
                    if inverted_zigzag:
                        zigzag *= -1
                    return zigzag
                case Waveform.SQUARE:
                    return np.where(phase_sin >= 0.0, 1, -1)
                case Waveform.SINE:
                    if inverted_sine:
                        return -phase_sin
                    else:
                        return phase_sin
                case Waveform.SAW:
                    if saw_rising:
                        return (2.0 * cycle) - 1.0
                    else:
                        return 1.0 - (2.0 * cycle)

        interp = np.full(seq_len, interp, dtype=np.float64)
        interp = np.clip(interp, 0.0, 1.0)
        result1 = wave(waveform1)
        result2 = wave(waveform2)
        s1 = np.sin((1 - interp) * np.pi / 2)
        s2 = np.sin(interp * np.pi / 2)
        result = (s1 * result1) + (s2 * result2)
        result = np.clip(result, -1, 1)

        wave = scale * result

        # wrapped phase angle scaled for the model
        # sawtooth from -5V to +5V
        sawtooth_phase = np.mod(phase / np.pi + 1.0, 2.0) - 1.0  # +/- 1 => 10V
        sawtooth_phase /= 2  # +/- 0.5 => 5V

        return sawtooth_phase, wave

    def random_freq(self):
        return sample_freq(
            FREQS[self.min_note], FREQS[self.max_note], alpha=self.rng.random()
        )

    def random_phase(self):
        return self.rng.random() * 2 * np.pi

    # def _sample_single_wave(self, seq_len, w1):
    #     data = self.calculate_wave(
    #         self.random_freq(),
    #         seq_len,
    #         self.random_phase(),
    #         w1,
    #         waveform2=None,
    #     )
    #     embed_pt = w1.to_embed_pt()
    #     return data, embed_pt

    # def _sample_interpolated_wave(self, seq_len, w1, w2, interp_start, interp_end):
    #     data = self.calculate_wave(
    #         self.random_freq(),
    #         seq_len,
    #         self.random_phase(),
    #         w1,
    #         w2,
    #         interp_start,
    #         interp_end,
    #     )
    #     interp = data["interp"].astype(np.float32)
    #     embed_pt = ((1.0 - interp)[:, None] * w1.to_embed_pt()) + (
    #         interp[:, None] * w2.to_embed_pt()
    #     )
    #     return data, embed_pt

    # def _xy_from_data(self, data, embed_pt):
    #     # TODO: this could be a map in tf
    #     N = len(data["phase"])
    #     x = np.zeros((N, IN_D), dtype=np.float32)
    #     y = np.zeros((N, OUT_D), dtype=np.float32)
    #     x[:, 0] = data["phase"]
    #     if np.ndim(embed_pt) == 1:
    #         x[:, 1] = embed_pt[0]
    #         x[:, 2] = embed_pt[1]
    #     else:
    #         x[:, 1] = embed_pt[:, 0]
    #         x[:, 2] = embed_pt[:, 1]

    #     y[:, 0] = data["wave"]

    #     return x, y

    def pt_to_phase_waveform(self, pt, freq, phase, seq_len: int = 1024):

        # map it to two waveforms and interpolation amount
        w1, w2, interp = Waveform.pairs_from_pt(pt)

        # derive waveform and wrapped phase
        sawtooth_phase, waveform = self.calculate_wave(
            freq,
            seq_len,
            phase,
            w1,
            w2,
            interp,
        )

        return sawtooth_phase, waveform

    def tf_dataset(
        self,
        batch_size: int,
        seq_len: int,
        num_samples: int,
        # emit_endpt_samples: bool = True,
        # emit_interpolated_samples: bool = True,
        # emit_double_interpolated_samples: bool = False,
        # emit_specific_wave: Waveform = None,
    ):
        """
        Generate num_samples samples of shape (batch_size, seq_len, 4)

        Args:
            batch_size: dim 0 of output
            seq_len: second axis for batch
            num_samples: total number of batches generated
        """

        # TODO: this switching of generators is kinda clumsy :/

        def gen_interp_waves():
            while True:
                pt = (self.rng.random() * 2.0) - 1.0
                freq = self.random_freq()
                phase = self.random_phase()
                sawtooth_phase, waveform = self.pt_to_phase_waveform(pt, freq, phase)
                x = np.zeros((seq_len, self.in_d()), dtype=np.float32)
                x[:, 0] = sawtooth_phase
                x[:, 1] = pt
                x[:, 2] = 0.0
                y = np.zeros((seq_len, self.out_d()), dtype=np.float32)
                y[:, 0] = waveform
                yield x, y

        def gen_limited_number():
            g = gen_interp_waves()
            for _ in range(num_samples):
                yield next(g)

        ds = tf.data.Dataset.from_generator(
            gen_limited_number,
            output_signature=(
                tf.TensorSpec(shape=(seq_len, self.in_d()), dtype=tf.float32),
                tf.TensorSpec(shape=(seq_len, self.out_d()), dtype=tf.float32),
            ),
        )
        #        ds = ds.shuffle(batch_size * 5)
        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    import argparse
    import warnings
    import matplotlib.pyplot as plt
    from pathlib import Path

    warnings.filterwarnings(
        "ignore",
        message=".*",
        category=FutureWarning,
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    Embed3DData.add_args(parser)
    parser.add_argument("--seq-len", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=str, default="tmp/embed_3d_plots")
    opts = parser.parse_args()
    print("opts", opts)

    data_source = Embed3DData(
        min_note="A4",
        max_note="A4",
        sample_rate_khz=48,
        harsh=True,
        seed=opts.seed,
    )

    def plot(
        sawtooth_phase,
        waveform,
        pt,
        sample_rate_hz=48000,
        width=800,
        height=300,
        dpi=100,
    ):
        import io
        from PIL import Image

        time_s = np.arange(len(sawtooth_phase), dtype=np.float32) / sample_rate_hz
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax.plot(time_s, sawtooth_phase, linewidth=1.0, label="phase")
        ax.plot(time_s, waveform, linewidth=1.0, label="wave")
        ax.axhline(pt, linestyle="--", linewidth=1.0, color="C2", label=f"pt={pt:.3f}")
        ax.set_xlabel("time_s")
        ax.set_ylabel("value")
        ax.set_ylim([-1.0, 1.0])
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    for i, pt in enumerate(np.linspace(-1, 1, num=21)):
        sawtooth_phase, waveform = data_source.pt_to_phase_waveform(
            pt, FREQS["A4"], phase=0, seq_len=opts.seq_len
        )
        plot_img = plot(
            sawtooth_phase, waveform, pt, sample_rate_hz=data_source.sample_rate_hz
        )
        out_dir = Path(opts.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_img.save(out_dir / f"eg_{i:02d}.png")
