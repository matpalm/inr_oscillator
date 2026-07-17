import numpy as np
from enum import Enum
import random
import tensorflow as tf

from qkeras import quantized_bits

FREQS = {"A2": 110, "A3": 220, "A4": 440, "A5": 880, "A6": 1760}

IN_D = 3
OUT_D = 1


class Waveform(Enum):
    ZIGZAG = "triangle"
    SQUARE = "square"
    SINE = "sine"
    SAW = "saw"

    def to_embed_pt(self):
        return {
            Waveform.ZIGZAG: np.array([1, 1]),
            Waveform.SQUARE: np.array([1, -1]),
            Waveform.SINE: np.array([-1, -1]),
            Waveform.SAW: np.array([-1, 1]),
        }[self]


WAVE_EDGES = [
    (Waveform.ZIGZAG, Waveform.SQUARE),
    (Waveform.SQUARE, Waveform.SINE),
    (Waveform.SINE, Waveform.SAW),
    (Waveform.SAW, Waveform.ZIGZAG),
]


def soft_clip(x, drive: float):
    x_ref = max(1.0, np.max(np.abs(x))) + 1e-10
    x_rescaled = x / x_ref
    return np.tanh(drive * x_rescaled) / np.tanh(drive)


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


class Embed2DQuadratureData(object):

    def __init__(
        self,
        min_note: str,
        max_note: str,
        sample_rate_khz: float,
        fp_int: int = 4,
        fp_frac: int = 12,
        quantise_y: bool = False,
        harsh: bool = False,
        soft_clip: bool = False,
        seed: int = 123,
    ):
        self.min_note = min_note
        self.max_note = max_note
        if sample_rate_khz > 1_000:
            print("WARNING sample_rate_khz! not sample_rate_hz")
        self.sample_rate_hz = sample_rate_khz * 1000
        self.harsh = harsh
        self.soft_clip = soft_clip
        self.rng = random.Random(seed)
        self.y_quantiser = quantized_bits(
            bits=fp_int + fp_frac, integer=fp_int, alpha=1
        )
        self.quantise_y = quantise_y

    def calculate_wave(
        self,
        frequency_hz: float,
        seq_len: int,
        starting_phase: float,
        waveform1: Waveform,
        waveform2: Waveform = None,
        interp_start: float = 0,
        interp_end: float = None,
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

        if waveform2 is None:
            result = wave(waveform1)
            interp = None
        else:
            if interp_end is None:
                interp_end = interp_start
            interp = np.full(seq_len, interp_start, dtype=np.float64)
            i25 = int(0.25 * seq_len)
            i75 = int(0.75 * seq_len)
            if i75 > i25:
                interp[i25:i75] = np.linspace(interp_start, interp_end, i75 - i25)
            interp[i75:] = interp_end
            interp = np.clip(interp, 0.0, 1.0)
            result1 = wave(waveform1)
            result2 = wave(waveform2)
            s1 = np.sin((1 - interp) * np.pi / 2)
            s2 = np.sin(interp * np.pi / 2)
            result = (s1 * result1) + (s2 * result2)

        # note: combos of interp don't ensure values stay in (-1, 1) so clip
        if self.soft_clip:
            result = soft_clip(result, drive=2)
        else:
            result = np.clip(result, -1, 1)

        # scale and quantise for output
        phase_sin *= scale
        phase_cos *= scale

        wave = scale * result
        if self.quantise_y:
            wave = self.y_quantiser(wave)

        # wrapped phase angle scaled to [-1, 1) fed directly to the model
        phase_wrapped = np.mod(phase / np.pi + 1.0, 2.0) - 1.0

        return {
            "phase": phase_wrapped,
            "phase_sin": phase_sin,
            "phase_cos": phase_cos,
            "wave": wave,
            "interp": interp,
        }

    def random_freq(self):
        return sample_freq(
            FREQS[self.min_note], FREQS[self.max_note], alpha=self.rng.random()
        )

    def random_phase(self):
        return self.rng.random() * 2 * np.pi

    def _sample_single_wave(self, seq_len, w1):
        data = self.calculate_wave(
            self.random_freq(),
            seq_len,
            self.random_phase(),
            w1,
            waveform2=None,
        )
        embed_pt = w1.to_embed_pt()
        return data, embed_pt

    def _sample_interpolated_wave(self, seq_len, w1, w2, interp_start, interp_end):
        data = self.calculate_wave(
            self.random_freq(),
            seq_len,
            self.random_phase(),
            w1,
            w2,
            interp_start,
            interp_end,
        )
        interp = data["interp"].astype(np.float32)
        embed_pt = ((1.0 - interp)[:, None] * w1.to_embed_pt()) + (
            interp[:, None] * w2.to_embed_pt()
        )
        return data, embed_pt

    def _xy_from_data(self, data, embed_pt):
        # TODO: this could be a map in tf
        N = len(data["phase"])
        x = np.zeros((N, IN_D), dtype=np.float32)
        y = np.zeros((N, OUT_D), dtype=np.float32)
        x[:, 0] = data["phase"]
        if np.ndim(embed_pt) == 1:
            x[:, 1] = embed_pt[0]
            x[:, 2] = embed_pt[1]
        else:
            x[:, 1] = embed_pt[:, 0]
            x[:, 2] = embed_pt[:, 1]

        y[:, 0] = data["wave"]

        return x, y

    def tf_dataset(
        self,
        batch_size: int,
        seq_len: int,
        num_samples: int,
        emit_endpt_samples: bool = True,
        emit_interpolated_samples: bool = True,
        emit_double_interpolated_samples: bool = False,
        emit_specific_wave: Waveform = None,
    ):
        """
        Generate num_samples samples of shape (batch_size, seq_len, 4)

        Args:
            batch_size: dim 0 of output
            seq_len: second axis for batch
            num_samples: total number of batches generated
            emit_endpt_samples: if true there's a chance of sampling ( uninterpolated ) endpts
            emit_interpolated_samples: if true there's a cchange of sampling an interpolate wave ( on edge )
            emit_double_interpolated_samples: if true then vary interpolation over seq_len for interp samples
            emit_specific_wave: if set just emit this wave
        """

        # TODO: this switching of generators is kinda clumsy :/

        def gen_specific_wave(wave):
            while True:
                yield self._sample_single_wave(seq_len, wave)

        def gen_interp_waves():
            assert emit_endpt_samples or emit_interpolated_samples

            samples_types = []
            if emit_endpt_samples:
                samples_types.append("endpt")
            if emit_interpolated_samples:
                samples_types.append("interp")

            while True:
                sample_type = self.rng.choice(samples_types)
                match sample_type:
                    case "endpt":
                        wave = self.rng.choice(list(Waveform))
                        yield self._sample_single_wave(seq_len, wave)
                    case "interp":
                        # samples an edge of the simplex
                        w1, w2 = self.rng.choice(WAVE_EDGES)
                        # emit interpolated; either with fixed start/end interp
                        # or a blend from start -> end if emit_double_interpolated_samples
                        interp_start = self.rng.random()
                        interp_end = (
                            self.rng.random()
                            if emit_double_interpolated_samples
                            else None
                        )
                        yield self._sample_interpolated_wave(
                            seq_len, w1, w2, interp_start, interp_end
                        )

        def gen_limited_number():
            if type(emit_specific_wave) == str:
                g = gen_specific_wave(Waveform(emit_specific_wave))
            else:
                g = gen_interp_waves()
            for _ in range(num_samples):
                data, embed_pt = next(g)
                yield self._xy_from_data(data, embed_pt)

        ds = tf.data.Dataset.from_generator(
            gen_limited_number,
            output_signature=(
                tf.TensorSpec(shape=(seq_len, IN_D), dtype=tf.float32),
                tf.TensorSpec(shape=(seq_len, OUT_D), dtype=tf.float32),
            ),
        )
        #        ds = ds.shuffle(batch_size * 5)
        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    import argparse
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import io

    warnings.filterwarnings(
        "ignore",
        message=".*",
        category=FutureWarning,
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--min-note", type=str, default="A4")
    parser.add_argument("--max-note", type=str, default="A4")
    parser.add_argument("--fp-int", type=int, default=4)
    parser.add_argument("--fp-frac", type=int, default=12)
    parser.add_argument("--starting-phase", type=float, default=0)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--harsh", action="store_true")
    parser.add_argument("--soft-clip", action="store_true")
    parser.add_argument("--output-png", type=str, default="foo.png")
    opts = parser.parse_args()
    print("opts", opts)

    # data_source = Embed2DQuadratureData(
    #     min_note=opts.min_note,
    #     max_note=opts.max_note,
    #     sample_rate_hz=opts.sample_rate_hz,
    #     seed=123,
    # )

    # ds = data_source.tf_dataset(
    #     batch_size=1,
    #     seq_len=opts.num_samples,
    #     num_samples=300,
    #     emit_endpt_samples=True,
    #     emit_interpolated_samples=True,
    # )

    PLOT_W = 320
    PLOT_H = 240
    from PIL import Image, ImageDraw

    # NxN images, with only border set
    N = opts.grid_size
    collage = Image.new(size=(PLOT_W * N, PLOT_H * N), mode="RGB")
    plot_data_source = Embed2DQuadratureData(
        min_note=opts.min_note,
        max_note=opts.max_note,
        sample_rate_khz=192,
        fp_int=opts.fp_int,
        fp_frac=opts.fp_frac,
        quantise_y=False,
        harsh=opts.harsh,
        soft_clip=opts.soft_clip,
        seed=123,
    )

    def plot_interp(w1, w2, interp):
        data = plot_data_source.calculate_wave(
            frequency_hz=FREQS["A4"],
            seq_len=opts.seq_len,
            starting_phase=0,
            waveform1=w1,
            waveform2=w2,
            interp_start=interp,
            interp_end=interp,
        )
        df = pd.DataFrame()
        df["n"] = range(len(data["wave"]))
        df["wave"] = data["wave"]
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        with io.BytesIO() as b:
            p = sns.lineplot(df, x="n", y="wave", linewidth=5, ax=ax)
            p.set(xticklabels=[])
            p.set(xlabel=None)
            p.set(yticklabels=[])
            p.set(ylabel=None)
            p.tick_params(bottom=False, left=False)
            p.set(ylim=(-2, 2))
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(
                b,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                facecolor="white",
                edgecolor="white",
                transparent=False,
            )
            b.seek(0)
            pil_img = Image.open(b).convert("RGB").copy()
            pil_img = pil_img.resize((PLOT_W, PLOT_H))
            pil_img.save("FFFF.png")
        plt.close(fig)
        return pil_img

    def plot_single(w1):
        return plot_interp(w1, None, None)

    # points based on grid
    top_left = Waveform.SAW
    top_right = Waveform.ZIGZAG
    bottom_right = Waveform.SQUARE
    bottom_left = Waveform.SINE

    # corners
    collage.paste(plot_single(top_left), (0, 0))
    collage.paste(plot_single(top_right), ((N - 1) * PLOT_W, 0))
    collage.paste(plot_single(bottom_right), ((N - 1) * PLOT_W, (N - 1) * PLOT_H))
    collage.paste(plot_single(bottom_left), (0, (N - 1) * PLOT_H))

    # edges
    for i in [k / (N - 1) for k in range(1, N - 1)]:
        interp_img = plot_interp(top_left, top_right, interp=i)
        collage.paste(interp_img, (int(PLOT_W * (N - 1) * i), 0))
        interp_img = plot_interp(top_right, bottom_right, interp=i)
        collage.paste(interp_img, ((N - 1) * PLOT_W, int(PLOT_H * (N - 1) * i)))
        interp_img = plot_interp(bottom_left, bottom_right, interp=i)
        collage.paste(interp_img, (int(PLOT_W * (N - 1) * i), PLOT_H * (N - 1)))
        interp_img = plot_interp(top_left, bottom_left, interp=i)
        collage.paste(interp_img, (0, int(PLOT_H * (N - 1) * i)))

    # draw borders between outputs
    draw = ImageDraw.Draw(collage)
    max_x = (N * PLOT_W) - 1
    max_y = (N * PLOT_H) - 1
    for k in range(1, N):
        x = k * PLOT_W
        y = k * PLOT_H
        draw.line([(x, 0), (x, max_y)], fill="black", width=1)
        draw.line([(0, y), (max_x, y)], fill="black", width=1)

    # top_left = Waveform.SAW  #: np.array([-1, 1]),
    # top_right = Waveform.TRIANGLE  #: np.array([1, 1]),  # top
    # bottom_right = Waveform.SQUARE  #: np.array([1, -1]),
    # bottom_left = Waveform.SINE  #: np.array([-1, -1]),

    collage.save(opts.output_png)

    # GRID_SIZE = 5

    # for i0, e0 in enumerate(np.linspace(-1, 1, GRID_SIZE)):

    #     for i1, e1 in enumerate(np.linspace(-1, 1, GRID_SIZE)):

    #         print("i", i0, i1, "=> e", e0, e1)
    #         x[0, :, 2] = e0
    #         x[0, :, 3] = e1

    #         y_pred = test_model.predict(x)

    #         # axis 0 ; just take first element ( single batch )
    #         # axis 1 ; drop first receptive field items ( warm up )
    #         # axis 2 ; just first element ( single dim output )
    #         y_pred = y_pred[0, RECEPTIVE_FIELD_SIZE:, 0]

    #         # save plot
    #         df = pd.DataFrame()
    #         df["n"] = range(len(y_pred))
    #         df["y_pred"] = y_pred
    #         with warnings.catch_warnings():
    #             warnings.simplefilter(action="ignore", category=FutureWarning)
    #             p = sns.lineplot(df, x="n", y="y_pred", linewidth=5)
    #             p.set(xticklabels=[])
    #             p.set(xlabel=None)
    #             p.set(yticklabels=[])
    #             p.set(ylabel=None)
    #             p.tick_params(bottom=False, left=False)
    #             p.set(ylim=(-2, 2))
    #             plt_fname = f"foo_{i0:02d}_{i1:02d}.png"
    #             print("saving plot to", plt_fname)
    #             plt.savefig(plt_fname)
    #             plt.clf()
