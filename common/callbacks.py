from pathlib import Path
import io
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


def setup_beta_stft_var_and_update_callback(
    epochs: int,
    beta_stft_warmup: int,
    beta_stft_ramp: int,
    beta_stft: float,
):
    """
    Args:
        epochs: total epochs being run
        beta_stft_warmup: number of epochs to keep beta_stft at 0
        beta_stft_ramp: number of epochs to ramp up, after warmup
        beta_stft: final target beta_stft value
    Returns:
        callback, beta_stft_var

        callback to adjust beta_stft or None if beta_stft_var static
        beta_stft_var for passing to loss_fn
    """

    if beta_stft_warmup > epochs or beta_stft_ramp > epochs:
        raise Exception(
            "--beta-stft-warmup & --beta-stft-ramp must not exceed --epochs"
        )

    use_beta_schedule = beta_stft_warmup > 0 or beta_stft_ramp > 0
    beta_stft_init = beta_stft if not use_beta_schedule else 0.0
    beta_stft_var = tf.Variable(beta_stft_init, trainable=False, dtype=tf.float32)

    if not use_beta_schedule:
        # no callback
        return None, beta_stft_var

    class RampBetaStft(tf.keras.callbacks.Callback):

        def __init__(
            self,
            beta_var: tf.Variable,
            target: float,
            warmup_epochs: int,
            ramp_epochs: int,
        ):
            self.beta_var = beta_var
            self.target = float(target)
            self.warmup_epochs = max(0, int(warmup_epochs))
            self.ramp_epochs = max(1, int(ramp_epochs))

        def on_epoch_begin(self, epoch, logs=None):
            # epoch is 0-indexed.
            if epoch < self.warmup_epochs:
                value = 0.0
            elif self.ramp_epochs == 1:
                value = self.target
            else:
                ramp_epoch = epoch - self.warmup_epochs
                value = self.target * min(ramp_epoch / (self.ramp_epochs - 1), 1.0)
            self.beta_var.assign(value)

    ramp_callback = RampBetaStft(
        beta_var=beta_stft_var,
        target=beta_stft,
        warmup_epochs=beta_stft_warmup,
        ramp_epochs=beta_stft_ramp,
    )
    return ramp_callback, beta_stft_var


class LogLrAndBetaStft(tf.keras.callbacks.Callback):
    """put before tb callback"""

    def __init__(self, beta_stft_var: tf.Variable):
        self.beta_stft_var = beta_stft_var

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        lr_f = float(tf.convert_to_tensor(lr).numpy())
        beta_stft_f = float(self.beta_stft_var.numpy())
        logs["learning_rate"] = lr_f
        logs["beta_stft"] = beta_stft_f


class CheckYPred(tf.keras.callbacks.Callback):

    def __init__(self, tb_dir, dataset):
        self.summary_writer = tf.summary.create_file_writer(tb_dir)

        # TODO: clumsy to assume run in tb dir :/
        run_dir = Path(tb_dir).parent
        self.run = int(run_dir.name)
        self.validation_plots_dir = run_dir / "validation_plots"
        self.validation_plots_dir.mkdir(parents=True, exist_ok=True)

        for x, y in dataset:
            self.x = x
            self.y_true = y
            break  # just one batch

    def _plot_as_numpy(self, x, y_true, y_pred):
        df = pd.DataFrame()
        df["y_true"] = y_true[:, 0]
        df["y_pred"] = y_pred[:, 0]
        df["phase"] = x[:, 0]  # already in [-1, 1]
        df["n"] = range(len(x))
        wide_df = pd.melt(
            df,
            id_vars=["n"],
            value_vars=["y_pred", "y_true", "phase"],
        )
        with io.BytesIO() as img_buffer:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                fig, ax = plt.subplots(figsize=(30, 5))
                sns.lineplot(wide_df, x="n", y="value", hue="variable", ax=ax)
                ax.set_ylim((-1.1, 1.1))
                ax.set_title(f"run {self.run:04d}")
                fig.savefig(img_buffer, format="png")
                plt.close(fig)
            img_buffer.seek(0)
            pil_img = Image.open(img_buffer).convert("RGB")
        return np.array(pil_img)

    def on_epoch_end(self, epoch, logs=None):
        with self.summary_writer.as_default():
            with tf.name_scope("validation") as scope:
                y_pred = self.model(self.x)

                # tb pagination dft is 12, so take at most 2 pages
                plot_x = self.x[:12]
                y_pred = y_pred[:12]
                plot_y_true = self.y_true[:12]

                imgs = []
                for i in range(len(plot_x)):
                    img = self._plot_as_numpy(plot_x[i], plot_y_true[i], y_pred[i])
                    imgs.append(img)
                    save_path = (
                        self.validation_plots_dir
                        / f"r{self.run:04d}_e{epoch:04d}_eg{i:02d}.jpg"
                    )
                    Image.fromarray(img).save(save_path)
                imgs = np.stack(imgs)
                tf.summary.image(
                    "check_ypred", imgs, max_outputs=len(plot_x), step=epoch
                )
