import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import io

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; avoids tkinter main-loop errors
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import warnings


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
