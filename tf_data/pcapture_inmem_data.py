from pathlib import Path
import numpy as np
import json
import tensorflow as tf
import pandas as pd

from common.sample_db import SampleDB
from common.zarr_util import zarr_base_path_for, zarr_buffer_fields

IGNORE_FADE_LEN = 500

# generate samples based on materialised sampling probabilities / importance sampling
# weights based on converged model loss. dataset includes y_teacher as possible output


class ParametricCaptureStaticData(object):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--capture-run", type=str)
        parser.add_argument("--keras-model", type=str)

    def __init__(
        self,
        capture_run: str,
        keras_model: str,
        seed: int = 123,
        uniform_sampling_floor: float = 0.2,
    ):
        if capture_run is None or keras_model is None:
            raise Exception(
                "capture_run and keras_model are both required for ParametricCaptureStaticData"
            )

        db = SampleDB()
        loss_rows = db.losses_for(capture_run, keras_model)
        self.losses = np.array([l.loss for l in loss_rows], dtype=np.float64)
        if len(self.losses) == 0:
            raise Exception(
                f"no scores in db for run={capture_run} model={keras_model} ?"
            )
        del db

        self.capture_run = capture_run
        # memmapped numpy array of shape (n_chunks, TOTAL_SEQ_LEN, FEATURES)
        self.inr_model_data = np.load(
            zarr_base_path_for(capture_run) / "inr_model_data.npy", mmap_mode="r"
        )
        if self.inr_model_data.ndim != 3:
            raise Exception(
                f"expected (n_chunks, seq_len, features) array, got shape {self.inr_model_data.shape}"
            )
        self.n_chunks = self.inr_model_data.shape[0]
        self.seq_len = self.inr_model_data.shape[1]

        print(
            "capture_run",
            self.capture_run,
            "n_chunks",
            self.n_chunks,
            "chunk_len",
            self.seq_len,
            "|losses|",
            len(self.losses),
        )

        if len(self.losses) != self.n_chunks:
            raise Exception(
                "|losses| != n_chunks; either we have wrong losses or chunk_size of dest is wrong"
            )

        self.rng = np.random.default_rng(seed=seed)

        if uniform_sampling_floor < 0.0 or uniform_sampling_floor > 1.0:
            raise ValueError("uniform_sampling_floor must be in [0, 1]")

        # compute static priorities
        # TODO: try high_loss_skew in 0.4, 0.7 range
        #  0.0 => uniform ( ignore loss )
        #  1.0 => denotes skewing proportional to loss
        alpha_high_loss_skew = 0.4
        f64eps = np.finfo(np.float64).eps
        static_priorities = self.losses**alpha_high_loss_skew + f64eps

        # convert priorities to sampling probabilities by normalization, then
        # mix in a uniform floor so hard-mined examples cannot dominate.
        # p' = (1-lambda)*p + lambda*(1/N)
        raw_sampling_probabilities = static_priorities / static_priorities.sum()
        uniform_prob = np.full_like(raw_sampling_probabilities, 1.0 / self.n_chunks)
        self.sampling_probabilies = (
            1.0 - uniform_sampling_floor
        ) * raw_sampling_probabilities + uniform_sampling_floor * uniform_prob

        # since we are leaning heavily on converged ( ish ) loss of a large model
        # we can try to just calculate importance weights purely on that loss
        # i.e. regardless of where they came from; sobol, is_weights, uniform etc
        # this might be super naive... we'll see...
        # TODO: try bias_correction in 0.5, 1.0
        #  0 => w_i=1 for all => keeps all bias from sampling prio
        #  1 => full correction => weighting cancels out sampling prio
        beta_bias_correction = 0.6
        num_examples = len(self.sampling_probabilies)
        unnormalised_static_importance_weights = (
            1.0 / (num_examples * self.sampling_probabilies)
        ) ** beta_bias_correction
        self.static_importance_weights = (
            unnormalised_static_importance_weights
            / unnormalised_static_importance_weights.max()
        )

        # read in debug mapping for src_runs ( which gives the src_run of each index )
        # with open(zarr_base_path_for(capture_run) / "src_runs.json", "r") as f:
        #    src_runs = json.load(f)
        # write key arrays for debugging
        # df = pd.DataFrame(
        #     zip(src_runs, self.sampling_probabilies, self.static_importance_weights),
        #     columns=["run", "sampling_probability", "static_importance_weight"],
        # )
        # df.to_csv("/tmp/weights.tsv", sep="\t", index=False)

        if (
            self.n_chunks
            != len(self.sampling_probabilies)
            != len(self.static_importance_weights)
        ):
            raise Exception(
                "mismatch between n_chunks, sampling_probabilies, static_importance_weights"
            )

    def num_examples(self):
        return self.n_chunks

    def in_d(self):
        return 4  # ( phase, *cv_values )

    def out_d(self):
        return 1

    def model_data_block_to_xs_ys(
        self,
        data,
    ):
        """
        Args:
            data: chunk from inr_model_data (seq_len, features)
        """

        f = zarr_buffer_fields("inr_model_data.z")
        xs = data[..., [f.x_phase, f.x_a_cv, f.x_b_cv, f.x_morph_cv]]
        ys = data[..., f.y_true : f.y_true + 1]
        return xs, ys

    def slice_window(
        self,
        idx: int,
        seq_from: int,
        seq_len: int,
    ):
        seq_to = seq_from + seq_len
        data = self.inr_model_data[idx, seq_from:seq_to]
        xs, ys = self.model_data_block_to_xs_ys(data)
        return xs.astype(np.float32), ys.astype(np.float32)

    def tf_training_dataset(
        self,
        seq_len: int,
        num_batches: int,
        batch_size: int,
        emit_weights: bool,
        deterministic: bool = False,
    ):
        """
        Generate num_samples samples of shape (batch_size, seq_len, 4)
        sampling is done with statically derived importance sampling probabilities
        and importance weights.

        Args:
            seq_len: second axis for batch
            num_batches: total number of batches generated
            batch_size: batch size
            emit_weight: if set we return _weight as 3rd tuple element
            deterministic: true if we want same first set each time
        """

        if deterministic:
            local_rng = np.random.default_rng(seed=123)
        else:
            local_rng = self.rng

        num_examples = num_batches * batch_size
        sampled_idxs = local_rng.choice(
            self.n_chunks,
            size=num_examples,
            p=self.sampling_probabilies,
        ).astype(np.int32)
        sampled_seq_from = local_rng.integers(
            low=IGNORE_FADE_LEN,
            high=self.seq_len - IGNORE_FADE_LEN - seq_len,
            size=num_examples,
        ).astype(np.int32)

        sampled_weights = tf.convert_to_tensor(
            self.static_importance_weights.astype(np.float32)
        )

        ds = tf.data.Dataset.from_tensor_slices((sampled_idxs, sampled_seq_from))

        def fetch_record(idx, seq_from):

            def lookup_py(idx_np, seq_from_np):
                return self.slice_window(
                    idx=int(idx_np),
                    seq_from=int(seq_from_np),
                    seq_len=seq_len,
                )

            xs, ys = tf.py_function(
                func=lookup_py,
                inp=[idx, seq_from],
                Tout=[tf.float32, tf.float32],
            )
            xs.set_shape((seq_len, self.in_d()))
            ys.set_shape((seq_len, self.out_d()))

            if emit_weights:
                weight = tf.gather(sampled_weights, idx)
                weight = tf.cast(weight, tf.float32)
                return xs, ys, weight

            return xs, ys

        if not deterministic:
            ds = ds.shuffle(512)

        ds = ds.map(
            fetch_record,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def tf_inference_dataset(
        self,
        batch_size: int = 1,
        cache_fname: str = None,
        return_sample_info: bool = False,
    ):
        """
        Generate all samples, once, with full returned shape
        x - (1, SAMPLE_LEN, 4)
        y - (1, SAMPLE_LEN, 1)

        Args:
            seq_len: second axis for batch
            num_batches: total number of batches generated
            batch_size: batch size
            return_sample_info: if True return (x, y, model_data_z, idx, static_weight) otherwise return normal (x, y)
        """

        def sample_generator():
            for c in range(self.n_chunks):
                data = self.inr_model_data[c]
                weight = self.static_importance_weights[c]
                if return_sample_info:
                    yield (
                        *self.model_data_block_to_xs_ys(data),
                        self.capture_run,
                        c,
                        weight,
                    )
                else:
                    yield self.model_data_block_to_xs_ys(data)

        output_signature = [
            tf.TensorSpec(shape=(self.seq_len, self.in_d()), dtype=tf.float32),
            tf.TensorSpec(shape=(self.seq_len, self.out_d()), dtype=tf.float32),
        ]
        if return_sample_info:
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.string))
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.int32))
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))

        ds = tf.data.Dataset.from_generator(
            sample_generator, output_signature=tuple(output_signature)
        )

        if cache_fname is not None:
            ds = ds.cache(cache_fname)
        if batch_size is not None:
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--uniform-sampling-floor", type=float, default=0.2)
    opts = parser.parse_args()
    print("opts", opts)
    pd = ParametricCaptureStaticData(
        capture_run=opts.run,
        keras_model=opts.model,
        uniform_sampling_floor=opts.uniform_sampling_floor,
    )

    ds = pd.tf_training_dataset(seq_len=64, num_batches=4, batch_size=4)
    for xs, ys, weights in ds:
        print(xs.shape, ys.shape, weights)

    # for x, y, idxs, weights in pd.tf_training_dataset(
    #     seq_len=100, num_batches=5, batch_size=8
    # ):
    #     print("idxs", idxs)
    #     print("weights", weights)
    #     print("x", x.shape, "y", y.shape)
