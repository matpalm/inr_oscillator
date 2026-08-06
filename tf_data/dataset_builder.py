from .embed_2d_data import Embed2DData
from .embed_3d_data import Embed3DData
from .pcapture_inmem_data import ParametricCaptureStaticData
from .pcapture_uniform_data import ParametricCaptureUniformData


def dataset_types():
    return ["embed_2d", "embed_3d", "pcapture", "pcapture_uniform"]


def add_dataset_parser_args(parser):
    Embed2DData.add_args(parser.add_argument_group("Embed2DData"))
    Embed3DData.add_args(parser.add_argument_group("Embed3DData"))
    ParametricCaptureStaticData.add_args(
        parser.add_argument_group("ParametricCaptureStaticData")
    )


def build_datasets(opts):

    train_seq_len = int(opts.train_seq_mult * opts.base_stft_win_length)

    assert opts.dataset_type in dataset_types()

    if opts.dataset_type == "embed_2d":
        data = Embed2DData(
            min_note=opts.min_note,
            max_note=opts.max_note,
            sample_rate_khz=opts.sample_rate_khz,
            harsh=opts.harsh,
            seed=opts.seed,
        )
        train_ds = data.tf_dataset(
            batch_size=opts.batch_size,
            seq_len=train_seq_len,
            num_samples=opts.num_train_samples,
            emit_endpt_samples=True,
            emit_interpolated_samples=True,
        )
        validate_ds = data.tf_dataset(
            batch_size=opts.batch_size,
            seq_len=opts.test_seq_len,
            num_samples=opts.num_validate_samples,
            emit_endpt_samples=True,
            emit_interpolated_samples=True,
        )
        return data, train_ds, validate_ds

    if opts.dataset_type == "embed_3d":
        data = Embed3DData(
            min_note=opts.min_note,
            max_note=opts.max_note,
            sample_rate_khz=opts.sample_rate_khz,
            harsh=opts.harsh,
            seed=opts.seed,
        )
        train_ds = data.tf_dataset(
            batch_size=opts.batch_size,
            seq_len=train_seq_len,
            num_samples=opts.num_train_samples,
        )
        validate_ds = data.tf_dataset(
            batch_size=opts.batch_size,
            seq_len=opts.test_seq_len,
            num_samples=opts.num_validate_samples,
        )
        return data, train_ds, validate_ds

    if opts.dataset_type == "pcapture":
        data = ParametricCaptureStaticData(
            capture_run=opts.capture_run,
            keras_model=opts.keras_model,
            seed=opts.seed,
        )
        train_ds = data.tf_training_dataset(
            seq_len=train_seq_len,
            num_batches=opts.num_train_samples // opts.batch_size,
            batch_size=opts.batch_size,
            emit_weights=True,
            deterministic=False,
        )
        validate_ds = data.tf_training_dataset(
            seq_len=opts.test_seq_len,
            num_batches=opts.num_train_samples // opts.batch_size,
            batch_size=opts.batch_size,
            emit_weights=False,
            deterministic=True,
        )
        return data, train_ds, validate_ds

    if opts.dataset_type == "pcapture_uniform":
        data = ParametricCaptureUniformData(
            capture_run=opts.capture_run,
            seed=opts.seed,
        )
        train_ds = data.tf_training_dataset(
            seq_len=train_seq_len,
            num_batches=opts.num_train_samples // opts.batch_size,
            batch_size=opts.batch_size,
            deterministic=False,
        )
        validate_ds = data.tf_training_dataset(
            seq_len=opts.test_seq_len,
            num_batches=opts.num_train_samples // opts.batch_size,
            batch_size=opts.batch_size,
            deterministic=True,
        )
        return data, train_ds, validate_ds

    assert False, opts.dataset_type
