import tensorflow as tf


def huber(alpha=0.1, reduce_mean: bool = True):
    """
    Calculates the Huber loss

    Parameters:
        alpha: Huber delta; threshold when the loss transitions from quadratic to linear
    Returns:
        keras loss function
    """

    # TODO: generalise with MSE

    def loss_fn(y_true, y_pred):
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape[-1] == 1, "expected (batch, sequence_length, output_dim=1)"
        # huber loss per element
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, alpha)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + alpha * linear
        # average over elements of y
        huber = tf.reduce_mean(huber, axis=-1)
        # return per-example average over sequence, optionally reduced over batch
        huber = tf.reduce_mean(huber, axis=-1)
        return tf.reduce_mean(huber) if reduce_mean else huber

    return loss_fn


def mse(reduce_mean: bool = True):
    """
    Calculates mean square error

    Returns:
        keras loss function
    """

    def loss_fn(y_true, y_pred):
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape[-1] == 1, "expected (batch, sequence_length, output_dim=1)"
        # average over elements of y
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        # return per-example average over sequence, optionally reduced over batch
        mse = tf.reduce_mean(mse, axis=-1)
        return tf.reduce_mean(mse) if reduce_mean else mse

    return loss_fn


def slope(reduce_mean: bool = True):
    """
    L1 penalty for first-difference (slope) => sharpens edges that MSE smoothes over
    """

    def loss_fn(y_true, y_pred):
        # first difference along S
        d_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        d_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        l1 = tf.abs(d_true - d_pred)
        # average over elements of y...
        l1 = tf.reduce_mean(l1, axis=-1)
        # ...then per-example mean over S -> (B,)
        l1 = tf.reduce_mean(l1, axis=-1)
        return tf.reduce_mean(l1) if reduce_mean else l1

    return loss_fn


def dc(reduce_mean: bool = True):
    """
    L1 penalty on the DC/mean offset => waveshaper should stay 0 mean
    """

    def loss_fn(y_true, y_pred):
        # per-example mean over S -> (B, out_d)
        m_true = tf.reduce_mean(y_true, axis=1)
        m_pred = tf.reduce_mean(y_pred, axis=1)
        l1 = tf.abs(m_true - m_pred)
        # average over elements of y -> (B,)
        l1 = tf.reduce_mean(l1, axis=-1)
        return tf.reduce_mean(l1) if reduce_mean else l1

    return loss_fn


def multires_stft_loss(
    fft_sizes=(256, 128, 64),
    hop_sizes=(64, 32, 16),
    win_lengths=(256, 128, 64),
    w_mag=0.325,
    w_sc=0.675,
    reduce_mean: bool = True,
    seq_len: int = None,
):
    """
    Calculates multi-resolution STFT loss

    Args:
        fft_sizes: FFT sizes used at each STFT res
        hop_sizes: STFT hop sizes for each res
        win_lengths: STFT window lengths for each res
        w_mag: Weight for log-magnitude spectral term
        w_sc: Weight for spectral-convergence term
        seq_len: if set we drop resolutions long than this training length
    """

    assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

    # drop resolutions that can't fit the signal, otherwise the
    # STFT collapses to a single padded frame and the "multi-res" is a no-op.
    if seq_len is not None:
        avail = seq_len
        kept = [
            (f, h, w)
            for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
            if w <= avail
        ]
        if not kept:
            # fall back to the largest power-of-two window that fits
            win = 1
            while win * 2 <= max(avail, 2):
                win *= 2
            kept = [(win, max(1, win // 4), win)]
        fft_sizes, hop_sizes, win_lengths = (
            tuple(f for f, _, _ in kept),
            tuple(h for _, h, _ in kept),
            tuple(w for _, _, w in kept),
        )

    print(
        "multires_stft_loss resolutions"
        f" fft_sizes={tuple(fft_sizes)}"
        f" hop_sizes={tuple(hop_sizes)}"
        f" win_lengths={tuple(win_lengths)}"
    )

    def _stft_mag(x, fft_size, hop, win):
        s = tf.signal.stft(
            x,
            frame_length=win,
            frame_step=hop,
            fft_length=fft_size,
            window_fn=tf.signal.hann_window,
            # pad short tails (and very short sequences) to avoid empty STFT outputs
            pad_end=True,
        )
        return tf.abs(s)

    def loss_fn(y_true, y_pred):
        assert y_true.shape == y_pred.shape
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape[-1] == 1, "expected (batch, sequence_length, output_dim=1)"

        # collapse channel dim for STFT (assuming 1 selected channel)
        y_true_1d = tf.squeeze(y_true, axis=-1)
        y_pred_1d = tf.squeeze(y_pred, axis=-1)

        mr_mag = tf.constant(0.0, dtype=tf.float32)
        mr_sc = tf.constant(0.0, dtype=tf.float32)
        eps = tf.constant(1e-6, dtype=tf.float32)
        n = tf.constant(float(len(fft_sizes)), dtype=tf.float32)

        for fft_size, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
            m_true = _stft_mag(y_true_1d, fft_size, hop, win)
            m_pred = _stft_mag(y_pred_1d, fft_size, hop, win)

            # log-mag L1 is usually more perceptual than linear-mag MSE
            log_true = tf.math.log(m_true + eps)
            log_pred = tf.math.log(m_pred + eps)
            mr_mag += tf.reduce_mean(tf.abs(log_true - log_pred), axis=[-2, -1])

            # spectral convergence
            num = tf.norm(m_true - m_pred, ord="euclidean", axis=[-2, -1])
            den = tf.norm(m_true, ord="euclidean", axis=[-2, -1]) + eps
            mr_sc += tf.math.divide_no_nan(num, den)

        mr_mag = tf.math.divide_no_nan(mr_mag, n)
        mr_sc = tf.math.divide_no_nan(mr_sc, n)

        total = w_mag * mr_mag + w_sc * mr_sc
        total = tf.where(tf.math.is_finite(total), total, tf.zeros_like(total))
        return tf.reduce_mean(total) if reduce_mean else total

    return loss_fn


def combined_loss(
    alpha_mse: float,
    alpha_huber: float,
    beta_stft: float,
    stft_fft_sizes=(256, 128, 64),
    stft_win_lengths=(256, 128, 64),
    stft_hop_sizes=(64, 32, 16),
    reduce_mean: bool = True,
    seq_len: int = None,
    gamma_slope: float = 0.0,
    delta_dc: float = 0.0,
):
    combined_fn = combined_loss_terms(
        alpha_mse=alpha_mse,
        alpha_huber=alpha_huber,
        beta_stft=beta_stft,
        reduce_mean=reduce_mean,
        seq_len=seq_len,
        stft_fft_sizes=stft_fft_sizes,
        stft_win_lengths=stft_win_lengths,
        stft_hop_sizes=stft_hop_sizes,
        gamma_slope=gamma_slope,
        delta_dc=delta_dc,
    )[0]
    return combined_fn


def combined_loss_terms(
    alpha_mse: float,
    alpha_huber: float,
    beta_stft: float,
    stft_fft_sizes=(256, 128, 64),
    stft_win_lengths=(256, 128, 64),
    stft_hop_sizes=(64, 32, 16),
    gamma_slope: float = 0.0,
    delta_dc: float = 0.0,
    reduce_mean: bool = True,
    seq_len: int = None,
):

    mse_fn = mse(reduce_mean=reduce_mean)
    huber_fn = huber(reduce_mean=reduce_mean)
    stft_fn = multires_stft_loss(
        fft_sizes=stft_fft_sizes,
        hop_sizes=stft_hop_sizes,
        win_lengths=stft_win_lengths,
        reduce_mean=reduce_mean,
        seq_len=seq_len,
    )
    slope_fn = slope(reduce_mean=reduce_mean)
    dc_fn = dc(reduce_mean=reduce_mean)

    @tf.function
    def loss_fn(y_true, y_pred):
        loss = alpha_mse * mse_fn(y_true, y_pred)
        loss += alpha_huber * huber_fn(y_true, y_pred)
        loss += beta_stft * stft_fn(y_true, y_pred)
        loss += gamma_slope * slope_fn(y_true, y_pred)
        loss += delta_dc * dc_fn(y_true, y_pred)
        return loss

    @tf.function
    def mse_component(y_true, y_pred):
        return mse_fn(y_true, y_pred)

    @tf.function
    def huber_component(y_true, y_pred):
        return huber_fn(y_true, y_pred)

    @tf.function
    def stft_component(y_true, y_pred):
        return stft_fn(y_true, y_pred)

    @tf.function
    def slope_component(y_true, y_pred):
        return slope_fn(y_true, y_pred)

    @tf.function
    def dc_component(y_true, y_pred):
        return dc_fn(y_true, y_pred)

    # fix named metric names for tb / keras etc
    loss_fn.__name__ = "combined_loss"
    mse_component.__name__ = "mse"
    huber_component.__name__ = "huber"
    stft_component.__name__ = "stft"
    slope_component.__name__ = "slope"
    dc_component.__name__ = "dc"

    return (
        loss_fn,
        mse_component,
        huber_component,
        stft_component,
        slope_component,
        dc_component,
    )
