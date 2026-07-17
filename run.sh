set -ex


#{"run": "089", "learning_rate": 0.001, "weight_decay": 0.001, "epochs": 20,
# "batch_size": 128, "min_note": "A3", "max_note": "A5", "harsh": true,
# "sample_rate_khz": 192, "seed": 123, "num_train_samples": 20000,
# "num_validate_samples": 100, "num_fourier_features": 64, "rff_scale": 1.0,
# "rff_seed": 0, "mlp_layers": 3, "mlp_width": 16, "alpha_mse": 0.9, "alpha_huber": 0.1,
#  "beta_stft": 0.001, "beta_stft_warmup": 5, "beta_stft_ramp": 5, "base_stft_fft_size": 2048,
#   "base_stft_win_length": 256, "base_stft_hop_size": null}


# uv run -m qkeras_v.train \
#  --run 078 \
#  --harsh \
#  --io-fp-int 1 --io-fp-frac 15 \
#  --num-fourier-features 64 --rff-scale 1.0 \
#  --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
#  --mlp-layers 3 --mlp-width 16 \
#  --alpha-mse 1.0 \
#  --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --epochs 20 --num-train-samples 20_000 --batch-size 128

#  uv run -m test_equivalence.rff_equivalence --weights-pkl runs/078/weights/qkeras/latest.pkl

export RUN_DIR=$PWD/runs
export WEIGHTS_PKL=$RUN_DIR/078/weights/qkeras/latest.pkl
export LUT_SIZE=1024
pushd ../tiliqua/gateware
pdm inr_waveshaper build --hw r3 --name foo --fs-192khz
popd
# openFPGALoader -c dirtyJtag build/foo-r3/top.bit
