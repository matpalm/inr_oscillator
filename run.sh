set -ex


#{"run": "089", "learning_rate": 0.001, "weight_decay": 0.001, "epochs": 20,
# "batch_size": 128, "min_note": "A3", "max_note": "A5", "harsh": true,
# "sample_rate_khz": 192, "seed": 123, "num_train_samples": 20000,
# "num_validate_samples": 100, "num_fourier_features": 64, "rff_scale": 1.0,
# "rff_seed": 0, "mlp_layers": 3, "mlp_dim": 16, "alpha_mse": 0.9, "alpha_huber": 0.1,
#  "beta_stft": 0.001, "beta_stft_warmup": 5, "beta_stft_ramp": 5, "base_stft_fft_size": 2048,
#   "base_stft_win_length": 256, "base_stft_hop_size": null}


export RUN=080

uv run -m qkeras_v.train \
 --run $RUN \
 --harsh \
 --io-fp-int 1 --io-fp-frac 15 \
 --num-fourier-features 64 --rff-scale 1.0 \
 --io-fp-int 1 --io-fp-frac 15 \
 --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
 --mlp-layers 3 --mlp-dim 16 \
 --alpha-mse 1.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 20 --num-train-samples 100_000 --batch-size 128

export WEIGHTS_PKL=$PWD/runs/$RUN/weights/qkeras/latest.pkl
# uv run -m unittest test_equivalences.test_rff_equivalence
# uv run -m unittest test_equivalences.test_rff_network_equivalence

rm -rf /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3/
pushd ~/dev/tiliqua/gateware
time pdm inr_waveshaper build --hw r3 --name inr_waveshaper --fs-192khz
popd
cp -r /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3 runs/$RUN/tiliqua_build
openFPGALoader -c dirtyJtag runs/$RUN/tiliqua_build/top.bit