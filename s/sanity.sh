set -ex


export RUN=996_k_sanity
time uv run -m keras_v.train \
 --run $RUN \
 --dataset-type embed2d --harsh \
 --num-fourier-features 128 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
 --mlp-dims 8 8 8 --film-layers 1 \
 --alpha-mse 0.01 --alpha-huber 1.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 20 --learning-rate 1e-3  \
 --num-train-samples 5_000 --batch-size 32

export RUN=997_qk_sanity
time uv run -m qkeras_v.train \
 --run $RUN \
 --dataset-type embed2d --harsh \
 --init-from-run ${RUN}/kv --num-fourier-features 64 --rff-lut-size 4096 \
 --io-fp-int 1 --io-fp-frac 15 \
 --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
 --mlp-dims 8 8 8 --film-layers 1 \
 --alpha-mse 0.01 --alpha-huber 1.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 20 --learning-rate 1e-3  \
 --num-train-samples 5_000 --batch-size 32

export WEIGHTS_PKL=$PWD/runs/$RUN/weights/qkeras/latest.pkl
uv run -m unittest discover -s test_equivalences

# rm -rf /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3/
# pushd ~/dev/tiliqua/gateware
# time pdm inr_waveshaper build --hw r3 --name inr_waveshaper --fs-192khz
# popd
# rm -rf runs/$RUN/tiliqua_build
# cp -r /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3 runs/$RUN/tiliqua_build
# openFPGALoader -c dirtyJtag runs/$RUN/tiliqua_build/top.bit
