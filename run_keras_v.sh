set -ex

# compare effect of gamma_slope for rounding harsh edges

# 153a baseline --gamma-slope = --delta-dc = 0

uv run -m keras_v.train \
 --run 154a_dc_0.1 \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --num-fourier-features 200 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
 --mlp-dims 24 24 24 --mlp-activation relu \
 --alpha-mse 0.01 --alpha-huber 1.0 --lambda-morph-consistency 0.1 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --gamma-slope 0 --delta-dc 0.1 \
 --epochs 20 --learning-rate 1e-3 --cosine-schedule \
 --num-train-samples 100_000 --batch-size 128

uv run -m keras_v.train \
 --run 154b_dc_1.0 \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --num-fourier-features 200 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
 --mlp-dims 24 24 24 --mlp-activation relu \
 --alpha-mse 0.01 --alpha-huber 1.0 --lambda-morph-consistency 0.1 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --gamma-slope 0.1 --delta-dc 1.0 \
 --epochs 20 --learning-rate 1e-3 --cosine-schedule \
 --num-train-samples 100_000 --batch-size 128
