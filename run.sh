set -ex

uv run -m keras_v.train \
 --run 077 \
 --harsh \
 --num-fourier-features 32 --rff-scale 9.0 \
 --mlp-layers 2 --mlp-width 16 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --alpha-mse 1 --alpha-huber 0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --epochs 20 --num-train-samples 20_000

uv run -m keras_v.train \
 --run 077 \
 --harsh \
 --num-fourier-features 32 --rff-scale 9.0 \
 --mlp-layers 2 --mlp-width 16 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --alpha-mse 1 --alpha-huber 0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --epochs 20 --num-train-samples 20_000

uv run -m keras_v.train \
 --run 077 \
 --harsh \
 --num-fourier-features 32 --rff-scale 9.0 \
 --mlp-layers 2 --mlp-width 16 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --alpha-mse 1 --alpha-huber 0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --epochs 20 --num-train-samples 20_000
