set -ex

uv run -m keras_v.train \
 --run 060 \
 --harsh \
 --num-fourier-features 32 --rff-scale 3.0 --mlp-layers 2 --mlp-width 16 \
 --beta-stft 0.0001 \
 --epochs 10 --num-train-samples 20_000
