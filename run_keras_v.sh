set -ex

# uv run -m keras_v.train \
#  --run 155_film \
#  --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
#  --num-fourier-features 200 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
#  --mlp-dims 24 24 24 --mlp-activation relu \
#  --film \
#  --alpha-mse 0.01 --alpha-huber 1.0 --lambda-morph-consistency 0.1 \
#  --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --gamma-slope 0.1 --delta-dc 0.0 \
#  --epochs 20 --learning-rate 1e-3 --cosine-schedule \
#  --num-train-samples 100_000 --batch-size 128

# uv run -m keras_v.train \
#  --run 156_embed_2d_regression \
#  --dataset-type embed2d --harsh\
#  --num-fourier-features 200 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
#  --mlp-dims 24 24 24 --mlp-activation relu \
#  --film \
#  --alpha-mse 0.01 --alpha-huber 1.0 \
#  --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --gamma-slope 0.1 --delta-dc 0.0 \
#  --epochs 20 --learning-rate 1e-3 --cosine-schedule \
#  --num-train-samples 100_000 --batch-size 128

uv run -m keras_v.train \
 --run 157_embed_2d_smaller \
 --dataset-type embed2d --harsh \
 --num-fourier-features 64 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
 --mlp-dims 8 8 8 --mlp-activation relu --film \
 --alpha-mse 0.01 --alpha-huber 1.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --gamma-slope 0.1 --delta-dc 1e-4 \
 --epochs 20 --learning-rate 1e-3 --cosine-schedule \
 --num-train-samples 100_000 --batch-size 128