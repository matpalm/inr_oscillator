set -ex

# try different basis init for rff B

# uv run -m keras_v.train \
#  --run 150_gaussian_basis \
#  --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
#  --num-fourier-features 80 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
#  --mlp-dims 24 24 24 --mlp-activation relu \
#  --alpha-mse 0.01 --alpha-huber 1.0 --lambda-morph-consistency 0.1 \
#  --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --epochs 20 --learning-rate 1e-3 --cosine-schedule  \
#  --num-train-samples 100_000 --batch-size 128

# uv run -m keras_v.train \
#  --run 151_harmonic_basis \
#  --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
#  --num-fourier-features 80 --rff-basis harmonic \
#  --mlp-dims 24 24 24 --mlp-activation relu \
#  --alpha-mse 0.01 --alpha-huber 1.0 --lambda-morph-consistency 0.1 \
#  --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --epochs 20 --learning-rate 1e-3 --cosine-schedule  \
#  --num-train-samples 100_000 --batch-size 128

uv run -m keras_v.train \
 --run 152_harmonic_basis_no_morph_c \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --num-fourier-features 80 --rff-basis harmonic \
 --mlp-dims 24 24 24 --mlp-activation relu \
 --alpha-mse 0.01 --alpha-huber 1.0 --lambda-morph-consistency 0.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 20 --learning-rate 1e-3 --cosine-schedule  \
 --num-train-samples 100_000 --batch-size 128
