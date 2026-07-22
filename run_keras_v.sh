set -ex
#export RUN=990_morph_c_0
export RUN=135
uv run -m keras_v.train \
 --run $RUN \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --num-fourier-features 1024 --rff-l1 1e-5 --rff-scale-min 0.01 --rff-scale-max 10.0 \
 --mlp-dims 24 24 24 \
 --alpha-mse 0.01 --alpha-huber 1.0 --lambda-morph-consistency 0.1 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 20 --learning-rate 1e-3 --cosine-schedule  \
 --num-train-samples 100_000 --batch-size 32

