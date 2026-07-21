set -ex

export RUN=117

# time uv run -m qkeras_v.train \
#  --run $RUN \
#  --dataset-type embed2d --harsh \
#  --num-fourier-features 80 --rff-scale 0.25 --rff-lut-size 4096 \
#  --io-fp-int 1 --io-fp-frac 15 \
#  --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
#  --mlp-dims 16 16 16 \
#  --alpha-mse 0.0 --alpha-huber 1.0 \
#  --beta-stft-warmup 5 --beta-stft-ramp 20 --beta-stft 0.01 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --epochs 40 --learning-rate 1e-3 --cosine-schedule \
#  --num-train-samples 100_000 --batch-size 128

time uv run -m qkeras_v.train \
 --run $RUN \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --num-fourier-features 80 --rff-scale 0.25 --rff-lut-size 4096 \
 --io-fp-int 1 --io-fp-frac 15 \
 --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
 --mlp-dims 16 16 16 \
 --alpha-mse 0.0 --alpha-huber 1.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 20 --beta-stft 0.01 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 40 --learning-rate 1e-3 --cosine-schedule \
 --num-train-samples 100_000 --batch-size 128


export WEIGHTS_PKL=$PWD/runs/$RUN/weights/qkeras/latest.pkl

# uv run -m unittest test_equivalences.test_dense_equivalence
# uv run -m unittest test_equivalences.test_rff_equivalence
# uv run -m unittest test_equivalences.test_rff_network_equivalence

# pushd ~/dev/tiliqua/gateware
# rm -rf build/inr_waveshaper-r3/
# time pdm inr_waveshaper build --hw r3 --name inr_waveshaper --fs-192khz
# popd
# rm -rf runs/$RUN/tiliqua_build
# cp -r /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3 runs/$RUN/tiliqua_build
# openFPGALoader -c dirtyJtag runs/$RUN/tiliqua_build/top.bit

# pdm flash archive build/tiliqua_build/inr-waveshaper*.tar.gz --slot 1 --noconfirm