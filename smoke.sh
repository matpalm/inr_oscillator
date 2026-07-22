set -ex

export RUN=999
export WEIGHTS_PKL=$PWD/runs/$RUN/weights/qkeras/latest.pkl

# time uv run -m qkeras_v.train \
#  --run $RUN \
#  --dataset-type embed2d --harsh \
#  --num-fourier-features 32 --rff-scale-min 0.25 --rff-scale-max 0.25 --rff-lut-size 2048 \
#  --io-fp-int 1 --io-fp-frac 15 \
#  --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
#  --mlp-dims 4 4 \
#  --alpha-mse 0.0 --alpha-huber 1.0 \
#  --beta-stft-warmup 0 --beta-stft-ramp 0 --beta-stft 0.001 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --epochs 1 --learning-rate 1e-3  \
#  --num-train-samples 500 --batch-size 128

time uv run -m qkeras_v.train \
 --run $RUN \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --num-fourier-features 32 --rff-scale-min 0.25 --rff-scale-max 0.25 --rff-lut-size 2048 \
 --io-fp-int 1 --io-fp-frac 15 \
 --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
 --mlp-dims 4 4 \
 --alpha-mse 0.0 --alpha-huber 1.0 \
 --beta-stft-warmup 0 --beta-stft-ramp 0 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 1 --learning-rate 1e-3  \
 --num-train-samples 100 --batch-size 32



# uv run -m unittest test_equivalences.test_dense_equivalence
# uv run -m unittest test_equivalences.test_rff_equivalence
# uv run -m unittest test_equivalences.test_rff_network_equivalence

# rm -rf /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3/
# pushd ~/dev/tiliqua/gateware
# time pdm inr_waveshaper build --hw r3 --name inr_waveshaper --fs-192khz
# popd
# rm -rf runs/$RUN/tiliqua_build
# cp -r /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3 runs/$RUN/tiliqua_build
# openFPGALoader -c dirtyJtag runs/$RUN/tiliqua_build/top.bit
