set -ex

export RUN=084

time uv run -m qkeras_v.train \
 --run $RUN \
 --harsh \
 --io-fp-int 1 --io-fp-frac 15 \
 --num-fourier-features 64 --rff-scale 0.5 --rff-lut-size 2048 \
 --io-fp-int 1 --io-fp-frac 15 \
 --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
 --mlp-layers 3 --mlp-dim 16 \
 --alpha-mse 0.0 --alpha-huber 1.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --epochs 20 --learning-rate 1e-3 --num-train-samples 200_000 --batch-size 128

export WEIGHTS_PKL=$PWD/runs/$RUN/weights/qkeras/latest.pkl
uv run -m unittest test_equivalences.test_dense_equivalence
uv run -m unittest test_equivalences.test_rff_equivalence
uv run -m unittest test_equivalences.test_rff_network_equivalence

rm -rf /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3/
pushd ~/dev/tiliqua/gateware
time pdm inr_waveshaper build --hw r3 --name inr_waveshaper --fs-192khz
popd
cp -r /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3 runs/$RUN/tiliqua_build
openFPGALoader -c dirtyJtag runs/$RUN/tiliqua_build/top.bit