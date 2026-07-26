set -ex

export RUN=174_pcapture_concat

# run initial keras_v model with large RFF bank and rff-l1
# use FiLM for MLP0 only
uv run -m keras_v.train \
 --run ${RUN}/kv \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --num-fourier-features 1024 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 --rff-l1 1e-4 \
 --mlp-dims 16 16 16 \
 --alpha-mse 0.01 --alpha-huber 1.0 \
 --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --gamma-slope 0.1 --delta-dc 1e-4 --lambda-morph-consistency 0.1 \
 --epochs 20 --learning-rate 1e-3 --cosine-schedule \
 --num-train-samples 20_000 --batch-size 128

# continue with qkeras_v; init from 160_keras_v but only take top 64 RFF entries
uv run -m qkeras_v.train \
 --run ${RUN}/qkv \
 --dataset-type pcapture --capture-run 600 --keras-model 232_keras/i9 \
 --init-from-run ${RUN}/kv --num-fourier-features 64 --rff-lut-size 4096 \
 --io-fp-int 1 --io-fp-frac 15 \
 --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
 --mlp-dims 16 16 16 \
 --alpha-mse 0.01 --alpha-huber 1.0 \
 --beta-stft-warmup 2 --beta-stft-ramp 2 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --gamma-slope 0.1 --delta-dc 1e-4 --lambda-morph-consistency 0.1 \
 --epochs 10 --learning-rate 1e-3 --cosine-schedule  \
 --num-train-samples 10_000 --batch-size 128

export WEIGHTS_PKL=$PWD/runs/$RUN/qkv/weights/qkeras/latest.pkl

#uv run -m unittest test_equivalences.test_dense_equivalence
#uv run -m unittest test_equivalences.test_rff_equivalence
#uv run -m unittest test_equivalences.test_rff_network_equivalence

pushd ~/dev/tiliqua/gateware
rm -rf build/inr_waveshaper-r3/
time pdm inr_waveshaper build --hw r3 --name inr_waveshaper # --fs-192khz
popd
export TILIQUA_BUILD=runs/${RUN}/tiliqua
rm -rf $TILIQUA_BUILD
cp -r /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3 $TILIQUA_BUILD
uv run -m analysis.parse_top_tim --top-tim $TILIQUA_BUILD

openFPGALoader -c dirtyJtag $TILIQUA_BUILD/top.bit

# pdm flash archive build/tiliqua_build/inr-waveshaper*.tar.gz --slot 1 --noconfirm

