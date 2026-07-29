set -ex

export RUN=185_siren_test_2

# c.f. 180 relu

# run initial keras_v model
# uv run -m keras_v.train \
#  --run $RUN/kv \
#  --dataset-type embed2d --harsh \
#  --num-fourier-features 512 --rff-basis gaussian --rff-scale-min 0.1 --rff-scale-max 5.0 \
#  --mlp-dims 16 8 8 --film-layers 1 \
#  --alpha-mse 0.01 --alpha-huber 1.0 \
#  --mlp-activation siren \
#  --beta-stft-warmup 5 --beta-stft-ramp 5 --beta-stft 0.01 \
#  --base-stft-fft-size 2048 --base-stft-win-length 256 \
#  --gamma-slope 0.1 --delta-dc 1e-4 \
#  --epochs 20 --learning-rate 1e-3 --cosine-schedule \
#  --num-train-samples 20_000 --batch-size 128

# continue with qkeras_v;
uv run -m qkeras_v.train \
 --run $RUN/qkv \
 --dataset-type embed2d --harsh \
 --init-from-run $RUN/kv --num-fourier-features 512 --rff-lut-size 4096 \
 --io-fp-int 1 --io-fp-frac 14 \
 --mlp-fp-int 3 --mlp-fp-frac 13 --relu-upper-bound 8 \
 --mlp-dims 16 8 8 --film-layers 1 \
 --alpha-mse 0.01 --alpha-huber 1.0 \
 --mlp-activation siren \
 --beta-stft-warmup 2 --beta-stft-ramp 2 --beta-stft 0.001 \
 --base-stft-fft-size 2048 --base-stft-win-length 256 \
 --gamma-slope 0.1 --delta-dc 1e-4 \
 --epochs 20 --learning-rate 1e-3 --cosine-schedule  \
 --num-train-samples 20_000 --batch-size 128

# TODO: these should move into a json config and be loaded like everything elsez

export WEIGHTS_PKL=$PWD/runs/$RUN/qkv/weights/qkeras/latest.pkl
export PHASE_H_BIN=$PWD/runs/$RUN/qkv/weights/qkeras/phase_h_lut.bin
export PHASE_H_INDEX_BITS=13

uv run -m qkeras_v.export_phase_h_bin \
 --weights-pkl $WEIGHTS_PKL \
 --out-bin $PHASE_H_BIN \
 --index-bits $PHASE_H_INDEX_BITS

uv run -m unittest test_equivalences.test_phase_h_lut_bin_equivalence

# # uv run -m unittest discover -s test_equivalences

pushd ~/dev/tiliqua/gateware
rm -rf build/inr_waveshaper-r3/
time pdm inr_waveshaper build --hw r3 --name inr_waveshaper --fs-192khz
popd
export TILIQUA_BUILD=runs/$RUN/tiliqua
rm -rf $TILIQUA_BUILD
cp -r /home/mat/dev/tiliqua/gateware/build/inr_waveshaper-r3 $TILIQUA_BUILD
uv run -m analysis.parse_top_tim --top-tim $TILIQUA_BUILD
# openFPGALoader -c dirtyJtag $TILIQUA_BUILD/top.bit
pushd ~/dev/tiliqua/gateware
ARCHIVE_PATH="$(ls -t build/inr_waveshaper-r3/*.tar.gz | head -n1)"
pdm flash archive "$ARCHIVE_PATH" --slot 1 --noconfirm
popd
