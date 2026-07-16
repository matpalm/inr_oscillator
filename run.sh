set -ex
uv run -m keras_v.train \
 --run 003 \
 --harsh \
 --mlp-layers 3 \
 --epochs 100 --batch-size 128 --num-train-samples 100_000
 --learning-rate 1e-4

