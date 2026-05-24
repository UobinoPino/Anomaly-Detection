#!/usr/bin/env bash
# FastFlow sweep — 2D normalizing flows on backbone features.
#
# Three variants:
#   exp15   : WRN50-2 multi-scale (layers 1+2+3), 8 flow blocks per scale,
#             hidden_ratio=1.0, clamp=2.0. FastFlow paper default.
#   exp15b  : WRN50-2, 12 flow blocks, hidden_ratio=2.0. More capacity.
#   exp15c  : DINOv2 ViT-B/14 reg, blocks 3+6+9+11, 8 flow blocks.
#             Same backbone family as the rest of the dnv2reg sweep.
#
# Total walltime estimate on a single NVIDIA L4 (24 GB):
#   exp15   ~55 min (8 classes)
#   exp15b  ~80 min
#   exp15c  ~110 min (ViT-B fwd is ~2x ViT-S, flow params 4x at
#                    768-channel inputs)
#
# AMP NOTE:
#   --amp-flow is OFF by default (and intentionally not added below).
#   Flow forward + backward through exp/log produces NaNs in fp16
#   early in training. The backbone forward is still in AMP, so the
#   speedup is preserved where it's safe.

set -euo pipefail

DATA=/workspace/anomaly-detection/data
OUT=/workspace/anomaly-detection/baseline_out

# ── EXPERIMENT 15 — FastFlow paper default ──────────────────────────────────
#echo "================================================================"
#echo "EXPERIMENT 15 — FastFlow @ WRN50-2, 8 blocks, hr=1.0"
#echo "================================================================"
#
#uv run python fastflow_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --backbone wide_resnet50_2 \
#    --feature-layers 1 2 3 \
#    --input-size 384 \
#    --n-flow-blocks 8 \
#    --hidden-ratio 1.0 \
#    --clamp 2.0 \
#    --total-iters 2500 \
#    --batch-size 32 \
#    --lr 1e-3 \
#    --weight-decay 1e-5 \
#    --score-batch-size 16 \
#    --smooth-sigma 1.5 \
#    --tta hvflip \
#    --num-workers 8 \
#    --seed 0 \
#    --run-tag "exp15-fastflow-wrn50-b8h1"
#
## ── EXPERIMENT 15b — Higher capacity flow ───────────────────────────────────
#echo "================================================================"
#echo "EXPERIMENT 15b — FastFlow @ WRN50-2, 12 blocks, hr=2.0"
#echo "================================================================"
#
#uv run python fastflow_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --backbone wide_resnet50_2 \
#    --feature-layers 1 2 3 \
#    --input-size 384 \
#    --n-flow-blocks 12 \
#    --hidden-ratio 2.0 \
#    --clamp 2.0 \
#    --total-iters 2500 \
#    --batch-size 32 \
#    --lr 1e-3 \
#    --weight-decay 1e-5 \
#    --score-batch-size 16 \
#    --smooth-sigma 1.5 \
#    --tta hvflip \
#    --num-workers 8 \
#    --seed 0 \
#    --run-tag "exp15b-fastflow-wrn50-b12h2"

# ── EXPERIMENT 15c — DINOv2 ViT-B/14 reg variant ────────────────────────────
# Channel dim per scale = 768 (ViT-B) vs 384 (ViT-S). Flow params
# scale ~quadratically in channel dim at hr=1.0, so batch must drop.
# batch=4 / score=2 fits at input 518 on a 24 GB L4 with headroom.
# Bump batch to 6 if you've got an A100/H100 to play with.
echo "================================================================"
echo "EXPERIMENT 15c — FastFlow @ DINOv2 ViT-B/14 reg, blocks 3+6+9+11"
echo "================================================================"

uv run python fastflow_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vitb14_reg \
    --feature-layers 3 6 9 11 \
    --input-size 518 \
    --n-flow-blocks 8 \
    --hidden-ratio 1.0 \
    --clamp 2.0 \
    --total-iters 2500 \
    --batch-size 4 \
    --lr 1e-3 \
    --weight-decay 1e-5 \
    --score-batch-size 2 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --num-workers 8 \
    --seed 0 \
    --run-tag "exp15c-fastflow-dnv2b14reg-b8h1"

echo
echo "================================================================"
echo "DONE — rows appended to $OUT/ablation_master.csv"
echo
echo "To add the best FastFlow run to the XGBoost stacker, append the"
echo "winning run directory to COMMON_ARGS in stack.sh:"
echo "    --runs        \$FASTFLOW_RUN/submission.csv"
echo "    --local-preds \$FASTFLOW_RUN/local_predictions.npz"
echo "================================================================"