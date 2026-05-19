#!/usr/bin/env bash
# FastFlow sweep — 2D normalizing flows on backbone features.
#
# Three variants:
#   exp15   : WRN50-2 multi-scale (layers 1+2+3), 8 flow blocks per scale,
#             hidden_ratio=1.0, clamp=2.0. FastFlow paper default.
#   exp15b  : WRN50-2, 12 flow blocks, hidden_ratio=2.0. More capacity —
#             tests whether the paper default is bottlenecked on
#             Spacepresso's relatively complex feature distributions.
#   exp15c  : DINOv2 ViT-S/14, blocks 3+6+9+11, 8 flow blocks. Strong
#             features + density estimation; should be the best variant
#             on textured classes (coffee, pistachio).
#
# Total walltime estimate on a single NVIDIA L4 (24 GB):
#   exp15   ~55 min (8 classes)
#   exp15b  ~80 min  (12 blocks instead of 8, 2x hidden)
#   exp15c  ~75 min  (DINOv2 forward is slower)
#   Sweep total: ~3.5 hours.
#
# AMP NOTE:
#   --amp-flow is OFF by default (and intentionally not added below).
#   Flow forward + backward through exp/log produces NaNs in fp16
#   early in training. The backbone forward is still in AMP (controlled
#   by the default --amp=True), so we get the speedup where it's safe.

set -euo pipefail
# hello
DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

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

# ── EXPERIMENT 15c — DINOv2 variant ─────────────────────────────────────────
echo "================================================================"
echo "EXPERIMENT 15c — FastFlow @ DINOv2 ViT-S/14, blocks 3+6+9+11"
echo "================================================================"

uv run python fastflow_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vits14 \
    --feature-layers 3 6 9 11 \
    --input-size 518 \
    --n-flow-blocks 8 \
    --hidden-ratio 1.0 \
    --clamp 2.0 \
    --total-iters 2500 \
    --batch-size 8 \
    --lr 1e-3 \
    --weight-decay 1e-5 \
    --score-batch-size 4 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --num-workers 8 \
    --seed 0 \
    --run-tag "exp15c-fastflow-dnv2s14-b8h1"

echo
echo "================================================================"
echo "DONE — three rows appended to $OUT/ablation_master.csv"
echo
echo "To add the best FastFlow run to the XGBoost stacker, append the"
echo "winning run directory to COMMON_ARGS in stack.sh:"
echo "    --runs        \$FASTFLOW_RUN/submission.csv"
echo "    --local-preds \$FASTFLOW_RUN/local_predictions.npz"
echo "================================================================"