#!/usr/bin/env bash
# Step 8 (v2) — CutPaste with fast iteration budget.
#
# v1 used --epochs 256, which produced ~20k iters/class on Spacepresso's
# 2000+ train_good images per class (10-50x more data than MVTec). Training
# accuracy plateaued by epoch 32-64 anyway. v2 fixes the iteration budget at
# 2500 (paper's effective count) and bumps batch + workers.
#
# Speedup vs v1: ~8-10x → total walltime ~40-60 min for 8 classes.
#
#   exp8a:  ResNet-18 (paper standard; fastest)
#   exp8b:  ResNet-50 (richer features, ~2-3x slower)
#
# Note: DINOv2 is intentionally NOT a CutPaste backbone — CutPaste's CNN
# training recipe (SGD lr=0.03) would destroy DINOv2's SSL features.

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 8a ───────────────────────────────────────────────────────────
echo "================================================================"
echo "EXPERIMENT 8a — CutPaste @ ResNet-18 (paper standard, v2 budget)"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone resnet18 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 64 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --projection-dim 100 \
    --padim-eps 0.01 \
    --score-batch-size 16 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8a-cutpaste-rn18-fast"

# ── EXPERIMENT 8b ───────────────────────────────────────────────────────────
# ResNet-50 needs a smaller batch to keep activation memory in check on L4.
# 32 → effective 96 (same as v1 RN18 setting). With AMP this fits comfortably.
echo "================================================================"
echo "EXPERIMENT 8b — CutPaste @ ResNet-50 (richer features, v2 budget)"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone resnet50 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 32 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --projection-dim 100 \
    --padim-eps 0.01 \
    --score-batch-size 12 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8b-cutpaste-rn50-fast"

echo
echo "================================================================"
echo "DONE — check $OUT/ablation_master.csv for the two new rows."
echo "Pick the better of (8a, 8b) by AP_overall for fusion (step 9)."
echo "================================================================"