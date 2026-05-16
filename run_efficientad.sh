#!/bin/bash

set -e

# =========================================================
# Paths
# =========================================================


ROOT=/work/u10813429/anomaly-detection/
DATA=$ROOT/data
OUT=$ROOT/baseline_out

cd $ROOT

# =========================================================
# Run 1: Standard EfficientAD (NO multiview)
# =========================================================
echo "====================================================="
echo "Running EfficientAD WITHOUT multiview"
echo "====================================================="

uv run python efficientad_baseline.py \
    --data-root $DATA \
    --report-dir $OUT \
    --input-size 256 \
    --total-iters 2500 \
    --batch-size 16 \
    --score-batch-size 32 \
    --tta hvflip \
    --multiview none \
    --run-tag effad_nomv

# =========================================================
# Run 2: EfficientAD WITH sibling-bank multiview
# =========================================================
echo "====================================================="
echo "Running EfficientAD WITH sibling-bank multiview"
echo "====================================================="

uv run python efficientad_baseline.py \
    --data-root $DATA \
    --report-dir $OUT \
    --input-size 256 \
    --total-iters 2500 \
    --batch-size 16 \
    --score-batch-size 32 \
    --tta hvflip \
    --multiview sibling-bank \
    --mv-alpha 0.5 \
    --run-tag effad_multiview

echo "====================================================="
echo "All runs completed"
echo "====================================================="