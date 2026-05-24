#!/usr/bin/env bash
# Ablation sweep #2 — resolution 224 → 384.
#
# Updated:
#   Uses FAST minibatch coreset selection (batch=128),
#   validated against exact greedy with negligible AP difference
#   and ~27x faster runtime.

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 5 ────────────────────────────────────────────────────────────
# input_size 384 + hvflip TTA
# fast minibatch coreset (batch=128)

echo "================================================================"
echo "EXPERIMENT 5 — input 384 + minibatch coreset (128)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --input-size 384 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --coreset-fp16 \
    --memory-dtype fp16 \
    --batch-size 16 \
    --score-batch-size 8 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --aggressive-cleanup \
    --run-tag "exp5-input384-mb128"

# ── EXPERIMENT 6 ────────────────────────────────────────────────────────────
# d4 TTA version
#
# Uncomment if desired.

echo "================================================================"
echo "EXPERIMENT 6 — d4 TTA + minibatch coreset (128)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --input-size 384 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --coreset-fp16 \
    --memory-dtype fp16 \
    --batch-size 16 \
    --score-batch-size 8 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta d4 \
    --aggressive-cleanup \
    --no-save-banks \
    --run-tag "exp6-d4tta-mb128"

echo
echo "================================================================"
echo "ALL DONE — see $OUT/ablation_master.csv for the new rows."
echo "================================================================"