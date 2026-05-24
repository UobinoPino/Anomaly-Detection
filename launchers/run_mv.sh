#!/usr/bin/env bash
# Ablation sweep — sibling-bank multiview variants of the best PatchCore runs.
#
# The non-multiview baselines (exp5, exp7, etc.) already live in
# baseline_out/ablation_master.csv. This script adds the multiview-enabled
# counterparts. The new run directories are distinct (the run_id digest
# changes when --multiview is on) so nothing is overwritten.
#
# Recipe diff vs the originals:
#   --multiview sibling-bank
#   --mv-alpha 0.5            # balanced blend; try 0.3 / 0.7 to A/B
#
# At test time, all views of a sample are batched together, each view's
# patch features form a sibling bank for the others, and view-inconsistent
# pixels get boosted. See module docstring in patchcore_baseline_v2.py.

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 5-MV ─────────────────────────────────────────────────────────
# WRN50-2, multi-scale (1+2+3), input 384, hvflip TTA, + sibling-bank MV.
echo "================================================================"
echo "EXPERIMENT 5-MV — input 384 + minibatch coreset (128) + multiview"
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
    --multiview sibling-bank \
    --mv-alpha 0.3 \
    --aggressive-cleanup \
    --run-tag "exp5-mv-input384-mb128"

# ── EXPERIMENT 7-MV ─────────────────────────────────────────────────────────
# DINOv2 ViT-S/14, multi-scale (blocks 3+6+9+11), input 518, hvflip + MV.
echo "================================================================"
echo "EXPERIMENT 7-MV — DINOv2 ViT-S/14 (input 518) + multiview"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vits14 \
    --feature-layers 3 6 9 11 \
    --target-layer 3 \
    --input-size 518 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --coreset-fp16 \
    --memory-dtype fp16 \
    --batch-size 8 \
    --score-batch-size 4 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --multiview sibling-bank \
    --mv-alpha 0.3 \
    --aggressive-cleanup \
    --no-save-banks \
    --run-tag "exp7-mv-dinov2s14-in518"

echo
echo "================================================================"
echo "DONE — new rows in $OUT/ablation_master.csv."
echo "New run dirs: $OUT/runs/*_mv-a0.50_exp{5,7}-mv-*"
echo "================================================================"