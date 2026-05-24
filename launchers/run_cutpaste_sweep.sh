##!/usr/bin/env bash
## Step 8 (v2) — CutPaste with fast iteration budget.
##
## v1 used --epochs 256, which produced ~20k iters/class on Spacepresso's
## 2000+ train_good images per class (10-50x more data than MVTec). Training
## accuracy plateaued by epoch 32-64 anyway. v2 fixes the iteration budget at
## 2500 (paper's effective count) and bumps batch + workers.
##
## Speedup vs v1: ~8-10x → total walltime ~40-60 min for 8 classes.
##
##   exp8a:  ResNet-18 (paper standard; fastest)
##   exp8b:  ResNet-50 (richer features, ~2-3x slower)
##
## Note: DINOv2 is intentionally NOT a CutPaste backbone — CutPaste's CNN
## training recipe (SGD lr=0.03) would destroy DINOv2's SSL features.
#
#set -euo pipefail
#
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


#!/usr/bin/env bash
# Step 8 (v3) — CutPaste with PatchCore-NN scoring on the trained features.
#
# The v2 PaDiM-scoring runs got ~0.25 overall (per-position Gaussian breaks
# on Spacepresso's multi-view data). v3 keeps the same CutPaste training
# but scores test images with a PatchCore-style memory bank built over the
# trained features. Expected to match or exceed vanilla PatchCore on
# texture-heavy classes (coffee, pistachio) while providing genuine fusion
# diversity via anomaly-aware features.
#
#   exp8c:  CutPaste + PatchCore-NN @ ResNet-18
#   exp8d:  CutPaste + PatchCore-NN @ ResNet-50
#
# Estimated walltime on a single NVIDIA L4 (24 GB):
#   exp8c (RN18):  ~45-60 min for 8 classes
#   exp8d (RN50):  ~75-100 min for 8 classes
#
# Both depend on patchcore_baseline_v2.py being in the SAME DIR (the
# cutpaste_baseline.py imports patchify_and_combine and greedy_coreset
# from it).



# ── EXPERIMENT 8c ───────────────────────────────────────────────────────────
echo "================================================================"
echo "EXPERIMENT 8c — CutPaste @ ResNet-18 + PatchCore-NN scoring"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --scorer patchcore \
    --backbone resnet18 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 64 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --memory-dtype fp16 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --score-batch-size 16 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8c-cutpaste-rn18-pcnn"

# ── EXPERIMENT 8d ───────────────────────────────────────────────────────────
echo "================================================================"
echo "EXPERIMENT 8d — CutPaste @ ResNet-50 + PatchCore-NN scoring"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --scorer patchcore \
    --backbone resnet50 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 32 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --memory-dtype fp16 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --score-batch-size 12 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8d-cutpaste-rn50-pcnn"

echo
echo "================================================================"
echo "DONE — check $OUT/ablation_master.csv for the new rows."
echo
echo "Then for the fusion (step 9), pick the BEST CutPaste run and use:"
echo "  uv run python score_fusion.py \\"
echo "      --runs <wrn50_exp5>/submission.csv <dinov2_exp7>/submission.csv <best_cutpaste>/submission.csv \\"
echo "      --local-evals <wrn50_exp5>/local_eval.csv <dinov2_exp7>/local_eval.csv <best_cutpaste>/local_eval.csv \\"
echo "      --weights local_ap \\"
echo "      --out $OUT/runs/fusion_v2/submission.csv"
echo
echo "NOTE: do NOT pass --rank-normalise (submissions already calibrated)."
echo "================================================================"