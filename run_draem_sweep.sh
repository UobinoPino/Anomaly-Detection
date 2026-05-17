#!/usr/bin/env bash
# DRAEM sweep — single run, TTA only (no multi-view, by design).
#
# DRAEM has no shared feature space across views; the sibling-bank
# multi-view mechanism used in EfficientAD / UniAD doesn't apply.
# If you want score-level multi-view aggregation across this run's
# submission, that's already available in postprocess_submission.py
# (--multiview-floor) and is method-agnostic.
#
# Estimated walltime on a single NVIDIA L4 (24 GB):
#   - Training per class:  ~5-6 min @ bs=8, 2500 iters, base=32
#   - Local eval + test:   ~30-40 s per class
#   Full 8-class run:      ~50 min
#
# Dependencies:
#   - draem_baseline.py and patchcore_baseline_v2.py in same directory
#   - local_preds_saver.py in same directory
#   - no extra pip installs (Perlin noise is pure-numpy)

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 13 ──────────────────────────────────────────────────────────
# Standard DRAEM with the defaults that worked well on MVTec.
# input 256 (multiple of 64 for the 6-level U-Net), base=32 (~63M params),
# anomaly_prob=0.5 (half clean + half synth-corrupted batches),
# focal alpha=0.5 (balanced).
echo "================================================================"
echo "EXPERIMENT 13 — DRAEM @ in256, base=32, 2500 iters"
echo "================================================================"

uv run python draem_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --input-size 256 \
    --unet-base 32 \
    --total-iters 2500 \
    --batch-size 8 \
    --lr 5e-5 \
    --weight-decay 0.0 \
    --anomaly-prob 0.5 \
    --focal-gamma 2.0 \
    --focal-alpha 0.5 \
    --num-workers 8 \
    --score-batch-size 16 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --seed 0 \
    --run-tag "exp13-draem-in256-b32"

# ── EXPERIMENT 13b (OPTIONAL, COMMENTED) ──────────────────────────────────
# Smaller config: base=24 -> ~35M params -> ~2x faster training. Use
# this if you want to A/B against base=32 and confirm the larger model
# is actually helping. Uncomment to run alongside exp13.
#
 echo "================================================================"
 echo "EXPERIMENT 13b — DRAEM @ in256, base=24, 2500 iters"
 echo "================================================================"
 uv run python draem_baseline.py \
     --data-root  "$DATA" \
     --report-dir "$OUT" \
     --input-size 256 \
     --unet-base 24 \
     --total-iters 2500 \
     --batch-size 12 \
     --lr 5e-5 \
     --anomaly-prob 0.5 \
     --focal-gamma 2.0 \
     --focal-alpha 0.5 \
     --num-workers 8 \
     --score-batch-size 16 \
     --smooth-sigma 1.5 \
     --tta hvflip \
     --run-tag "exp13b-draem-in256-b24"

echo
echo "================================================================"
echo "DONE — new row in $OUT/ablation_master.csv"
echo
echo "If the run's local AP is competitive (>= ~0.40 overall), add to"
echo "the stacker by appending these to COMMON_ARGS in"
echo "run_xgb_stacker_full.sh:"
echo
echo "    --runs"
echo "        \$DRAEM_RUN/submission.csv"
echo "    --local-preds"
echo "        \$DRAEM_RUN/local_predictions.npz"
echo
echo "where DRAEM_RUN is the directory printed above (latest exp13)."
echo "================================================================"