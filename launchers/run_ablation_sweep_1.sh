#!/usr/bin/env bash
# Ablation sweep #1 — runs experiments 2, 3, 4 inside a single GPU job.
# Each call writes its own row to baseline_out/ablation_master.csv
# and its own submission under baseline_out/runs/<run_id>/submission.zip.
#
# Usage on the cluster, inside qsub -I -q gpu ...:
#     cd /work/u10813429/anomaly-detection
#     bash run_ablation_sweep_1.sh
#
# Estimated walltime budget (NVIDIA L4):
#   exp2 (coreset 10%):              ~75 min
#   exp3 (+ multi-scale 1+2+3):      ~80 min
#   exp4 (+ TTA hflip+vflip):       ~110 min   (3x inference)
#   ─────────────────────────────────────────
#   total                          ~265 min  ≈ 4h 25min   (fits in 8h walltime)

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 2 ────────────────────────────────────────────────────────────
# One-variable change vs the baseline: coreset 1% → 10%
echo "================================================================"
echo "EXPERIMENT 2 — coreset 10% (rest unchanged)"
echo "================================================================"
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 2 3 \
    --input-size 224 \
    --coreset-frac 0.05 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta none \
    --run-tag "exp2-coreset10"

# ── EXPERIMENT 3 ────────────────────────────────────────────────────────────
# One-variable change vs exp2: add layer 1 to the feature stack
echo "================================================================"
echo "EXPERIMENT 3 — + multi-scale (layers 1+2+3)"
echo "================================================================"
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --input-size 224 \
    --coreset-frac 0.05 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta none \
    --run-tag "exp3-multiscale"

# ── EXPERIMENT 4 ────────────────────────────────────────────────────────────
# One-variable change vs exp3: TTA on
echo "================================================================"
echo "EXPERIMENT 4 — + TTA hflip+vflip"
echo "================================================================"
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --input-size 224 \
    --coreset-frac 0.05 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp4-tta-hvflip"

echo
echo "================================================================"
echo "ALL DONE — see $OUT/ablation_master.csv for the three new rows."
echo "================================================================"