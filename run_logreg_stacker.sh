#!/usr/bin/env bash
# Logistic-regression stacker on the three best Spacepresso runs.
# Replaces score_fusion.py's heuristic local_ap weighting with a
# per-class linear model fitted on local-validation pixels.
#
# Prerequisites:
#   1. local_preds_saver.py and logreg_stacker.py in the project dir
#   2. EACH of the three runs below must have a local_predictions.npz
#      file in its run directory (see local_preds_saver.py docstring
#      for the 3-line integration into patchcore_baseline_v2.py /
#      cutpaste_baseline.py)
#   3. scikit-learn installed:  uv pip install scikit-learn
#
# Walltime: ~3-5 minutes on a CPU node (decoding + logreg + encoding).
# No GPU needed.

set -euo pipefail

RUNS=/work/u10813429/anomaly-detection/baseline_out/runs
DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out/runs/stacker_logreg_v1

EXP7=$RUNS/20260512-230929_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
EXP8C=$RUNS/20260513-160337_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0
EXP8D=$RUNS/20260513-181141_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458


# ── Sanity: every run must have BOTH submission.csv AND local_predictions.npz
for d in "$EXP7" "$EXP8C" "$EXP8D"; do
    if [[ ! -f "$d/submission.csv" ]]; then
        echo "[FATAL] missing $d/submission.csv" >&2; exit 1
    fi
    if [[ ! -f "$d/local_predictions.npz" ]]; then
        echo "[FATAL] missing $d/local_predictions.npz" >&2
        echo "        See local_preds_saver.py docstring for how to" >&2
        echo "        produce it. You will need to re-run that method" >&2
        echo "        ONCE with the saver hook added (~30-90 min)." >&2
        exit 1
    fi
done


# ── Quick inspection (optional but cheap, confirms the npz files are well-formed)
echo "================================================================"
echo "Local-predictions inspection"
echo "================================================================"
for d in "$EXP7" "$EXP8C" "$EXP8D"; do
    echo
    echo "--- $(basename "$d") ---"
    uv run python local_preds_saver.py inspect "$d/local_predictions.npz"
done


# ── Train + predict
echo
echo "================================================================"
echo "Running logreg stacker (exp7 + exp8c + exp8d)"
echo "================================================================"

uv run python logreg_stacker.py \
    --runs        "$EXP7/submission.csv" \
                  "$EXP8C/submission.csv" \
                  "$EXP8D/submission.csv" \
    --local-preds "$EXP7/local_predictions.npz" \
                  "$EXP8C/local_predictions.npz" \
                  "$EXP8D/local_predictions.npz" \
    --data-root   "$DATA" \
    --rank-normalise \
    --C 1.0 \
    --neg-per-pos 30 \
    --seed 0 \
    --out         "$OUT/submission.csv" \
    --run-tag     "stacker-logreg-exp7-exp8c-exp8d"


echo
echo "================================================================"
echo "Done. Submit:  $OUT/submission.zip"
echo
echo "Tune-up suggestions if it doesn't beat 0.7268:"
echo "  - try --C 0.1   (stronger regularisation, less per-class overfit)"
echo "  - try --C 10    (let per-class models diverge more)"
echo "  - try --no-rank-normalise   (sanity check; usually worse)"
echo "  - try --neg-per-pos 100     (more balanced, more memory)"
echo "================================================================"