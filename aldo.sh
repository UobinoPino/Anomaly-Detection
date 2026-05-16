#!/usr/bin/env bash
# Spacepresso — XGBoost stacker
# TEST ONLY:
#   RUN B = per-class Optuna tuning, no calibration
#
# Models:
#   exp7  = DINOv2
#   exp8c = CutPaste RN18 PCNN
#   exp8d = CutPaste RN50 PCNN

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT=/mnt/c/Users/Francoo/PycharmProjects/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/xgboost_stacker.py

# ─────────────────────────────────────────────────────────────────────────────
# Run directories
# ─────────────────────────────────────────────────────────────────────────────

EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a

EXP8C=$RUNS_DIR/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0

EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458

# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

for D in "$EXP7" "$EXP8C" "$EXP8D"; do
    if [[ ! -f "$D/submission.csv" ]]; then
        echo "ERROR: missing $D/submission.csv"
        exit 1
    fi

    if [[ ! -f "$D/local_predictions.npz" ]]; then
        echo "ERROR: missing $D/local_predictions.npz"
        exit 1
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Common args
# ─────────────────────────────────────────────────────────────────────────────

COMMON_ARGS=(
    --runs
        "$EXP7/submission.csv"
        "$EXP8C/submission.csv"
        "$EXP8D/submission.csv"

    --local-preds
        "$EXP7/local_predictions.npz"
        "$EXP8C/local_predictions.npz"
        "$EXP8D/local_predictions.npz"

    --data-root "$DATA"

    --seed 0
    --neg-per-pos 30
)

TS=$(date +%Y%m%d-%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────
# RUN B — per-class Optuna tuning, no calibration
# ─────────────────────────────────────────────────────────────────────────────

echo
echo "================================================================"
echo "RUN B — per-class tuning, no calibration"
echo "================================================================"

OUT_B=$RUNS_DIR/${TS}_stacker_xgb_B_exp7_exp8c_exp8d

uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --tune-mode per-class \
    --n-trials 40 \
    --tune-cv loao \
    --tune-timeout-min 20 \
    --calibrate none \
    --out "$OUT_B/submission.csv" \
    --run-tag "stacker-xgb-B-exp7-exp8c-exp8d"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo
echo "================================================================"
echo "RUN COMPLETE"
echo "================================================================"

ls -la "$OUT_B/submission.zip" 2>/dev/null || true

echo
echo "Artifacts:"
echo "  submission.csv + .zip"
echo "  stacker_config.json"
echo "  oof_predictions.npz"
echo "  run_log.txt"