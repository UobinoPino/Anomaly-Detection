#!/usr/bin/env bash
# Spacepresso — XGBoost stacker B only

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT=/work/u10813429/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/xgboost_stacker.py

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

EXP2=$RUNS_DIR/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46
EXP3=$RUNS_DIR/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18
EXP4=$RUNS_DIR/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df
EXP5=$RUNS_DIR/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2
EXP6=$RUNS_DIR/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb
EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
EXP8C=$RUNS_DIR/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0
EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458

NEW2=$RUNS_DIR/20260516-143518_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce
NEW3=$RUNS_DIR/20260516-145732_effad_rn18_in256_it2500_bs16_MV-a0.50_tta-hvflip_effad_multiview_e08748
NEW4=$RUNS_DIR/20260516-190801_uniad_effb4_in256_d256_e4d4_nr1_it2500_bs16_noMV_tta-hvflip_exp12-uniad-effb4-noMV_2f4704
NEW5=$RUNS_DIR/20260516-192355_uniad_effb4_in256_d256_e4d4_nr1_it2500_bs16_MV-a0.50_tta-hvflip_exp12mv-uniad-effb4-MV_80c1e5

# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

for D in \
    "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EXP6" "$EXP7" "$EXP8C" "$EXP8D" \
    "$NEW2" "$NEW3" "$NEW4" "$NEW5"; do
    [[ -f "$D/submission.csv" ]] || { echo "Missing submission.csv in $D"; exit 1; }
    [[ -f "$D/local_predictions.npz" ]] || { echo "Missing local_predictions.npz in $D"; exit 1; }
done

# ─────────────────────────────────────────────────────────────────────────────
# Common args
# ─────────────────────────────────────────────────────────────────────────────

COMMON_ARGS=(
    --runs
        "$EXP2/submission.csv"
        "$EXP3/submission.csv"
        "$EXP4/submission.csv"
        "$EXP5/submission.csv"
        "$EXP6/submission.csv"
        "$EXP7/submission.csv"
        "$EXP8C/submission.csv"
        "$EXP8D/submission.csv"
        "$NEW2/submission.csv"
        "$NEW3/submission.csv"
        "$NEW4/submission.csv"
        "$NEW5/submission.csv"

    --local-preds
        "$EXP2/local_predictions.npz"
        "$EXP3/local_predictions.npz"
        "$EXP4/local_predictions.npz"
        "$EXP5/local_predictions.npz"
        "$EXP6/local_predictions.npz"
        "$EXP7/local_predictions.npz"
        "$EXP8C/local_predictions.npz"
        "$EXP8D/local_predictions.npz"
        "$NEW2/local_predictions.npz"
        "$NEW3/local_predictions.npz"
        "$NEW4/local_predictions.npz"
        "$NEW5/local_predictions.npz"

    --data-root "$DATA"
    --seed 0
    --neg-per-pos 30
)

TS=$(date +%Y%m%d-%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────
# RUN B — tuning only
# ─────────────────────────────────────────────────────────────────────────────

echo "RUN B — tuning"

OUT_B=$RUNS_DIR/${TS}_stacker_xgb_B

uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --tune-mode per-class \
    --n-trials 60 \
    --tune-cv loao \
    --tune-timeout-min 30 \
    --calibrate none \
    --out "$OUT_B/submission.csv" \
    --run-tag "stacker-xgb-B"

echo "DONE"

