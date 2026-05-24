##!/usr/bin/env bash
## Spacepresso — XGBoost stacker v2 (roadmap steps 11 + 12)
## Now using: exp2 + exp3 + exp4 + exp5 + exp6 + exp7 + exp8c + exp8d
#
#set -euo pipefail
#
## ─────────────────────────────────────────────────────────────────────────────
## Paths
## ─────────────────────────────────────────────────────────────────────────────
#ROOT=/work/u10813429/anomaly-detection
#RUNS_DIR=$ROOT/baseline_out/runs
#DATA=$ROOT/data
#STACKER=$ROOT/xgboost_stacker.py
#
## ─────────────────────────────────────────────────────────────────────────────
## EXPERIMENTS (UPDATED)
## ─────────────────────────────────────────────────────────────────────────────
#
#EXP2=$RUNS_DIR/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46
#EXP3=$RUNS_DIR/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18
#EXP4=$RUNS_DIR/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df
#EXP5=$RUNS_DIR/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2
#EXP6=$RUNS_DIR/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb
#EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
#EXP8C=$RUNS_DIR/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0
#EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458
#
## Sanity check (all required files exist)
#for D in "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EXP6" "$EXP7" "$EXP8C" "$EXP8D"; do
#    if [[ ! -f "$D/submission.csv" ]]; then
#        echo "ERROR: missing $D/submission.csv"; exit 1
#    fi
#    if [[ ! -f "$D/local_predictions.npz" ]]; then
#        echo "ERROR: missing $D/local_predictions.npz"; exit 1
#    fi
#done
#
## Common args
#COMMON_ARGS=(
#    --runs
#        "$EXP2/submission.csv"
#        "$EXP3/submission.csv"
#        "$EXP4/submission.csv"
#        "$EXP5/submission.csv"
#        "$EXP6/submission.csv"
#        "$EXP7/submission.csv"
#        "$EXP8C/submission.csv"
#        "$EXP8D/submission.csv"
#
#    --local-preds
#        "$EXP2/local_predictions.npz"
#        "$EXP3/local_predictions.npz"
#        "$EXP4/local_predictions.npz"
#        "$EXP5/local_predictions.npz"
#        "$EXP6/local_predictions.npz"
#        "$EXP7/local_predictions.npz"
#        "$EXP8C/local_predictions.npz"
#        "$EXP8D/local_predictions.npz"
#
#    --data-root "$DATA"
#    --seed 0
#    --neg-per-pos 30
#)
#
#TS=$(date +%Y%m%d-%H%M%S)
#
### ─────────────────────────────────────────────────────────────────────────────
### RUN A — baseline
### ─────────────────────────────────────────────────────────────────────────────
##echo "RUN A — default"
##OUT_A=$RUNS_DIR/${TS}_stacker_xgb_A
##uv run python "$STACKER" \
##    "${COMMON_ARGS[@]}" \
##    --tune-mode none \
##    --calibrate none \
##    --out "$OUT_A/submission.csv" \
##    --run-tag "stacker-xgb-A"
##
### ─────────────────────────────────────────────────────────────────────────────
### RUN B — tuning only
### ─────────────────────────────────────────────────────────────────────────────
##echo "RUN B — tuning"
##OUT_B=$RUNS_DIR/${TS}_stacker_xgb_B
##uv run python "$STACKER" \
##    "${COMMON_ARGS[@]}" \
##    --tune-mode per-class \
##    --n-trials 40 \
##    --tune-cv loao \
##    --tune-timeout-min 20 \
##    --calibrate none \
##    --out "$OUT_B/submission.csv" \
##    --run-tag "stacker-xgb-B"
##
### ─────────────────────────────────────────────────────────────────────────────
### RUN C — tuning + isotonic
### ─────────────────────────────────────────────────────────────────────────────
##echo "RUN C — tuning + isotonic"
##OUT_C=$RUNS_DIR/${TS}_stacker_xgb_C
##uv run python "$STACKER" \
##    "${COMMON_ARGS[@]}" \
##    --tune-mode per-class \
##    --n-trials 40 \
##    --tune-cv loao \
##    --tune-timeout-min 20 \
##    --calibrate isotonic \
##    --out "$OUT_C/submission.csv" \
##    --run-tag "stacker-xgb-C"
#
## ─────────────────────────────────────────────────────────────────────────────
## RUN D — tuning + platt
## ─────────────────────────────────────────────────────────────────────────────
#echo "RUN D — tuning + platt"
#OUT_D=$RUNS_DIR/${TS}_stacker_xgb_D
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --tune-mode per-class \
#    --n-trials 40 \
#    --tune-cv loao \
#    --tune-timeout-min 20 \
#    --calibrate platt \
#    --out "$OUT_D/submission.csv" \
#    --run-tag "stacker-xgb-D"
#
#echo "DONE"


#!/usr/bin/env bash
# Spacepresso — XGBoost stacker v2 (roadmap steps 11 + 12)
# Now using: exp2 + exp3 + exp4 + exp5 + exp6 + exp7 + exp8c + exp8d
#          + exp12mv (uniad) + exp14c (cfa) + exp15c (fastflow) + effad

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT=/work/u10813429/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/xgboost_stacker.py

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS (UPDATED)
# ─────────────────────────────────────────────────────────────────────────────

EXP2=$RUNS_DIR/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46
EXP3=$RUNS_DIR/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18
EXP4=$RUNS_DIR/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df
EXP5=$RUNS_DIR/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2
EXP6=$RUNS_DIR/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb
EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
EXP8C=$RUNS_DIR/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0
EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458

# NEW additions
EXP12MV=$RUNS_DIR/20260517-015616_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_MV-a0.50_tta-hvflip_exp12mv-uniad-dnv2s14-MV_7534b5
EXP14C=$RUNS_DIR/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373
EXP15C=$RUNS_DIR/20260517-110227_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70
EFFAD=$RUNS_DIR/20260516-143518_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce

# Sanity check (all required files exist)
for D in "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EXP6" "$EXP7" "$EXP8C" "$EXP8D" \
         "$EXP12MV" "$EXP14C" "$EXP15C" "$EFFAD"; do
    if [[ ! -f "$D/submission.csv" ]]; then
        echo "ERROR: missing $D/submission.csv"; exit 1
    fi
    if [[ ! -f "$D/local_predictions.npz" ]]; then
        echo "ERROR: missing $D/local_predictions.npz"; exit 1
    fi
done

# Common args
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
        "$EXP12MV/submission.csv"
        "$EXP14C/submission.csv"
        "$EXP15C/submission.csv"
        "$EFFAD/submission.csv"

    --local-preds
        "$EXP2/local_predictions.npz"
        "$EXP3/local_predictions.npz"
        "$EXP4/local_predictions.npz"
        "$EXP5/local_predictions.npz"
        "$EXP6/local_predictions.npz"
        "$EXP7/local_predictions.npz"
        "$EXP8C/local_predictions.npz"
        "$EXP8D/local_predictions.npz"
        "$EXP12MV/local_predictions.npz"
        "$EXP14C/local_predictions.npz"
        "$EXP15C/local_predictions.npz"
        "$EFFAD/local_predictions.npz"

    --data-root "$DATA"
    --seed 0
    --neg-per-pos 30
)

TS=$(date +%Y%m%d-%H%M%S)

## ─────────────────────────────────────────────────────────────────────────────
## RUN A — baseline
## ─────────────────────────────────────────────────────────────────────────────
#echo "RUN A — default"
#OUT_A=$RUNS_DIR/${TS}_stacker_xgb_A
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --tune-mode none \
#    --calibrate none \
#    --out "$OUT_A/submission.csv" \
#    --run-tag "stacker-xgb-A"
#
# ─────────────────────────────────────────────────────────────────────────────
# RUN B — tuning only
# ─────────────────────────────────────────────────────────────────────────────
echo "RUN B — tuning"
OUT_B=$RUNS_DIR/${TS}_stacker_xgb_B
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --tune-mode per-class \
    --n-trials 40 \
    --tune-cv loao \
    --tune-timeout-min 2 \
    --calibrate none \
    --out "$OUT_B/submission.csv" \
    --run-tag "stacker-xgb-B23"

## ─────────────────────────────────────────────────────────────────────────────
## RUN C — tuning + isotonic
## ─────────────────────────────────────────────────────────────────────────────
#echo "RUN C — tuning + isotonic"
#OUT_C=$RUNS_DIR/${TS}_stacker_xgb_C
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --tune-mode per-class \
#    --n-trials 40 \
#    --tune-cv loao \
#    --tune-timeout-min 20 \
#    --calibrate isotonic \
#    --out "$OUT_C/submission.csv" \
#    --run-tag "stacker-xgb-C"

# ─────────────────────────────────────────────────────────────────────────────
# RUN D — tuning + platt
# ─────────────────────────────────────────────────────────────────────────────
#echo "RUN D — tuning + platt"
#OUT_D=$RUNS_DIR/${TS}_stacker_xgb_D
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --tune-mode per-class \
#    --n-trials 40 \
#    --tune-cv loao \
#    --tune-timeout-min 20 \
#    --calibrate platt \
#    --out "$OUT_D/submission.csv" \
#    --run-tag "stacker-xgb-D"

echo "DONE"