#!/usr/bin/env bash
# Spacepresso — XGBoost stacker v6, 14-method bundle.
# Memory-efficient successor to v3/v5; preserves their CLI.

set -euo pipefail

ROOT=/work/u10813429/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/xgboost_stacker_v6.py

# Cache dir (v6).  Stores: aligned val, decoded test (uint8), rank-normed
# test (uint16).  Keyed on file mtimes + small-cc spec — safe to share
# across runs.  Delete to force a clean rebuild.
CACHE=$ROOT/baseline_out/stacker_cache

# Updated experiment paths
#EXP2=$RUNS_DIR/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46
#EXP3=$RUNS_DIR/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18
#EXP4=$RUNS_DIR/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df
#EXP5=$RUNS_DIR/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2
#EXP6=$RUNS_DIR/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb
#EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
#EXP8C=$RUNS_DIR/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0
#EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458
#EXP12MV=$RUNS_DIR/20260518-185822_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_noMV_tta-hvflip_exp12-uniad-dnv2s14-noMV_512a6d
#EXP14C=$RUNS_DIR/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373
#EXP15C=$RUNS_DIR/20260518-163849_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70
#EFFAD=$RUNS_DIR/20260518-205727_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce
#TEXTAD=$RUNS_DIR/20260519-155930_textad_in256_b16_it1000_bs8_nd1-3_p0.70_tta-hvflip_baseline_90bd53
#DRAEM=$RUNS_DIR/20260519-142933_draem_in256_b16_it2500_bs8_tta-hvflip_exp13-draem-in256-b32_addbf8

EXP2=$RUNS_DIR/20260521-030758_dnv2s14r_L9_11_T9_in392_cs05_mb128_retrain-exp2-vit-dnv2reg_dc7aca
EXP3=$RUNS_DIR/20260521-031759_dnv2s14r_L3_6_9_11_T3_in392_cs05_mb128_retrain-exp3-vit-dnv2reg_3cf57f
EXP4=$RUNS_DIR/20260521-032831_dnv2s14r_L3_6_9_11_T3_in392_cs05_mb128_tta-hvflip_retrain-exp4-vit-dnv2reg_86a995
EXP5=$RUNS_DIR/20260521-034122_dnv2s14r_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_retrain-exp5-vit-dnv2reg_11a312
EXP6=$RUNS_DIR/20260521-041020_dnv2s14r_L3_6_9_11_T3_in518_cs05_mb128_tta-d4_retrain-exp6-vit-dnv2reg_5f096c
EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
EXP8C=$RUNS_DIR/20260521-081419_cutpaste-pcnn_dnv2s14r_in392_it2500_bs16_pc_cs05mb128_tta-hvflip_retrain-exp8c-vit-dnv2reg_32406b
EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458
EXP12MV=$RUNS_DIR/20260518-185822_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_noMV_tta-hvflip_exp12-uniad-dnv2s14-noMV_512a6d
EXP14C=$RUNS_DIR/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373
EXP15C=$RUNS_DIR/20260518-163849_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70
EFFAD=$RUNS_DIR/20260518-205727_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce
TEXTAD=$RUNS_DIR/20260519-155930_textad_in256_b16_it1000_bs8_nd1-3_p0.70_tta-hvflip_baseline_90bd53
DRAEM=$RUNS_DIR/20260519-142933_draem_in256_b16_it2500_bs8_tta-hvflip_exp13-draem-in256-b32_addbf8


# Sanity check
for D in "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EXP6" "$EXP7" "$EXP8C" "$EXP8D" \
         "$EXP12MV" "$EXP14C" "$EXP15C" "$EFFAD" "$TEXTAD" "$DRAEM"; do
    [[ -f "$D/submission.csv"        ]] || { echo "missing $D/submission.csv";        exit 1; }
    [[ -f "$D/local_predictions.npz" ]] || { echo "missing $D/local_predictions.npz"; exit 1; }
done

mkdir -p "$CACHE"

COMMON_ARGS=(
    --runs
        "$EXP2/submission.csv"  "$EXP3/submission.csv"  "$EXP4/submission.csv"
        "$EXP5/submission.csv"  "$EXP6/submission.csv"  "$EXP7/submission.csv"
        "$EXP8C/submission.csv" "$EXP8D/submission.csv"
        "$EXP12MV/submission.csv" "$EXP14C/submission.csv"
        "$EXP15C/submission.csv"  "$EFFAD/submission.csv"
        "$TEXTAD/submission.csv"  "$DRAEM/submission.csv"
    --local-preds
        "$EXP2/local_predictions.npz"  "$EXP3/local_predictions.npz"  "$EXP4/local_predictions.npz"
        "$EXP5/local_predictions.npz"  "$EXP6/local_predictions.npz"  "$EXP7/local_predictions.npz"
        "$EXP8C/local_predictions.npz" "$EXP8D/local_predictions.npz"
        "$EXP12MV/local_predictions.npz" "$EXP14C/local_predictions.npz"
        "$EXP15C/local_predictions.npz"  "$EFFAD/local_predictions.npz"
        "$TEXTAD/local_predictions.npz"  "$DRAEM/local_predictions.npz"
    --data-root "$DATA"
    --seed 0
    --neg-per-pos 30
    --cache-dir "$CACHE"
)

TS=$(date +%Y%m%d-%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────
# RUN V6-A — smoke test: v6 defaults (uint16 storage, per-class lifecycle,
# drop-decoded during fusion).  No Optuna, no calibration.
# First run: warms the cache.  Subsequent runs skip decode + rank-norm.
# ─────────────────────────────────────────────────────────────────────────────
echo "RUN V6-A — v6 defaults, no tuning, no calibration"
OUT_A=$RUNS_DIR/${TS}_stacker_xgb_v7
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --rank-norm per-class \
    --tune-mode none \
    --calibrate none \
    --out "$OUT_A/submission.csv" \
    --run-tag "stacker-xgb-v7"

# ─────────────────────────────────────────────────────────────────────────────
# RUN V3-B — v3 + per-class Optuna under the pooled CV objective.
# ─────────────────────────────────────────────────────────────────────────────
echo "RUN V6-B — v5 fixes + per-class tuning"
OUT_B=$RUNS_DIR/${TS}_stacker_xgb_v6_B
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --tune-mode per-class \
    --n-trials 100 \
    --tune-cv loio \
    --tune-timeout-min 15 \
    --calibrate none \
    --out "$OUT_B/submission.csv" \
    --run-tag "stacker-xgb-v6"