#!/usr/bin/env bash
# Spacepresso — XGBoost stacker v7, 14-method bundle.
#
# Differences from stacker_v6.sh:
#   • --rank-norm  flag removed (anchors are now train_good-based, implicit)
#   • --small-cc   flag removed (CCs are features now, not a hard filter)
#   • --prior-heatmaps-dir  NEW  (per-class + per-(class,type) heatmaps)
#   • cache dir is a SEPARATE folder from v6 (different feature shapes,
#     different decoded layouts; safer not to share).
#
# Same I/O contract as v6: feed the same N submission.csv + N
# local_predictions.npz, get out a fused submission.csv + .zip +
# stacker_config.json + oof_predictions.npz + run_log.txt and an
# ablation_master.csv row.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit ROOT to match your environment.
#   /workspace/anomaly-detection  — original v6 location (Docker / container)
#   /work/u10813429/anomaly-detection — user's cluster path
# ─────────────────────────────────────────────────────────────────────────────
ROOT=/work/u10813429/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/xgboost_stacker_v7.py

# Pre-computed spatial heatmaps (see analyze_spacepresso_dataset.py § 6).
# Expected files in this dir:
#     06_heat_global.npy
#     06_heat_class_NN.npy                (one per class)
#     06_heat_class_NN_anomaly_MM.npy     (one per (class, anomaly_type))
# If missing, --no-spatial-prior is forced and zeros are used.
PRIORS=$ROOT/analysis_out/tables

# v7 cache dir.  Holds: aligned val, anchors, top-method picks,
# decoded test (uint8).  Keyed on file mtimes; delete to force rebuild.
CACHE=$ROOT/baseline_out/stacker_cache_v7

# ─────────────────────────────────────────────────────────────────────────────
# Model paths (unchanged from v6 — same 14 base learners)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
for D in "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EXP6" "$EXP7" "$EXP8C" "$EXP8D" \
         "$EXP12MV" "$EXP14C" "$EXP15C" "$EFFAD" "$TEXTAD" "$DRAEM"; do
    [[ -f "$D/submission.csv"        ]] || { echo "missing $D/submission.csv";        exit 1; }
    [[ -f "$D/local_predictions.npz" ]] || { echo "missing $D/local_predictions.npz"; exit 1; }
done

[[ -f "$STACKER" ]] || { echo "missing stacker script: $STACKER"; exit 1; }
[[ -d "$DATA"    ]] || { echo "missing data root: $DATA";         exit 1; }

if [[ ! -d "$PRIORS" ]]; then
    echo "[warn] priors dir not found: $PRIORS"
    echo "       run analyze_spacepresso_dataset.py first, or rerun"
    echo "       with --no-spatial-prior to use zero priors."
fi

mkdir -p "$CACHE"

# ─────────────────────────────────────────────────────────────────────────────
# Common args — all 14 methods.
# Order matters: this is the (mi) ordering used everywhere downstream.
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --runs
        "$EXP2/submission.csv"   "$EXP3/submission.csv"   "$EXP4/submission.csv"
        "$EXP5/submission.csv"   "$EXP6/submission.csv"   "$EXP7/submission.csv"
        "$EXP8C/submission.csv"  "$EXP8D/submission.csv"
        "$EXP12MV/submission.csv" "$EXP14C/submission.csv"
        "$EXP15C/submission.csv"  "$EFFAD/submission.csv"
        "$TEXTAD/submission.csv"  "$DRAEM/submission.csv"
    --local-preds
        "$EXP2/local_predictions.npz"   "$EXP3/local_predictions.npz"   "$EXP4/local_predictions.npz"
        "$EXP5/local_predictions.npz"   "$EXP6/local_predictions.npz"   "$EXP7/local_predictions.npz"
        "$EXP8C/local_predictions.npz"  "$EXP8D/local_predictions.npz"
        "$EXP12MV/local_predictions.npz" "$EXP14C/local_predictions.npz"
        "$EXP15C/local_predictions.npz"  "$EFFAD/local_predictions.npz"
        "$TEXTAD/local_predictions.npz"  "$DRAEM/local_predictions.npz"
    --data-root "$DATA"
    --prior-heatmaps-dir "$PRIORS"
    --cache-dir "$CACHE"
    --seed 0
    --neg-per-pos 30
)

TS=$(date +%Y%m%d-%H%M%S)

# ═════════════════════════════════════════════════════════════════════════════
# RUN V7-A — smoke test (all features ON, no tuning, no calibration)
# Use this first to verify the pipeline + cache + LB-proxy.
# Expected runtime: ~10-25 min (depending on val size).  Warms the cache.
# ═════════════════════════════════════════════════════════════════════════════
 echo "RUN V7-A — v7 smoke test (defaults, no tuning, no calibration)"
 OUT_A=$RUNS_DIR/${TS}_stacker_xgb_v7_A
 uv run python "$STACKER" \
     "${COMMON_ARGS[@]}" \
     --tune-mode none \
     --calibrate none \
     --out "$OUT_A/submission.csv" \
     --run-tag "stacker-xgb-v7-A-smoke"

# ═════════════════════════════════════════════════════════════════════════════
# RUN V7-B — main run: per-class Optuna under LOIO + no calibration
# This mirrors v6-B (the best-scoring v6 config) and is the default
# recommended invocation.  Cache from V7-A (if it ran) is reused.
# ═════════════════════════════════════════════════════════════════════════════
echo "RUN V7-B — v7 main: per-class tuning under LOAO CV"
OUT_B=$RUNS_DIR/${TS}_stacker_xgb_v7_B
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --tune-mode per-class \
    --n-trials 100 \
    --tune-cv loao \
    --tune-timeout-min 5 \
    --calibrate none \
    --out "$OUT_B/submission.csv" \
    --run-tag "stacker-xgb-v7-B-perclass-loao"

# ═════════════════════════════════════════════════════════════════════════════
# RUN V7-C — ablation: same as V7-B but WITHOUT the new v7 features.
# Used to measure the marginal LB contribution of anchors+MV+CC+priors
# vs a "v6-style" stacker.  Enable to A/B test.
# ═════════════════════════════════════════════════════════════════════════════
# echo "RUN V7-C — ablation: v7 with v7-features DISABLED (v6-style)"
# OUT_C=$RUNS_DIR/${TS}_stacker_xgb_v7_C_v6style
# uv run python "$STACKER" \
#     "${COMMON_ARGS[@]}" \
#     --no-normed \
#     --no-tail-excess \
#     --no-mv-image-level \
#     --no-mv-window-max \
#     --no-cc-features \
#     --no-spatial-prior \
#     --tune-mode per-class \
#     --n-trials 100 \
#     --tune-cv loio \
#     --tune-timeout-min 15 \
#     --calibrate none \
#     --out "$OUT_C/submission.csv" \
#     --run-tag "stacker-xgb-v7-C-v6style"

# ═════════════════════════════════════════════════════════════════════════════
# RUN V7-D — calibrated variant (isotonic on OOF).
# Often helps when LB AP is sensitive to score scale.  Costs ~3-5 min more.
# ═════════════════════════════════════════════════════════════════════════════
# echo "RUN V7-D — v7 main + isotonic calibration"
# OUT_D=$RUNS_DIR/${TS}_stacker_xgb_v7_D_iso
# uv run python "$STACKER" \
#     "${COMMON_ARGS[@]}" \
#     --tune-mode per-class \
#     --n-trials 100 \
#     --tune-cv loio \
#     --tune-timeout-min 15 \
#     --calibrate isotonic \
#     --out "$OUT_D/submission.csv" \
#     --run-tag "stacker-xgb-v7-D-iso"

# ═════════════════════════════════════════════════════════════════════════════
# RUN V7-E — feature-pillar ablation grid.
# Toggles each v7 pillar OFF in turn (anchors / MV / CC / priors).
# Use to identify which pillar contributes most LB.
# ═════════════════════════════════════════════════════════════════════════════
# for PILLAR in no-normed no-mv-image-level no-mv-window-max \
#               no-cc-features no-spatial-prior; do
#     echo "RUN V7-E[$PILLAR] — ablation: --$PILLAR"
#     OUT_E=$RUNS_DIR/${TS}_stacker_xgb_v7_E_${PILLAR//-/_}
#     uv run python "$STACKER" \
#         "${COMMON_ARGS[@]}" \
#         --$PILLAR \
#         --tune-mode none \
#         --calibrate none \
#         --out "$OUT_E/submission.csv" \
#         --run-tag "stacker-xgb-v7-E-${PILLAR}"
# done

echo ""
echo "Done.  Check the per-class verification table near the bottom of"
echo "    $OUT_B/run_log.txt"
echo "and the LB-PROXY POOLED PIXEL-AP block right after it — that"
echo "number is what predicts the public LB score."