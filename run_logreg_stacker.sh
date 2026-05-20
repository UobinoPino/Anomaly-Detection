#set -euo pipefail
#
#RUNS_DIR=/work/u10813429/anomaly-detection/baseline_out/runs
#DATA=/work/u10813429/anomaly-detection/data
#OUT=/work/u10813429/anomaly-detection/baseline_out/runs/stacker_logreg_v61
#
## ─────────────────────────────────────────────────────────────────────────────
## SAME 14-MODEL BUNDLE AS XGBOOST STACKER V6
## ─────────────────────────────────────────────────────────────────────────────
#
#EXP2=$RUNS_DIR/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46
#EXP3=$RUNS_DIR/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18
#EXP4=$RUNS_DIR/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df
#EXP5=$RUNS_DIR/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2
#EXP6=$RUNS_DIR/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb
#EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
#EXP12MV=$RUNS_DIR/20260518-185822_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_noMV_tta-hvflip_exp12-uniad-dnv2s14-noMV_512a6d
#EXP14C=$RUNS_DIR/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373
#EXP15C=$RUNS_DIR/20260518-163849_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70
#EFFAD=$RUNS_DIR/20260518-205727_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce
#TEXTAD=$RUNS_DIR/20260519-155930_textad_in256_b16_it1000_bs8_nd1-3_p0.70_tta-hvflip_baseline_90bd53
#DRAEM=$RUNS_DIR/20260519-142933_draem_in256_b16_it2500_bs8_tta-hvflip_exp13-draem-in256-b32_addbf8
#
## ─────────────────────────────────────────────────────────────────────────────
## SANITY CHECK
## ─────────────────────────────────────────────────────────────────────────────
#for d in "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EXP6" "$EXP7" \
#         "$EXP12MV" "$EXP14C" "$EXP15C" "$EFFAD" "$TEXTAD" "$DRAEM"; do
#    [[ -f "$d/submission.csv" ]] || { echo "missing $d/submission.csv"; exit 1; }
#    [[ -f "$d/local_predictions.npz" ]] || { echo "missing $d/local_predictions.npz"; exit 1; }
#done
#
#mkdir -p "$OUT"
#
#echo "================================================================"
#echo "Running LogReg stacker on FULL v6 ensemble (14 models)"
#echo "================================================================"
#
#uv run python log_stacker_v2.py \
#    --runs \
#        "$EXP2/submission.csv"  "$EXP3/submission.csv"  "$EXP4/submission.csv" \
#        "$EXP5/submission.csv"  "$EXP6/submission.csv"  "$EXP7/submission.csv" \
#        "$EXP12MV/submission.csv" "$EXP14C/submission.csv" \
#        "$EXP15C/submission.csv" "$EFFAD/submission.csv" \
#        "$TEXTAD/submission.csv" "$DRAEM/submission.csv" \
#    --local-preds \
#        "$EXP2/local_predictions.npz"  "$EXP3/local_predictions.npz"  "$EXP4/local_predictions.npz" \
#        "$EXP5/local_predictions.npz"  "$EXP6/local_predictions.npz"  "$EXP7/local_predictions.npz" \
#        "$EXP12MV/local_predictions.npz" "$EXP14C/local_predictions.npz" \
#        "$EXP15C/local_predictions.npz" "$EFFAD/local_predictions.npz" \
#        "$TEXTAD/local_predictions.npz" "$DRAEM/local_predictions.npz" \
#    --data-root "$DATA" \
#    --rank-norm per-class \
#    --C 1.0 \
#    --neg-per-pos 30 \
#    --seed 0 \
#    --out "$OUT/submission.csv" \
#    --run-tag "stacker-logreg-v6-full2"
#
#echo
#echo "Done → $OUT/submission.csv"
#
#echo
#echo "Suggested tuning if needed:"
#echo "  --C 0.1   (stronger regularisation)"
#echo "  --C 10    (more flexible per-class fusion)"
#echo "  --neg-per-pos 100 (stabilise rare anomaly structure)"



#!/usr/bin/env bash
# logreg_stacker_v2.sh — overfitting diagnostic against xgboost_stacker_v6.
#
# Uses the SAME 14 base models as stacker_v6.sh so the CV pooled-AP rows
# in ablation_master.csv are directly comparable. Reuses the same
# stacker_cache directory so the aligned-val and decoded-test steps are
# instant if v6 has already run.
#
# This script runs three sweeps. After they finish, look at
# ablation_master.csv and compare these notes columns:
#
#   STACKER_XGB_V6   pooled_AP=X.XXXX    (your v6 result)
#   STACKER_LR_V2    pooled_AP=Y.YYYY    (run A: same features, tuned LR)
#   STACKER_LR_V2    pooled_AP=Z.ZZZZ    (run B: leaky features removed)
#
# Interpretation:
#   X - Y  small (≤ 0.01)  → XGB CV gain is mostly overfit. Submit LR.
#   X - Y  large (≥ 0.02)  → XGB doing real work. Submit XGB.
#   Y - Z  large           → CV is partially inflated by feature leakage.
#                            Z is the "honest" CV ceiling — use that as
#                            your LB expectation. Drop the same flagged
#                            features from v6 too (--no-mahalanobis,
#                            --no-cc-features --no-per-method-zrank-top).

set -euo pipefail

ROOT=/work/u10813429/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/log_stacker_v2.py
CACHE=$ROOT/baseline_out/stacker_cache   # SHARED with xgboost_stacker_v6

# EXACT same experiment paths as stacker_v6.sh — comparison is invalid
# if these drift, so the script asserts they exist and aborts otherwise.
EXP2=$RUNS_DIR/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46
EXP3=$RUNS_DIR/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18
EXP4=$RUNS_DIR/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df
EXP5=$RUNS_DIR/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2
EXP6=$RUNS_DIR/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb
EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
EXP8C=$RUNS_DIR/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0
EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458
EXP12MV=$RUNS_DIR/20260518-185822_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_noMV_tta-hvflip_exp12-uniad-dnv2s14-noMV_512a6d
EXP14C=$RUNS_DIR/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373
EXP15C=$RUNS_DIR/20260518-163849_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70
EFFAD=$RUNS_DIR/20260518-205727_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce
TEXTAD=$RUNS_DIR/20260519-155930_textad_in256_b16_it1000_bs8_nd1-3_p0.70_tta-hvflip_baseline_90bd53
DRAEM=$RUNS_DIR/20260519-142933_draem_in256_b16_it2500_bs8_tta-hvflip_exp13-draem-in256-b32_addbf8

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
    --rank-norm per-class
)

TS=$(date +%Y%m%d-%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────
# RUN LR-A — baseline LR with v6 features, fixed C=0.1, L2.
# Reference number: this is the "with all v6 features, no tuning" LR result.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "╭─ RUN LR-A — LR, v6 features, no tuning, no leak fixes ─────────────────╮"
OUT_A=$RUNS_DIR/${TS}_stacker_lr_v2_A
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --C 0.1 \
    --penalty l2 \
    --tune-mode none \
    --calibrate none \
    --out "$OUT_A/submission.csv" \
    --run-tag "stacker-lr-v2-A-baseline"

# ─────────────────────────────────────────────────────────────────────────────
# RUN LR-B — LR with per-class C+penalty tuning.
# If LR-B ≈ XGB v6, XGB v6 is overfitting CV.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "╭─ RUN LR-B — LR, v6 features, per-class C/penalty tuning ───────────────╮"
OUT_B=$RUNS_DIR/${TS}_stacker_lr_v2_B
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --tune-mode per-class \
    --n-trials 30 \
    --tune-cv loao \
    --tune-timeout-min 10 \
    --calibrate none \
    --out "$OUT_B/submission.csv" \
    --run-tag "stacker-lr-v2-B-tuned"

# ─────────────────────────────────────────────────────────────────────────────
# RUN LR-C — LR with leaky features removed.
# This is the HONEST CV: if LR-A or LR-B are well above LR-C, that delta is
# the magnitude of in-sample leakage in your evaluation, NOT real signal.
# Apply the same flags to xgboost_stacker_v6 to get an honest XGB number too.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "╭─ RUN LR-C — LR, leaky features dropped (HONEST CV) ────────────────────╮"
OUT_C=$RUNS_DIR/${TS}_stacker_lr_v2_C
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --tune-mode per-class \
    --n-trials 30 \
    --tune-cv loao \
    --tune-timeout-min 10 \
    --drop-leaky-top-methods \
    --drop-leaky-mahalanobis \
    --calibrate none \
    --out "$OUT_C/submission.csv" \
    --run-tag "stacker-lr-v2-C-honest"

echo
echo "════════════════════════════════════════════════════════════════════════"
echo " Diagnostic complete. Now inspect ablation_master.csv:"
echo
echo "   grep -E 'STACKER_XGB_V6|STACKER_LR_V2' \\"
echo "        $ROOT/baseline_out/ablation_master.csv | tail -10"
echo
echo " Read the 'notes' column — find each row's pooled_AP=X.XXXX field."
echo
echo " Comparisons that matter:"
echo "   XGB v6  vs  LR-B    → measures XGB nonlinear gain over LR"
echo "   LR-B    vs  LR-C    → measures feature-leak inflation"
echo "   XGB v6  vs  LR-C    → measures total inflated CV margin"
echo
echo " If LR-C is close to XGB v6, your LB will land near LR-C, not XGB v6."
echo "════════════════════════════════════════════════════════════════════════"