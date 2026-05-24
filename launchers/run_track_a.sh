#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Track A — TransFusion at input 384 for class_03 and class_07
# ─────────────────────────────────────────────────────────────────────────────
# What this script does (top → bottom):
#   1. Trains TransFusion per-class on class_03 and class_07 (input 384).
#      Produces 1 run dir under baseline_out/runs/ containing both classes'
#      predictions (test + val) in standard submission.csv / local_predictions.npz
#      format, BUT only for the target classes (not full coverage).
#
#   2. Splices the TransFusion output INTO an existing 14-method run
#      (we use exp5-input384-mb128 = WRN50@384 as the "filler" for the other
#      6 classes). This produces a 15th method that has full 8-class coverage
#      and can be passed to the v6 stacker without breaking alignment.
#
#   3. Re-runs the v6 stacker with the 14 existing methods + the new spliced
#      method (15 total). The stacker's per-class top-3 selection will pick
#      TransFusion automatically on class_03/07 if it's strong there.
#
# Expected run time on a 4090: ~5h (3h train × 2 classes + 30m inference + 1h stacker)
# Expected VRAM: ~12 GB at training, ~6 GB at inference.
# Adjust ITERS down to 25_000 if you want to test the pipeline end-to-end
# faster before committing to a full run.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Configurable paths (CHANGE THESE TO MATCH YOUR LAYOUT) ──────────────────
ROOT=/workspace/anomaly-detection
DATA_ROOT=${ROOT}/data
BASELINE_OUT=${ROOT}/baseline_out
RUNS_DIR=${BASELINE_OUT}/runs
CACHE_DIR=${BASELINE_OUT}/stacker_cache

# The base method used as filler for the non-target classes in the splice.
# Choose one that's reasonable across all 8 classes (NOT one that's already
# best on c03/c07, since that one's signal will be partially overwritten).
# Default: exp5 (WRN50@384) — middle-of-the-pack, won't dominate anything.
BASE_RUN=${RUNS_DIR}/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2

# Iteration count: 50k is the recommended full run. 25k is enough to
# diagnose if TransFusion will help at all (use this first if uncertain).
ITERS=${ITERS:-12500}

# Training hyperparameters
INPUT_SIZE=${INPUT_SIZE:-384}
BATCH=${BATCH:-8}
LR=${LR:-2e-4}
T_MAX=${T_MAX:-20}
SEED=${SEED:-0}

# ─── 1. Train TransFusion on class_03 and class_07 ──────────────────────────
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  STEP 1: Training TransFusion on class_03 + class_07"
echo "═══════════════════════════════════════════════════════════════════════════"

python transfusion_baseline.py \
    --data-root        "${DATA_ROOT}" \
    --classes          class_03 class_07 \
    --input-size       "${INPUT_SIZE}" \
    --iters            "${ITERS}" \
    --batch-size       "${BATCH}" \
    --lr               "${LR}" \
    --T-max            "${T_MAX}" \
    --base-width       64 \
    --tta-hvflip \
    --inference-t-steps 5 10 15 20 \
    --seed             "${SEED}" \
    --out-root         "${RUNS_DIR}" \
    --run-tag          "track-a-in${INPUT_SIZE}"

# Capture the run directory we just created (TransFusion's mainfile prints it,
# but we also identify it as the newest directory matching the pattern).
TF_RUN=$(ls -td "${RUNS_DIR}"/*transfusion_in${INPUT_SIZE}_class_03_class_07_*track-a-in${INPUT_SIZE}* | head -1)
echo ""
echo "TransFusion run: ${TF_RUN}"

# ─── 2. Splice into base method to produce full-coverage method ─────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  STEP 2: Splicing into ${BASE_RUN##*/}"
echo "═══════════════════════════════════════════════════════════════════════════"

# Output naming: same date prefix as TF run, but tagged "_spliced"
TF_RUN_NAME=$(basename "${TF_RUN}")
SPLICED_RUN="${RUNS_DIR}/${TF_RUN_NAME}_spliced_on_$(basename "${BASE_RUN}")"

python splice_for_stacker.py \
    --base-run         "${BASE_RUN}" \
    --tf-run           "${TF_RUN}" \
    --target-classes   class_03 class_07 \
    --data-root        "${DATA_ROOT}" \
    --out-dir          "${SPLICED_RUN}"

echo ""
echo "Spliced run: ${SPLICED_RUN}"

# ─── 3. Re-run v6 stacker with all 14 existing methods + this 15th ──────────
#echo ""
#echo "═══════════════════════════════════════════════════════════════════════════"
#echo "  STEP 3: Re-running v6 stacker (14 + 1 = 15 methods)"
#echo "═══════════════════════════════════════════════════════════════════════════"
#
## IMPORTANT: ${RUNS_15} below is the same list of 14 runs you currently use in
## stacker_v6.sh, with the new ${SPLICED_RUN} appended at the end. Edit the
## list below to match your current 14 runs (these names come from your latest
## run_log.txt).
#RUNS_15=(
#    "${RUNS_DIR}/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46"
#    "${RUNS_DIR}/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18"
#    "${RUNS_DIR}/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df"
#    "${RUNS_DIR}/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2"
#    "${RUNS_DIR}/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb"
#    "${RUNS_DIR}/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a"
#    "${RUNS_DIR}/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0"
#    "${RUNS_DIR}/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458"
#    "${RUNS_DIR}/20260518-185822_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_noMV_tta-hvflip_exp12-uniad-dnv2s14-noMV_512a6d"
#    "${RUNS_DIR}/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373"
#    "${RUNS_DIR}/20260518-163849_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70"
#    "${RUNS_DIR}/20260518-205727_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce"
#    "${RUNS_DIR}/20260519-155930_textad_in256_b16_it1000_bs8_nd1-3_p0.70_tta-hvflip_baseline_90bd53"
#    "${RUNS_DIR}/20260519-142933_draem_in256_b16_it2500_bs8_tta-hvflip_exp13-draem-in256-b32_addbf8"
#    "${SPLICED_RUN}"
#)
#
## Build --runs and --local-preds CLI args
#RUNS_ARGS=()
#LOCAL_PREDS_ARGS=()
#for r in "${RUNS_15[@]}"; do
#    RUNS_ARGS+=("${r}/submission.csv")
#    LOCAL_PREDS_ARGS+=("${r}/local_predictions.npz")
#done
#
#STACKER_OUT_TS=$(date +%Y%m%d-%H%M%S)
#STACKER_OUT_DIR="${RUNS_DIR}/${STACKER_OUT_TS}_stacker_xgb_v6_with_transfusion"
#mkdir -p "${STACKER_OUT_DIR}"
#
#python xgboost_stacker_v6.py \
#    --runs           "${RUNS_ARGS[@]}" \
#    --local-preds    "${LOCAL_PREDS_ARGS[@]}" \
#    --data-root      "${DATA_ROOT}" \
#    --rank-norm      per-class-view \
#    --neg-per-pos    30 \
#    --seed           0 \
#    --out            "${STACKER_OUT_DIR}/submission.csv" \
#    --cache-dir      "${CACHE_DIR}_v15" \
#    --master-csv     "${BASELINE_OUT}/ablation_master.csv" \
#    --run-tag        "stacker-xgb-v6-15methods-track-a"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  TRACK A COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Outputs:"
echo "  TransFusion run     : ${TF_RUN}"
echo "  Spliced run         : ${SPLICED_RUN}"
echo "  New stacker run     : ${STACKER_OUT_DIR}"
echo ""
echo "Next: inspect ${STACKER_OUT_DIR}/run_log.txt to check:"
echo "  1. Does 'Picking top-3 methods per class' show the spliced TF in"
echo "     top-3 for class_03 and class_07?"
echo "  2. Did class_03 STACKER pooled-AP move from 0.357 ?"
echo "  3. Did class_07 STACKER pooled-AP move from 0.176 ?"
echo "  4. Did LB-PROXY POOLED pixel-AP move from 0.6048 ?"