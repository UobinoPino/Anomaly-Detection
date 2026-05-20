#!/usr/bin/env bash
# ensemble_tier0.sh — Tier-0 PDF-native ensemble across the 14 base models
# used by xgboost_stacker_v6. NO XGBoost, NO logistic regression.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/work/u10813429/anomaly-detection"
DATA_ROOT="${PROJECT_ROOT}/data"
REPORT_DIR="${PROJECT_ROOT}/baseline_out"
CACHE_DIR="${REPORT_DIR}/stacker_cache"
PRIORS_DIR="${PROJECT_ROOT}/analysis_out/tables"
RUNS_DIR="${REPORT_DIR}/runs"

# ─────────────────────────────────────────────────────────────────────────────
# FIXED BASE MODEL RUNS (your corrected paths)
# ─────────────────────────────────────────────────────────────────────────────

EXP2="${RUNS_DIR}/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46"
EXP3="${RUNS_DIR}/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18"
EXP4="${RUNS_DIR}/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df"
EXP5="${RUNS_DIR}/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2"
EXP6="${RUNS_DIR}/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb"
EXP7="${RUNS_DIR}/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a"

EXP8C="${RUNS_DIR}/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0"
EXP8D="${RUNS_DIR}/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458"

EXP12MV="${RUNS_DIR}/20260518-185822_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_noMV_tta-hvflip_exp12-uniad-dnv2s14-noMV_512a6d"
EXP14C="${RUNS_DIR}/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373"
EXP15C="${RUNS_DIR}/20260518-163849_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70"

EFFAD="${RUNS_DIR}/20260518-205727_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce"
TEXTAD="${RUNS_DIR}/20260519-155930_textad_in256_b16_it1000_bs8_nd1-3_p0.70_tta-hvflip_baseline_90bd53"
DRAEM="${RUNS_DIR}/20260519-142933_draem_in256_b16_it2500_bs8_tta-hvflip_exp13-draem-in256-b32_addbf8"

# ─────────────────────────────────────────────────────────────────────────────
# RUN LIST
# ─────────────────────────────────────────────────────────────────────────────
RUNS=(
    "${EXP2}" "${EXP3}" "${EXP4}" "${EXP5}" "${EXP6}" "${EXP7}"
    "${EXP8C}" "${EXP8D}"
    "${EXP12MV}" "${EXP14C}" "${EXP15C}"
    "${EFFAD}" "${TEXTAD}" "${DRAEM}"
)

echo "[ensemble_tier0.sh] base-model runs (${#RUNS[@]}):"
for r in "${RUNS[@]}"; do
    echo "    $(basename "${r}")"
done

# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────
for r in "${RUNS[@]}"; do
    if [[ ! -f "${r}/local_predictions.npz" ]]; then
        echo "[FATAL] missing local_predictions.npz in: ${r}" >&2
        exit 1
    fi
done
echo "[ensemble_tier0.sh] artefact check OK."

# ─────────────────────────────────────────────────────────────────────────────
# Common args
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --data-root  "${DATA_ROOT}"
    --report-dir "${REPORT_DIR}"
    --cache-dir  "${CACHE_DIR}"
    --runs       "${RUNS[@]}"
    --seed       0
    --device     cuda

    --smooth-sigma   1.0
    --prior-gamma    0.5
    --multiview-mode max_minus_std
    --multiview-beta 0.3
    --calibrate-lo-pct 0.1
    --calibrate-hi-pct 99.9
    --submission-h   224
    --submission-w   224
    --no-within-image-rank
)

if [[ -d "${PRIORS_DIR}" ]]; then
    COMMON_ARGS+=( --priors-dir "${PRIORS_DIR}" )
else
    echo "[ensemble_tier0.sh] note: priors disabled"
fi

# ─────────────────────────────────────────────────────────────────────────────
# TIER 0-A
# ─────────────────────────────────────────────────────────────────────────────
echo "════════ TIER0-A PADIM ════════"
uv run python ensemble_tier0.py \
    "${COMMON_ARGS[@]}" \
    --families padim \
    --force-family padim \
    --skip-loao \
    --run-tag A-padim-only

# ─────────────────────────────────────────────────────────────────────────────
# TIER 0-B
# ─────────────────────────────────────────────────────────────────────────────
echo "════════ TIER0-B LOAO 4 families ════════"
uv run python ensemble_tier0.py \
    "${COMMON_ARGS[@]}" \
    --families padim gmm patchcore svdd \
    --gmm-K 3 \
    --patchcore-k 5 \
    --patchcore-bank 1500 \
    --svdd-epochs 50 \
    --run-tag B-four-families

# ─────────────────────────────────────────────────────────────────────────────
# TIER 0-C
# ─────────────────────────────────────────────────────────────────────────────
echo "════════ TIER0-C FULL STACK ════════"
uv run python ensemble_tier0.py \
    "${COMMON_ARGS[@]}" \
    --families padim gmm patchcore svdd student_teacher \
    --enable-student-teacher \
    --gmm-K 3 \
    --patchcore-k 5 \
    --patchcore-bank 1500 \
    --svdd-epochs 50 \
    --st-epochs 25 \
    --run-tag C-five-families

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo "════════ ALL TIER-0 RUNS COMPLETE ════════"
echo "Results → ${REPORT_DIR}/tier0_ablation_master.csv"
echo "Submissions → ${REPORT_DIR}/tier0_ensembles/"