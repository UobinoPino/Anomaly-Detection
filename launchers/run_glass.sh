#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Spacepresso GLASS runner — Chen et al. ECCV 2024 +  text-guided synthesis
# ─────────────────────────────────────────────────────────────────────────────
#  Three backbone profiles:
#     PROFILE=wrn50    : wide_resnet50_2 + L2/L3      (paper default, ~3 min/cls)
#     PROFILE=dnv2reg  : dinov2_vits14_reg + b6/b9    (~4 min/cls)
#     PROFILE=dnv3     : dinov3_vits16 + b6/b9        (~5 min/cls)
#
#  Three recipes per profile (set RECIPE=...):
#     RECIPE=default        : full GLASS + CSV-driven mode profile
#     RECIPE=uniform        : same, but uniform mode sampling (ablation
#                              of the text-as-a-lantern extension)
#     RECIPE=simplenet      : λ_global=0  (no PGD ascent → SimpleNet-style
#                              with rich synth — keeps everything else)
#
#  Example invocations:
#     bash run_glass.sh                                  # wrn50 / default
#     PROFILE=dnv2reg bash run_glass.sh                  # dinov2-reg
#     PROFILE=dnv3 RECIPE=uniform bash run_glass.sh
#     PROFILE=wrn50 RECIPE=simplenet bash run_glass.sh
#     ONLY="class_06 class_07" bash run_glass.sh         # subset of classes
#
#  Runtime estimate on a single 4090:
#     wrn50    : ~3 min/class  → ~25 min for 8 classes
#     dnv2reg  : ~4 min/class  → ~35 min for 8 classes
#     dnv3     : ~5 min/class  → ~40 min for 8 classes
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/u10813429/anomaly-detection}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_ROOT}/baseline_out}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="${PROFILE:-wrn50}"
RECIPE="${RECIPE:-default}"
ONLY="${ONLY:-}"
SEED="${SEED:-0}"

# ── Backbone profile ─────────────────────────────────────────────────────────
case "${PROFILE}" in
  wrn50)
    BACKBONE="wide_resnet50_2"
    FEATURE_LAYERS=(2 3)
    TARGET_LAYER="2"
    INPUT_SIZE=288
    BATCH_SIZE=16
    SCORE_BATCH=16
    TOTAL_ITERS=3000
    ;;
  dnv2reg)
    BACKBONE="dinov2_vits14_reg"
    FEATURE_LAYERS=(6 9)
    TARGET_LAYER="auto"
    INPUT_SIZE=392       # = 28 * 14
    BATCH_SIZE=16
    SCORE_BATCH=16
    TOTAL_ITERS=3000
    ;;
  dnv3)
    BACKBONE="dinov3_vits16"
    FEATURE_LAYERS=(6 9)
    TARGET_LAYER="auto"
    INPUT_SIZE=384       # = 24 * 16
    BATCH_SIZE=16
    SCORE_BATCH=16
    TOTAL_ITERS=3000
    ;;
  *)
    echo "[FATAL] unknown PROFILE=${PROFILE} (use: wrn50 | dnv2reg | dnv3)"
    exit 2 ;;
esac

# ── Recipe (ablation switches) ───────────────────────────────────────────────
EXTRA_ARGS=()
RUN_TAG=""
case "${RECIPE}" in
  default)
    # Paper-style training + CSV-driven per-class synthesis mode mix.
    RUN_TAG="default"
    ;;
  uniform)
    # Ablation: drop the text-guided per-class mode profile, sample
    # all 8 synthesis modes uniformly. Same as vanilla GLASS would do
    # if it had multiple modes.
    EXTRA_ARGS+=(--no-anom-profile)
    RUN_TAG="uniform"
    ;;
  simplenet)
    # Ablation: drop the PGD global-anomaly term (λ_global=0). This
    # leaves the BCE on normal + local synthesis, which is essentially
    # SimpleNet-with-richer-synth. Useful as a control for measuring
    # what the global term actually buys you.
    EXTRA_ARGS+=(--lambda-global 0.0)
    RUN_TAG="simplenet"
    ;;
  *)
    echo "[FATAL] unknown RECIPE=${RECIPE} (use: default | uniform | simplenet)"
    exit 2 ;;
esac

ONLY_ARG=()
if [[ -n "${ONLY}" ]]; then
  # shellcheck disable=SC2206
  ONLY_ARG=(--only-classes ${ONLY})
fi

# ── Anomaly description CSV (text-guided synthesis profile) ─────────────────
ANOM_CSV="${ANOM_CSV:-${SCRIPT_DIR}/anomaly_descriptions.csv}"
ANOM_CSV_ARG=()
if [[ -f "${ANOM_CSV}" ]]; then
  ANOM_CSV_ARG=(--anom-profile-csv "${ANOM_CSV}")
  echo "[info] using anomaly profile: ${ANOM_CSV}"
else
  echo "[info] no anomaly_descriptions.csv at ${ANOM_CSV}; using script-dir lookup or uniform"
fi

# ── Print plan ───────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════════"
echo "  GLASS  profile=${PROFILE}  recipe=${RECIPE}"
echo "  backbone=${BACKBONE}  layers=${FEATURE_LAYERS[*]}  input=${INPUT_SIZE}"
echo "  total_iters=${TOTAL_ITERS}  bs=${BATCH_SIZE}  seed=${SEED}"
echo "  data_root=${DATA_ROOT}"
echo "  report_dir=${REPORT_DIR}"
[[ -n "${ONLY}" ]] && echo "  classes=${ONLY}"
echo "════════════════════════════════════════════════════════════════════════"

# ── Launch ───────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

python3 -u glass_baseline.py \
  --data-root        "${DATA_ROOT}" \
  --report-dir       "${REPORT_DIR}" \
  --backbone         "${BACKBONE}" \
  --feature-layers   "${FEATURE_LAYERS[@]}" \
  --target-layer     "${TARGET_LAYER}" \
  --input-size       "${INPUT_SIZE}" \
  --total-iters      "${TOTAL_ITERS}" \
  --batch-size       "${BATCH_SIZE}" \
  --score-batch-size "${SCORE_BATCH}" \
  --lr               2e-4 \
  --weight-decay     1e-5 \
  --lambda-local     1.0 \
  --lambda-global    1.0 \
  --attack-epsilon   0.05 \
  --attack-n-steps   4 \
  --warmup-iters     400 \
  --discriminator-hidden 1024 \
  --discriminator-layers 2 \
  --synth-intensity-min 0.5 \
  --synth-intensity-max 1.0 \
  --tta              hvflip \
  --smooth-sigma     1.5 \
  --num-workers      8 \
  --seed             "${SEED}" \
  --save-checkpoints \
  --run-tag          "${PROFILE}_${RUN_TAG}" \
  "${ANOM_CSV_ARG[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${ONLY_ARG[@]}"

echo
echo "════════════════════════════════════════════════════════════════════════"
echo "  Done. Find submission.zip + local_predictions.npz + test_predictions.npz"
echo "  under ${REPORT_DIR}/runs/<run_id>/"
echo "════════════════════════════════════════════════════════════════════════"