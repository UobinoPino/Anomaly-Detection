#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_textad_sweep.sh — TextAD sweep launcher
#
# Mirrors run_draem_sweep.sh: one entry point that runs textad_baseline.py
# across all 8 classes, writing submission.csv + local_predictions.npz +
# an ablation_master row that xgboost_stacker_v5 picks up.
#
# Per-class budget (L4 24 GB):
#   ~3 min training + ~10 s eval + ~25 s test  →  ~25 min for 8 classes.
#
# Usage:
#   ./run_textad_sweep.sh                # default baseline (1 seed)
#   ./run_textad_sweep.sh seeds          # 3-seed sweep for stacker diversity
#   ./run_textad_sweep.sh quick          # smoke test: 2 classes, 500 iters
#   ./run_textad_sweep.sh long           # 5000 iters, more defects/image
#
# Env overrides:
#   PYTHON, PROJECT_ROOT, DATA_ROOT, REPORT_DIR, DESCRIPTIONS_CSV
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

# ─── Paths ──────────────────────────────────────────────────────────────────
PYTHON=${PYTHON:-python}
PROJECT_ROOT=${PROJECT_ROOT:-/work/u10813429/anomaly-detection}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
REPORT_DIR=${REPORT_DIR:-${PROJECT_ROOT}/baseline_out}
DESCRIPTIONS_CSV=${DESCRIPTIONS_CSV:-${DATA_ROOT}/anomaly_descriptions.csv}

SCRIPT=textad_baseline.py

# Pre-flight: fail loudly if anything obvious is missing rather than letting
# Python sputter halfway through a 25-minute run.
[[ -f "${SCRIPT}" ]]            || { echo "[FATAL] ${SCRIPT} not found in $(pwd)"; exit 2; }
[[ -d "${DATA_ROOT}" ]]         || { echo "[FATAL] data root not found: ${DATA_ROOT}"; exit 2; }
[[ -f "${DESCRIPTIONS_CSV}" ]]  || echo "[warn] descriptions CSV missing: ${DESCRIPTIONS_CSV} — defect_library will fall back to the default mixture for every class."

mkdir -p "${REPORT_DIR}"

# ─── Common args ────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --data-root         "${DATA_ROOT}"
    --report-dir        "${REPORT_DIR}"
    --descriptions-csv  "${DESCRIPTIONS_CSV}"
    --input-size        256
    --unet-base         16
    --batch-size        8
    --score-batch-size  8
    --lr                1e-4
    --anomaly-prob      0.7
    --n-defects         1 3
    --focal-gamma       2.0
    --focal-alpha       0.5
    --smooth-sigma      1.5
    --tta               hvflip
    --num-workers       16
)

# ─── Mode selector ──────────────────────────────────────────────────────────
MODE=${1:-default}

run() {
    # Echo the full command for the run log, then execute.
    echo "▶ ${PYTHON} ${SCRIPT} $*"
    "${PYTHON}" "${SCRIPT}" "$@"
}

case "${MODE}" in
    default)
        echo "═══ TextAD baseline — single seed, full taxonomy, 2500 iters/class ═══"
        run "${COMMON_ARGS[@]}" \
            --total-iters 1000 \
            --seed 0 \
            --run-tag baseline
        ;;

    seeds)
        echo "═══ TextAD seed sweep — 3 seeds for stacker diversity ═══"
        for SEED in 0 1 2; do
            echo
            echo "──────────────── seed=${SEED} ────────────────"
            run "${COMMON_ARGS[@]}" \
                --total-iters 2500 \
                --seed "${SEED}" \
                --run-tag "seed${SEED}"
        done
        ;;

    quick)
        echo "═══ TextAD smoke test — 2 classes, 500 iters, no zip ═══"
        run "${COMMON_ARGS[@]}" \
            --total-iters 500 \
            --only-classes class_01 class_02 \
            --seed 0 \
            --no-zip \
            --run-tag smoke
        ;;

    long)
        echo "═══ TextAD long run — 5000 iters/class, n_defects 2–4 ═══"
        run "${COMMON_ARGS[@]/--n-defects 1 3/--n-defects 2 4}" \
            --total-iters 5000 \
            --seed 0 \
            --run-tag long
        ;;

    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 [default|seeds|quick|long]"
        exit 1
        ;;
esac

echo
echo "✓ done."
echo "    ablation_master.csv : ${REPORT_DIR}/ablation_master.csv"
echo "    run dirs            : ${REPORT_DIR}/runs/"