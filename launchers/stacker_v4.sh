#!/usr/bin/env bash
# Spacepresso — XGBoost stacker v4, mirrored to your existing 12-method bundle.
# v4 = v3 features + --gpu (CUDA training) + --parallel-classes N (concurrent
# per-class Optuna tuning in subprocesses sharing the same GPU).
#
# Drop next to xgboost_stacker_v4.py and run.

set -euo pipefail

ROOT=/work/u10813429/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/xgboost_stacker_v4.py

# v4 GPU knobs ───────────────────────────────────────────────────────────────
# PARALLEL_CLASSES: how many classes to tune concurrently in subprocesses
#   4 = safe on a 24GB L4 (each worker holds ~2-3GB GPU). Comfortable.
#   8 = all classes at once. Tight on memory; watch `nvidia-smi` first run,
#       drop back to 4 if a worker dies with a CUDA OOM error.
PARALLEL_CLASSES=8

# Updated experiment paths
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

# Sanity check
for D in "$EXP2" "$EXP3" "$EXP4" "$EXP5" "$EXP6" "$EXP7" "$EXP8C" "$EXP8D" \
         "$EXP12MV" "$EXP14C" "$EXP15C" "$EFFAD"; do
    [[ -f "$D/submission.csv"        ]] || { echo "missing $D/submission.csv";        exit 1; }
    [[ -f "$D/local_predictions.npz" ]] || { echo "missing $D/local_predictions.npz"; exit 1; }
done

# Sanity check: GPU visible (only matters if --gpu is passed below)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s) available:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[warn] nvidia-smi not found — --gpu runs will fail if no CUDA"
fi

COMMON_ARGS=(
    --runs
        "$EXP2/submission.csv"  "$EXP3/submission.csv"  "$EXP4/submission.csv"
        "$EXP5/submission.csv"  "$EXP6/submission.csv"  "$EXP7/submission.csv"
        "$EXP8C/submission.csv" "$EXP8D/submission.csv"
        "$EXP12MV/submission.csv" "$EXP14C/submission.csv"
        "$EXP15C/submission.csv"  "$EFFAD/submission.csv"
    --local-preds
        "$EXP2/local_predictions.npz"  "$EXP3/local_predictions.npz"  "$EXP4/local_predictions.npz"
        "$EXP5/local_predictions.npz"  "$EXP6/local_predictions.npz"  "$EXP7/local_predictions.npz"
        "$EXP8C/local_predictions.npz" "$EXP8D/local_predictions.npz"
        "$EXP12MV/local_predictions.npz" "$EXP14C/local_predictions.npz"
        "$EXP15C/local_predictions.npz"  "$EFFAD/local_predictions.npz"
    --data-root "$DATA"
    --seed 0
    --neg-per-pos 30
)

TS=$(date +%Y%m%d-%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────
# RUN V4-A — smoke test: all v3/v4 fixes ON, no Optuna, no calibration,
# but GPU on for the final per-class fits + test inference.
# Runtime should be a few minutes total; --parallel-classes is a no-op here
# because there's no tuning happening.
# ─────────────────────────────────────────────────────────────────────────────
#echo "RUN V4-A — v4 fixes, no tuning, no calibration, GPU on"
#OUT_A=$RUNS_DIR/${TS}_stacker_xgb_v4_A
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --rank-norm per-class \
#    --tune-mode none \
#    --calibrate none \
#    --gpu \
#    --out "$OUT_A/submission.csv" \
#    --run-tag "stacker-xgb-v4-AA"

# ─────────────────────────────────────────────────────────────────────────────
# RUN V4-B — v4 + per-class Optuna under pooled CV, GPU, parallel.
# Wall-clock budget with --parallel-classes=4 and timeout=5min/class:
#   8 classes / 4 workers × 5 min = ~10 min wall (vs ~40 min CPU sequential).
# With more trials per class (--n-trials 80 instead of 40) the budget is
# the true bottleneck, not the trial count — the timeout will cap it.
# ─────────────────────────────────────────────────────────────────────────────
echo "RUN V4-B — v4 fixes + per-class tuning, GPU + parallel($PARALLEL_CLASSES)"
OUT_B=$RUNS_DIR/${TS}_stacker_xgb_v4_B
uv run python "$STACKER" \
    "${COMMON_ARGS[@]}" \
    --rank-norm per-class \
    --tune-mode per-class \
    --n-trials 80 \
    --tune-cv loio \
    --tune-timeout-min 15 \
    --calibrate none \
    --gpu \
    --parallel-classes "$PARALLEL_CLASSES" \
    --out "$OUT_B/submission.csv" \
    --run-tag "stacker-xgb-v4-BB"

## ─────────────────────────────────────────────────────────────────────────────
## RUN V4-C — v4 + tuning + isotonic calibration.
## Calibration is monotonic (won't change per-class OOF AP) but realigns
## scales across classes, which can change the GLOBAL leaderboard ranking.
## ─────────────────────────────────────────────────────────────────────────────
#echo "RUN V4-C — v4 fixes + tuning + isotonic, GPU + parallel($PARALLEL_CLASSES)"
#OUT_C=$RUNS_DIR/${TS}_stacker_xgb_v4_C
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --rank-norm per-class \
#    --tune-mode per-class \
#    --n-trials 80 \
#    --tune-cv loao \
#    --tune-timeout-min 5 \
#    --calibrate isotonic \
#    --gpu \
#    --parallel-classes "$PARALLEL_CLASSES" \
#    --out "$OUT_C/submission.csv" \
#    --run-tag "stacker-xgb-v4-C"

## ─────────────────────────────────────────────────────────────────────────────
## RUN V4-D — ABLATION: v4 with global rank-norm (i.e. only pooled-CV +
## small-CC + consensus + CVD changes; the rank-norm change is reverted).
## Useful to isolate how much of v3/v4's gain comes from per-class rank-norm.
## GPU still on so it's quick.
## ─────────────────────────────────────────────────────────────────────────────
#echo "RUN V4-D — ablation: v4 but global rank-norm, GPU on"
#OUT_D=$RUNS_DIR/${TS}_stacker_xgb_v4_D
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --rank-norm global \
#    --tune-mode none \
#    --calibrate none \
#    --gpu \
#    --out "$OUT_D/submission.csv" \
#    --run-tag "stacker-xgb-v4-D-ablation"

## ─────────────────────────────────────────────────────────────────────────────
## RUN V4-E — like V4-B but pushes to all 8 classes in parallel + more trials.
## ONLY run this after V4-B works and you've checked nvidia-smi during a run;
## 8 simultaneous CUDA contexts on a 24GB card is workable but tight.
## ─────────────────────────────────────────────────────────────────────────────
#echo "RUN V4-E — v4 fixes + per-class tuning, GPU + parallel(8) aggressive"
#OUT_E=$RUNS_DIR/${TS}_stacker_xgb_v4_E
#uv run python "$STACKER" \
#    "${COMMON_ARGS[@]}" \
#    --rank-norm per-class \
#    --tune-mode per-class \
#    --n-trials 100 \
#    --tune-cv loao \
#    --tune-timeout-min 5 \
#    --calibrate none \
#    --gpu \
#    --parallel-classes 8 \
#    --out "$OUT_E/submission.csv" \
#    --run-tag "stacker-xgb-v4-E-par8"

echo "DONE"