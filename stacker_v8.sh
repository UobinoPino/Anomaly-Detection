set -euo pipefail

ROOT=/workspace/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
STACKER=$ROOT/xgboost_stacker_v8.py

CACHE=$ROOT/baseline_out/stacker_cache

#EXP2=$RUNS_DIR/20260521-111908_dnv2b14r_L9_11_T9_in392_cs05_mb128_retrain-exp2-vit-dnv2reg_d674ef
EXP3=$RUNS_DIR/20260521-112617_dnv2b14r_L3_6_9_11_T3_in392_cs05_mb128_retrain-exp3-vit-dnv2reg_8557de
#EXP4=$RUNS_DIR/20260521-113404_dnv2b14r_L3_6_9_11_T3_in392_cs05_mb128_tta-hvflip_retrain-exp4-vit-dnv2reg_7e1fc2
EXP5=$RUNS_DIR/20260521-114333_dnv2b14r_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_retrain-exp5-vit-dnv2reg_1581f2
EXP6=$RUNS_DIR/20260521-120103_dnv2b14r_L3_6_9_11_T3_in518_cs05_mb128_tta-d4_retrain-exp6-vit-dnv2reg_9ebcc7
EXP8C=$RUNS_DIR/20260521-122829_cutpaste-pcnn_dnv2b14r_in392_it2500_bs16_pc_cs05mb128_tta-hvflip_retrain-exp8c-vit-dnv2reg_81b562
EFFAD=$RUNS_DIR/20260521-131445_effad_dnv2b14r_L9_in392_it5000_bs8_noMV_tta-hvflip_retrain-effad-vit-dnv2reg_b140c3
RD=$RUNS_DIR/20260521-135556_rd_dnv2b14r_L9_in392_it2500_bs8_lr5e-04_mul_tta-hvflip_rd-vit-dnv2reg_38a4ff
EXP15C=$RUNS_DIR/20260521-140612_fastflow_dnv2b14r_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs4_tta-hvflip_exp15c-fastflow-dnv2b14reg-b8h1_64dc9f
ADINO=$RUNS_DIR/20260521-161953_adino_dnv2s14r_b11_in392_k1_fg75_tta-hvflip_adino-paper-dnv2reg_cd8663
DPMM=$RUNS_DIR/20260521-165643_dpmm_dnv2b14r_b9_in392_pca64_K30_diag_p0.01_tta-hvflip_dpmm-diag64-dnv2reg_252889

# Sanity check
for D in  "$EXP3"  "$EXP5" "$EXP6" \
         "$EXP8C" "$EFFAD" "$RD" "$EXP15C" "$ADINO" "$DPMM"; do
    [[ -f "$D/submission.csv"        ]] || { echo "missing $D/submission.csv";        exit 1; }
    [[ -f "$D/local_predictions.npz" ]] || { echo "missing $D/local_predictions.npz"; exit 1; }
done

mkdir -p "$CACHE"

COMMON_ARGS=(
    --runs
         "$EXP3/submission.csv"
        "$EXP5/submission.csv"  "$EXP6/submission.csv"
        "$EXP8C/submission.csv" "$EFFAD/submission.csv"
        "$RD/submission.csv"    "$EXP15C/submission.csv"
        "$ADINO/submission.csv" "$DPMM/submission.csv"
    --local-preds
          "$EXP3/local_predictions.npz"
        "$EXP5/local_predictions.npz"  "$EXP6/local_predictions.npz"
        "$EXP8C/local_predictions.npz" "$EFFAD/local_predictions.npz"
        "$RD/local_predictions.npz"    "$EXP15C/local_predictions.npz"
        "$ADINO/local_predictions.npz" "$DPMM/local_predictions.npz"
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
echo ".................................."
OUT_C=$RUNS_DIR/${TS}_stacker_xgb_v8
uv run python "$ROOT/xgboost_stacker_v8.py" \
    "${COMMON_ARGS[@]}" \
    --tune-mode global  \
    --tune-cv kfold-stratified --tune-cv-k 5 \
    --tune-timeout-min 10 \
    --calibrate isotonic \
    --final-rank-norm global \
    --gpu auto \
    --out "$OUT_C/submission.csv" \
    --run-tag "stacker-xgb-v8-lb-aligned"