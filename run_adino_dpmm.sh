#!/usr/bin/env bash
# Run AnomalyDINO + DINO-DPMM with sensible defaults for an RTX 4090.
#
# Both methods are very cheap:
#   - AnomalyDINO:  ~3 min / 8 classes (training-free, only feature extr.)
#   - DINO-DPMM:    ~5-6 min / 8 classes (PCA + variational fit on CPU)
#
# Two BACKBONE_PROFILE choices, mirroring your existing run_dinov3_retrain.sh:
#   dnv2reg : DINOv2 with registers (no gating; ICLR 2024 improvement)
#   dnv3    : DINOv3 ViT-B (Meta gated; needs DINOV3_REPO + DINOV3_WEIGHTS or
#             huggingface-cli login with the license accepted)
#
# After both finish, the printed run_dirs can be added to your stacker
# command (xgboost_stacker_v7.py --runs ... --local-preds ...) for the
# ensemble.

set -euo pipefail

ROOT=${ROOT:-/work/u10813429/anomaly-detection}
DATA=$ROOT/data
OUT=$ROOT/baseline_out

cd "$ROOT"

PROFILE=${BACKBONE_PROFILE:-dnv2reg}

if [[ "$PROFILE" == "dnv2reg" ]]; then
    # vits14 has 12 blocks (0..11); block 11 = last, block 9 = late-mid.
    ADINO_BACKBONE=dinov2_vits14_reg
    ADINO_BLOCK=11                # last block — most semantic
    ADINO_INPUT=392               # 28x28 = 784 tokens

    DPMM_BACKBONE=dinov2_vitb14_reg
    DPMM_BLOCKS="9"               # single mid-late block
    DPMM_INPUT=392
elif [[ "$PROFILE" == "dnv3" ]]; then
    ADINO_BACKBONE=dinov3_vits16
    ADINO_BLOCK=11
    ADINO_INPUT=384               # 24x24 = 576 tokens

    DPMM_BACKBONE=dinov3_vitb16
    DPMM_BLOCKS="9"
    DPMM_INPUT=384
else
    echo "unknown BACKBONE_PROFILE=$PROFILE (use dnv2reg or dnv3)"
    exit 1
fi

echo "================================================================"
echo "AnomalyDINO + DINO-DPMM   PROFILE=$PROFILE"
echo "  AnomalyDINO : $ADINO_BACKBONE block=$ADINO_BLOCK in=$ADINO_INPUT"
echo "  DINO-DPMM   : $DPMM_BACKBONE blocks=$DPMM_BLOCKS in=$DPMM_INPUT"
echo "================================================================"

# ─── AnomalyDINO sweep ───────────────────────────────────────────────────────
echo
echo "================================================================"
echo "ANOMALYDINO"
echo "================================================================"

# Recipe A — paper-faithful: single last block, k=1, foreground mask on, hvflip
#uv run python anomalydino_baseline.py \
#    --data-root  "$DATA" --report-dir "$OUT" \
#    --backbone "$ADINO_BACKBONE" \
#    --block-idx "$ADINO_BLOCK" \
#    --input-size "$ADINO_INPUT" \
#    --knn-k 1 \
#    --foreground-keep-pct 75.0 \
#    --bank-dtype fp16 \
#    --score-chunk 4096 --memory-chunk 32768 \
#    --batch-size 8 --score-batch-size 8 \
#    --tta hvflip \
#    --smooth-sigma 1.5 \
#    --num-workers 8 \
#    --run-tag "adino-paper-$PROFILE"

## Recipe B — robust k-NN, k=4 averaging (less sensitive to single bad neighbour)
#uv run python anomalydino_baseline.py \
#    --data-root  "$DATA" --report-dir "$OUT" \
#    --backbone "$ADINO_BACKBONE" \
#    --block-idx "$ADINO_BLOCK" \
#    --input-size "$ADINO_INPUT" \
#    --knn-k 4 \
#    --foreground-keep-pct 75.0 \
#    --bank-dtype fp16 \
#    --score-chunk 4096 --memory-chunk 32768 \
#    --batch-size 8 --score-batch-size 8 \
#    --tta hvflip \
#    --smooth-sigma 1.5 \
#    --num-workers 8 \
#    --run-tag "adino-k4-$PROFILE"

# Recipe C — no foreground mask (ablation; useful if classes are object-only)
#uv run python anomalydino_baseline.py \
#    --data-root  "$DATA" --report-dir "$OUT" \
#    --backbone "$ADINO_BACKBONE" \
#    --block-idx "$ADINO_BLOCK" \
#    --input-size "$ADINO_INPUT" \
#    --knn-k 1 \
#    --no-foreground-mask \
#    --bank-dtype fp16 \
#    --score-chunk 4096 --memory-chunk 32768 \
#    --batch-size 8 --score-batch-size 8 \
#    --tta hvflip \
#    --smooth-sigma 1.5 \
#    --num-workers 8 \
#    --run-tag "adino-noFG-$PROFILE"

# ─── DINO-DPMM sweep ─────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "DINO-DPMM"
echo "================================================================"

# Recipe A — default: diag covariance, pca=64, K_max=30, late-mid block
uv run python dino_dpmm_baseline.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$DPMM_BACKBONE" \
    --block-indices $DPMM_BLOCKS \
    --input-size "$DPMM_INPUT" \
    --pca-dim 64 \
    --max-components 30 \
    --covariance-type diag \
    --weight-conc-prior 0.01 \
    --max-iter 200 \
    --dpmm-fit-subsample 30000 \
    --batch-size 8 --score-batch-size 8 \
    --tta hvflip \
    --smooth-sigma 1.5 \
    --num-workers 8 \
    --run-tag "dpmm-diag64-$PROFILE"

# Recipe B — full covariance on a smaller PCA dim (captures correlated modes)
uv run python dino_dpmm_baseline.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$DPMM_BACKBONE" \
    --block-indices $DPMM_BLOCKS \
    --input-size "$DPMM_INPUT" \
    --pca-dim 32 \
    --max-components 20 \
    --covariance-type full \
    --weight-conc-prior 0.01 \
    --max-iter 300 \
    --dpmm-fit-subsample 20000 \
    --batch-size 8 --score-batch-size 8 \
    --tta hvflip \
    --smooth-sigma 1.5 \
    --num-workers 8 \
    --run-tag "dpmm-full32-$PROFILE"

# Recipe C — multi-block fused features (more capacity, slower fit)
uv run python dino_dpmm_baseline.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$DPMM_BACKBONE" \
    --block-indices 3 6 9 11 \
    --input-size "$DPMM_INPUT" \
    --pca-dim 96 \
    --max-components 30 \
    --covariance-type diag \
    --weight-conc-prior 0.01 \
    --max-iter 200 \
    --dpmm-fit-subsample 30000 \
    --batch-size 8 --score-batch-size 8 \
    --tta hvflip \
    --smooth-sigma 1.5 \
    --num-workers 8 \
    --run-tag "dpmm-multiblock-$PROFILE"

echo
echo "================================================================"
echo "DONE."
echo
echo "Next: scan baseline_out/runs/ for the new run_dirs (search for"
echo "tags 'adino-*' and 'dpmm-*') and add them to your stacker command:"
echo
echo "  python xgboost_stacker_v7.py \\"
echo "      --runs <existing runs...> <new ADINO and DPMM run dirs> \\"
echo "      --local-preds <matching local_predictions.npz files> \\"
echo "      --out <new ensemble out>"
echo "================================================================"