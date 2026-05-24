set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

#echo "================================================================"
#echo "EXPERIMENT 15c — FastFlow @ DINOv2 ViT-S/14, blocks 3+6+9+11"
#echo "================================================================"
#
#uv run python fastflow_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --backbone dinov2_vits14 \
#    --feature-layers 3 6 9 11 \
#    --input-size 518 \
#    --n-flow-blocks 8 \
#    --hidden-ratio 1.0 \
#    --clamp 2.0 \
#    --total-iters 2500 \
#    --batch-size 8 \
#    --lr 1e-3 \
#    --weight-decay 1e-5 \
#    --score-batch-size 4 \
#    --smooth-sigma 1.5 \
#    --tta hvflip \
#    --num-workers 8 \
#    --seed 0 \
#    --run-tag "exp15c-fastflow-dnv2s14-b8h1"
#
#
#echo "================================================================"
#echo "EXPERIMENT 12 — UniAD @ DINOv2 ViT-S/14 block 9, no multi-view"
#echo "================================================================"
#
#uv run python uniad_baseline.py \
#    "${COMMON[@]}" \
#    --multiview none \
#    --run-tag "exp12-uniad-dnv2s14-noMV"
#

echo "====================================================="
echo "Running EfficientAD WITHOUT multiview"
echo "====================================================="

uv run python efficientad_baseline.py \
    --data-root $DATA \
    --report-dir $OUT \
    --input-size 256 \
    --total-iters 2500 \
    --batch-size 16 \
    --score-batch-size 32 \
    --tta hvflip \
    --multiview none \
    --run-tag effad_nomv



echo
echo "================================================================"
echo "DONE — check $OUT/ablation_master.csv for the new rows."
echo
echo "Then for the fusion (step 9), pick the BEST CutPaste run and use:"
echo "  uv run python score_fusion.py \\"
echo "      --runs <wrn50_exp5>/submission.csv <dinov2_exp7>/submission.csv <best_cutpaste>/submission.csv \\"
echo "      --local-evals <wrn50_exp5>/local_eval.csv <dinov2_exp7>/local_eval.csv <best_cutpaste>/local_eval.csv \\"
echo "      --weights local_ap \\"
echo "      --out $OUT/runs/fusion_v2/submission.csv"
echo
echo "NOTE: do NOT pass --rank-normalise (submissions already calibrated)."
echo "================================================================"