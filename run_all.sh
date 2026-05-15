set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 2 ────────────────────────────────────────────────────────────
# One-variable change vs the baseline: coreset 1% → 10%
echo "================================================================"
echo "EXPERIMENT 2 — coreset 10% (rest unchanged)"
echo "================================================================"
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 2 3 \
    --input-size 224 \
    --coreset-frac 0.05 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta none \
    --run-tag "exp2-coreset10"

# ── EXPERIMENT 3 ────────────────────────────────────────────────────────────
# One-variable change vs exp2: add layer 1 to the feature stack
echo "================================================================"
echo "EXPERIMENT 3 — + multi-scale (layers 1+2+3)"
echo "================================================================"
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --input-size 224 \
    --coreset-frac 0.05 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta none \
    --run-tag "exp3-multiscale"

# ── EXPERIMENT 4 ────────────────────────────────────────────────────────────
# One-variable change vs exp3: TTA on
echo "================================================================"
echo "EXPERIMENT 4 — + TTA hflip+vflip"
echo "================================================================"
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --input-size 224 \
    --coreset-frac 0.05 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp4-tta-hvflip"

echo
echo "================================================================"
echo "ALL DONE — see $OUT/ablation_master.csv for the three new rows."
echo "================================================================"

echo "================================================================"
echo "EXPERIMENT 5 — input 384 + minibatch coreset (128)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --input-size 384 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --coreset-fp16 \
    --memory-dtype fp16 \
    --batch-size 16 \
    --score-batch-size 8 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --aggressive-cleanup \
    --run-tag "exp5-input384-mb128"

# ── EXPERIMENT 6 ────────────────────────────────────────────────────────────
# d4 TTA version
#
# Uncomment if desired.

echo "================================================================"
echo "EXPERIMENT 6 — d4 TTA + minibatch coreset (128)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone wide_resnet50_2 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --input-size 384 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --coreset-fp16 \
    --memory-dtype fp16 \
    --batch-size 16 \
    --score-batch-size 8 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta d4 \
    --aggressive-cleanup \
    --no-save-banks \
    --run-tag "exp6-d4tta-mb128"

echo
echo "================================================================"
echo "ALL DONE — see $OUT/ablation_master.csv for the new rows."
echo "================================================================"


echo "================================================================"
echo "EXPERIMENT 7 — DINOv2 ViT-S/14 backbone (input 518, multi-scale)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vits14 \
    --feature-layers 3 6 9 11 \
    --target-layer 3 \
    --input-size 518 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --coreset-fp16 \
    --memory-dtype fp16 \
    --batch-size 8 \
    --score-batch-size 4 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --aggressive-cleanup \
    --no-save-banks \
    --run-tag "exp7-dinov2s14-in518"

# ── EXPERIMENT 7b (OPTIONAL) ────────────────────────────────────────────────
# Apples-to-apples comparison vs exp5 spatial grid: input 392 (28x28 = 784
# tokens, similar to WRN50@384 layer-2 grid). Uncomment to run alongside.
#
echo "================================================================"
echo "EXPERIMENT 7b — DINOv2 ViT-S/14 (input 392, exp5-equivalent grid)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vits14 \
    --feature-layers 3 6 9 11 \
    --target-layer 3 \
    --input-size 392 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --coreset-fp16 \
    --memory-dtype fp16 \
    --batch-size 16 \
    --score-batch-size 8 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --knn-k 9 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --aggressive-cleanup \
    --no-save-banks \
    --run-tag "exp7b-dinov2s14-in392"

echo
echo "================================================================"
echo "DONE — check $OUT/ablation_master.csv for the new row(s)."
echo "Submission(s): $OUT/runs/<run_id>/submission.zip"
echo "================================================================"


echo "================================================================"
echo "EXPERIMENT 8a — CutPaste @ ResNet-18 (paper standard, v2 budget)"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone resnet18 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 64 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --projection-dim 100 \
    --padim-eps 0.01 \
    --score-batch-size 16 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8a-cutpaste-rn18-fast"

# ── EXPERIMENT 8b ───────────────────────────────────────────────────────────
# ResNet-50 needs a smaller batch to keep activation memory in check on L4.
# 32 → effective 96 (same as v1 RN18 setting). With AMP this fits comfortably.
echo "================================================================"
echo "EXPERIMENT 8b — CutPaste @ ResNet-50 (richer features, v2 budget)"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone resnet50 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 32 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --projection-dim 100 \
    --padim-eps 0.01 \
    --score-batch-size 12 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8b-cutpaste-rn50-fast"

echo
echo "================================================================"
echo "DONE — check $OUT/ablation_master.csv for the two new rows."
echo "Pick the better of (8a, 8b) by AP_overall for fusion (step 9)."
echo "================================================================"

echo "================================================================"
echo "EXPERIMENT 8c — CutPaste @ ResNet-18 + PatchCore-NN scoring"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --scorer patchcore \
    --backbone resnet18 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 64 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --memory-dtype fp16 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --score-batch-size 16 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8c-cutpaste-rn18-pcnn"

# ── EXPERIMENT 8d ───────────────────────────────────────────────────────────
echo "================================================================"
echo "EXPERIMENT 8d — CutPaste @ ResNet-50 + PatchCore-NN scoring"
echo "================================================================"

uv run python cutpaste_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --scorer patchcore \
    --backbone resnet50 \
    --input-size 256 \
    --feature-layers 1 2 3 \
    --target-layer 2 \
    --total-iters 2500 \
    --batch-size 32 \
    --num-workers 8 \
    --lr 0.03 \
    --weight-decay 3e-5 \
    --coreset-frac 0.05 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --memory-dtype fp16 \
    --score-chunk 4096 \
    --memory-chunk 16384 \
    --score-batch-size 12 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp8d-cutpaste-rn50-pcnn"


echo "================================================================"
echo "EXPERIMENT 10 — Reverse Distillation @ WRN50-2 in 256 (mul amap)"
echo "================================================================"

uv run python reverse_distillation_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --input-size 256 \
    --amap-mode mul \
    --total-iters 2500 \
    --batch-size 16 \
    --num-workers 8 \
    --lr 0.005 \
    --beta1 0.5 \
    --beta2 0.999 \
    --weight-decay 0.0 \
    --score-batch-size 16 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp10-rd-wrn50-in256"

# ── EXPERIMENT 10b (OPTIONAL) ──────────────────────────────────────────────
# Alternative: 'sum' amap. Sometimes better when one scale's cosine sim
# saturates close to 1.0 (mul then collapses that pixel even if other
# scales flag a real anomaly). Uncomment to A/B against exp10.

 echo "================================================================"
 echo "EXPERIMENT 10b — Reverse Distillation @ WRN50-2 in 256 (sum amap)"
 echo "================================================================"

 uv run python reverse_distillation_baseline.py \
     --data-root  "$DATA" \
     --report-dir "$OUT" \
     --input-size 256 \
     --amap-mode sum \
     --total-iters 2500 \
     --batch-size 16 \
     --num-workers 8 \
     --lr 0.005 \
     --beta1 0.5 \
     --beta2 0.999 \
     --weight-decay 0.0 \
     --score-batch-size 16 \
     --smooth-sigma 1.5 \
     --tta hvflip \
     --run-tag "exp10b-rd-wrn50-in256-sum"

# ── EXPERIMENT 10c (OPTIONAL) ──────────────────────────────────────────────
# Higher-resolution variant matching exp5's spatial grid. Useful if the
# 256-input run underperforms on the small-defect classes (class_03 / 08).
# Halve --batch-size to keep activation memory in check.

 echo "================================================================"
 echo "EXPERIMENT 10c — Reverse Distillation @ WRN50-2 in 384 (mul amap)"
 echo "================================================================"

 uv run python reverse_distillation_baseline.py \
     --data-root  "$DATA" \
     --report-dir "$OUT" \
     --input-size 384 \
     --amap-mode mul \
     --total-iters 2500 \
     --batch-size 8 \
     --num-workers 8 \
     --lr 0.005 \
     --beta1 0.5 \
     --beta2 0.999 \
     --weight-decay 0.0 \
     --score-batch-size 8 \
     --smooth-sigma 1.5 \
     --tta hvflip \
     --run-tag "exp10c-rd-wrn50-in384"

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