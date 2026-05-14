#!/usr/bin/env bash
# Step 10 of the roadmap — Reverse Distillation (RD4AD).
#
# Adds a third independent track for the score-fusion stack:
#   1. PatchCore WRN50-2 @ 384  (exp5)
#   2. PatchCore DINOv2 ViT-S/14 @ 518  (exp7)
#   3. Reverse Distillation @ 256  (this step, exp10)
#
# Why this is a useful fusion partner:
#   - Same teacher backbone as exp5 (WRN50-2) but a fundamentally different
#     scoring paradigm: instead of nearest-neighbour in a frozen feature
#     space, RD learns a one-class bottleneck + decoder that fails to
#     reconstruct anomalous regions. Error modes are uncorrelated with
#     PatchCore's, which is what fusion needs.
#   - Different from CutPaste (exp8) too — RD trains on train_good only,
#     no synthetic anomalies, and uses the WRN50 teacher rather than a
#     fine-tuned RN18/50 encoder.
#
# Memory/speed on a single NVIDIA L4 (24 GB):
#   Per class @ input 256, batch 16, 2500 iters:
#     - Training:   ~6 min  (Adam + AMP, ~150 ms/iter)
#     - Inference:  ~1.5 min  (TTA hvflip = 3 passes, ~100 ms/batch)
#     - Per class:  ~8 min
#   8 classes total: ~65 min
#
# Peak VRAM:
#   - Teacher (frozen, fp16 forward):  ~50 MB activations
#   - OCBE + Decoder + optimizer:      ~1.2 GB (weights + Adam state + grad)
#   - Batch-16 activations:            ~150 MB
#   - Total:                           ~1.5 GB / 24 GB available
#
# Dependencies:
#   reverse_distillation_baseline.py imports from patchcore_baseline_v2.py
#   (FeatureExtractor, scan_dataset, RLE helpers). They must live in the
#   same directory.

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 10 ──────────────────────────────────────────────────────────
# Standard RD4AD recipe: WRN50-2 teacher, input 256, 2500 iters @ batch 16,
# Adam(0.5, 0.999) at lr=5e-3, hvflip TTA, 'mul' anomaly map (paper default).
#echo "================================================================"
#echo "EXPERIMENT 10 — Reverse Distillation @ WRN50-2 in 256 (mul amap)"
#echo "================================================================"
#
#uv run python reverse_distillation_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --input-size 256 \
#    --amap-mode mul \
#    --total-iters 2500 \
#    --batch-size 16 \
#    --num-workers 8 \
#    --lr 0.005 \
#    --beta1 0.5 \
#    --beta2 0.999 \
#    --weight-decay 0.0 \
#    --score-batch-size 16 \
#    --smooth-sigma 1.5 \
#    --tta hvflip \
#    --run-tag "exp10-rd-wrn50-in256"
#
## ── EXPERIMENT 10b (OPTIONAL) ──────────────────────────────────────────────
## Alternative: 'sum' amap. Sometimes better when one scale's cosine sim
## saturates close to 1.0 (mul then collapses that pixel even if other
## scales flag a real anomaly). Uncomment to A/B against exp10.
#
# echo "================================================================"
# echo "EXPERIMENT 10b — Reverse Distillation @ WRN50-2 in 256 (sum amap)"
# echo "================================================================"
#
# uv run python reverse_distillation_baseline.py \
#     --data-root  "$DATA" \
#     --report-dir "$OUT" \
#     --input-size 256 \
#     --amap-mode sum \
#     --total-iters 2500 \
#     --batch-size 16 \
#     --num-workers 8 \
#     --lr 0.005 \
#     --beta1 0.5 \
#     --beta2 0.999 \
#     --weight-decay 0.0 \
#     --score-batch-size 16 \
#     --smooth-sigma 1.5 \
#     --tta hvflip \
#     --run-tag "exp10b-rd-wrn50-in256-sum"
#
## ── EXPERIMENT 10c (OPTIONAL) ──────────────────────────────────────────────
## Higher-resolution variant matching exp5's spatial grid. Useful if the
## 256-input run underperforms on the small-defect classes (class_03 / 08).
## Halve --batch-size to keep activation memory in check.
#
# echo "================================================================"
# echo "EXPERIMENT 10c — Reverse Distillation @ WRN50-2 in 384 (mul amap)"
# echo "================================================================"
#
# uv run python reverse_distillation_baseline.py \
#     --data-root  "$DATA" \
#     --report-dir "$OUT" \
#     --input-size 384 \
#     --amap-mode mul \
#     --total-iters 2500 \
#     --batch-size 8 \
#     --num-workers 8 \
#     --lr 0.005 \
#     --beta1 0.5 \
#     --beta2 0.999 \
#     --weight-decay 0.0 \
#     --score-batch-size 8 \
#     --smooth-sigma 1.5 \
#     --tta hvflip \
#     --run-tag "exp10c-rd-wrn50-in384"
#
#echo
#echo "================================================================"
#echo "DONE — check $OUT/ablation_master.csv for the new row(s)."
#echo
#echo "For step 11 / final 4-way fusion (when WinCLIP+ is also done):"
#echo "  uv run python score_fusion.py \\"
#echo "      --runs        <exp5_wrn50>/submission.csv \\"
#echo "                    <exp7_dinov2>/submission.csv \\"
#echo "                    <best_cutpaste>/submission.csv \\"
#echo "                    <exp10_rd>/submission.csv \\"
#echo "      --local-evals <exp5_wrn50>/local_eval.csv \\"
#echo "                    <exp7_dinov2>/local_eval.csv \\"
#echo "                    <best_cutpaste>/local_eval.csv \\"
#echo "                    <exp10_rd>/local_eval.csv \\"
#echo "      --weights local_ap \\"
#echo "      --out $OUT/runs/fusion_v3/submission.csv"
#echo "================================================================"

# ── EXPERIMENT 10d ──────────────────────────────────────────────────────────
# RD + DINOv2 ViT-S/14 @ input 518. Most expensive of the four but most
# likely to be the BEST RD result (mirrors why exp7 beat exp5).
echo "================================================================"
echo "EXPERIMENT 10d — RD @ DINOv2 ViT-S/14, input 518 (mul amap)"
echo "================================================================"

uv run python reverse_distillation_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vits14 \
    --feature-layers 3 6 9 11 \
    --input-size 518 \
    --amap-mode mul \
    --total-iters 2500 \
    --batch-size 8 \
    --num-workers 8 \
    --lr 0.005 \
    --beta1 0.5 \
    --beta2 0.999 \
    --weight-decay 0.0 \
    --score-batch-size 4 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp10d-rd-dinov2s14-in518"


# ── EXPERIMENT 10e ──────────────────────────────────────────────────────────
# RD + DINOv2 ViT-S/14 @ input 392. Cheaper ViT variant for A/B vs 10a.
echo "================================================================"
echo "EXPERIMENT 10e — RD @ DINOv2 ViT-S/14, input 392 (mul amap)"
echo "================================================================"

uv run python reverse_distillation_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vits14 \
    --feature-layers 3 6 9 11 \
    --input-size 392 \
    --amap-mode mul \
    --total-iters 2500 \
    --batch-size 16 \
    --num-workers 8 \
    --lr 0.005 \
    --beta1 0.5 \
    --beta2 0.999 \
    --weight-decay 0.0 \
    --score-batch-size 8 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --run-tag "exp10e-rd-dinov2s14-in392"
