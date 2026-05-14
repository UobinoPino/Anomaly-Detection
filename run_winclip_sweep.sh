#!/usr/bin/env bash
# Step 11 of the roadmap — WinCLIP+ (zero-/few-shot via CLIP ViT-B/16).
#
# Adds the FOURTH track to the score-fusion stack:
#   1. PatchCore WRN50-2 @ 384            (exp5)
#   2. PatchCore DINOv2 ViT-S/14 @ 518    (exp7)
#   3. CutPaste-NN @ ResNet-18 / -50      (exp8c / exp8d)
#   4. Reverse Distillation @ WRN50 256   (exp10b)
#   5. ─── WinCLIP+ @ CLIP ViT-B/16 ───   (exp11 — this script)
#
# Why this is a useful fifth ensemble member:
#   - All four existing tracks use vision-only feature distance. They
#     share systematic failure modes on small, semantically-distinct
#     defects (gear teeth fractures, capsule cracks, etc.) that look
#     "in-distribution" in CNN/ViT feature space.
#   - WinCLIP+ scores by text-image ALIGNMENT instead, using the
#     anomaly_descriptions.csv as a structured defect catalog. Its
#     error modes are orthogonal to NN/reconstruction approaches — the
#     property fusion-by-AP weighting will exploit on class_03 (gear)
#     and class_08 (capsule), where the descriptions are richest.
#
# One-time pre-flight (run on login node, with internet):
#     uv pip install "open_clip_torch>=2.20"
#     python -c "import open_clip; \
#         open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')"
# This caches the ~350 MB checkpoint at $HOME/.cache/clip so compute
# nodes can load it offline.
#
# Memory / speed on a single NVIDIA L4 (24 GB):
#   ViT-B/16 forward @ batch 32:  ~5 ms / image
#   Per class (~740 test imgs):   ~8 sec    (no TTA)
#   Full 8-class run:             ~1.5 min  + ~30 sec model load
#   With hflip TTA: ~2x  -> ~3.5 min total
#
# Dependencies:
#   winclip_baseline.py imports from patchcore_baseline_v2.py
#   (ImageRecord, scan_dataset, RLE helpers, append_to_ablation_master).
#   They must live in the same directory.

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out
CSV=/work/u10813429/anomaly-detection/data/anomaly_descriptions.csv

# ── EXPERIMENT 11 — WinCLIP+ main run ───────────────────────────────────────
# Paper-default knobs:
#   - CLIP ViT-B/16 (OpenAI weights, 224 input, 14x14 patch grid)
#   - Window sizes {2, 3} patches (32x32 and 48x48 px regions)
#   - alpha=0.5 (equal weight on zero-shot text scoring vs few-shot NN)
#   - K=8 reference normals from train/good (one per sample_id when possible)
#   - TTA off for speed; exp11c below adds hflip TTA
#echo "================================================================"
#echo "EXPERIMENT 11 — WinCLIP+ @ CLIP ViT-B/16 (windows 2+3, K=8, alpha=0.5)"
#echo "================================================================"
#
#uv run python winclip_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --csv        "$CSV" \
#    --clip-model ViT-B-16 \
#    --clip-pretrained openai \
#    --window-sizes 2 3 \
#    --alpha 0.5 \
#    --k-shot 8 \
#    --score-batch-size 16 \
#    --num-workers 4 \
#    --smooth-sigma 1.5 \
#    --tta none \
#    --run-tag "exp11-winclip-vitb16-k8"
#
## ── EXPERIMENT 11b (ablation) — zero-shot only, K=0 ─────────────────────────
## Useful as a diagnostic: tells you how much of the AP comes from text
## alignment alone vs the few-shot bank. If 11b >> 11, the bank is hurting;
## if 11 >> 11b, fusion will lean on visual reference matching too.
## Cheap to run (~30 sec for all 8 classes), so worth keeping in.
#echo "================================================================"
#echo "EXPERIMENT 11b — WinCLIP zero-shot ABLATION (K=0)"
#echo "================================================================"
#
#uv run python winclip_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --csv        "$CSV" \
#    --clip-model ViT-B-16 \
#    --clip-pretrained openai \
#    --window-sizes 2 3 \
#    --alpha 1.0 \
#    --k-shot 0 \
#    --score-batch-size 16 \
#    --num-workers 4 \
#    --smooth-sigma 1.5 \
#    --tta none \
#    --run-tag "exp11b-winclip-vitb16-zeroshot"

# ── EXPERIMENT 11c — WinCLIP+ with hflip TTA ────────────────────────────────
# hflip is the only direction-preserving TTA for most Spacepresso classes
# (objects aren't all rotation-invariant — screws/capsules have a clear
# axis). +0.5-1 AP typical for this kind of method.
echo "================================================================"
echo "EXPERIMENT 11c — WinCLIP+ @ CLIP ViT-B/16 + hflip TTA"
echo "================================================================"

uv run python winclip_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --csv        "$CSV" \
    --clip-model ViT-B-16 \
    --clip-pretrained openai \
    --window-sizes 2 3 \
    --alpha 0.5 \
    --k-shot 8 \
    --score-batch-size 16 \
    --num-workers 4 \
    --smooth-sigma 1.5 \
    --tta hflip \
    --run-tag "exp11c-winclip-vitb16-k8-hflip"

echo
echo "================================================================"
echo "DONE — three rows appended to $OUT/ablation_master.csv."
echo
echo "Pick the best WinCLIP+ run (likely 11c if hflip helps locally),"
echo "then run the 5-way fusion (replace <RUN_IDs> with actuals):"
echo
echo "  uv run python score_fusion.py \\"
echo "      --runs        \$OUT/runs/<exp5_wrn50>/submission.csv \\"
echo "                    \$OUT/runs/<exp7_dinov2>/submission.csv \\"
echo "                    \$OUT/runs/<exp8c_or_d_cutpaste>/submission.csv \\"
echo "                    \$OUT/runs/<exp10b_rd>/submission.csv \\"
echo "                    \$OUT/runs/<best_winclip>/submission.csv \\"
echo "      --local-evals \$OUT/runs/<exp5_wrn50>/local_eval.csv \\"
echo "                    \$OUT/runs/<exp7_dinov2>/local_eval.csv \\"
echo "                    \$OUT/runs/<exp8c_or_d_cutpaste>/local_eval.csv \\"
echo "                    \$OUT/runs/<exp10b_rd>/local_eval.csv \\"
echo "                    \$OUT/runs/<best_winclip>/local_eval.csv \\"
echo "      --weights local_ap \\"
echo "      --out \$OUT/runs/fusion_5way_localap/submission.csv"
echo "================================================================"