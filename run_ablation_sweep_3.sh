#!/usr/bin/env bash
# Ablation sweep #3 — Step 7 of the roadmap: DINOv2 ViT-S/14 backbone swap.
#
# Same recipe as exp5 (multi-scale, coreset 5% fp16, hvflip TTA, fp16
# memory bank), only the backbone changes.
#
# Note: d4 TTA was found to perform WORSE than hvflip on this dataset,
# so step 6 is skipped — we keep --tta hvflip.
#
# DINOv2 specifics:
#   - Input must be a multiple of 14. We use 518 (37x37 = 1369 tokens),
#     DINOv2's native pretraining resolution. The exp5-equivalent 392
#     (28x28 = 784 tokens) is provided commented out below for an
#     apples-to-apples spatial-grid comparison.
#   - --feature-layers indexes transformer blocks (0..11 for ViT-S/14).
#     We use [3, 6, 9, 11] — early / mid-early / mid-late / final —
#     matching the "multi-scale" spirit of WRN50 layers [1, 2, 3].
#   - fused feature dim = 4 * 384 = 1536  (vs WRN50 [1,2,3] = 1792).
#   - --no-save-banks keeps disk usage low (we only need submission.csv
#     and local_eval.csv for score fusion at step 9 — banks are useless
#     post-training).
#
# Pre-flight (run ONCE on a node with internet, e.g. the login node):
#   python -c "import torch; \
#       torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', \
#                       trust_repo=True)"
# This caches the ~88 MB checkpoint at $TORCH_HOME/hub/checkpoints so
# the compute node can read it offline. If TORCH_HOME isn't set, the
# default is ~/.cache/torch.
#
# Optional speed boost (~1.5-2x faster ViT inference):
#   uv pip install xformers
# DINOv2 auto-detects xformers; falls back to standard attention if absent.
#
# Estimated walltime on a single NVIDIA L4 (24 GB):
#   exp7 @ input 518 + hvflip TTA:  ~70-90 min for all 8 classes
#   (DINOv2 ViT-S/14 forward is ~3x faster than WRN50, but TTA triples it)

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# ── EXPERIMENT 7 ────────────────────────────────────────────────────────────
# DINOv2 ViT-S/14, multi-scale (blocks 3+6+9+11), input 518, hvflip TTA.
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