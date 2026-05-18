##!/usr/bin/env bash
## CFA sweep — Coupled-hypersphere Feature Adaptation.
##
## Three variants:
##   exp14   : WRN50-2 multi-scale (matches exp5 backbone), patch_size=1
##             (CFA paper default), 5% coreset, hvflip TTA.
##   exp14b  : WRN50-2 multi-scale, patch_size=3 (PatchCore-style local
##             pooling). A/B test: does the local aggregation help or hurt
##             CFA's contrastive training?
##   exp14c  : DINOv2 ViT-S/14 multi-scale (matches exp7), patch_size=1.
##             Higher-quality features but different scoring paradigm
##             than exp7's raw NN -> independent stacker signal.
##
## Total walltime estimate on a single NVIDIA L4 (24 GB):
##   exp14   ~55 min (8 classes)
##   exp14b  ~55 min
##   exp14c  ~70 min (DINOv2 forward is slower than WRN50 in fp16)
##   Sweep total: ~3 hours.
##
## Dependencies:
##   cfa_baseline.py and patchcore_baseline_v2.py in same directory
##   local_preds_saver.py in same directory
##
## Pre-flight (if you haven't already cached DINOv2 weights for exp14c):
##   python -c "import torch; \
##       torch.hub.load('facebookresearch/dinov2','dinov2_vits14', \
##                      trust_repo=True, source='github')"
#
#set -euo pipefail
#
#DATA=/work/u10813429/anomaly-detection/data
#OUT=/work/u10813429/anomaly-detection/baseline_out
#
### ── EXPERIMENT 14 — CFA paper default ────────────────────────────────────────
##echo "================================================================"
##echo "EXPERIMENT 14 — CFA @ WRN50-2 multi-scale, patch_size=1"
##echo "================================================================"
##
##uv run python cfa_baseline.py \
##    --data-root  "$DATA" \
##    --report-dir "$OUT" \
##    --backbone wide_resnet50_2 \
##    --feature-layers 1 2 3 \
##    --target-layer 2 \
##    --input-size 384 \
##    --patch-size 1 \
##    --coreset-frac 0.05 \
##    --coreset-algo minibatch \
##    --coreset-batch 128 \
##    --fit-batch-size 16 \
##    --hidden-dim 0 \
##    --total-iters 2500 \
##    --batch-size 16 \
##    --lr 1e-3 \
##    --weight-decay 1e-4 \
##    --k-att 3 \
##    --k-rep 3 \
##    --radius-sq 0.5 \
##    --alpha 0.5 \
##    --score-batch-size 16 \
##    --score-chunk 4096 \
##    --k-test 1 \
##    --smooth-sigma 1.5 \
##    --tta hvflip \
##    --num-workers 8 \
##    --seed 0 \
##    --run-tag "exp14-cfa-wrn50-p1"
##
### ── EXPERIMENT 14b — patch_size=3 ablation ──────────────────────────────────
##echo "================================================================"
##echo "EXPERIMENT 14b — CFA @ WRN50-2, patch_size=3 (PatchCore-style)"
##echo "================================================================"
##
##uv run python cfa_baseline.py \
##    --data-root  "$DATA" \
##    --report-dir "$OUT" \
##    --backbone wide_resnet50_2 \
##    --feature-layers 1 2 3 \
##    --target-layer 2 \
##    --input-size 384 \
##    --patch-size 3 \
##    --coreset-frac 0.05 \
##    --coreset-algo minibatch \
##    --coreset-batch 128 \
##    --fit-batch-size 16 \
##    --hidden-dim 0 \
##    --total-iters 2500 \
##    --batch-size 16 \
##    --lr 1e-3 \
##    --weight-decay 1e-4 \
##    --k-att 3 \
##    --k-rep 3 \
##    --radius-sq 0.5 \
##    --alpha 0.5 \
##    --score-batch-size 16 \
##    --score-chunk 4096 \
##    --k-test 1 \
##    --smooth-sigma 1.5 \
##    --tta hvflip \
##    --num-workers 8 \
##    --seed 0 \
##    --run-tag "exp14b-cfa-wrn50-p3"
#
## ── EXPERIMENT 14c — DINOv2 variant ─────────────────────────────────────────
#echo "================================================================"
#echo "EXPERIMENT 14c — CFA @ DINOv2 ViT-S/14, input 518"
#echo "================================================================"
#
#uv run python cfa_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --backbone dinov2_vits14 \
#    --feature-layers 3 6 9 11 \
#    --target-layer 3 \
#    --input-size 518 \
#    --patch-size 1 \
#    --coreset-frac 0.05 \
#    --coreset-algo minibatch \
#    --coreset-batch 128 \
#    --fit-batch-size 8 \
#    --hidden-dim 0 \
#    --total-iters 2500 \
#    --batch-size 8 \
#    --lr 1e-3 \
#    --weight-decay 1e-4 \
#    --k-att 3 \
#    --k-rep 3 \
#    --radius-sq 0.5 \
#    --alpha 0.5 \
#    --score-batch-size 4 \
#    --score-chunk 4096 \
#    --k-test 1 \
#    --smooth-sigma 1.5 \
#    --tta hvflip \
#    --num-workers 8 \
#    --seed 0 \
#    --run-tag "exp14c-cfa-dnv2s14-p1"
#
#echo
#echo "================================================================"
#echo "DONE — three rows appended to $OUT/ablation_master.csv"
#echo
#echo "To add the best CFA run to the XGBoost stacker, find the winning"
#echo "run directory and append it to COMMON_ARGS in stack.sh:"
#echo "    --runs        \$CFA_RUN/submission.csv"
#echo "    --local-preds \$CFA_RUN/local_predictions.npz"
#echo "================================================================"


#!/usr/bin/env bash
# CFA sweep — Coupled-hypersphere Feature Adaptation.
#
# Key knobs added in the speedup pass:
#   --mem-refresh-every K   refresh cached descriptor(memory) every K
#                           iters (default 10). Larger = faster but
#                           slightly staler targets.
#   --loss-chunk N          query patches per chunk of the d² matmul
#                           (default 8192). Lower if you OOM.
#   --no-amp-loss           disable bf16 matmul in cfa_loss.
#                           DON'T pass this unless debugging — it is
#                           the 10× speedup.
#
# Further speedup options if the L4 is still bottlenecking:
#   --coreset-frac 0.015    1.5% instead of 5%; for DINOv2 this drops
#                           M from ~178K to ~53K (3.3× faster matmul,
#                           same downstream AP within noise).
#   --mem-refresh-every 20  refresh half as often.
#
# Expected per-class walltime after the speedup pass (NVIDIA L4 24 GB):
#   exp14   ~5-7 min   (WRN50, M ~75K)
#   exp14b  ~5-7 min   (WRN50 patch=3)
#   exp14c  ~8-12 min  (DINOv2, M ~178K with coreset 0.05)
#       -> ~3-4 min  (DINOv2, M ~53K with coreset 0.015)

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

## ── EXPERIMENT 14 — CFA paper default ────────────────────────────────────────
#echo "================================================================"
#echo "EXPERIMENT 14 — CFA @ WRN50-2 multi-scale, patch_size=1"
#echo "================================================================"
#
#uv run python cfa_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --backbone wide_resnet50_2 \
#    --feature-layers 1 2 3 \
#    --target-layer 2 \
#    --input-size 384 \
#    --patch-size 1 \
#    --coreset-frac 0.05 \
#    --coreset-algo minibatch \
#    --coreset-batch 128 \
#    --fit-batch-size 16 \
#    --hidden-dim 0 \
#    --total-iters 2500 \
#    --batch-size 16 \
#    --lr 1e-3 \
#    --weight-decay 1e-4 \
#    --k-att 3 \
#    --k-rep 3 \
#    --radius-sq 0.5 \
#    --alpha 0.5 \
#    --loss-chunk 8192 \
#    --mem-refresh-every 10 \
#    --score-batch-size 16 \
#    --score-chunk 4096 \
#    --k-test 1 \
#    --smooth-sigma 1.5 \
#    --tta hvflip \
#    --num-workers 8 \
#    --seed 0 \
#    --run-tag "exp14-cfa-wrn50-p1"
#
## ── EXPERIMENT 14b — patch_size=3 ablation ──────────────────────────────────
#echo "================================================================"
#echo "EXPERIMENT 14b — CFA @ WRN50-2, patch_size=3 (PatchCore-style)"
#echo "================================================================"
#
#uv run python cfa_baseline.py \
#    --data-root  "$DATA" \
#    --report-dir "$OUT" \
#    --backbone wide_resnet50_2 \
#    --feature-layers 1 2 3 \
#    --target-layer 2 \
#    --input-size 384 \
#    --patch-size 3 \
#    --coreset-frac 0.05 \
#    --coreset-algo minibatch \
#    --coreset-batch 128 \
#    --fit-batch-size 16 \
#    --hidden-dim 0 \
#    --total-iters 2500 \
#    --batch-size 16 \
#    --lr 1e-3 \
#    --weight-decay 1e-4 \
#    --k-att 3 \
#    --k-rep 3 \
#    --radius-sq 0.5 \
#    --alpha 0.5 \
#    --loss-chunk 8192 \
#    --mem-refresh-every 10 \
#    --score-batch-size 16 \
#    --score-chunk 4096 \
#    --k-test 1 \
#    --smooth-sigma 1.5 \
#    --tta hvflip \
#    --num-workers 8 \
#    --seed 0 \
#    --run-tag "exp14b-cfa-wrn50-p3"

# ── EXPERIMENT 14c — DINOv2 variant ─────────────────────────────────────────
echo "================================================================"
echo "EXPERIMENT 14c — CFA @ DINOv2 ViT-S/14, input 518"
echo "================================================================"

uv run python cfa_baseline.py \
    --data-root  "$DATA" \
    --report-dir "$OUT" \
    --backbone dinov2_vits14 \
    --feature-layers 3 6 9 11 \
    --target-layer 3 \
    --input-size 518 \
    --patch-size 1 \
    --coreset-frac 0.025 \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --fit-batch-size 8 \
    --hidden-dim 0 \
    --total-iters 2500 \
    --batch-size 8 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --k-att 3 \
    --k-rep 3 \
    --radius-sq 0.5 \
    --alpha 0.5 \
    --loss-chunk 2048 \
    --mem-refresh-every 10 \
    --score-batch-size 4 \
    --score-chunk 4096 \
    --k-test 1 \
    --smooth-sigma 1.5 \
    --tta hvflip \
    --num-workers 8 \
    --seed 0 \
    --run-tag "exp14c-cfa-dnv2s14-p1"

echo
echo "================================================================"
echo "DONE — three rows appended to $OUT/ablation_master.csv"
echo
echo "To add the best CFA run to the XGBoost stacker, find the winning"
echo "run directory and append it to COMMON_ARGS in stack.sh:"
echo "    --runs        \$CFA_RUN/submission.csv"
echo "    --local-preds \$CFA_RUN/local_predictions.npz"
echo "================================================================"