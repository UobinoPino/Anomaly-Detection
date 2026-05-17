##!/usr/bin/env bash
## UniAD sweep — runs two variants A/B-style:
##
##   exp12  : standard scoring (no multi-view)
##   exp12mv: same training, sibling-bank multi-view at inference
##
## Total walltime estimate on a single NVIDIA L4 (24 GB):
##   - Training per class:  ~6-8 min
##   - Standard inference:  ~25-35 s eval + 25-35 s test
##   - Multi-view inference: ~40-50 s eval + 40-50 s test
##   Full sweep (2 runs x 8 classes): ~140 min
##
## Add `--only-classes class_01` to either invocation if you just want
## to smoke-test the pipeline before committing the full run.
##
## Dependencies:
##   - uniad_baseline.py and patchcore_baseline_v2.py in same directory
##   - local_preds_saver.py in same directory
##   - torchvision >= 0.13 (for EfficientNet_B4_Weights)
##
## Pre-flight (on a node with internet, e.g. login node):
##   python -c "from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights; \
##               efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)"
## This caches the ~75 MB checkpoint under $TORCH_HOME/hub/checkpoints.
#
#set -euo pipefail
#
#DATA=/work/u10813429/anomaly-detection/data
#OUT=/work/u10813429/anomaly-detection/baseline_out
#
## Shared knobs (tuned for L4 at input 256).
#COMMON=(
#    --data-root "$DATA"
#    --report-dir "$OUT"
#    --input-size 256
#    --model-dim 256
#    --n-heads 8
#    --n-enc-layers 4
#    --n-dec-layers 4
#    --neighbour-radius 1
#    --jitter-sigma 0.0
#    --total-iters 2500
#    --batch-size 16
#    --lr 1e-4
#    --weight-decay 1e-4
#    --num-workers 8
#    --score-batch-size 32
#    --smooth-sigma 1.5
#    --tta hvflip
#    --seed 0
#)
#
## ── EXPERIMENT 12 ──────────────────────────────────────────────────────────
## Standard scoring (no multi-view). Establishes the per-class AP baseline
## for this architecture so the multi-view A/B is clean.
#echo "================================================================"
#echo "EXPERIMENT 12 — UniAD @ EfficientNet-B4, no multi-view"
#echo "================================================================"
#
#uv run python uniad_baseline.py \
#    "${COMMON[@]}" \
#    --multiview none \
#    --run-tag "exp12-uniad-effb4-noMV"
#
#
## ── EXPERIMENT 12MV ────────────────────────────────────────────────────────
## Same model spec, sibling-bank multi-view at inference. Boost weight
## alpha=0.5 starts mid-range; if local AP regresses on textured classes
## (coffee, pistachio) try alpha=0.3, and if it gains a lot on
## small-defect classes (gear, screw) try alpha=0.7.
#echo "================================================================"
#echo "EXPERIMENT 12MV — UniAD @ EfficientNet-B4, multi-view sibling-bank"
#echo "================================================================"
#
#uv run python uniad_baseline.py \
#    "${COMMON[@]}" \
#    --multiview sibling-bank \
#    --mv-alpha 0.5 \
#    --run-tag "exp12mv-uniad-effb4-MV"
#
#
## ── Add to stacker ─────────────────────────────────────────────────────────
#echo
#echo "================================================================"
#echo "DONE — new rows in $OUT/ablation_master.csv"
#echo
#echo "Once both runs complete and you've compared local AP (noMV vs MV),"
#echo "add the WINNING run's submission.csv + local_predictions.npz to"
#echo "run_xgb_stacker_full.sh in the COMMON_ARGS lists, then retrain"
#echo "the stacker. Keep the LOSER as a comparison row but don't bother"
#echo "stacking both -- they're correlated."
#echo "================================================================"

#!/usr/bin/env bash
# UniAD sweep — DINOv2 ViT-S/14 teacher with L2-normalised features.
#
# Architecture changes vs the failed exp12 (EffNet-B4, AP=0.1589):
#   • Teacher: dinov2_vits14 block 9 (proven on this dataset by PatchCore
#     exp7, AP=0.4624). Single block (not multi-layer fusion) so the
#     reconstruction target is clean.
#   • L2-normalise teacher features along channel dim. MSE on unit
#     vectors is bounded in [0, ~2] and channel-balanced — no more
#     domination by a handful of high-magnitude channels.
#   • Input 392 (28x28 = 784 tokens) instead of 256 (16x16 = 256
#     tokens). 3x finer spatial resolution for small defects.
#   • 5000 iters instead of 2500 — L2-normed loss converges from ~1.0
#     to ~0.05-0.10; halfway through 2500 it would still be at ~0.3.
#   • batch_size 8 (was 16) — DINOv2 forward + 784-token attention
#     uses more memory than EffNet-B4 + 256-token attention.
#
# Total walltime estimate on a single NVIDIA L4 (24 GB):
#   • Training per class:  ~7-9 min
#   • Standard inference:  ~25-40 s eval + 25-40 s test
#   • Multi-view inference: ~40-60 s eval + 40-60 s test
#   Full sweep (2 runs x 8 classes): ~170 min
#
# Add `--only-classes class_01` to either invocation to smoke-test the
# pipeline (~12 min for one class) before committing the full sweep.
#
# Dependencies:
#   • uniad_baseline.py and patchcore_baseline_v2.py in same directory
#   • local_preds_saver.py in same directory
#   • DINOv2 cached: pre-fetch on a node with internet
#       python -c "import torch; \
#           torch.hub.load('facebookresearch/dinov2','dinov2_vits14', \
#                          trust_repo=True, source='github')"
#     Caches ~85 MB to ~/.cache/torch/hub/. You likely already have it
#     from PatchCore exp7.

set -euo pipefail

DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

# Shared knobs (tuned for L4 at input 392, DINOv2 ViT-S/14).
COMMON=(
    --data-root "$DATA"
    --report-dir "$OUT"
    --backbone dinov2_vits14
    --block-idx 9
    --input-size 392
    --model-dim 256
    --n-heads 8
    --n-enc-layers 4
    --n-dec-layers 4
    --neighbour-radius 7
    --jitter-sigma 0.0
    --total-iters 5000
    --batch-size 8
    --lr 2e-4
    --weight-decay 1e-4
    --num-workers 8
    --score-batch-size 16
    --smooth-sigma 1.5
    --tta hvflip
    --seed 0
)

# ── EXPERIMENT 12 ──────────────────────────────────────────────────────────
# Standard scoring (no multi-view). This is the main run — it's the one
# that should be roughly competitive with PatchCore exp7 (AP=0.4624)
# and is the one to add to the stacker.
echo "================================================================"
echo "EXPERIMENT 12 — UniAD @ DINOv2 ViT-S/14 block 9, no multi-view"
echo "================================================================"

uv run python uniad_baseline.py \
    "${COMMON[@]}" \
    --multiview none \
    --run-tag "exp12-uniad-dnv2s14-noMV"


# ── EXPERIMENT 12MV ────────────────────────────────────────────────────────
# Sibling-bank multi-view at inference. WARNING: this path is known to
# underperform `none` on this dataset because the 5 views are different
# camera angles (top/side/bottom/...), not perturbations of the same
# scene. Cross-view feature similarity for "good" tokens is dominated
# by viewpoint geometry, not defect signal. Kept here only as an A/B
# comparison to confirm the regression is reproducible on UniAD as it
# was on PatchCore.
#
# If you only want the production run, comment this block out.
echo "================================================================"
echo "EXPERIMENT 12MV — UniAD @ DINOv2 ViT-S/14 block 9, sibling-bank MV"
echo "================================================================"

uv run python uniad_baseline.py \
    "${COMMON[@]}" \
    --multiview sibling-bank \
    --mv-alpha 0.5 \
    --run-tag "exp12mv-uniad-dnv2s14-MV"


# ── Done ───────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "DONE — new rows in $OUT/ablation_master.csv"
echo
echo "Expected results vs the failed EffNet-B4 run (exp12: AP=0.1589):"
echo "  exp12 noMV   should land around AP=0.35-0.45 (similar to"
echo "               PatchCore exp7=0.4624; UniAD is an independent"
echo "               signal so the stacker still gains even if it's"
echo "               slightly below)."
echo "  exp12mv MV   should land 0.01-0.05 below exp12 noMV. If it"
echo "               beats noMV, something is unusual about the data"
echo "               and you should run the consensus-v3-style ablation"
echo "               we did for PatchCore."
echo
echo "Next step (assuming exp12 noMV wins):"
echo "  • add its submission.csv + local_predictions.npz to"
echo "    run_xgb_stacker_full.sh in the COMMON_ARGS lists"
echo "  • retrain the stacker; UniAD's reconstruction signal is"
echo "    independent from PatchCore (distance) and CutPaste"
echo "    (classification), so the stacker should pick up"
echo "    complementary information"
echo "================================================================"