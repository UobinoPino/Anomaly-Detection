#!/usr/bin/env bash
# UniAD sweep — DINOv2 ViT-B/14 (with registers) teacher, L2-normed features.
#
# Architecture changes vs the failed exp12 (EffNet-B4, AP=0.1589):
#   • Teacher: dinov2_vitb14_reg block 9 — same backbone used by the
#     dnv2reg PatchCore/CutPaste/EffAD/RD sweep, so all ViT-based
#     methods share representations and the stacker sees one consistent
#     feature family across heads.
#   • L2-normalise teacher features along channel dim. MSE on unit
#     vectors is bounded in [0, ~2] and channel-balanced.
#   • Input 392 (28x28 = 784 tokens). 3x finer spatial resolution
#     than the EffNet-B4 16x16 baseline.
#   • 5000 iters: L2-normed loss converges from ~1.0 to ~0.05-0.10.
#   • batch_size 4 (was 8 for ViT-S): ViT-B channel dim is 768 vs
#     384, attention activations scale linearly, and the UniAD
#     decoder cross-attends back into 784 tokens. Bump to 8 if you
#     see GPU memory headroom; drop to 2 if OOM.
#
# Total walltime estimate on a single NVIDIA L4 (24 GB):
#   • Training per class:  ~12-15 min (ViT-B fwd is ~2x ViT-S)
#   • Standard inference:  ~45-60 s eval + 45-60 s test
#   • Multi-view inference: ~70-90 s eval + 70-90 s test
#   Full sweep (just exp12 noMV x 8 classes): ~2 h
#
# Add `--only-classes class_01` to smoke-test (~18 min for one class).
#
# Dependencies:
#   • uniad_baseline.py and patchcore_baseline_v2.py in same directory
#   • local_preds_saver.py in same directory
#   • DINOv2 cached: pre-fetch on a node with internet
#       python -c "import torch; \
#           torch.hub.load('facebookresearch/dinov2','dinov2_vitb14_reg', \
#                          trust_repo=True, source='github')"
#     Caches ~330 MB to ~/.cache/torch/hub/. You likely already have it
#     from the dnv2reg PatchCore retrain sweep.

set -euo pipefail

DATA=/workspace/anomaly-detection/data
OUT=/workspace/anomaly-detection/baseline_out

# Shared knobs (tuned for L4 at input 392, DINOv2 ViT-B/14 reg).
COMMON=(
    --data-root "$DATA"
    --report-dir "$OUT"
    --backbone dinov2_vitb14_reg
    --block-idx 9
    --input-size 392
    --model-dim 256
    --n-heads 8
    --n-enc-layers 4
    --n-dec-layers 4
    --neighbour-radius 7
    --jitter-sigma 0.0
    --total-iters 5000
    --batch-size 4
    --lr 2e-4
    --weight-decay 1e-4
    --num-workers 8
    --score-batch-size 8
    --smooth-sigma 1.5
    --tta hvflip
    --seed 0
)

# ── EXPERIMENT 12 ──────────────────────────────────────────────────────────
# Standard scoring (no multi-view). Main run — the one to compare
# against PatchCore exp7/equivalent and feed into the stacker.
echo "================================================================"
echo "EXPERIMENT 12 — UniAD @ DINOv2 ViT-B/14 reg block 9, no multi-view"
echo "================================================================"

uv run python uniad_baseline.py \
    "${COMMON[@]}" \
    --multiview none \
    --run-tag "exp12-uniad-dnv2b14reg-noMV"


# ── EXPERIMENT 12MV ────────────────────────────────────────────────────────
# Sibling-bank multi-view at inference. Disabled by default — same
# rationale as before: different camera angles, cross-view "good"
# similarity dominated by viewpoint geometry, not defect signal.
#echo "================================================================"
#echo "EXPERIMENT 12MV — UniAD @ DINOv2 ViT-B/14 reg block 9, sibling-bank MV"
#echo "================================================================"
#
#uv run python uniad_baseline.py \
#    "${COMMON[@]}" \
#    --multiview sibling-bank \
#    --mv-alpha 0.5 \
#    --run-tag "exp12mv-uniad-dnv2b14reg-MV"


echo
echo "================================================================"
echo "DONE — new rows in $OUT/ablation_master.csv"
echo
echo "Expected vs EffNet-B4 exp12 (AP=0.1589) and the ViT-S/14 variant:"
echo "  exp12 noMV   should land around AP=0.40-0.50. ViT-B with"
echo "               registers is the stronger backbone family used"
echo "               by the dnv2reg sweep, so it should at least"
echo "               match ViT-S and is independent from PatchCore"
echo "               (distance) and CutPaste (classification)."
echo
echo "Next step (assuming exp12 noMV wins):"
echo "  • add its submission.csv + local_predictions.npz to the"
echo "    XGBoost stacker COMMON_ARGS lists"
echo "  • retrain the stacker; UniAD's reconstruction signal is"
echo "    independent from the other ViT-B/14-reg heads"
echo "================================================================"