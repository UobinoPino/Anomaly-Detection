set -euo pipefail

# ─── Paths (edit ROOT to match your env) ─────────────────────────────────────
ROOT=${ROOT:-/workspace/anomaly-detection}
DATA=$ROOT/data
OUT=$ROOT/baseline_out

cd "$ROOT"

# ─── Profile + backbone selection ────────────────────────────────────────────
PROFILE=${BACKBONE_PROFILE:-dnv2reg}

if [[ "$PROFILE" == "dnv2reg" ]]; then
    PC_BACKBONE=dinov2_vitb14_reg
    PC_LAYERS="3 6 9 11"      # mid + late blocks (ViT-B has 12 blocks like S)
    PC_INPUT=518               # multiple of 14
    PC_INPUT_LO=392
    CUTPASTE_BACKBONE=dinov2_vitb14_reg
    CUTPASTE_INPUT=392
    EFFAD_TEACHER=dinov2_vitb14_reg
    EFFAD_TLAYER=9
    EFFAD_INPUT=392
    RD_TEACHER=dinov2_vitb14_reg
    RD_TLAYER=9
    RD_INPUT=392
elif [[ "$PROFILE" == "dnv3" ]]; then
    PC_BACKBONE=dinov3_vitb16
    PC_LAYERS="3 6 9 11"      # vitb16 has 12 blocks (indices 0..11)
    PC_INPUT=512               # multiple of 16
    PC_INPUT_LO=384
    CUTPASTE_BACKBONE=dinov3_vits16
    CUTPASTE_INPUT=384
    EFFAD_TEACHER=dinov3_vitb16
    EFFAD_TLAYER=11
    EFFAD_INPUT=384
    RD_TEACHER=dinov3_vitb16
    RD_TLAYER=11
    RD_INPUT=384
else
    echo "unknown BACKBONE_PROFILE=$PROFILE (use dnv2reg or dnv3)"
    exit 1
fi

echo "================================================================"
echo "A-DINOv3/v2-reg retrain sweep  PROFILE=$PROFILE"
echo "  PatchCore   : $PC_BACKBONE  layers=$PC_LAYERS  in=$PC_INPUT"
echo "  CutPaste    : $CUTPASTE_BACKBONE  in=$CUTPASTE_INPUT"
echo "  EfficientAD : teacher=$EFFAD_TEACHER  L$EFFAD_TLAYER  in=$EFFAD_INPUT"
echo "  RD          : teacher=$RD_TEACHER  L$RD_TLAYER  in=$RD_INPUT"
echo "================================================================"

# ─── Pre-flight: DINOv3 access (only when profile = dnv3) ───────────────────
if [[ "$PROFILE" == "dnv3" ]]; then
    HAVE_REPO=0
    [[ -d "${DINOV3_REPO:-/nonexistent}/hubconf.py" || -f "${DINOV3_REPO:-/nonexistent}/hubconf.py" ]] && HAVE_REPO=1
    HAVE_WEIGHTS=0
    if [[ -n "${DINOV3_WEIGHTS:-}" && -d "$DINOV3_WEIGHTS" ]]; then
        HAVE_WEIGHTS=$(ls "$DINOV3_WEIGHTS" 2>/dev/null | grep -c dinov3 || true)
    fi
    if [[ $HAVE_REPO -eq 0 && $HAVE_WEIGHTS -eq 0 ]]; then
        echo "[warn] DINOv3 torch.hub path not configured."
        echo "       Loader will fall back to HuggingFace transformers."
        echo "       Make sure 'huggingface-cli login' worked and"
        echo "       you accepted the dinov3 license on HF Hub."
    fi
fi



# ─── PatchCore retrains (exp2..exp6 equivalents) ────────────────────────────
echo
echo "================================================================"
echo "PATCHCORE RETRAINS (replacing exp2..exp6)"
echo "================================================================"

# exp2 equivalent — minimal layers, low input (debug-fast)
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$PC_BACKBONE" \
    --feature-layers 9 11 --target-layer 9 \
    --input-size "$PC_INPUT_LO" \
    --coreset-frac 0.05 --coreset-fp16 --memory-dtype fp16 \
    --coreset-algo minibatch --coreset-batch 128 \
    --batch-size 16 --score-batch-size 8 \
    --tta none \
    --num-workers 8 \
    --run-tag "retrain-exp2-vit-$PROFILE"

# exp3 equivalent — multi-scale, low input, no TTA
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$PC_BACKBONE" \
    --feature-layers $PC_LAYERS --target-layer 3 \
    --input-size "$PC_INPUT_LO" \
    --coreset-frac 0.05 --coreset-fp16 --memory-dtype fp16 \
    --coreset-algo minibatch --coreset-batch 128 \
    --batch-size 16 --score-batch-size 8 \
    --tta none \
    --num-workers 8 \
    --run-tag "retrain-exp3-vit-$PROFILE"

# exp4 equivalent — multi-scale + hvflip TTA
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$PC_BACKBONE" \
    --feature-layers $PC_LAYERS --target-layer 3 \
    --input-size "$PC_INPUT_LO" \
    --coreset-frac 0.05 --coreset-fp16 --memory-dtype fp16 \
    --coreset-algo minibatch --coreset-batch 128 \
    --batch-size 16 --score-batch-size 8 \
    --tta hvflip \
    --num-workers 8 \
    --run-tag "retrain-exp4-vit-$PROFILE"

# exp5 equivalent — high input + hvflip TTA
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$PC_BACKBONE" \
    --feature-layers $PC_LAYERS --target-layer 3 \
    --input-size "$PC_INPUT" \
    --coreset-frac 0.05 --coreset-fp16 --memory-dtype fp16 \
    --coreset-algo minibatch --coreset-batch 128 \
    --batch-size 8 --score-batch-size 4 \
    --tta hvflip \
    --num-workers 8 \
    --run-tag "retrain-exp5-vit-$PROFILE"

# exp6 equivalent — high input + d4 TTA
uv run python patchcore_baseline_v2.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$PC_BACKBONE" \
    --feature-layers $PC_LAYERS --target-layer 3 \
    --input-size "$PC_INPUT" \
    --coreset-frac 0.05 --coreset-fp16 --memory-dtype fp16 \
    --coreset-algo minibatch --coreset-batch 128 \
    --batch-size 8 --score-batch-size 4 \
    --tta d4 \
    --num-workers 8 \
    --run-tag "retrain-exp6-vit-$PROFILE"

# ─── CutPaste retrains (exp8c/d equivalents) ────────────────────────────────
echo
echo "================================================================"
echo "CUTPASTE RETRAINS (replacing exp8c, exp8d) — ViT frozen + head"
echo "================================================================"

# exp8c equivalent — base ViT, AdamW on the projection head
uv run python cutpaste_baseline.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --backbone "$CUTPASTE_BACKBONE" \
    --input-size "$CUTPASTE_INPUT" \
    --feature-layers 3 6 9 11 --target-layer 9 \
    --total-iters 2500 --batch-size 16 --num-workers 8 \
    --coreset-frac 0.05 --coreset-algo minibatch --coreset-batch 128 \
    --memory-dtype fp16 \
    --score-chunk 4096 --memory-chunk 16384 \
    --score-batch-size 8 \
    --smooth-sigma 1.5 --tta hvflip \
    --run-tag "retrain-exp8c-vit-$PROFILE"

# exp8d equivalent — same ViT-B as exp8c but with smaller batch.
# NOTE: in the dnv2reg profile this is now redundant with exp8c (both
# are vitb14_reg). For a real step-up replace the line below with
#   BIG_VIT=dinov2_vitl14_reg; BIG_IN=518
# and change --feature-layers and --target-layer in the next call to
#   --feature-layers 6 12 18 23 --target-layer 18
# Or simply comment out this entire exp8d block.
#if [[ "$PROFILE" == "dnv3" ]]; then
#    BIG_VIT=dinov3_vitb16; BIG_IN=384
#else
#    BIG_VIT=dinov2_vitb14_reg; BIG_IN=392
#fi
#uv run python cutpaste_baseline.py \
#    --data-root  "$DATA" --report-dir "$OUT" \
#    --backbone "$BIG_VIT" \
#    --input-size "$BIG_IN" \
#    --feature-layers 3 6 9 11 --target-layer 9 \
#    --total-iters 2500 --batch-size 8 --num-workers 8 \
#    --coreset-frac 0.05 --coreset-algo minibatch --coreset-batch 128 \
#    --memory-dtype fp16 \
#    --score-chunk 4096 --memory-chunk 16384 \
#    --score-batch-size 4 \
#    --smooth-sigma 1.5 --tta hvflip \
#    --run-tag "retrain-exp8d-vit-$PROFILE"

# ─── EfficientAD retrain (replaces effad_nomv) ──────────────────────────────
echo
echo "================================================================"
echo "EFFICIENTAD RETRAIN (ViT teacher)"
echo "================================================================"
uv run python efficientad_baseline.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --teacher-backbone "$EFFAD_TEACHER" \
    --teacher-layer "$EFFAD_TLAYER" \
    --input-size "$EFFAD_INPUT" \
    --total-iters 5000 \
    --batch-size 8 --score-batch-size 16 \
    --tta hvflip \
    --multiview none \
    --num-workers 8 \
    --run-tag "retrain-effad-vit-$PROFILE"

# ─── Reverse Distillation (NEW track with ViT teacher) ──────────────────────
echo
echo "================================================================"
echo "REVERSE DISTILLATION (new ViT teacher track)"
echo "================================================================"
uv run python reverse_distillation_baseline.py \
    --data-root  "$DATA" --report-dir "$OUT" \
    --teacher-backbone "$RD_TEACHER" \
    --teacher-layer "$RD_TLAYER" \
    --input-size "$RD_INPUT" \
    --amap-mode mul \
    --total-iters 2500 --batch-size 8 \
    --lr 5e-4 --beta1 0.5 --beta2 0.999 \
    --score-batch-size 8 \
    --tta hvflip \
    --num-workers 8 \
    --run-tag "rd-vit-$PROFILE"

# ─── Optional: re-run your DINOv2-S winners with DINOv3-L for A/B ───────────
if [[ "${ALSO_RERUN_DNV2_WINNERS:-0}" == "1" && "$PROFILE" == "dnv3" ]]; then
    echo
    echo "================================================================"
    echo "OPTIONAL: rerunning DINOv2-S winners with DINOv3-L for A/B"
    echo "================================================================"

    # PatchCore exp7 with dnv3-L
    uv run python patchcore_baseline_v2.py \
        --data-root  "$DATA" --report-dir "$OUT" \
        --backbone dinov3_vitl16 \
        --feature-layers 5 11 17 23 --target-layer 11 \
        --input-size 512 \
        --coreset-frac 0.05 --coreset-fp16 --memory-dtype fp16 \
        --coreset-algo minibatch --coreset-batch 128 \
        --batch-size 4 --score-batch-size 2 \
        --tta hvflip \
        --num-workers 8 \
        --run-tag "exp7-rerun-dnv3l16"
fi

echo
echo "================================================================"
echo "ALL RETRAINS DONE."
echo
echo "Next: update stacker_v6.sh — replace EXP2..EXP6, EXP8C, EXP8D, and"
echo "EFFAD with the new run_dirs (printed at the top of each run_log),"
echo "and add the new RD run_dir as a 15th method. Then run:"
echo "    ./stacker_v6.sh"
echo "================================================================"

