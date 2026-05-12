DATA=/work/u10813429/anomaly-detection/data
OUT=/work/u10813429/anomaly-detection/baseline_out

COMMON_ARGS=(
    --data-root "$DATA"
    --report-dir "$OUT"
    --only-classes class_01

    --backbone wide_resnet50_2
    --feature-layers 1 2 3
    --target-layer 2

    --input-size 384
    --coreset-frac 0.05
    --coreset-fp16
    --memory-dtype fp16

    --batch-size 16
    --score-batch-size 8
    --score-chunk 4096
    --memory-chunk 16384

    --knn-k 9
    --smooth-sigma 1.5
    --tta hvflip

    --aggressive-cleanup
)

echo "================================================================"
echo "SANITY — exact greedy coreset"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    "${COMMON_ARGS[@]}" \
    --coreset-algo exact \
    --run-tag sanity-exact


echo "================================================================"
echo "SANITY — minibatch coreset (128)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    "${COMMON_ARGS[@]}" \
    --coreset-algo minibatch \
    --coreset-batch 128 \
    --run-tag sanity-mb128


echo "================================================================"
echo "SANITY — minibatch coreset (64)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    "${COMMON_ARGS[@]}" \
    --coreset-algo minibatch \
    --coreset-batch 64 \
    --run-tag sanity-mb64


echo "================================================================"
echo "SANITY — minibatch coreset (32)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    "${COMMON_ARGS[@]}" \
    --coreset-algo minibatch \
    --coreset-batch 32 \
    --run-tag sanity-mb32


echo "================================================================"
echo "SANITY — minibatch coreset (16)"
echo "================================================================"

uv run python patchcore_baseline_v2.py \
    "${COMMON_ARGS[@]}" \
    --coreset-algo minibatch \
    --coreset-batch 16 \
    --run-tag sanity-mb16