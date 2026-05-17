uv run python patchcore_baseline_v2.py $COMMON \
    --multiview consensus-v3 --mv-alpha 0 --mv-beta 0 \
    --run-tag "cv3-B-only"

uv run python patchcore_baseline_v2.py $COMMON \
    --multiview consensus-v3 --mv-alpha 0.5 --mv-beta 0 \
    --run-tag "cv3-B+D"

uv run python patchcore_baseline_v2.py $COMMON \
    --multiview consensus-v3 --mv-alpha 0 --mv-beta 0.4 \
    --run-tag "cv3-B+A"

uv run python patchcore_baseline_v2.py $COMMON \
    --multiview consensus-v3 --mv-alpha 0.5 --mv-beta 0.4 \
    --run-tag "cv3-default"

uv run python patchcore_baseline_v2.py $COMMON \
    --multiview consensus-v3 --mv-alpha 0.7 --mv-beta 0.4 --good-keep-frac 0.5 \
    --run-tag "cv3-aggressive-D"

uv run python patchcore_baseline_v2.py $COMMON \
    --multiview consensus-v3 --mv-alpha 0.5 --mv-beta 0.7 --agreement-thresh 0.85 \
    --run-tag "cv3-high-A"