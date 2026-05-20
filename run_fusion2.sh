#!/usr/bin/env bash
# make_five_fusions.sh
# ────────────────────
# Generates five DIVERSE fusion submissions over the 14 detectors.
#
# Each strategy fails in a different way, so the resulting five
# submissions are complementary. Submit all five and keep the one with
# the best public-LB pixel-AP, or — if the LB lets you — pick the best
# 2-3 based on your local_eval.
#
#   1. rank_wmean — calibrated weighted mean (the safe baseline)
#   2. max        — aggressive any-flag-is-flag, with strong sparsity
#   3. median     — robust to a few catastrophic methods
#   4. agreement  — mean of top-K scoring methods per pixel
#   5. consensus  — geometric mean (rewards multi-method agreement)
#
# Tip: re-run with --multiview-pool soft and --multiview-regex set, once
# you know your dataset's view-suffix pattern (see commented section).

set -euo pipefail

ROOT=/work/u10813429/anomaly-detection
RUNS_DIR=$ROOT/baseline_out/runs
DATA=$ROOT/data
FUSION=$ROOT/score_fusion2.py
OUT_BASE=$RUNS_DIR
MASTER_CSV=$ROOT/baseline_out/ablation_master.csv

mkdir -p "$OUT_BASE"

# ─── 14 method paths (from your bash) ────────────────────────────────────────
EXP2=$RUNS_DIR/20260515-093627_wrn50_L23_T2_in224_cs05_mb64_exp2-coreset10_979f46
EXP3=$RUNS_DIR/20260515-094637_wrn50_L123_T2_in224_cs05_mb64_exp3-multiscale_3fcc18
EXP4=$RUNS_DIR/20260515-095705_wrn50_L123_T2_in224_cs05_mb64_tta-hvflip_exp4-tta-hvflip_6fa3df
EXP5=$RUNS_DIR/20260515-100850_wrn50_L123_T2_in384_cs05_mb128_tta-hvflip_exp5-input384-mb128_1fe0a2
EXP6=$RUNS_DIR/20260515-110430_wrn50_L123_T2_in384_cs05_mb128_tta-d4_exp6-d4tta-mb128_f3c3bb
EXP7=$RUNS_DIR/20260515-122231_dnv2s14_L3_6_9_11_T3_in518_cs05_mb128_tta-hvflip_exp7-dinov2s14-in518_d4576a
EXP8C=$RUNS_DIR/20260515-183145_cutpaste-patchcore_rn18_in256_it2500_bs64_pc_cs05mb128_tta-hvflip_exp8c-cutpaste-rn18-pcnn_d2a7c0
EXP8D=$RUNS_DIR/20260515-202731_cutpaste-patchcore_rn50_in256_it2500_bs32_pc_cs05mb128_tta-hvflip_exp8d-cutpaste-rn50-pcnn_e9a458
EXP12MV=$RUNS_DIR/20260518-185822_uniad_dnv2s14_b9_in392_d256_e4d4_nr7_it5000_bs8_noMV_tta-hvflip_exp12-uniad-dnv2s14-noMV_512a6d
EXP14C=$RUNS_DIR/20260517-162955_cfa_dnv2s14_L3_6_9_11_T3_in518_p1_cs02_it2500_bs8_ka3kr3_r0.5a0.5_kt1_tta-hvflip_exp14c-cfa-dnv2s14-p1_fb1373
EXP15C=$RUNS_DIR/20260518-163849_fastflow_dnv2s14_L3_6_9_11_in518_nb8_hr1_c2_it2500_bs8_tta-hvflip_exp15c-fastflow-dnv2s14-b8h1_bebc70
EFFAD=$RUNS_DIR/20260518-205727_effad_rn18_in256_it2500_bs16_noMV_tta-hvflip_effad_nomv_7437ce
TEXTAD=$RUNS_DIR/20260519-155930_textad_in256_b16_it1000_bs8_nd1-3_p0.70_tta-hvflip_baseline_90bd53
DRAEM=$RUNS_DIR/20260519-142933_draem_in256_b16_it2500_bs8_tta-hvflip_exp13-draem-in256-b32_addbf8

SUBS=(
  "$EXP2/submission.csv"   "$EXP3/submission.csv"   "$EXP4/submission.csv"
  "$EXP5/submission.csv"   "$EXP6/submission.csv"   "$EXP7/submission.csv"
  "$EXP8C/submission.csv"  "$EXP8D/submission.csv"  "$EXP12MV/submission.csv"
  "$EXP14C/submission.csv" "$EXP15C/submission.csv" "$EFFAD/submission.csv"
  "$TEXTAD/submission.csv" "$DRAEM/submission.csv"
)
EVALS=(
  "$EXP2/local_eval.csv"   "$EXP3/local_eval.csv"   "$EXP4/local_eval.csv"
  "$EXP5/local_eval.csv"   "$EXP6/local_eval.csv"   "$EXP7/local_eval.csv"
  "$EXP8C/local_eval.csv"  "$EXP8D/local_eval.csv"  "$EXP12MV/local_eval.csv"
  "$EXP14C/local_eval.csv" "$EXP15C/local_eval.csv" "$EFFAD/local_eval.csv"
  "$TEXTAD/local_eval.csv" "$DRAEM/local_eval.csv"
)

# ─── Optional multi-view pooling (Advice 12/05) ──────────────────────────────
# The Spacepresso dataset has 5 views per object. If your IDs encode the
# view (e.g.  "class_5_obj0042_view2"), set MV_REGEX to a regex with ONE
# capture group that extracts the object_id without the view suffix, and
# set MV_POOL to "soft" (50% per-view + 50% across-view mean — usually
# best for pixel-AP).
#
# Examples:
#   MV_REGEX='^(.+)_view[0-9]+$'   # for "..._view0", "..._view1"
#   MV_REGEX='^(.+)_v[0-9]+$'       # for "..._v0", "..._v1"
#   MV_REGEX='^(.+)_[0-4]$'         # for "...0", "...1" ... "...4"
#
# Leave empty to disable.
MV_REGEX=''
MV_POOL='none'   # none | mean | max | soft

# ─── Fusion launcher ─────────────────────────────────────────────────────────
run_fusion () {
  local name=$1; shift
  local out_dir=$OUT_BASE/fusion_v3_${name}
  echo
  echo "════════════════════════════════════════════════════════════════════════"
  echo "  FUSION [$name]  →  $out_dir"
  echo "════════════════════════════════════════════════════════════════════════"
  # Pass MV args only if regex is set, so the python script ignores them
  local mv_args=()
  if [[ -n "$MV_REGEX" && "$MV_POOL" != "none" ]]; then
    mv_args=( --multiview-regex "$MV_REGEX" --multiview-pool "$MV_POOL" )
  fi
  python "$FUSION" \
    --runs "${SUBS[@]}" \
    --local-evals "${EVALS[@]}" \
    --weights local_ap \
    --data-root "$DATA" \
    --master-csv "$MASTER_CSV" \
    --out "$out_dir/submission.csv" \
    --run-tag "fusion_v3_${name}" \
    "${mv_args[@]}" \
    "$@"
}

# ─── 1. rank_wmean ───────────────────────────────────────────────────────────
# Calibrated weighted mean. Smoothest of the five.
# Light background suppression — keep the bottom 30% at 0 to clean up
# constant-low background noise without hurting any true positives.
run_fusion rank_wmean \
  --strategy rank_wmean \
  --rank-normalise \
  --suppress-below 30

# ─── 2. max ──────────────────────────────────────────────────────────────────
# Aggressive: any TRUSTED method's flag is the fused flag.
# Top-7 methods only (per class) to keep the worst detectors out of the
# max. Strong sparsity post-processing to silence the small ghosts.
run_fusion max \
  --strategy max \
  --rank-normalise \
  --top-k-methods-per-class 7 \
  --suppress-below 85 \
  --remove-small-cc 8

# ─── 3. median ───────────────────────────────────────────────────────────────
# Robust to up to ⌊M/2⌋ catastrophically wrong methods. Conservative,
# tends to high precision. Moderate background suppression.
run_fusion median \
  --strategy median \
  --rank-normalise \
  --suppress-below 50

# ─── 4. agreement ────────────────────────────────────────────────────────────
# Mean of top-K=⌈0.45*M⌉=7 scoring methods per pixel. A "soft consensus":
# pixel needs ~half the methods to agree it's anomalous to score high,
# without requiring all of them.
run_fusion agreement \
  --strategy agreement \
  --rank-normalise \
  --top-k-frac 0.45 \
  --suppress-below 65 \
  --remove-small-cc 5

# ─── 5. consensus ────────────────────────────────────────────────────────────
# Weighted geometric mean. Near-zero where ANY trusted method disagrees,
# so very high precision but can miss complementary detections — that's
# its job (different failure mode from max). Restrict to top-9 methods
# so a weak detector can't tank the geomean.
run_fusion consensus \
  --strategy consensus \
  --rank-normalise \
  --top-k-methods-per-class 9 \
  --suppress-below 20

echo
echo "════════════════════════════════════════════════════════════════════════"
echo "  ALL FIVE FUSIONS COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo
echo "Submission zips:"
for name in rank_wmean max median agreement consensus; do
  echo "  $OUT_BASE/fusion_v3_${name}/submission.zip"
done
echo
echo "Next steps:"
echo "  1. Run your local evaluator on each fusion's submission.csv."
echo "  2. Submit the best 1-3 to the public LB."
echo "  3. If your IDs encode the 5 views per object, set MV_REGEX +"
echo "     MV_POOL='soft' at the top of this script and rerun — that"
echo "     usually adds another point or two of pixel-AP."