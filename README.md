# Spacepresso Anomaly Detection

Pixel-level anomaly detection on the Spacepresso dataset (8 industrial classes:
resistor, inductor, gear, screw, nut, coffee, pistachio, capsule). The pipeline
runs a portfolio of independent detectors and combines them with a stacker
tuned against the leaderboard metric — globally pooled pixel-AP.

```bash
uv sync
uv run spacepresso list                                          # what's available
uv run spacepresso run patchcore --backbone dinov2_vitb14_reg    # one detector
uv run spacepresso stack --runs baseline_out/runs/a baseline_out/runs/b
```

---

## Layout

```
src/spacepresso/
  core/          records, q8rle codec, metrics, imaging, submission, logging
  config.py      RuntimeConfig (where things live) + DetectorConfig (the science)
  data/          datasets, transforms, synthetic-anomaly generation
  backbones/     ResNet and DINOv2/v3, behind one interface
  detectors/     the 14 detectors, each just an algorithm
  postprocess/   TTA, per-view calibration, spatial filters
  stacking/      one configurable stacker
  runner/        the experiment loop and the CLI
  analysis/      spatial priors, pre-submission checks
configs/         YAML run configurations
tests/           unit, architecture and end-to-end tests
```

Dependencies point strictly inward:

```
core  <-  data, backbones  <-  detectors, postprocess  <-  stacking, runner
```

`core` imports nothing from the project and no detector imports another.
Those rules are enforced by `tests/unit/test_architecture.py`, not just
documented — along with "no `sys.path` hacks", "no hardcoded absolute paths",
and "no `print()` in library code".

---

## Detectors

| name | idea |
|---|---|
| `patchcore` | k-NN distance to a coreset of normal patch features |
| `efficientad` | student-teacher plus autoencoder, hard-pixel mined |
| `fastflow` | normalizing-flow density on backbone features |
| `reverse_distillation` | student decodes the teacher's features from a bottleneck |
| `cfa` | learned hypersphere adaptation of the feature space, then k-NN |
| `draem` | reconstruct a synthetically corrupted image, then discriminate |
| `cutpaste` | self-supervised pretext task, then nearest-neighbour scoring |
| `glass` | per-patch discriminator on local + gradient-ascent synthetic anomalies |
| `uniad` | transformer reconstruction with neighbour-masked attention |
| `dino_dpmm` | Dirichlet-process Gaussian mixture over DINO patch features |
| `anomalydino` | training-free k-NN on foreground-masked DINO features |
| `winclip` | CLIP text prompts + windowed few-shot reference matching |
| `textad` | segmentation U-Net on language-grounded synthetic defects |
| `transfusion` | transparency-conditioned diffusion |

Adding one means writing a `Detector` subclass and a config dataclass, then
adding a row to the registry in `detectors/__init__.py`. Everything else —
the per-class loop, evaluation, the submission, the run id, the CLI flags —
comes from the runner.

```python
class MyDetector(Detector[MyConfig]):
    name = "mine"
    config_type = MyConfig

    def fit(self, train_good): ...            # learn what normal looks like
    def score_batch(self, images): ...        # (B, C, H, W) -> (B, H, W)
```

A detector never sees ground-truth masks: the runner loads them and computes
the metric.

---

## Running

### One detector

```bash
uv run spacepresso run patchcore \
    --backbone dinov2_vitb14_reg \
    --feature-layers 3 6 9 11 \
    --input-size 392 \
    --coreset-frac 0.10 \
    --tta hvflip \
    --run-tag my-run
```

Or from a config file:

```bash
uv run spacepresso run --config configs/detectors/patchcore-dinov2.yaml
```

CLI flags override the config file, which overrides the environment, which
overrides the repo layout. Detector flags are derived from the config
dataclass, so `--help` is always in step with the code.

### The stacker

After two or more detectors have produced `local_predictions.npz` and
`submission.csv`:

```bash
uv run spacepresso stack \
    --config configs/stacking/xgboost.yaml \
    --runs baseline_out/runs/<a> baseline_out/runs/<b> baseline_out/runs/<c>
```

Estimator (`xgboost` or `logreg`), features, rank-normalisation scope,
cross-validation grouping and calibration are all configuration.

### Where things land

```
baseline_out/
  runs/<run_id>/
    run_log.txt              full log
    config.json              exact settings + the fingerprint they hash to
    local_eval.csv           per-(class, anomaly_type) pixel-AP
    local_predictions.npz    raw val scores + masks, for the stacker
    test_predictions.npz     raw test scores, for the stacker
    submission.csv / .zip    q8rle-encoded, ready to upload
  stacks/<run_id>/           the same, for a stacker run
  ablation_master.csv        one row per detector run
  stack_master.csv           one row per stacker run
```

### Paths

No path is hardcoded. Resolution order: `--data-root` / `--report-dir`, then
`$SPACEPRESSO_DATA_ROOT` / `$SPACEPRESSO_REPORT_DIR`, then `./data` and
`./baseline_out` relative to the repo, then the Kaggle mount when running in a
Kaggle notebook.

The dataset layout is
`data/class_XX/{train/good, train/anomaly_YY, ground_truth_train/anomaly_YY, test}/`.

---

## Two things worth knowing about the metric

**Pooled, not averaged.** The leaderboard pools every pixel of every image
into one ranking. A detector can win the per-image average and still lose the
leaderboard, because per-image AP is blind to whether image A's scores are
comparable with image B's. `core.metrics` names both so the distinction is
visible at the call site, and the runner reports both.

**Calibration is global.** `write_submission` maps scores to [0, 1] with one
percentile pair across every row, for the same reason. Per-image
normalisation would destroy exactly what is being measured.

---

## Development

```bash
uv run pytest tests -m "not slow"    # unit + architecture, a few seconds
uv run pytest tests                  # + end-to-end on a synthetic fixture
uv run ruff check src tests
uv run ruff format src tests
```

The end-to-end tests build a tiny synthetic dataset and run each detector with
randomly-initialised backbones, so they need no network and no GPU. They check
that the pipeline produces well-formed artifacts, not that the numbers are
good.

---

## Notes

- **Reproducibility.** A run's id is `<timestamp>_<slug>_<digest>`, where the
  digest hashes only the settings that change the result. Re-running the same
  experiment with different `--num-workers` or `--run-tag` gives the same
  digest; changing `--tta` does not.
- **Debugging one class.** `--only-classes class_03` on any detector.
- **Memory.** Each detector run fits in ~20 GB RAM and finishes in about an
  hour on one GPU. The stacker fits the same envelope.
- **DINOv3 weights** are gated. Set `DINOV3_REPO` and `DINOV3_WEIGHTS`, or
  install `transformers` and accept the licence on HuggingFace; the loader
  prints the exact steps if neither is available. DINOv2 needs nothing.
- **WinCLIP** needs `open_clip_torch`, which is not a default dependency.
