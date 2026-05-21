# Spacepresso Anomaly Detection

Pixel-level anomaly detection on the Spacepresso dataset (8 industrial classes: resistor, inductor, gear, screw, nut, coffee, pistachio, capsule). The pipeline runs a portfolio of independent anomaly detectors and combines them with a GPU-accelerated XGBoost stacker tuned against the leaderboard metric (global pooled pixel-AP).

## What's in here

A collection of training-free and trainable anomaly detection baselines:

- **PatchCore v8** (`patchcore_baseline_v2.py`) — k-NN coreset on ResNet / DINOv2 / DINOv3 features
- **CutPaste v5** (`cutpaste_baseline.py`) — self-supervised classifier + PatchCore-NN scoring
- **FastFlow** (`fastflow_baseline.py`) — normalizing flow density on backbone features
- **EfficientAD v2** (`efficientad_baseline.py`) — student-teacher + autoencoder
- **Reverse Distillation v2** (`reverse_distillation_baseline.py`) — teacher-student cosine
- **DINO-DPMM** (`dino_dpmm_baseline.py`) — Dirichlet Process GMM on DINO features
- **AnomalyDINO**, **UniAD**, **DRAEM**, **CFA**, **TextAD** — additional tracks
- **XGBoost stacker v7** (`xgboost_stacker_v8.py`) — LB-aligned ensemble with isotonic calibration and global rank-norm

Every model produces a `submission.csv` (per-pixel anomaly scores in q8rle encoding) plus a `local_predictions.npz` (per-pixel scores + GT masks on local validation) that the stacker consumes.

## Resource budget

Each individual baseline run finishes within **~1 hour** on a single GPU and uses **at most ~20 GB of RAM**. The stacker fits in the same envelope. Every run writes a complete log to `<run_dir>/run_log.txt` so you can audit training curves, per-class AP, and timing post-hoc.

## Output layout

All artifacts land under `baseline_out/runs/<run_id>/`:

```
baseline_out/runs/<run_id>/
├── run_log.txt              # full stdout from the run
├── config.json              # exact hyperparameters
├── local_eval.csv           # per-(class, anomaly_type) pixel-AP
├── local_predictions.npz    # raw score maps + GT for the stacker
├── submission.csv           # per-pixel scores, q8rle encoded
└── submission.zip           # zipped CSV, ready to upload
```

Offline, every run saves both the CSV and its zipped version so you can submit directly without re-zipping. A master `baseline_out/ablation_master.csv` is appended on each run for cross-experiment tracking.

---

## Running locally with uv

[uv](https://docs.astral.sh/uv/) handles the Python environment and dependencies.

### One-time setup

```bash
git clone <this-repo>
cd anomaly-detection
uv sync              # installs the locked environment from pyproject.toml / uv.lock
```

Place the dataset at `./data/` (or pass `--data-root` to override). The expected layout is `data/class_XX/{train/good, train/anomaly_YY, ground_truth_train/anomaly_YY, test}/`.

### Running a single baseline

Each baseline is a self-contained CLI. Example: PatchCore with DINOv2-reg backbone:

```bash
uv run python patchcore_baseline_v2.py \
    --data-root ./data \
    --report-dir ./baseline_out \
    --backbone dinov2_vits14_reg \
    --feature-layers 3 6 9 11 \
    --target-layer 3 \
    --input-size 392 \
    --tta hvflip \
    --run-tag my-patchcore-run
```

The ready-made shell scripts (`run_efficientad.sh`, `run_adino_dpmm.sh`, etc.) wrap typical configurations. Inspect and run them directly:

```bash
bash run_efficientad.sh
```

### Running the stacker

After ≥2 baselines have produced `submission.csv` + `local_predictions.npz`, stack them. Edit `stacker_v8.sh` to point at your run directories, then:

```bash
bash stacker_v8.sh
```

The stacker handles GPU detection automatically (`--gpu auto`) and falls back to CPU if CUDA isn't available. The final fused `submission.zip` is ready to upload to the leaderboard.

---

## Running on Kaggle

To run end-to-end inside a Kaggle notebook:

1. **Create a new Kaggle notebook**, attach the Spacepresso dataset as input (it will be mounted at `/kaggle/input/spacepresso/`), and enable GPU (Settings → Accelerator → GPU T4/P100).

2. **Clone this repo into the notebook's workspace**:
   ```python
   !git clone <this-repo-url> /kaggle/working/repo
   %cd /kaggle/working/repo
   ```

3. **Install dependencies**. Kaggle's base image ships with most of what's needed; install the rest with pip (uv isn't preinstalled, and pip is fine for one-shot use):
   ```python
   !pip install -q xgboost optuna scikit-learn
   ```

4. **Point the scripts at the Kaggle paths**:
   ```python
   !python patchcore_baseline_v2.py \
       --data-root /kaggle/input/spacepresso/data \
       --report-dir /kaggle/working/baseline_out \
       --backbone dinov2_vits14_reg \
       --feature-layers 3 6 9 11 --target-layer 3 \
       --input-size 392 --tta hvflip \
       --run-tag kaggle-pc
   ```

5. **Mind the time budget**. Each baseline takes ~1 hour, so plan on 2–4 baselines + stacker per notebook session.

6. **Submit**. The `submission.zip` under `/kaggle/working/baseline_out/runs/<run_id>/` is downloadable from the notebook's *Output* panel, or you can submit directly from the notebook with the Kaggle API:
   ```python
   !kaggle competitions submit -c <competition-slug> \
       -f /kaggle/working/baseline_out/runs/<run_id>/submission.zip \
       -m "stacker v7"
   ```

---

## Tips

- **Reproducibility**: every run is keyed by a content hash of its config, so re-running with identical args reuses the same `run_id`.
- **Caching**: the stacker caches decoded test submissions and rank-normalized inputs under `baseline_out/stacker_cache/`. The first run warms the cache; subsequent runs skip decode and rank-norm.
- **Debugging a single class**: pass `--only-classes class_03` to any baseline to iterate fast on one category.
- **Logs**: if a run looks off, `cat baseline_out/runs/<run_id>/run_log.txt` — full per-epoch loss curves, per-anomaly-type AP, and runtime are all there.
