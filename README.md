# Deepfake detection pipeline

Runnable implementation of the eight-stage pipeline described in
Figure 1 and Section 3 of the accompanying dissertation, "Leveraging
Machine Learning for AI-Deepfake Detection in Strengthening Digital
Forensic Evidence Authentication."

## Stage -> file -> research basis

| # | Stage (Figure 1) | File | Implements / cites |
|---|---|---|---|
| 1 | Media upload | `stage1_upload.py` | True-type (not extension-based) file validation |
| 2 | Preprocessing | `stage2_preprocessing.py` | Frame extraction (OpenCV), normalisation, augmentation |
| 3 | Forensic metadata capture | `stage3_forensic_metadata.py` | SHA-256 hashing, EXIF extraction, Error Level Analysis |
| 4 | Face detection & alignment | `stage4_face_detection.py` | MTCNN -- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). *IEEE Signal Processing Letters*, 23(10), 1499-1503 |
| 5 | Feature extraction | `stage5_feature_extraction.py` | Spatial (CNN) + radially-averaged FFT frequency-domain artifacts |
| 6 | Deepfake classification | `stage6_classification.py` | Xception -- Chollet, F. (2017). *CVPR*, 1251-1258, + LSTM temporal head for video |
| 7 | Explainability & scoring | `stage7_explainability.py` | Grad-CAM -- Selvaraju, R. R. et al. (2017). *ICCV*, 618-626 |
| 8 | Forensic report | `stage8_report.py` | Signed PDF report (ReportLab) combining verdict, hash, metadata, heatmap |

`datasets.py` loads the benchmark data for training/evaluation and
`evaluation.py` computes the metrics from Section 3.11. `pipeline.py`
wires all eight stages together end to end for a single uploaded file.

`face_cache.py` and `clip_dataset.py` support `train.py` (below): the
former extracts + caches aligned faces per video (so a multi-epoch
run doesn't re-run MTCNN every epoch), the latter turns that cache
into fixed-length clip tensors for `TemporalDeepfakeClassifier`.

## Dataset -- wired to this machine's local copy

`datasets.DEFAULT_ROOT_DIR` points at the actual local download, so
`load_faceforensics()` works with no arguments. **Both collections are
now downloaded on this machine** -- `train.py` defaults to
`--collections actors,youtube` accordingly:

```
/Users/annetnabukenya/Downloads/Annet Research/FaceForensics++_data/
├── original_sequences/youtube/c23/videos                (1000 real clips)
├── original_sequences/actors/c23/videos                 (363 real clips)
├── manipulated_sequences/Deepfakes/c23/videos            (1000 fake clips)
└── manipulated_sequences/DeepFakeDetection/c23/videos    (128 fake clips)
```

| Collection | Original videos | Manipulation method | Role |
|---|---|---|---|
| `youtube` | `original_sequences/youtube` (1000 clips) | `Deepfakes` (1000 clips) | real + fake |
| `actors` | `original_sequences/actors` (363 clips) | `DeepFakeDetection` (128 clips) | real + fake |

With both collections in scope, `leave_one_method_out_split` / Table
5's cross-manipulation generalisation test is meaningful (2 methods:
`Deepfakes`, `DeepFakeDetection`). `train.py`'s default
`--max-videos 80` stratified sample (`datasets.sample_for_quick_run`)
keeps a representative slice of *both* methods rather than dropping
one, so Table 5 still populates even on a fast run; if it ever doesn't
(e.g. `--max-videos` set very low, or a `--collections` value with
only one method), `train.py` detects this via
`datasets.methods_present()` and writes an explanatory note into
`results/experiment_summary.md` instead of guessing.

`load_faceforensics(collections=("actors",))` restricts loading to
just one collection; pass `collections=None` (the default in
`load_faceforensics` itself, though not in `train.py`) to load every
collection declared in `COLLECTIONS`.

These are **not** identity-matched against each other -- `youtube` and
`actors` use different source videos and different filename
conventions, so `datasets.py` keeps them in separate identity spaces
and only pairs real/fake clips *within* a collection (see
`datasets._match_identity`).

If this pipeline runs on a different machine, either move the data to
the same path or pass `root_dir=...` explicitly:
```python
samples = load_faceforensics(root_dir="/some/other/path/FaceForensics++_data")
```

- **FaceForensics++ / DeepFakeDetection** -- Roessler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Niessner, M. (2019). *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 1-11. Request: https://github.com/ondyari/FaceForensics

Downloading requires agreeing to the authors' terms of use, appropriate
for a dataset containing real people's likenesses.

The `images/` folder bundled alongside this pipeline contains the
publicly shared FaceForensics++ example/preview stills and manipulation
GIFs (from the FaceForensics-master repository) and is used only for
illustration -- it is not training data and is not read by any script
in this pipeline.

## Setup

**Use an isolated virtual environment, not your conda `base` environment.**
`base` (e.g. miniconda's default environment) usually ships with
`numpy`/`pandas`/`scikit-learn` pre-installed via `conda`; installing
this project's pinned versions with `pip` on top of that can leave
`pandas`'s compiled extensions built against a different NumPy ABI
than the one now on the path, causing a `ValueError: numpy.dtype size
changed, may indicate binary incompatibility` at import time
(see "Troubleshooting" below if you hit this).

```bash
cd /path/to/this/project
python3 -m venv venv
source venv/bin/activate      # macOS/Linux; on Windows: venv\Scripts\activate
which python3                 # sanity check: should print .../venv/bin/python3, NOT .../miniconda/...
pip install --upgrade pip
pip install -r requirements.txt
```

## Troubleshooting

**`ValueError: numpy.dtype size changed... Expected 96... got 88`** (or
any traceback ending inside `pandas/_libs/...` or `sklearn/utils/...`)
-- this is an environment problem, not a bug in this pipeline: NumPy
and pandas were compiled against incompatible versions of each other,
almost always because packages were installed by a mix of `conda` and
`pip` into the same environment (commonly conda's `base`). Fix:

```bash
conda deactivate    # if a conda environment is currently active
cd /path/to/this/project
python3 -m venv venv
source venv/bin/activate
which python3        # must point inside venv/, not miniconda/
pip install --upgrade pip
pip install -r requirements.txt
python train.py --smoke-test
```

If you'd rather not create a new environment, forcing a consistent
reinstall of the three packages in the existing one usually also
works, but is more fragile long-term than the venv above:
```bash
pip install --upgrade --force-reinstall numpy pandas scikit-learn
```

## Running the pipeline on a single file

```bash
python pipeline.py path/to/video.mp4 --output-dir output
```

This prints progress for each of the eight stages and writes
`output/forensic_report.pdf` (plus `output/gradcam_overlay.png` when a
heatmap could be generated).

## Loading and splitting the dataset

```python
from datasets import load_faceforensics, stratified_identity_split, class_balance, collection_summary

samples = load_faceforensics(collections=("actors",))  # uses DEFAULT_ROOT_DIR; pass root_dir=... to override
print(collection_summary(samples))
# e.g. {'actors': {'real': 363, 'DeepFakeDetection': 128}}

train_set, val_set, test_set = stratified_identity_split(samples)
counts = class_balance(train_set)
```

`train.py` (below) does all of this, plus training, evaluation, and
figure generation, end to end -- these snippets are for interactive
exploration only.

## Reproducing the Chapter 4 results (`train.py`)

`train.py` is the single entry point that trains the classifier and
writes everything Sections 4.2-4.7 need.

**By default it runs fast, not on the full dataset.** The slow part of
this pipeline is face detection/caching, which scales roughly linearly
with video count, so `train.py` stratified-samples down to `--max-videos`
(default 80, proportional across collection and manipulation method,
seed-controlled for reproducibility -- see `datasets.sample_for_quick_run`)
*before* caching anything. Pass `--max-videos 0` to disable this and
process every video `load_faceforensics` finds instead -- expect that
to take hours, not minutes, since it repeats MTCNN face detection
across the full dataset.

**Before even that**, do an even smaller ~2-5 minute sanity check to
confirm your environment, paths, and dependencies are all correct:

```bash
python train.py --smoke-test
```

This samples down to 16 videos and trains for 2 epochs, exercising
every stage (caching, training, evaluation, plotting, Grad-CAM, the
sample report), so a real problem (missing dependency, bad
`--root-dir`, zero detectable faces) surfaces in minutes instead of
hours into a full run.

Once that passes, run the default fast experiment (both collections,
<=80 videos, ~10-30 minutes on a laptop CPU depending on hardware):

```bash
python train.py
```

Useful flags (all optional, defaults shown):

```bash
python train.py \
  --root-dir "/Users/annetnabukenya/Downloads/Annet Research/FaceForensics++_data" \
  --collections actors,youtube --max-videos 80 \
  --output-dir results \
  --epochs 8 --batch-size 8 --lr 1e-4 --patience 2 \
  --frames-per-clip 8 --max-frames-per-video 12
```

Raise `--max-videos` (and, if you have the time budget, `--epochs`)
once the fast run looks right and you want a larger-sample result for
the final write-up -- nothing else needs to change.

The first run against a given `--max-videos`/`--max-frames-per-video`
combination is the slow(er) one: each selected video's frames are
extracted and run through MTCNN once, then cached as aligned-face
JPEGs under `face_cache/` (see `face_cache.py`). Re-running `train.py`
with different `--epochs`/`--lr` reuses that cache and skips straight
to training. Delete `face_cache/` (or pass a new `--cache-dir`) to
force a rebuild -- needed if you change `--max-frames-per-video` or
switch which videos are in scope.

### Output -> report section mapping

`train.py` never lets one bad video abort the run: unreadable files,
videos with zero detectable faces, and optional-figure failures
(plotting/Grad-CAM/report generation) are caught, logged, and skipped
rather than raising -- core training/evaluation errors still surface
normally, since there's nothing to salvage if that step fails.

## Notes

- No sample data or pretrained weights are included; the dataset and
  weights must be sourced separately per FaceForensics++'s licence.
- Network access is required only for `pip install` and for the
  `pretrained=True` ImageNet weights download in Stage 6 -- the
  pipeline itself runs fully offline once dependencies and data are in
  place.
- `train.py` writes cached aligned-face JPEGs to `face_cache/`
  (roughly a few hundred MB to ~1GB for the actors-only scope,
  depending on `--max-frames-per-video`). It's safe to delete this
  directory at any time; it will be rebuilt on the next run.
