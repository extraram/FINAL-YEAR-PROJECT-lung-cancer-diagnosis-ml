# Lung Cancer Diagnosis AI — Complete Project Documentation

A deep-learning web application that classifies chest CT scan images as **Normal**, **Benign**, or **Malignant**, with Grad-CAM visual explanations.

This document is the single source of truth for the project: **what** it does, **why** each piece exists, **where** each file lives, and **how** to use it end-to-end.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [The problem and the approach](#2-the-problem-and-the-approach)
3. [Technology stack — what we use and why](#3-technology-stack--what-we-use-and-why)
4. [Project layout — where everything lives](#4-project-layout--where-everything-lives)
5. [The dataset](#5-the-dataset)
6. [Data pipeline — from disk to tensor](#6-data-pipeline--from-disk-to-tensor)
7. [Models — architectures and choices](#7-models--architectures-and-choices)
8. [Training — how the models learn](#8-training--how-the-models-learn)
9. [Evaluation — how we measure quality](#9-evaluation--how-we-measure-quality)
10. [Prediction & Grad-CAM — how a single image is diagnosed](#10-prediction--grad-cam--how-a-single-image-is-diagnosed)
11. [Web application — server, API, and frontend](#11-web-application--server-api-and-frontend)
12. [Configuration reference](#12-configuration-reference)
13. [Setup and installation](#13-setup-and-installation)
14. [How to use — common workflows](#14-how-to-use--common-workflows)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitations and disclaimer](#16-limitations-and-disclaimer)

---

## 1. What this project is

A complete, runnable pipeline that takes a chest CT scan image and tells you whether it looks **Normal**, **Benign**, or **Malignant**, along with:

- a confidence score for each class,
- a **Grad-CAM heatmap** showing which regions of the image the model attended to (so a clinician can sanity-check the prediction visually).

It ships with:

- a **training pipeline** (`train.py`) that produces two trained models from the bundled dataset,
- a **Flask web app** (`server.py`) that serves a single-page UI for uploading images and viewing predictions,
- an **HTTP API** for programmatic access,
- generated **reports** (metrics, confusion matrix, ROC curves, per-epoch history).

---

## 2. The problem and the approach

**Problem.** Given a 2D chest CT slice, assign one of three labels: `Normal cases`, `Benign cases`, `Malignant cases`. This is a 3-class image classification task.

**Approach.**

1. **Pre-augmented dataset on disk** — the project bundles a balanced, augmented dataset (~3,609 images, ~1,200 per class).
2. **Two complementary models** trained on the same data:
   - a **custom CNN** built from scratch (small, fast, learns class-specific texture/shape features),
   - a **ResNet50 transfer-learning** model (large, pretrained on ImageNet, frozen base + new dense head).
3. **Streaming `tf.data` pipeline** — images are read from disk on demand instead of loading 3,609 tensors into RAM up front. Memory stays bounded to a few mini-batches.
4. **Stratified path-level split** (70/15/15 train/val/test) — splitting happens on file paths before any decoding, which makes the pipeline cheap and keeps class proportions identical across splits.
5. **Grad-CAM** explainability at inference time — the gradient of the predicted class score is taken with respect to the last convolutional feature map and turned into a heatmap overlaid on the input.

---

## 3. Technology stack — what we use and why

| Layer | Technology | Where it's used | Why this choice |
|---|---|---|---|
| Modeling | **TensorFlow / Keras** | `src/models/`, `src/training/`, `src/prediction/` | Industry-standard, ships with ResNet50 + ImageNet weights, has native `tf.data` for streaming |
| Numerical / array | **NumPy** | everywhere | Standard for tensor math outside the graph |
| Image I/O & preprocessing | **OpenCV (`cv2`)** | `src/preprocessing/preprocessing.py`, `server.py` | Fast bilateral filtering, CLAHE contrast enhancement, color space conversions |
| Streaming data pipeline | **`tf.data`** | `src/preprocessing/preprocessing.py` (`build_tf_dataset`) | Disk-resident; only a few batches in memory; parallel decode via `AUTOTUNE` |
| Train/val/test split | **scikit-learn `train_test_split`** | `src/preprocessing/preprocessing.py` (`split_paths`) | Stratified sampling keeps class balance across splits |
| Metrics & plots | **scikit-learn + Matplotlib** | `src/utils/helpers.py` | Confusion matrix, ROC, precision/recall/F1 |
| Backend | **Flask + Flask-CORS** | `server.py` | Tiny, no build step, fits a single-file API server |
| Frontend | **Vanilla HTML/CSS/JS** | `static/` | No bundler, no framework, no build step — open the file and it runs |
| Persistence | **Keras `.h5`** for models, **JSON** for metrics/history, **PNG** for plots | `models/`, `reports/` | Portable formats that the frontend can consume directly |
| Logging | **Python `logging`** | `src/utils/helpers.py` (`setup_logging`) | File + console logs under `logs/` |
| Packaging | **`venv` + `requirements.txt`** | project root | Loose pin ranges so it installs on Python 3.8–3.11 |

---

## 4. Project layout — where everything lives

```
Lung Cancer/
├── train.py                        Top-level training entry point
├── server.py                       Flask backend (API + static serving)
├── requirements.txt                Python dependencies (loose pins, 3.8–3.11)
├── run_app.bat                     Windows launcher — starts the web app
├── run_training.bat                Windows launcher — creates venv, installs deps, trains
├── README.md                       Short overview (installation + quick start)
├── QUICK_START.md                  Five-minute walkthrough
├── INSTALLATION_GUIDE.md           Detailed Python/venv troubleshooting
├── PROJECT_DOCUMENTATION.md        ← This document
│
├── lung cancer dataset/            Training data (see § 5)
│   ├── Benign cases/               1,200 JPGs
│   ├── Malignant cases/            1,201 JPGs
│   ├── Normal cases/               1,208 JPGs
│   └── Augmentation Details.txt    Original vs augmented sample counts
│
├── src/
│   ├── config.py                   Single source of truth for paths, hyper-parameters, class names
│   ├── preprocessing/
│   │   ├── __init__.py             Re-exports public API
│   │   └── preprocessing.py        ImagePreprocessor + tf.data streaming pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py               build_cnn_model, build_resnet50_model, (ensemble + EfficientNet helpers)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py              ModelTrainer class + train_cnn_model, train_transfer_learning_model
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py            Predictor class (predict + generate_gradcam)
│   └── utils/
│       ├── __init__.py
│       └── helpers.py              setup_logging, plotting, metrics, format_time
│
├── static/                         Frontend — served by Flask at /
│   ├── index.html                  Single-page app shell
│   ├── style.css                   Light theme, Geologica font
│   └── app.js                      Vanilla JS — calls the JSON API
│
├── models/                         Trained .h5 files — populated by train.py
├── reports/                        Metrics JSON + history JSON + PNG plots
├── logs/                           App + training log files (and TensorBoard runs)
├── assets/                         Static assets used by docs/UI
├── data/processed/                 Reserved for any cached pre-processed arrays
├── notebooks/                      Optional Jupyter notebooks
└── venv/                           Local virtual environment (created by run_training.bat)
```

---

## 5. The dataset

**Source:** the IQ-OTH/NCCD Lung Cancer Dataset, bundled under [`lung cancer dataset/`](lung%20cancer%20dataset/).

### Image counts

| Class | Folder | Original samples | Augmented samples (used) |
|---|---|---:|---:|
| Benign | `Benign cases/` | 120 | **1,200** |
| Malignant | `Malignant cases/` | 561 | **1,201** |
| Normal | `Normal cases/` | 416 | **1,208** |
| **Total** | | **1,097** | **3,609** |

All images are `.jpg`. The augmentation pipeline (already applied to the bundled images) is documented in [`Augmentation Details.txt`](lung%20cancer%20dataset/Augmentation%20Details.txt):

> Horizontal Flip · Vertical Flip · Rotation · Color jitter · Contour Crop · Gaussian Blur · Sharpness · Contrast · Histogram Equalization

### Why this dataset shape matters

- It is **balanced** post-augmentation, so no extra class-weighting is needed at training time.
- Because augmentation has already been applied **to all images**, a naive random split could place augmented variants of the same source image into different splits and inflate test accuracy. The streaming pipeline mitigates this somewhat by splitting **by file path**, but for the strictest evaluation you would split by original-image identity before augmentation.

---

## 6. Data pipeline — from disk to tensor

**File:** [`src/preprocessing/preprocessing.py`](src/preprocessing/preprocessing.py)

Two parallel APIs live in this module:

### 6a. The in-memory API (legacy / single-image preprocessing)

Used by [`src/prediction/predictor.py`](src/prediction/predictor.py) for one-off inference on uploaded images.

- `ImagePreprocessor` — class with `load_image → resize → remove_noise (bilateral) → enhance_contrast (CLAHE) → normalize_image (ImageNet mean/std)`.
- `load_and_prepare_dataset` / `split_dataset` / `convert_labels_to_categorical` — kept for backwards compatibility; **not** used by training anymore.

### 6b. The streaming API (production training path)

Used by [`train.py`](train.py).

| Function | What it does | Why |
|---|---|---|
| `collect_image_paths(data_dir)` | Walks the three class folders, returns `(paths, integer_labels)`. **Decodes nothing.** | Lets us split and shuffle without touching pixels |
| `split_paths(paths, labels)` | Stratified 70/15/15 split on the `(path, label)` pairs using sklearn `train_test_split` | Class proportions preserved across train/val/test |
| `build_tf_dataset(paths, labels, batch_size, shuffle, augment)` | Builds a `tf.data.Dataset` that does `read_file → decode → resize(224, BICUBIC) → /255 → (x - mean) / std → optional augment → one-hot → batch → prefetch`. All map steps use `num_parallel_calls=AUTOTUNE`. | Memory stays bounded; decoding parallelised; GPU stays fed |

**Augmentation in the streaming path** is intentionally lightweight (`random_flip_left_right`, `random_brightness`, `random_contrast`) because the on-disk dataset is already heavily augmented offline. Adding more aggressive on-the-fly augmentation would just re-augment already-augmented images.

**Normalization** uses ImageNet statistics (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`) so the ResNet50 pretrained weights receive inputs in the distribution they were trained on.

---

## 7. Models — architectures and choices

**File:** [`src/models/models.py`](src/models/models.py)

### 7a. Custom CNN — `build_cnn_model()`

```
Input (224 × 224 × 3)
 → Conv(32) → BN → Conv(32) → MaxPool → Dropout(0.25)        # conv_1 block
 → Conv(64) → BN → Conv(64) → MaxPool → Dropout(0.25)        # conv_2 block
 → Conv(128) → BN → Conv(128) → MaxPool → Dropout(0.25)      # conv_3 block  ← Grad-CAM target
 → Conv(256) → BN → Conv(256) → MaxPool → Dropout(0.25)
 → GlobalAveragePooling2D
 → Dense(512) → BN → Dropout(0.5)
 → Dense(256) → BN → Dropout(0.5)
 → Dense(3, softmax)
```

~5 M parameters. Optimised with Adam at `lr = 1e-3`, categorical cross-entropy.

### 7b. ResNet50 transfer learning — `build_resnet50_model()`

```
Input (224 × 224 × 3)
 → ResNet50 base (ImageNet weights, FROZEN)                  # conv5_block3_out ← Grad-CAM target
 → GlobalAveragePooling2D
 → Dense(512) → BN → Dropout(0.4)
 → Dense(256) → BN → Dropout(0.4)
 → Dense(128) → BN → Dropout(0.4)
 → Dense(3, softmax)
```

~25 M parameters. Adam at `lr = 1e-4` (much lower because the base is frozen and we only train the head).

### 7c. Why two models?

| | Custom CNN | ResNet50 |
|---|---|---|
| Strengths | Small, fast, no external weights, learns dataset-specific features | Rich ImageNet priors generalise well from limited data |
| Weaknesses | Less expressive, more prone to overfitting | Larger, slower, ImageNet priors may not align perfectly with CT slices |
| When it wins | Lots of in-domain data | Smaller datasets / faster convergence |

The web app lets the user pick which model to query, so both stay shipped.

### 7d. Other helpers in this file

- `build_efficientnet_model()` — alternative transfer-learning option (not used by `train.py` by default).
- `build_ensemble_model(cnn, resnet)` — averages the softmax outputs of the two trained models.
- `save_model` / `load_model` — thin wrappers around Keras serialization.

---

## 8. Training — how the models learn

### Entry points

| Layer | File | Role |
|---|---|---|
| CLI | [`train.py`](train.py) | Orchestrates the full pipeline |
| Wrapper | `train_cnn_model`, `train_transfer_learning_model` in [`src/training/trainer.py`](src/training/trainer.py) | Build → train → evaluate → save → plot |
| Core | `ModelTrainer` class in the same file | `train(train_ds, val_ds)`, `evaluate(test_ds)`, callback wiring, report generation |

### What `train.py` does, step by step

1. **`[1/5] Indexing dataset`** — `collect_image_paths(DATA_DIR)` enumerates all `.jpg/.jpeg/.png` files under the three class folders. Logs counts per class. No image is decoded.
2. **`[2/5] Splitting paths`** — `split_paths(...)` does a stratified 70/15/15 split.
3. **`[3/5] Train custom CNN`** — builds train/val/test `tf.data.Dataset` objects (train shuffled + augmented), then calls `train_cnn_model(...)`.
4. **`[4/5] Train ResNet50`** — same pattern with a separate set of datasets (batch size from `TRANSFER_LEARNING_CONFIG`).
5. **`[5/5] Summary`** — compares test accuracies, logs the winner, prints total time.

### Callbacks (configured in `ModelTrainer.create_callbacks`)

| Callback | Purpose |
|---|---|
| `EarlyStopping(monitor='val_loss', patience=…, restore_best_weights=True)` | Stop when validation loss stops improving; revert to best weights |
| `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=…)` | Halve the learning rate when stuck |
| `ModelCheckpoint(…_best_<timestamp>.h5)` | Save best-val-accuracy snapshot to `models/` |
| `TensorBoard(log_dir=logs/tensorboard_<timestamp>)` | Per-epoch scalars + histograms |
| `CSVLogger(logs/<model>_training_<timestamp>.csv)` | Per-epoch CSV log |

### Outputs after a successful training run

| Path | What it is |
|---|---|
| `models/cnn_lung_cancer_model.h5` | Final CNN (overwrites previous) |
| `models/cnn_lung_cancer_model_best_<ts>.h5` | Best-val-acc CNN snapshot |
| `models/resnet50_lung_cancer_model.h5` | Final ResNet50 |
| `models/resnet50_lung_cancer_model_best_<ts>.h5` | Best-val-acc ResNet50 snapshot |
| `reports/<model>_metrics.json` | Test loss, accuracy, precision, recall, F1 |
| `reports/<model>_history.json` | Per-epoch loss + accuracy for train and val |
| `reports/<model>_training_history.png` | Loss/accuracy curves |
| `reports/<model>_confusion_matrix.png` | 3×3 confusion matrix on the test set |
| `reports/<model>_roc_curve.png` | Per-class one-vs-rest ROC curves |
| `logs/<model>_training_<ts>.csv` | Per-epoch CSV |
| `logs/tensorboard_<ts>/` | TensorBoard event files |
| `logs/app.log` | Aggregated training log |

---

## 9. Evaluation — how we measure quality

**Implemented in:** `ModelTrainer.evaluate(test_ds)` (in [`src/training/trainer.py`](src/training/trainer.py)) and `calculate_metrics` (in [`src/utils/helpers.py`](src/utils/helpers.py)).

The test pipeline:

1. Call `model.evaluate(test_ds)` → returns `test_loss`, `test_accuracy`.
2. Stream the same test dataset a second time, batch by batch, collecting `model.predict(batch)` probabilities and ground-truth labels. (Avoids keeping the full test set in RAM.)
3. Compute precision / recall / F1 (weighted) via scikit-learn.
4. Plot the confusion matrix and per-class ROC curves to `reports/`.

The metrics JSON looks like:

```json
{
  "test_loss": 0.214,
  "test_accuracy": 0.937,
  "precision": 0.939,
  "recall": 0.937,
  "f1": 0.937
}
```

These JSON files are consumed by the **Analytics page** in the web app.

---

## 10. Prediction & Grad-CAM — how a single image is diagnosed

**File:** [`src/prediction/predictor.py`](src/prediction/predictor.py)

### Inference flow (`Predictor.predict(image_path)`)

1. `ImagePreprocessor.preprocess()` — same chain used during training conceptually, but applied to a single image:
   `imread → BGR→RGB → resize(224) → /255 → bilateral denoise → CLAHE → ImageNet normalize`.
2. Add a batch dimension → `model.predict()` → softmax vector of length 3.
3. Return `{predicted_class, confidence, all_predictions, image, image_batch}`.

### Grad-CAM (`Predictor.generate_gradcam(image_path)`)

For the predicted class:

1. Take the gradient of the class logit w.r.t. the last convolutional feature map:
   - **CNN** → layer `conv_3` (set via `GRAD_CAM_CNN_LAYER` in config)
   - **ResNet50** → layer `conv5_block3_out` (set via `GRAD_CAM_LAYER` in config)
2. Average those gradients channel-wise → per-channel weights.
3. Weighted sum of feature maps → take ReLU → normalize to [0, 1].
4. Upsample to 224×224, apply `COLORMAP_JET`, blend with the original image (55/45) → return the overlay as base64 PNG.

This overlay is what the frontend shows in the "Grad-CAM" panel.

---

## 11. Web application — server, API, and frontend

### Backend — [`server.py`](server.py)

Single-file Flask app. Serves the static frontend at `/` and exposes JSON endpoints.

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | Serves `static/index.html` |
| `GET` | `/api/models` | Lists trained models present in `models/` |
| `GET` | `/api/samples` | Lists sample image filenames (sample picker; degrades to empty list if folder absent) |
| `GET` | `/api/sample-image/<name>` | Returns a sample image file |
| `POST` | `/api/predict` | Multipart: `image` (file) **or** `sample` (name), plus `model` (`cnn` or `resnet50`). Returns prediction + Grad-CAM overlay |
| `GET` | `/api/analytics/<model>` | Returns the per-model metrics and history JSON for the Analytics page |

`Predictor` instances are cached in `_predictors` so a model is loaded into memory only once per process.

### Frontend — [`static/`](static/)

| File | Purpose |
|---|---|
| `index.html` | Three tabs: **Diagnosis**, **Analytics**, **About** |
| `style.css` | Light theme, Geologica font, card-based layout |
| `app.js` | Vanilla JS — drag-and-drop upload, sample picker, fetch calls to `/api/...`, canvas-rendered probability bars and history charts |

No build step. Edit the file, refresh the browser.

### Example API call

```bash
curl -X POST \
  -F "model=cnn" \
  -F "image=@my_scan.jpg" \
  http://localhost:8501/api/predict
```

Response:

```json
{
  "predicted_class": "Normal cases",
  "confidence": 0.658,
  "all_predictions": {
    "Normal cases": 0.658,
    "Benign cases": 0.301,
    "Malignant cases": 0.041
  },
  "gradcam": "data:image/png;base64,..."
}
```

---

## 12. Configuration reference

**File:** [`src/config.py`](src/config.py) — every tunable lives here.

| Group | Key | Default | Meaning |
|---|---|---|---|
| **Paths** | `DATA_DIR` | `lung cancer dataset/` | Where the training data is |
| | `MODELS_DIR` | `models/` | Where `.h5` files go |
| | `REPORTS_DIR` | `reports/` | Metrics, plots |
| | `LOGS_DIR` | `logs/` | Logs + TensorBoard runs |
| **Classes** | `CLASS_NAMES` | `["Normal cases", "Benign cases", "Malignant cases"]` | Ordered class list (index = label) |
| | `NUM_CLASSES` | `3` | Derived |
| **Image** | `IMAGE_SIZE` / `TARGET_SIZE` | `(224, 224)` | Input spatial size |
| | `NORMALIZE_MEAN` | `[0.485, 0.456, 0.406]` | ImageNet mean |
| | `NORMALIZE_STD` | `[0.229, 0.224, 0.225]` | ImageNet std |
| **Split** | `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | `0.70 / 0.15 / 0.15` | Stratified path split |
| | `RANDOM_SEED` | `42` | Reproducibility |
| **CNN training** | `CNN_CONFIG.epochs` | `10` | Upper bound (early stopping may stop sooner) |
| | `CNN_CONFIG.batch_size` | `32` | Mini-batch |
| | `CNN_CONFIG.learning_rate` | `0.001` | Adam LR |
| | `CNN_CONFIG.early_stopping_patience` | `4` | Epochs of no val improvement before stop |
| | `CNN_CONFIG.reduce_lr_patience` | `2` | Epochs before LR halving |
| **ResNet50 training** | `TRANSFER_LEARNING_CONFIG.epochs` | `8` | |
| | `TRANSFER_LEARNING_CONFIG.batch_size` | `32` | |
| | `TRANSFER_LEARNING_CONFIG.learning_rate` | `0.0001` | Lower because base is frozen |
| | `TRANSFER_LEARNING_CONFIG.freeze_base_layers` | `True` | Whether to update ImageNet weights |
| **Grad-CAM** | `GRAD_CAM_CNN_LAYER` | `"conv_3"` | Target layer for the custom CNN |
| | `GRAD_CAM_LAYER` | `"conv5_block3_out"` | Target layer for ResNet50 |

For a fast smoke-test run, drop CNN epochs to ~3 and ResNet50 epochs to ~2 and lower patience accordingly.

---

## 13. Setup and installation

### Requirements

- **Python** 3.8 – 3.11
- **RAM** 8 GB minimum (the streaming pipeline keeps usage low — only TF and the model live in RAM)
- **Disk** ~2 GB free
- **GPU** optional; CPU works, just slower

### Quick start (Windows)

```powershell
cd "C:\path\to\Lung Cancer"
.\run_training.bat        # creates venv, installs deps, trains both models
.\run_app.bat             # launches web app at http://localhost:8501
```

### Manual (any OS)

```bash
cd "Lung Cancer"
python -m venv venv

# Activate
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
python train.py                 # train both models
python server.py                # launch web app
```

Then open `http://localhost:8501`.

### Python 3.8 caveat

On Python 3.8 (Windows) the highest TensorFlow build is **2.13.0** (not 2.14). `requirements.txt` uses loose pins so a compatible version is picked automatically. For exact pins, use Python 3.9 – 3.11.

---

## 14. How to use — common workflows

### 14a. Train the models from scratch

```bash
python train.py
```

Expected log highlights:

```
[1/5] Indexing dataset (no images loaded yet)...
Found 3609 images across 3 classes
[2/5] Splitting paths into train/val/test (stratified)...
  Training: 2525 / Validation: 542 / Test: 542
[3/5] Training custom CNN model...
...
[4/5] Training ResNet50 transfer learning model...
...
[5/5] Training Pipeline Completed!
```

When it finishes you will find `.h5` files under `models/` and PNG/JSON reports under `reports/`.

### 14b. Run the web app

```bash
python server.py
# or set a custom port:
PORT=8502 python server.py            # macOS / Linux
$env:PORT="8502"; python server.py    # Windows PowerShell
```

Open `http://localhost:<port>`. The **Diagnosis** tab lets you upload an image or pick a sample; the **Analytics** tab shows metrics and training curves for whichever model you select.

### 14c. Predict from Python directly

```python
from src.prediction import Predictor

p = Predictor("models/cnn_lung_cancer_model.h5", model_type="cnn")
result = p.predict("path/to/scan.jpg")
print(result["predicted_class"], result["confidence"])
print(result["all_predictions"])
```

### 14d. Re-tune hyper-parameters

Edit [`src/config.py`](src/config.py) — change `CNN_CONFIG` / `TRANSFER_LEARNING_CONFIG`. No other files need updating.

### 14e. Inspect training curves in TensorBoard

```bash
tensorboard --logdir logs/
```

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: tensorflow` | venv not activated, or `pip install` ran outside it | Activate venv first, then re-install |
| `tensorflow==2.14.0` not found | Python 3.8 + Windows — TF tops out at 2.13 there | Use looser pin (already in `requirements.txt`) or upgrade to Python 3.9+ |
| "No trained models found" on the Diagnosis page | `models/` is empty | Run `python train.py` first |
| Out-of-memory during training | Batch size too large for GPU/RAM | Lower `batch_size` in `src/config.py` |
| Server won't start, port in use | Another service on 8501 | Set `PORT` env var (see § 14b) |
| Browser shows old UI after CSS/JS edits | Browser cached static files | Hard-refresh (Ctrl+F5 / Cmd+Shift+R) |
| Predictions look random | Models not yet trained, or trained for too few epochs | Re-train with default epochs |

---

## 16. Limitations and disclaimer

**Medical disclaimer.** This project is provided for **educational and research purposes only**. It is not a medical device, has not been clinically validated, and must not be used as a substitute for professional diagnosis. Always defer to qualified clinicians.

**Known limitations.**

- The bundled dataset is small (~1,097 originals) and the augmented variants are derived from those originals — claimed accuracy figures should be read with caution.
- The split is stratified by path but not by source-image identity, so some leakage between augmented variants is possible. For rigorous evaluation, hold out original images before augmentation.
- The model is trained on 2D slices, not 3D volumes — a single slice often does not contain enough context for a confident diagnosis.
- Grad-CAM shows what the model attended to, **not** what is medically relevant. Use it as a sanity check, not a clinical justification.
