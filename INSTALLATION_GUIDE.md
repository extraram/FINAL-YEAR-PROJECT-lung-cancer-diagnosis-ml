# Installation Guide

Detailed setup instructions for the Lung Cancer Diagnosis AI.

For a 5-minute walkthrough see [QUICK_START.md](QUICK_START.md). For the full reference see [README.md](README.md).

---

## Prerequisites

| | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 2 GB free | 5 GB free (SSD) |
| GPU | none | NVIDIA + CUDA |

### Verify Python

```bash
python --version
# Python 3.10.x  ← anything 3.8–3.11 works
```

If missing, install from <https://www.python.org/downloads/>.

---

## Step 1 — Get the project

Extract / clone the project so you have a folder containing `server.py`, `train.py`, `requirements.txt`, and the `src/` and `lung cancer dataset/` folders.

```bash
cd "Lung Cancer"
```

---

## Step 2 — Create a virtual environment

```bash
python -m venv venv
```

Then activate it:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows cmd
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

If PowerShell refuses with an execution-policy error, either:

```powershell
# A) one-shot bypass
powershell -ExecutionPolicy Bypass -NoExit -Command "cd '<project-path>'; .\venv\Scripts\Activate.ps1"

# B) skip activation entirely — invoke the venv's python directly
.\venv\Scripts\python.exe server.py
```

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Installation takes 3–10 minutes depending on bandwidth. TensorFlow + dependencies pull in ~400 MB.

### Python 3.8 caveat

On Python 3.8 (Windows in particular) `tensorflow==2.14.0` is **not** available — the latest cp38 wheel is **2.13.0**. The `requirements.txt` uses a loose pin (`tensorflow<2.14`) so pip will pick a compatible build automatically.

If pip falls back to an older TensorFlow you don't want, install a specific version explicitly:

```bash
pip install "tensorflow==2.13.0" "keras==2.13.1" "numpy<1.25" "typing-extensions<4.6"
```

### Verify the install

```bash
python -c "import tensorflow as tf, flask, cv2; print('TF', tf.__version__, '/ Flask', flask.__version__, '/ OpenCV', cv2.__version__)"
```

You should see something like:

```
TF 2.13.0 / Flask 3.0.3 / OpenCV 4.8.1
```

### Optional — GPU

If you have an NVIDIA GPU with current drivers + CUDA:

```bash
pip install "tensorflow[and-cuda]"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

CPU-only is fine for this project's dataset size; GPU just makes training faster.

---

## Step 4 — Train the models

```bash
python train.py
```

What happens:

1. Loads the dataset (~3 600 images across Normal / Benign / Malignant)
2. Splits 70 / 15 / 15
3. Trains the custom CNN with early stopping
4. Trains the ResNet50 transfer learning model
5. Saves `.h5` checkpoints to `models/` and metrics + plots to `reports/`

Expected wall time on CPU: **30 – 60 minutes** at the default settings (50 + 30 epochs with early stopping). For a smoke-test run, lower the epoch counts in [src/config.py](src/config.py).

You can interrupt with **Ctrl+C** — the most recent best checkpoint is already on disk.

---

## Step 5 — Launch the web app

```bash
python server.py
```

Server output:

```
  Lung Cancer Diagnosis AI
  Serving at http://localhost:8501
```

Open <http://localhost:8501> in your browser.

The frontend is plain HTML / CSS / JavaScript served from `static/` — no build step, no `npm`, no framework.

### Stopping

**Ctrl+C** in the terminal where `python server.py` is running.

### Changing the port

```bash
PORT=8502 python server.py            # macOS / Linux
$env:PORT="8502"; python server.py    # Windows PowerShell
```

---

## File layout after setup

```
Lung Cancer/
├── venv/                       created in step 2
├── models/                     populated by train.py
│   ├── cnn_lung_cancer_model.h5
│   └── resnet50_lung_cancer_model.h5
├── reports/                    populated by train.py
│   ├── cnn_lung_cancer_model_metrics.json
│   ├── cnn_lung_cancer_model_history.json
│   ├── cnn_lung_cancer_model_confusion_matrix.png
│   ├── cnn_lung_cancer_model_roc_curve.png
│   └── (same set for resnet50)
├── logs/                       app + training logs
├── static/                     frontend (committed; no build output)
├── src/                        Python package
├── server.py
├── train.py
├── requirements.txt
├── run_app.bat
└── run_training.bat
```

---

## Troubleshooting

**`Could not find a version that satisfies the requirement tensorflow==X.Y.Z`**

You're on a Python version with no matching TF wheel. Loosen the pin or use Python 3.10/3.11.

**`ImportError: cannot import name '...' from 'src.preprocessing'`**

A function isn't re-exported in `src/preprocessing/__init__.py`. Add it to the `from .preprocessing import (...)` block.

**`Class directory not found: .../Normal`**

`CLASS_NAMES` in [src/config.py](src/config.py) must match the dataset's folder names exactly. The bundled dataset uses `Normal cases`, `Benign cases`, `Malignant cases`.

**`Object of type float32 is not JSON serializable`**

Already patched in `src/training/trainer.py` (numpy scalars routed through a custom JSON encoder). If you see this again, you've reverted that fix.

**Streamlit-related references in old logs / scripts**

The project used Streamlit briefly. It's been removed. If you find leftover references in your local copy, they're safe to delete — the live stack is Flask + static HTML / CSS / JS.

**`Address already in use` on port 8501**

Another process is bound. Either stop it or set `PORT=...` as shown above.

**Out-of-memory during training**

Lower `batch_size` from 32 to 16 in [src/config.py](src/config.py) (both `CNN_CONFIG` and `TRANSFER_LEARNING_CONFIG`).

**Old UI persists after CSS / JS edits**

Hard-refresh: Ctrl+F5 (Win/Linux) or Cmd+Shift+R (macOS).

---

## Clean reinstall

```bash
# remove the venv
rm -rf venv               # macOS / Linux
rmdir /s /q venv          # Windows

# remove built artifacts (optional)
rm -rf models reports logs

# start over from step 2
```

---

## Verifying everything works

After install + train:

```bash
python -c "import tensorflow, flask, cv2"          # no errors
ls models/                                          # both .h5 files present
curl -s http://localhost:8501/api/models            # JSON listing both models
```

Then open <http://localhost:8501>, upload one of the images in `sample_test_images/`, and confirm you get a predicted class, probabilities, and a Grad-CAM image.
