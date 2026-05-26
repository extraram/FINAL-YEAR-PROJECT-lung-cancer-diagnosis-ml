# Lung Cancer Diagnosis AI

A deep learning web app that classifies chest X-ray and CT scan images as **Normal**, **Benign**, or **Malignant**, with Grad-CAM explainability.

- **Backend:** Python · TensorFlow / Keras · Flask
- **Frontend:** Vanilla HTML / CSS / JavaScript (no framework, no build step)
- **Models:** Custom CNN + ResNet50 transfer learning

---

## Project layout

```
Lung Cancer/
├── server.py                     Flask backend (API + static serving)
├── train.py                      Training entry point
├── requirements.txt              Python dependencies
├── run_app.bat                   Windows launcher (web app)
├── run_training.bat              Windows launcher (training)
│
├── static/                       Frontend (served by Flask at /)
│   ├── index.html
│   ├── style.css                 Light theme, Geologica font
│   └── app.js                    Vanilla JS, canvas charts
│
├── lung cancer dataset/
│   ├── Normal cases/
│   ├── Benign cases/
│   └── Malignant cases/
│
├── src/
│   ├── config.py
│   ├── preprocessing/preprocessing.py
│   ├── models/models.py
│   ├── training/trainer.py
│   ├── prediction/predictor.py
│   └── utils/helpers.py
│
├── models/                       Trained models (.h5) — populated by train.py
├── reports/                      Metrics, history, confusion matrix, ROC
├── logs/                         Training + server logs
└── sample_test_images/           Pre-bundled test images
```

---

## Dataset

The classifier is trained on the **IQ-OTH/NCCD Lung Cancer Dataset** (chest CT scans), bundled under [lung cancer dataset/](lung%20cancer%20dataset/). It ships pre-augmented and approximately balanced across the three classes.

### Image counts

| Class | Folder | Original | Augmented (used for training) |
|---|---|---:|---:|
| Benign | [Benign cases/](lung%20cancer%20dataset/Benign%20cases/) | 120 | **1,200** |
| Malignant | [Malignant cases/](lung%20cancer%20dataset/Malignant%20cases/) | 561 | **1,201** |
| Normal | [Normal cases/](lung%20cancer%20dataset/Normal%20cases/) | 416 | **1,208** |
| **Total** | | **1,097** | **3,609** |

All images are `.jpg`. Full augmentation pipeline (horizontal/vertical flip, rotation, color jitter, contour crop, Gaussian blur, sharpness, contrast, histogram equalization) is documented in [Augmentation Details.txt](lung%20cancer%20dataset/Augmentation%20Details.txt).

> **Note on splitting:** because augmentation is already applied to all images, a naive random 70/15/15 split can leak augmented variants of the same source image across train/val/test and inflate metrics. For a stricter evaluation, split by original image identity before augmentation.

---

## Requirements

- **Python** 3.8 – 3.11
- **OS:** Windows / macOS / Linux
- **RAM:** 8 GB minimum (16 GB recommended for training)
- **Disk:** ~2 GB free
- **GPU:** optional; CPU works, just slower

### Python 3.8 note

On Python 3.8 (Windows), the highest TensorFlow build is **2.13.0** — not 2.14. The `requirements.txt` is loose enough to pick a working version automatically. If you want the exact pinned versions, use Python 3.9 – 3.11.

---

## Quick start

### Windows

```powershell
cd "C:\path\to\Lung Cancer"
run_training.bat          # creates venv, installs deps, trains
run_app.bat               # launches web app at http://localhost:8501
```

### Manual (any OS)

```bash
cd "Lung Cancer"
python -m venv venv

# Activate venv
venv\Scripts\activate           # Windows PowerShell / cmd
source venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
python train.py                 # train models (~10–60 min depending on settings)
python server.py                # launch web app
```

Open <http://localhost:8501>.

---

## Training

`train.py` runs the full pipeline:

1. Loads images from `lung cancer dataset/{Normal,Benign,Malignant} cases/`
2. Splits 70 / 15 / 15 (train / validation / test)
3. Trains the **custom CNN** with early stopping
4. Trains **ResNet50 transfer learning** with early stopping
5. Saves `.h5` checkpoints to `models/` and metrics / plots to `reports/`

### Adjusting training

Edit [src/config.py](src/config.py):

```python
CNN_CONFIG = {
    "epochs": 50,                       # lower → faster
    "batch_size": 32,                   # lower → less memory
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    ...
}

TRANSFER_LEARNING_CONFIG = {
    "epochs": 30,
    ...
}
```

For a quick smoke-test run, drop epochs to 10 / 8 and patience to 4 / 3.

### Output files

| File | Purpose |
|---|---|
| `models/cnn_lung_cancer_model.h5` | Trained CNN |
| `models/resnet50_lung_cancer_model.h5` | Trained ResNet50 |
| `reports/<model>_metrics.json` | Test accuracy, precision, recall, F1 |
| `reports/<model>_history.json` | Per-epoch loss + accuracy |
| `reports/<model>_confusion_matrix.png` | Confusion matrix |
| `reports/<model>_roc_curve.png` | ROC curves |

---

## Web app

```bash
python server.py
```

Open <http://localhost:8501>.

### Pages

- **Diagnosis** — upload a JPG/PNG (drag-and-drop or click) or pick a sample; get predicted class, per-class probabilities, and a Grad-CAM heatmap.
- **Analytics** — test metrics + accuracy/loss line charts for the selected model.
- **About** — project overview and disclaimer.

### HTTP API

The frontend is just a client for these endpoints — call them yourself if you want:

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/models` | Lists available trained models |
| `GET` | `/api/samples` | Lists sample image filenames |
| `GET` | `/api/sample-image/<name>` | Returns a sample image |
| `POST` | `/api/predict` | Multipart form: `image` (file) **or** `sample` (name), `model` (`cnn` \| `resnet50`) |
| `GET` | `/api/analytics/<model>` | Returns metrics + training history JSON |

Example:

```bash
curl -X POST \
  -F "model=cnn" \
  -F "sample=sample_normal_1.jpg" \
  http://localhost:8501/api/predict
```

Response:

```json
{
  "predicted_class": "Normal cases",
  "confidence": 0.658,
  "all_predictions": { "Normal cases": 0.658, "Benign cases": 0.301, "Malignant cases": 0.041 },
  "gradcam": "data:image/png;base64,..."
}
```

---

## Models

### Custom CNN

```
Input (224×224×3)
 → 4 × [Conv → BatchNorm → MaxPool → Dropout(0.25)]   (32, 64, 128, 256 filters)
 → GlobalAveragePooling
 → Dense(512) → BatchNorm → Dropout(0.5)
 → Dense(256) → BatchNorm → Dropout(0.5)
 → Dense(3, softmax)
```

~5 M parameters.

### ResNet50 transfer learning

```
Input (224×224×3)
 → ResNet50 base (ImageNet weights, frozen)
 → GlobalAveragePooling
 → Dense(512) → BatchNorm → Dropout(0.4)
 → Dense(256) → BatchNorm → Dropout(0.4)
 → Dense(128) → BatchNorm → Dropout(0.4)
 → Dense(3, softmax)
```

~25 M parameters.

### Preprocessing pipeline

1. Read with OpenCV (BGR) → convert to RGB
2. Resize to 224 × 224 (cubic interpolation)
3. Denoise (bilateral filter, σ_spatial 9, σ_color 75)
4. CLAHE contrast enhancement (clip 2.0, tile 8×8)
5. Normalize using ImageNet mean / std

Training-time augmentation: rotation ±20°, width / height shift 20 %, shear 20 %, zoom 20 %, horizontal flip.

### Grad-CAM

For the predicted class, the gradient of the class score is taken with respect to the last convolutional feature map (`conv_3` for the custom CNN, `conv5_block3_out` for ResNet50). The channel-wise mean of those gradients weights the feature maps; the result is averaged, clamped to ≥ 0, and normalized to produce a heatmap that is overlaid on the input.

---

## Troubleshooting

**`ModuleNotFoundError`** — venv isn't activated, or you ran `pip install` outside it. Activate the venv first.

**`tensorflow==2.14.0` not found on Python 3.8** — expected. Use looser pin (`tensorflow<2.14`) or upgrade to Python 3.9+.

**Server won't start, port in use** — another process is on 8501. Either stop it, or:

```bash
PORT=8502 python server.py            # macOS / Linux
$env:PORT="8502"; python server.py    # Windows PowerShell
```

**Diagnosis page says "No trained models found"** — train first: `python train.py`.

**Out of memory during training** — lower `batch_size` in [src/config.py](src/config.py).

**Browser shows old UI after CSS / JS edits** — hard-refresh (Ctrl+F5 / Cmd+Shift+R).

---

## Medical disclaimer

For **educational and research use only**. Not a substitute for professional medical diagnosis. Always defer to qualified clinicians.
