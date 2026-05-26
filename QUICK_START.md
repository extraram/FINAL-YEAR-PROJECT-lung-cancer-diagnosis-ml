# Quick Start

Get the Lung Cancer Diagnosis AI running locally.

---

## Windows (easiest)

1. Double-click **`run_training.bat`** — creates the venv, installs dependencies, trains both models. Takes ~10–60 min depending on epoch settings and hardware.
2. Double-click **`run_app.bat`** — launches the web app at <http://localhost:8501>.

---

## Manual (any OS)

```bash
cd "Lung Cancer"

# 1. Create + activate virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (writes models/*.h5 and reports/*)
python train.py

# 4. Launch the web app
python server.py
```

Open <http://localhost:8501>.

---

## What you'll see

### Diagnosis page
- Drag-and-drop or click to upload a JPG / PNG
- Or pick from the bundled sample images
- Result shows: predicted class, confidence, per-class probabilities, Grad-CAM heatmap

### Analytics page
- Test accuracy, precision, recall, F1
- Accuracy + loss line charts (training vs validation)

### About page
- Project overview and disclaimer

---

## Folder cheat sheet

| Folder | What's in it |
|---|---|
| `models/` | Trained `.h5` files |
| `reports/` | Metrics JSON + matplotlib plots |
| `logs/` | Training and server logs |
| `sample_test_images/` | Pre-bundled images for the sidebar sample picker |
| `lung cancer dataset/` | Source images (Normal / Benign / Malignant cases) |

---

## Faster training (tradeoff: lower accuracy)

Edit [src/config.py](src/config.py):

```python
CNN_CONFIG["epochs"] = 10                       # was 50
CNN_CONFIG["early_stopping_patience"] = 4

TRANSFER_LEARNING_CONFIG["epochs"] = 8          # was 30
TRANSFER_LEARNING_CONFIG["early_stopping_patience"] = 3
```

Then re-run `python train.py`.

---

## Stopping the app

- **Server:** Ctrl+C in the terminal where `python server.py` is running.
- **Browser:** just close the tab.

---

## Common issues

| Problem | Fix |
|---|---|
| `tensorflow==2.14.0` won't install | You're on Python 3.8 — TensorFlow 2.13 works there; the loose pin in `requirements.txt` already handles this. Or upgrade to Python 3.9–3.11. |
| `ModuleNotFoundError` | Virtual environment isn't activated. |
| "No trained models found" in the UI | Run `python train.py` first. |
| Port 8501 in use | `PORT=8502 python server.py` (Linux/Mac) or `$env:PORT="8502"; python server.py` (PowerShell). |
| UI didn't update after CSS/JS edit | Hard-refresh: Ctrl+F5 (Win/Linux) or Cmd+Shift+R (macOS). |

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for more detail, [README.md](README.md) for the full reference.
