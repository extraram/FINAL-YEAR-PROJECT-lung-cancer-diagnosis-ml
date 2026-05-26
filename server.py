"""
Flask backend for Lung Cancer Diagnosis AI.

- Serves the static HTML/CSS/JS frontend at /
- Exposes JSON APIs for prediction, analytics, and model listing
"""

import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.config import CLASS_NAMES, MODELS_DIR, REPORTS_DIR
from src.prediction import Predictor
from src.preprocessing import is_medical_image
from src.utils import setup_logging

logger = setup_logging("server")

STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_DIR = Path(__file__).parent / "sample_test_images"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)

MODEL_FILES = {
    "cnn": "cnn_lung_cancer_model.h5",
    "resnet50": "resnet50_lung_cancer_model.h5",
}
MODEL_LABELS = {
    "cnn": "Custom CNN",
    "resnet50": "ResNet50 Transfer Learning",
}
MODEL_DESCRIPTIONS = {
    "cnn": "Lightweight model trained from scratch on this dataset. Faster to run; best when you want a quick second opinion.",
    "resnet50": "ImageNet-pretrained backbone fine-tuned on lung scans. Slower but generally more robust on unfamiliar scans.",
}

_predictors = {}


def get_predictor(model_key: str) -> Predictor:
    if model_key not in MODEL_FILES:
        raise ValueError(f"Unknown model key: {model_key}")
    if model_key not in _predictors:
        path = MODELS_DIR / MODEL_FILES[model_key]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        logger.info(f"Loading model: {model_key}")
        _predictors[model_key] = Predictor(str(path), model_key)
    return _predictors[model_key]


def safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read {path.name}: {e}")
        return None


def png_to_base64(rgb_array: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/models")
def list_models():
    available = []
    for key, filename in MODEL_FILES.items():
        if (MODELS_DIR / filename).exists():
            available.append({
                "key": key,
                "label": MODEL_LABELS[key],
                "description": MODEL_DESCRIPTIONS[key],
            })
    return jsonify({"models": available, "classes": CLASS_NAMES})


@app.route("/api/samples")
def list_samples():
    if not SAMPLE_DIR.exists():
        return jsonify({"samples": []})
    samples = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in SAMPLE_DIR.glob(ext):
            samples.append(p.name)
    samples.sort()
    return jsonify({"samples": samples})


@app.route("/api/sample-image/<path:name>")
def sample_image(name):
    return send_from_directory(str(SAMPLE_DIR), name)


@app.route("/api/predict", methods=["POST"])
def predict():
    model_key = request.form.get("model", "cnn")

    if "image" in request.files:
        file = request.files["image"]
        suffix = Path(file.filename or "upload.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
    elif request.form.get("sample"):
        sample_name = request.form["sample"]
        candidate = (SAMPLE_DIR / sample_name).resolve()
        if not candidate.is_file() or SAMPLE_DIR.resolve() not in candidate.parents:
            return jsonify({"error": "Sample not found"}), 404
        tmp_path = str(candidate)
    else:
        return jsonify({"error": "No image provided"}), 400

    # Reject obvious non-medical images (photos, screenshots, thumbnails)
    # before they reach the model. CT scans are grayscale; anything with
    # meaningful color is out-of-distribution and would get a nonsense label.
    ok, reason = is_medical_image(tmp_path)
    if not ok:
        if "image" in request.files:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return jsonify({"error": reason, "reason": "not_medical"}), 400

    try:
        predictor = get_predictor(model_key)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = predictor.predict(tmp_path)
        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        response = {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "all_predictions": result["all_predictions"],
        }

        try:
            heatmap = predictor.generate_gradcam(tmp_path)
            if heatmap is not None:
                resized = cv2.resize(heatmap, (224, 224))
                heatmap_u8 = (resized * 255).astype(np.uint8)
                colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
                colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

                original = cv2.imread(tmp_path)
                if original is not None:
                    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                    original = cv2.resize(original, (224, 224))
                    overlay = cv2.addWeighted(original.astype(np.uint8), 0.55, colored_rgb, 0.45, 0)
                    response["gradcam"] = png_to_base64(overlay)
                else:
                    response["gradcam"] = png_to_base64(colored_rgb)
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
            response["gradcam_error"] = str(e)

        return jsonify(response)
    finally:
        if "image" in request.files:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.route("/api/analytics/<model_key>")
def analytics(model_key):
    if model_key not in MODEL_FILES:
        return jsonify({"error": "Unknown model"}), 400

    base = f"{model_key}_lung_cancer_model"
    metrics = safe_load_json(REPORTS_DIR / f"{base}_metrics.json")
    history = safe_load_json(REPORTS_DIR / f"{base}_history.json")

    return jsonify(
        {
            "model": model_key,
            "label": MODEL_LABELS[model_key],
            "metrics": metrics,
            "history": history,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    print(f"\n  Lung Cancer Diagnosis AI")
    print(f"  Serving at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
