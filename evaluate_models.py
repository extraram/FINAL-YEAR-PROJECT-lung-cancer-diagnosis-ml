"""
Compute real test-set metrics for every saved model and write them to
reports/{model_name}_metrics.json, matching the format produced by
src.training.trainer.save_metrics().

Re-uses the exact same stratified split (RANDOM_SEED) and tf.data pipeline
that the trainer used, so the test set here equals the test set during
training. Without that, "test" accuracy would silently include training
images and be inflated.

Also derives reports/resnet50_lung_cancer_model_history.json from the
training CSV log, since the original run did not write the history JSON.
"""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR
from src.preprocessing import build_tf_dataset, collect_image_paths, split_paths
from src.utils.helpers import calculate_metrics


MODEL_SPECS = [
    ("cnn_lung_cancer_model", "cnn_lung_cancer_model.h5"),
    ("resnet50_lung_cancer_model", "resnet50_lung_cancer_model.h5"),
]


def evaluate_model(model_name: str, model_filename: str, test_ds: tf.data.Dataset) -> dict:
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        print(f"  [skip] {model_filename} not found")
        return None

    print(f"  loading {model_filename}")
    model = tf.keras.models.load_model(str(model_path))

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    probas, trues = [], []
    for batch_x, batch_y in test_ds:
        probas.append(model.predict(batch_x, verbose=0))
        trues.append(batch_y.numpy())
    y_pred_proba = np.concatenate(probas, axis=0)
    y_true_onehot = np.concatenate(trues, axis=0)

    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1) if y_true_onehot.ndim > 1 else y_true_onehot

    metrics = calculate_metrics(y_true, y_pred)
    metrics["test_loss"] = float(test_loss)
    metrics["test_accuracy"] = float(test_accuracy)

    out_path = REPORTS_DIR / f"{model_name}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, default=float)
    print(f"  wrote {out_path.name}: acc={metrics['test_accuracy']:.4f}, "
          f"prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}, "
          f"f1={metrics['f1']:.4f}")

    tf.keras.backend.clear_session()
    return metrics


def derive_history_from_csv(model_name: str):
    """
    The resnet50 run finished without saving history JSON. Reconstruct it
    from the per-epoch CSV that the CSVLogger wrote during training.
    """
    out_path = REPORTS_DIR / f"{model_name}_history.json"
    if out_path.exists():
        return

    candidates = sorted(LOGS_DIR.glob(f"{model_name}_training_*.csv"))
    if not candidates:
        print(f"  [skip history] no training CSV for {model_name}")
        return

    csv_path = candidates[-1]
    history = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key == "epoch":
                    continue
                history.setdefault(key, []).append(float(value))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"  wrote {out_path.name} from {csv_path.name}")


def main():
    print("[1/3] Collecting image paths...")
    paths, labels = collect_image_paths(DATA_DIR)
    print(f"  found {len(paths)} images across {len(set(labels))} classes")

    print("[2/3] Reproducing stratified train/val/test split (seeded)...")
    splits = split_paths(paths, labels)
    paths_test, y_test = splits["test"]
    print(f"  test set size: {len(paths_test)}")

    test_ds = build_tf_dataset(
        paths_test,
        y_test,
        batch_size=32,
        num_classes=len(CLASS_NAMES),
        shuffle=False,
        augment=False,
        cache=False,
    )

    print("[3/3] Evaluating models...")
    for model_name, model_filename in MODEL_SPECS:
        evaluate_model(model_name, model_filename, test_ds)
        derive_history_from_csv(model_name)


if __name__ == "__main__":
    main()
