"""
Main training script for lung cancer detection models.

Streams images from disk via tf.data so the full 3,609-image dataset never
needs to live in RAM at once.
"""

import time
import logging

from src.config import (
    DATA_DIR, MODELS_DIR, REPORTS_DIR, CLASS_NAMES,
    CNN_CONFIG, TRANSFER_LEARNING_CONFIG,
)
from src.preprocessing import (
    collect_image_paths,
    split_paths,
    build_tf_dataset,
)
from src.training import train_cnn_model, train_transfer_learning_model
from src.utils import setup_logging, format_time

logger = setup_logging("training")


def _make_datasets(paths_train, y_train, paths_val, y_val, paths_test, y_test, batch_size):
    """Build train (augmented + shuffled), val, and test tf.data pipelines."""
    train_ds = build_tf_dataset(
        paths_train, y_train,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
    )
    val_ds = build_tf_dataset(
        paths_val, y_val,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    test_ds = build_tf_dataset(
        paths_test, y_test,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    return train_ds, val_ds, test_ds


def main():
    start_time = time.time()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LUNG CANCER DETECTION - MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    logger.info("\n[1/5] Indexing dataset (no images loaded yet)...")
    try:
        paths, labels = collect_image_paths(DATA_DIR)
        logger.info(f"Found {len(paths)} images across {len(CLASS_NAMES)} classes")
        logger.info(f"  Classes: {', '.join(CLASS_NAMES)}")
    except Exception as e:
        logger.error(f"Failed to index dataset: {e}")
        return

    logger.info("\n[2/5] Splitting paths into train/val/test (stratified)...")
    try:
        splits = split_paths(paths, labels)
        paths_train, y_train = splits['train']
        paths_val, y_val = splits['val']
        paths_test, y_test = splits['test']
        logger.info(f"  Training: {len(paths_train)} images")
        logger.info(f"  Validation: {len(paths_val)} images")
        logger.info(f"  Test: {len(paths_test)} images")
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        return

    # CNN
    logger.info("\n[3/5] Training custom CNN model...")
    cnn_model = None
    cnn_metrics = None
    try:
        cnn_train_ds, cnn_val_ds, cnn_test_ds = _make_datasets(
            paths_train, y_train, paths_val, y_val, paths_test, y_test,
            batch_size=CNN_CONFIG['batch_size'],
        )
        cnn_model, cnn_history, cnn_metrics, cnn_predictions = train_cnn_model(
            cnn_train_ds, cnn_val_ds, cnn_test_ds,
        )
        logger.info("CNN model training completed")
        logger.info(f"  Test Accuracy: {cnn_metrics['test_accuracy']:.4f}")
        logger.info(f"  Test Loss: {cnn_metrics['test_loss']:.4f}")
        logger.info(f"  Precision: {cnn_metrics['precision']:.4f}")
        logger.info(f"  Recall: {cnn_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {cnn_metrics['f1']:.4f}")
    except Exception as e:
        logger.exception(f"CNN training failed: {e}")

    # Transfer learning
    logger.info("\n[4/5] Training ResNet50 transfer learning model...")
    tl_model = None
    tl_metrics = None
    try:
        tl_train_ds, tl_val_ds, tl_test_ds = _make_datasets(
            paths_train, y_train, paths_val, y_val, paths_test, y_test,
            batch_size=TRANSFER_LEARNING_CONFIG['batch_size'],
        )
        tl_model, tl_history, tl_metrics, tl_predictions = train_transfer_learning_model(
            tl_train_ds, tl_val_ds, tl_test_ds,
        )
        logger.info("Transfer learning model training completed")
        logger.info(f"  Test Accuracy: {tl_metrics['test_accuracy']:.4f}")
        logger.info(f"  Test Loss: {tl_metrics['test_loss']:.4f}")
        logger.info(f"  Precision: {tl_metrics['precision']:.4f}")
        logger.info(f"  Recall: {tl_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {tl_metrics['f1']:.4f}")
    except Exception as e:
        logger.exception(f"Transfer learning training failed: {e}")

    logger.info("\n[5/5] Training Pipeline Completed!")
    logger.info("=" * 80)

    if cnn_model and tl_model:
        logger.info("\nBoth models trained successfully")
        if cnn_metrics['test_accuracy'] > tl_metrics['test_accuracy']:
            logger.info("Best Model: CNN")
        else:
            logger.info("Best Model: ResNet50 Transfer Learning")
        logger.info(f"  CNN Accuracy: {cnn_metrics['test_accuracy']:.4f}")
        logger.info(f"  ResNet50 Accuracy: {tl_metrics['test_accuracy']:.4f}")
        logger.info(f"\nModels saved to: {MODELS_DIR}")
        logger.info(f"Reports saved to: {REPORTS_DIR}")

    logger.info(f"\nTotal training time: {format_time(time.time() - start_time)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
