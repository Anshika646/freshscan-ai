#!/usr/bin/env python3
"""
Reproducible end-to-end model training script for FreshScan AI.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.data_loader import (
    download_agml_raw_dataset,
    prepare_dataset_splits_and_cache,
)
from src.preprocessing import (
    LabelEncoderMapping,
    PreprocessingConfig,
    build_training_augmentation_layer,
    compute_balanced_class_weights,
)


def set_global_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def build_mobilenetv2_classifier(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.25,
):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape, name="rgb_normalized_input")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(192, activation="relu", name="head_dense_192")(x)
    x = tf.keras.layers.Dropout(0.20, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="freshscan_mobilenetv2")
    return model, base_model


def make_tf_datasets(
    train_npz_path: Path,
    val_npz_path: Path,
    batch_size: int = 32,
    seed: int = 42,
):
    import tensorflow as tf

    train_data = np.load(train_npz_path)
    val_data = np.load(val_npz_path)

    x_train = train_data["images"]
    y_train = train_data["labels"]
    x_val = val_data["images"]
    y_val = val_data["labels"]

    aug_layer = build_training_augmentation_layer(seed=seed)

    def _prep_train(images: tf.Tensor, labels: tf.Tensor):
        imgs_f32 = tf.cast(images, tf.float32)
        augmented = aug_layer(imgs_f32, training=True)
        normalized = (augmented / 127.5) - 1.0
        return normalized, labels

    def _prep_eval(images: tf.Tensor, labels: tf.Tensor):
        imgs_f32 = tf.cast(images, tf.float32)
        normalized = (imgs_f32 / 127.5) - 1.0
        return normalized, labels

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(_prep_train, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .map(_prep_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, y_train, y_val


def train_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    download_if_missing: bool = True,
    max_per_class: int | None = 80,
    stage1_epochs: int = 6,
    stage2_epochs: int = 5,
    batch_size: int = 32,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-4,
    fine_tune_layers: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    import tensorflow as tf

    set_global_reproducibility(seed)
    t_start = time.time()

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    mod_path = Path(models_dir)
    mod_path.mkdir(parents=True, exist_ok=True)

    config = PreprocessingConfig(random_seed=seed)

    train_npz = proc_path / "train_split.npz"
    val_npz = proc_path / "val_split.npz"
    splits_ready = (
        train_npz.exists()
        and val_npz.exists()
        and (mod_path / "class_indices.json").exists()
    )

    if not splits_ready:
        has_raw = raw_path.exists() and any(
            d.is_dir() and any(d.iterdir()) for d in raw_path.iterdir()
        )
        if not has_raw:
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Raw dataset directory '{raw_path}' is empty and --no-download was specified."
                )
            print("[train_model] Downloading raw Project-AgML fruit freshness dataset...")
            dl_stats = download_agml_raw_dataset(
                raw_dir=raw_path,
                max_per_class=max_per_class or 80,
            )
            print(f"[train_model] Download summary: {json.dumps(dl_stats, indent=2)}")

        print("[train_model] Inspecting raw dataset and preparing leakage-safe splits...")
        prepare_dataset_splits_and_cache(
            raw_dir=raw_path,
            processed_dir=proc_path,
            models_dir=mod_path,
            config=config,
        )

    encoder = LabelEncoderMapping.load(mod_path / "class_indices.json")
    train_ds, val_ds, y_train, y_val = make_tf_datasets(
        train_npz_path=train_npz,
        val_npz_path=val_npz,
        batch_size=batch_size,
        seed=seed,
    )
    class_weights = compute_balanced_class_weights(y_train, num_classes=encoder.num_classes)

    print(
        f"[train_model] Building MobileNetV2 transfer-learning model for "
        f"{encoder.num_classes} classes (train={len(y_train)}, val={len(y_val)})..."
    )
    model, base_model = build_mobilenetv2_classifier(
        num_classes=encoder.num_classes,
        input_shape=(config.input_height, config.input_width, config.channels),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=stage1_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks_s1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"[train_model] Starting Stage 1 (Frozen Backbone) for up to {stage1_epochs} epochs...")
    hist_s1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_s1,
        verbose=2,
    )

    combined_loss: List[float] = [float(x) for x in hist_s1.history.get("loss", [])]
    combined_val_loss: List[float] = [float(x) for x in hist_s1.history.get("val_loss", [])]
    combined_acc: List[float] = [float(x) for x in hist_s1.history.get("accuracy", [])]
    combined_val_acc: List[float] = [float(x) for x in hist_s1.history.get("val_accuracy", [])]
    stage1_completed_epochs = len(combined_loss)

    if stage2_epochs > 0:
        base_model.trainable = True
        total_base_layers = len(base_model.layers)
        freeze_until = max(0, total_base_layers - fine_tune_layers)

        for idx, layer in enumerate(base_model.layers):
            if idx < freeze_until or isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=stage2_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        callbacks_s2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        print(
            f"[train_model] Starting Stage 2 (Fine-tuning top {fine_tune_layers} non-BN layers) "
            f"for up to {stage2_epochs} epochs..."
        )
        hist_s2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=stage2_epochs,
            class_weight=class_weights,
            callbacks=callbacks_s2,
            verbose=2,
        )

        combined_loss.extend(float(x) for x in hist_s2.history.get("loss", []))
        combined_val_loss.extend(float(x) for x in hist_s2.history.get("val_loss", []))
        combined_acc.extend(float(x) for x in hist_s2.history.get("accuracy", []))
        combined_val_acc.extend(float(x) for x in hist_s2.history.get("val_accuracy", []))

    import zipfile

    model_out_path = mod_path / "freshscan_mobilenetv2.keras"
    tmp_raw_keras = mod_path / "_tmp_uncompressed.keras"
    inference_clone = tf.keras.models.clone_model(model)
    inference_clone.set_weights(model.get_weights())
    inference_clone.save(tmp_raw_keras, include_optimizer=False)
    with (
        zipfile.ZipFile(tmp_raw_keras, "r") as zin,
        zipfile.ZipFile(model_out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zout,
    ):
        for item in zin.infolist():
            zout.writestr(item.filename, zin.read(item.filename))
    tmp_raw_keras.unlink(missing_ok=True)

    total_duration_sec = round(time.time() - t_start, 2)
    epochs_records = [
        {
            "epoch": idx + 1,
            "stage": "head_warmup" if idx < stage1_completed_epochs else "fine_tuning",
            "loss": round(combined_loss[idx], 6),
            "val_loss": round(combined_val_loss[idx], 6),
            "accuracy": round(combined_acc[idx], 6),
            "val_accuracy": round(combined_val_acc[idx], 6),
        }
        for idx in range(len(combined_loss))
    ]

    history_payload = {
        "model_name": "FreshScan MobileNetV2 Transfer Classifier",
        "backbone": "MobileNetV2 (ImageNet weights)",
        "input_shape": [config.input_height, config.input_width, config.channels],
        "num_classes": encoder.num_classes,
        "random_seed": seed,
        "batch_size": batch_size,
        "stage1_lr": stage1_lr,
        "stage2_lr": stage2_lr,
        "fine_tune_top_layers": fine_tune_layers,
        "stage1_epochs_completed": stage1_completed_epochs,
        "total_epochs_completed": len(combined_loss),
        "training_duration_seconds": total_duration_sec,
        "model_file": str(model_out_path),
        "model_size_mb": round(model_out_path.stat().st_size / (1024 * 1024), 2),
        "best_val_accuracy": round(max(combined_val_acc) if combined_val_acc else 0.0, 6),
        "best_val_loss": round(min(combined_val_loss) if combined_val_loss else 0.0, 6),
        "epochs": epochs_records,
    }

    history_path = mod_path / "training_history.json"
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    print(
        f"[train_model] Training finished in {total_duration_sec}s. "
        f"Saved model ({history_payload['model_size_mb']} MB) to {model_out_path}."
    )
    return history_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FreshScan AI MobileNetV2 model.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=80,
        help="Images downloaded per class from raw benchmark (default: 80 = 1,280 raw images; max: 200 = 3,200).",
    )
    parser.add_argument("--stage1-epochs", type=int, default=6)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        download_if_missing=not args.no_download,
        max_per_class=args.max_per_class,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        fine_tune_layers=args.fine_tune_layers,
        seed=args.seed,
    )
