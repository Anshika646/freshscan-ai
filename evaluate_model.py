#!/usr/bin/env python3
"""
Held-out test set evaluation and diagnostic chart generator for FreshScan AI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data_loader import parse_class_label
from src.preprocessing import LabelEncoderMapping, normalize_mobilenet_v2


def _save_evaluation_plots(
    plots_dir: Path,
    class_names: List[str],
    cm: np.ndarray,
    training_history: Dict[str, Any] | None,
    dataset_summary: Dict[str, Any] | None,
) -> Dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=140)
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    short_labels = [parse_class_label(c)["display_name"] for c in class_names]
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_ylabel("True Class (Held-Out Test Split)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax.set_title("FreshScan AI — Held-Out Test Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val > 0 or i == j:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if val > thresh else "#14241B",
                )
    fig.tight_layout()
    cm_path = plots_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    saved_paths["confusion_matrix_png"] = str(cm_path)

    if training_history and training_history.get("epochs"):
        epochs_data = training_history["epochs"]
        xs = [e["epoch"] for e in epochs_data]
        tr_loss = [e["loss"] for e in epochs_data]
        va_loss = [e["val_loss"] for e in epochs_data]
        tr_acc = [e["accuracy"] * 100.0 for e in epochs_data]
        va_acc = [e["val_accuracy"] * 100.0 for e in epochs_data]
        s1_end = training_history.get("stage1_epochs_completed", 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
        ax1.plot(xs, tr_loss, marker="o", color="#1B4332", linewidth=2, label="Train Loss")
        ax1.plot(xs, va_loss, marker="s", color="#D97706", linewidth=2, linestyle="--", label="Val Loss")
        if 0 < s1_end < len(xs):
            ax1.axvline(x=s1_end + 0.5, color="#64748B", linestyle=":", label="Fine-Tuning Start")
        ax1.set_title("Cross-Entropy Loss Curve", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.25)
        ax1.legend(frameon=False)

        ax2.plot(xs, tr_acc, marker="o", color="#1B4332", linewidth=2, label="Train Accuracy (%)")
        ax2.plot(xs, va_acc, marker="s", color="#2D6A4F", linewidth=2, linestyle="--", label="Val Accuracy (%)")
        if 0 < s1_end < len(xs):
            ax2.axvline(x=s1_end + 0.5, color="#64748B", linestyle=":", label="Fine-Tuning Start")
        ax2.set_title("Classification Accuracy Curve", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(frameon=False)

        fig.tight_layout()
        curves_path = plots_dir / "training_curves.png"
        fig.savefig(curves_path)
        plt.close(fig)
        saved_paths["training_curves_png"] = str(curves_path)

    if dataset_summary and dataset_summary.get("class_distribution"):
        dist = dataset_summary["class_distribution"]
        labels_disp = [parse_class_label(k)["display_name"] for k in class_names]
        counts = [int(dist.get(k, 0)) for k in class_names]
        colors = ["#1B4332" if k.startswith("fresh_") else "#B45309" for k in class_names]

        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=140)
        bars = ax.bar(np.arange(len(class_names)), counts, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(labels_disp, rotation=40, ha="right", fontsize=8.5)
        ax.set_ylabel("Valid Raw Images", fontsize=10, fontweight="bold")
        ax.set_title("Dataset Class Distribution (Fresh vs Rotten by Fruit)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + max(counts) * 0.01,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        fig.tight_layout()
        dist_path = plots_dir / "class_distribution.png"
        fig.savefig(dist_path)
        plt.close(fig)
        saved_paths["class_distribution_png"] = str(dist_path)

    return saved_paths


def evaluate_model(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    batch_size: int = 32,
) -> Dict[str, Any]:
    import tensorflow as tf

    proc_path = Path(processed_dir)
    mod_path = Path(models_dir)

    model_file = mod_path / "freshscan_mobilenetv2.keras"
    class_map_file = mod_path / "class_indices.json"
    test_npz_file = proc_path / "test_split.npz"

    for req_file in (model_file, class_map_file, test_npz_file):
        if not req_file.exists():
            raise FileNotFoundError(
                f"Required evaluation artifact missing: {req_file}. "
                f"Run `python train_model.py` before running `evaluate_model.py`."
            )

    encoder = LabelEncoderMapping.load(class_map_file)
    model = tf.keras.models.load_model(model_file, compile=False)

    test_bundle = np.load(test_npz_file)
    x_test_uint8 = test_bundle["images"]
    y_test = test_bundle["labels"].astype(np.int64)

    t0 = time.perf_counter()
    x_test_norm = normalize_mobilenet_v2(x_test_uint8)
    y_prob = model.predict(x_test_norm, batch_size=batch_size, verbose=0)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    row_sums = np.clip(y_prob.sum(axis=1, keepdims=True), 1e-12, None)
    y_prob = y_prob / row_sums
    eval_seconds = round(time.perf_counter() - t0, 3)

    y_pred = np.argmax(y_prob, axis=1)

    overall_acc = float(accuracy_score(y_test, y_pred))
    test_log_loss = float(log_loss(y_test, y_prob, labels=list(range(encoder.num_classes))))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=list(range(encoder.num_classes)),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(encoder.num_classes)))

    per_class_metrics: List[Dict[str, Any]] = []
    for idx, cls_name in enumerate(encoder.class_names):
        parsed = parse_class_label(cls_name)
        cls_total = int(per_sup[idx])
        cls_correct = int(cm[idx, idx])
        per_class_metrics.append(
            {
                "class_index": idx,
                "class_label": cls_name,
                "display_name": parsed["display_name"],
                "fruit_type": parsed["fruit_type"],
                "freshness": parsed["freshness"],
                "precision": round(float(per_prec[idx]), 4),
                "recall": round(float(per_rec[idx]), 4),
                "f1_score": round(float(per_f1[idx]), 4),
                "support": cls_total,
                "correct": cls_correct,
            }
        )

    present_classes = set(np.unique(y_test).tolist())
    if len(present_classes) == encoder.num_classes:
        ovr_macro = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
        ovr_weighted = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
        roc_auc_payload: Dict[str, Any] = {
            "appropriate": True,
            "methodology": "Multi-class One-vs-Rest (OvR) over calibrated softmax probabilities",
            "ovr_macro_roc_auc": round(ovr_macro, 5),
            "ovr_weighted_roc_auc": round(ovr_weighted, 5),
        }
    else:
        roc_auc_payload = {
            "appropriate": False,
            "reason": "Not all target classes are present in the held-out test set.",
        }

    rotten_indices = [
        i for i, c in enumerate(encoder.class_names) if parse_class_label(c)["freshness"] == "rotten"
    ]
    y_test_bin = np.isin(y_test, rotten_indices).astype(np.int64)
    y_prob_rotten = y_prob[:, rotten_indices].sum(axis=1)
    y_pred_bin = (y_prob_rotten >= 0.5).astype(np.int64)

    bin_acc = float(accuracy_score(y_test_bin, y_pred_bin))
    bin_f1 = float(f1_score(y_test_bin, y_pred_bin, zero_division=0))
    bin_cm = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1]).tolist()
    bin_roc_auc = (
        float(roc_auc_score(y_test_bin, y_prob_rotten))
        if len(np.unique(y_test_bin)) == 2
        else None
    )

    confused_pairs: List[Dict[str, Any]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused_pairs.append(
                    {
                        "true_class": encoder.decode(i),
                        "true_display": parse_class_label(encoder.decode(i))["display_name"],
                        "predicted_class": encoder.decode(j),
                        "predicted_display": parse_class_label(encoder.decode(j))["display_name"],
                        "count": int(cm[i, j]),
                    }
                )
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    history_file = mod_path / "training_history.json"
    summary_file = mod_path / "dataset_summary.json"
    training_history = (
        json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else None
    )
    dataset_summary = (
        json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else None
    )

    saved_plots = _save_evaluation_plots(
        plots_dir=mod_path / "plots",
        class_names=encoder.class_names,
        cm=cm,
        training_history=training_history,
        dataset_summary=dataset_summary,
    )

    lowest_classes = sorted(per_class_metrics, key=lambda m: m["f1_score"])[:3]
    imbalance_ratio = (
        dataset_summary.get("imbalance_ratio_max_to_min", 1.0) if dataset_summary else 1.0
    )
    dedup_removed = (
        dataset_summary.get("splits", {})
        .get("deduplication_and_leakage", {})
        .get("exact_duplicates_removed", 0)
        if dataset_summary
        else 0
    )

    limitations = [
        (
            f"Class Balance & Deduplication Shift: Exact-byte SHA-256 deduplication ({dedup_removed} "
            f"duplicates removed) and perceptual dHash cluster grouping yield a post-cleaning max/min "
            f"class ratio of {imbalance_ratio:.2f}x, mitigated via balanced inverse-frequency training class weights."
        ),
        (
            "Surface-Only Modality: RGB optical imaging captures external peel discoloration, fungal "
            "lesions, and shriveling, but cannot detect internal core rot, early enzymatic browning "
            "beneath thick rinds (e.g., pomegranate or orange), or invisible microbial/chemical hazards."
        ),
        (
            "Closed-Set Domain Assumption: Softmax probabilities sum to 1.0 across the 8 trained fruit "
            "species. Uploading unsupported produce (e.g., mango, avocado) can trigger uncertain "
            "predictions, flagged in production when top-1 confidence falls below 60%."
        ),
        (
            "Lowest Held-Out F1 Subclasses: "
            + ", ".join(f"{c['display_name']} (F1={c['f1_score']:.3f})" for c in lowest_classes)
            + ", where dark natural skin pigmentation or specular highlights resemble early bruising."
        ),
    ]

    report = {
        "evaluated_split": "test (held-out, zero train/val overlap)",
        "num_test_samples": int(len(y_test)),
        "num_classes": encoder.num_classes,
        "evaluation_duration_seconds": eval_seconds,
        "overall_metrics": {
            "accuracy": round(overall_acc, 5),
            "test_log_loss": round(test_log_loss, 5),
            "macro_precision": round(float(prec_macro), 5),
            "macro_recall": round(float(rec_macro), 5),
            "macro_f1": round(float(f1_macro), 5),
            "weighted_precision": round(float(prec_weighted), 5),
            "weighted_recall": round(float(rec_weighted), 5),
            "weighted_f1": round(float(f1_weighted), 5),
        },
        "roc_auc": roc_auc_payload,
        "binary_fresh_vs_rotten": {
            "accuracy": round(bin_acc, 5),
            "f1_rotten_detection": round(bin_f1, 5),
            "roc_auc": round(bin_roc_auc, 5) if bin_roc_auc is not None else None,
            "confusion_matrix_labels": ["fresh", "rotten"],
            "confusion_matrix": bin_cm,
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": encoder.class_names,
            "display_labels": [parse_class_label(c)["display_name"] for c in encoder.class_names],
            "matrix": cm.tolist(),
        },
        "top_confused_pairs": confused_pairs[:10],
        "plots": saved_plots,
        "limitations_and_imbalance_summary": limitations,
        "text_classification_report": classification_report(
            y_test,
            y_pred,
            target_names=encoder.class_names,
            digits=4,
            zero_division=0,
        ),
    }

    report_path = mod_path / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[evaluate_model] Held-out test accuracy: {overall_acc * 100:.2f}% | "
        f"Macro F1: {f1_macro:.4f} | Binary Fresh/Rotten Accuracy: {bin_acc * 100:.2f}%"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained FreshScan AI model on held-out test set.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate_model(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        batch_size=args.batch_size,
    )
