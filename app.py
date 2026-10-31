#!/usr/bin/env python3
"""
FastAPI backend for FreshScan AI — Fruit Freshness Detection.

Exposes both clean REST routes (`/health`, `/model-info`, `/metrics`, `/predict`)
and gateway-prefixed aliases (`/api/ml/*`) for seamless full-stack & standalone deployment.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.predict import (
    FOOD_SAFETY_DISCLAIMER,
    FreshScanPredictor,
    ModelUnavailableError,
)
from src.preprocessing import ALLOWED_EXTENSIONS, InvalidImageError

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("FRESHSCAN_MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.environ.get("FRESHSCAN_DATA_DIR", BASE_DIR / "data"))
DIST_DIR = BASE_DIR / "dist"
MODEL_VERSION = "1.2.0-mobilenetv2"

ALLOWED_PLOTS = {
    "confusion_matrix.png",
    "training_curves.png",
    "class_distribution.png",
}


def _parse_cors_origins() -> List[str]:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(
    models_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    max_upload_mb: Optional[float] = None,
) -> FastAPI:
    m_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    d_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    upload_limit_mb = (
        float(max_upload_mb)
        if max_upload_mb is not None
        else float(os.environ.get("MAX_UPLOAD_MB", "15"))
    )
    max_upload_bytes = int(upload_limit_mb * 1024 * 1024)

    model_file = m_dir / "freshscan_mobilenetv2.keras"
    class_map_file = m_dir / "class_indices.json"
    prep_cfg_file = m_dir / "preprocessing_config.json"

    predictor = FreshScanPredictor(
        model_path=model_file,
        class_map_path=class_map_file,
        config_path=prep_cfg_file,
    )

    app = FastAPI(
        title="FreshScan AI Inference API",
        description="Real-time MobileNetV2 fruit freshness classification and evaluation analytics.",
        version=MODEL_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _iso_mtime(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        try:
            ts = path.stat().st_mtime
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except OSError:
            return None

    @app.get("/health")
    @app.get("/api/ml/health")
    def health_check() -> Dict[str, Any]:
        import tensorflow as tf

        ready = predictor.is_ready()
        missing = predictor.missing_artifacts()
        classes: List[str] = []
        if class_map_file.exists():
            try:
                classes = predictor.encoder.class_names
            except Exception:
                classes = []

        return {
            "status": "ok",
            "service": "freshscan-ml-backend",
            "model_version": MODEL_VERSION,
            "tensorflow_version": tf.__version__,
            "model_ready": ready,
            "missing_artifacts": missing,
            "supported_extensions": sorted(ALLOWED_EXTENSIONS),
            "max_upload_mb": upload_limit_mb,
            "num_classes": len(classes),
            "classes": classes,
            "food_safety_disclaimer": FOOD_SAFETY_DISCLAIMER,
        }

    @app.get("/model-info")
    @app.get("/api/ml/model-info")
    def get_model_info() -> Dict[str, Any]:
        ready = predictor.is_ready()
        train_hist = _read_json_if_exists(m_dir / "training_history.json")
        prep_cfg = _read_json_if_exists(prep_cfg_file)
        dataset_sum = _read_json_if_exists(m_dir / "dataset_summary.json")
        eval_file = m_dir / "evaluation_report.json"

        classes: List[str] = []
        supported_fruits: List[str] = []
        if class_map_file.exists():
            try:
                classes = predictor.encoder.class_names
                supported_fruits = sorted({c.split("_", 1)[1] for c in classes if "_" in c})
            except Exception:
                classes = []

        model_size_mb = (
            round(model_file.stat().st_size / (1024 * 1024), 2) if model_file.exists() else 0.0
        )

        return {
            "model_trained": ready,
            "status_message": "Trained MobileNetV2 model loaded and ready."
            if ready
            else "Model not trained yet. Run `python train_model.py` to generate artifacts.",
            "model_version": MODEL_VERSION,
            "architecture": train_hist.get("backbone", "MobileNetV2 (ImageNet transfer learning)")
            if train_hist
            else "MobileNetV2 (ImageNet transfer learning)",
            "input_shape": train_hist.get("input_shape", [224, 224, 3]) if train_hist else [224, 224, 3],
            "num_classes": len(classes),
            "supported_fruits": supported_fruits,
            "classes": classes,
            "model_size_mb": model_size_mb,
            "evaluated_at_utc": _iso_mtime(eval_file),
            "preprocessing": prep_cfg,
            "training_summary": {
                "stage1_epochs": train_hist.get("stage1_epochs_completed"),
                "total_epochs": train_hist.get("total_epochs_completed"),
                "fine_tune_top_layers": train_hist.get("fine_tune_top_layers"),
                "best_val_accuracy": train_hist.get("best_val_accuracy"),
                "random_seed": train_hist.get("random_seed"),
            }
            if train_hist
            else None,
            "dataset_provenance": dataset_sum.get("dataset_metadata") if dataset_sum else None,
            "food_safety_disclaimer": FOOD_SAFETY_DISCLAIMER,
        }

    @app.get("/metrics")
    @app.get("/api/ml/metrics")
    def get_metrics() -> Dict[str, Any]:
        eval_file = m_dir / "evaluation_report.json"
        evaluation_report = _read_json_if_exists(eval_file)
        training_history = _read_json_if_exists(m_dir / "training_history.json")

        if not evaluation_report:
            return {
                "available": False,
                "status_message": "Model not trained yet. Run `python train_model.py && python evaluate_model.py`.",
                "model_version": MODEL_VERSION,
                "evaluated_at_utc": None,
                "evaluation_report": None,
                "baseline_comparison": None,
            }

        epochs = training_history.get("epochs", []) if training_history else []
        epoch1_val_acc = float(epochs[0]["val_accuracy"]) if epochs else None

        baseline_comparison = {
            "uniform_random_16_class_accuracy": round(1.0 / max(1, evaluation_report.get("num_classes", 16)), 5),
            "frozen_backbone_epoch1_val_accuracy": round(epoch1_val_acc, 5)
            if epoch1_val_acc is not None
            else None,
            "final_mobilenetv2_test_accuracy": evaluation_report["overall_metrics"]["accuracy"],
            "final_mobilenetv2_macro_f1": evaluation_report["overall_metrics"]["macro_f1"],
        }

        return {
            "available": True,
            "status_message": "Verified held-out test set evaluation metrics.",
            "model_version": MODEL_VERSION,
            "evaluated_at_utc": _iso_mtime(eval_file),
            "evaluation_report": evaluation_report,
            "baseline_comparison": baseline_comparison,
        }

    @app.get("/api/ml/analytics")
    def get_analytics() -> Dict[str, Any]:
        dataset_summary = _read_json_if_exists(m_dir / "dataset_summary.json")
        training_history = _read_json_if_exists(m_dir / "training_history.json")
        eval_file = m_dir / "evaluation_report.json"
        evaluation_report = _read_json_if_exists(eval_file)

        available = bool(dataset_summary and training_history and evaluation_report)
        unavailable_reasons: List[str] = []
        if not dataset_summary:
            unavailable_reasons.append("Dataset summary (models/dataset_summary.json) has not been generated yet.")
        if not training_history:
            unavailable_reasons.append("Model training history (models/training_history.json) is unavailable.")
        if not evaluation_report:
            unavailable_reasons.append("Held-out evaluation report (models/evaluation_report.json) is unavailable.")

        return {
            "available": available,
            "model_version": MODEL_VERSION,
            "evaluated_at_utc": _iso_mtime(eval_file),
            "unavailable_reasons": unavailable_reasons,
            "dataset_summary": dataset_summary,
            "training_history": training_history,
            "evaluation_report": evaluation_report,
        }

    @app.get("/api/ml/plots/{plot_name}")
    def get_plot_image(plot_name: str):
        if plot_name not in ALLOWED_PLOTS:
            raise HTTPException(status_code=404, detail="Requested diagnostic chart is not supported.")
        plot_file = m_dir / "plots" / plot_name
        if not plot_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Plot '{plot_name}' has not been generated yet. Run evaluate_model.py first.",
            )
        return FileResponse(plot_file, media_type="image/png")

    @app.get("/api/ml/samples")
    def list_test_samples() -> Dict[str, Any]:
        manifest_path = d_dir / "sample_test_images" / "manifest.json"
        if not manifest_path.exists():
            return {"available": False, "samples": []}
        try:
            samples = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            samples = []
        return {"available": len(samples) > 0, "samples": samples}

    @app.get("/api/ml/samples/image/{filename}")
    def get_sample_image(filename: str):
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith(".jpg"):
            raise HTTPException(status_code=400, detail="Invalid sample image filename.")
        target = d_dir / "sample_test_images" / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="Sample image not found on disk.")
        return FileResponse(target, media_type="image/jpeg")

    @app.post("/predict")
    @app.post("/api/ml/predict")
    async def predict_uploaded_image(file: UploadFile = File(...)) -> Dict[str, Any]:
        raw_bytes = await file.read()
        if len(raw_bytes) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Uploaded file ({len(raw_bytes) / (1024 * 1024):.2f} MB) exceeds the "
                    f"{upload_limit_mb:.0f} MB maximum size limit."
                ),
            )
        try:
            result = predictor.predict_bytes(
                raw_bytes=raw_bytes,
                filename=file.filename or "upload.jpg",
                content_type=file.content_type,
            )
            result["model_version"] = MODEL_VERSION
            return result
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Inference pipeline error: {exc}",
            ) from exc

    @app.post("/api/ml/predict-sample/{sample_id}")
    def predict_heldout_sample(sample_id: str) -> Dict[str, Any]:
        manifest_path = d_dir / "sample_test_images" / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Held-out test samples have not been prepared yet.",
            )
        samples = json.loads(manifest_path.read_text(encoding="utf-8"))
        matched = next((s for s in samples if s["id"] == sample_id), None)
        if not matched:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

        img_path = d_dir / "sample_test_images" / matched["filename"]
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Sample image file is missing on disk.")

        try:
            prediction = predictor.predict_bytes(
                raw_bytes=img_path.read_bytes(),
                filename=matched["filename"],
                content_type="image/jpeg",
            )
            prediction["model_version"] = MODEL_VERSION
            prediction["sample_ground_truth"] = {
                "ground_truth_class": matched["ground_truth_class"],
                "display_name": matched["display_name"],
                "split": matched["split"],
                "matched_ground_truth": prediction["predicted_class"] == matched["ground_truth_class"],
            }
            return prediction
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
        assets_dir = DIST_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}")
        def serve_spa(full_path: str):
            candidate = DIST_DIR / full_path
            if full_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(DIST_DIR / "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ML_PORT", os.environ.get("PORT", "8050")))
    uvicorn.run(app, host="0.0.0.0", port=port)
