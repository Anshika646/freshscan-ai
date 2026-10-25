#!/usr/bin/env python3
"""
FastAPI backend for FreshScan AI — Fruit Freshness Detection.
"""

from __future__ import annotations

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

ALLOWED_PLOTS = {
    "confusion_matrix.png",
    "training_curves.png",
    "class_distribution.png",
}


def create_app(
    models_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> FastAPI:
    m_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    d_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    predictor = FreshScanPredictor(
        model_path=m_dir / "freshscan_mobilenetv2.keras",
        class_map_path=m_dir / "class_indices.json",
        config_path=m_dir / "preprocessing_config.json",
    )

    app = FastAPI(
        title="FreshScan AI Inference API",
        description="Real-time MobileNetV2 fruit freshness classification and evaluation analytics.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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

    @app.get("/api/ml/health")
    def health_check() -> Dict[str, Any]:
        import tensorflow as tf

        ready = predictor.is_ready()
        missing = predictor.missing_artifacts()
        classes: List[str] = []
        if (m_dir / "class_indices.json").exists():
            try:
                classes = predictor.encoder.class_names
            except Exception:
                classes = []

        return {
            "status": "ok",
            "service": "freshscan-ml-backend",
            "tensorflow_version": tf.__version__,
            "model_ready": ready,
            "missing_artifacts": missing,
            "supported_extensions": sorted(ALLOWED_EXTENSIONS),
            "num_classes": len(classes),
            "classes": classes,
            "food_safety_disclaimer": FOOD_SAFETY_DISCLAIMER,
        }

    @app.get("/api/ml/analytics")
    def get_analytics() -> Dict[str, Any]:
        dataset_summary = _read_json_if_exists(m_dir / "dataset_summary.json")
        training_history = _read_json_if_exists(m_dir / "training_history.json")
        evaluation_report = _read_json_if_exists(m_dir / "evaluation_report.json")

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

    @app.post("/api/ml/predict")
    async def predict_uploaded_image(file: UploadFile = File(...)) -> Dict[str, Any]:
        raw_bytes = await file.read()
        try:
            return predictor.predict_bytes(
                raw_bytes=raw_bytes,
                filename=file.filename or "upload.jpg",
                content_type=file.content_type,
            )
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
