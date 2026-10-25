"""
Genuine TensorFlow/Keras inference pipeline for FreshScan AI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.data_loader import parse_class_label
from src.preprocessing import (
    LabelEncoderMapping,
    PreprocessingConfig,
    preprocess_image_bytes,
)

LOW_CONFIDENCE_THRESHOLD = 0.60
MODERATE_CONFIDENCE_THRESHOLD = 0.80

FOOD_SAFETY_DISCLAIMER = (
    "Visual inspection cannot detect internal decay, microbial pathogens (such as Salmonella, "
    "E. coli, or Listeria), mycotoxins, or chemical contamination. Do not rely solely on image "
    "predictions to guarantee food safety—always inspect odor, firmness, and storage history."
)


class ModelUnavailableError(FileNotFoundError):
    """Raised when the trained model or required metadata files are missing on disk."""


class FreshScanPredictor:
    def __init__(
        self,
        model_path: Union[str, Path] = "models/freshscan_mobilenetv2.keras",
        class_map_path: Union[str, Path] = "models/class_indices.json",
        config_path: Union[str, Path] = "models/preprocessing_config.json",
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        self.config_path = Path(config_path)
        self.low_confidence_threshold = low_confidence_threshold

        self._model = None
        self._encoder: Optional[LabelEncoderMapping] = None
        self._config: Optional[PreprocessingConfig] = None

    def is_ready(self) -> bool:
        return (
            self.model_path.exists()
            and self.class_map_path.exists()
            and self.config_path.exists()
        )

    def missing_artifacts(self) -> List[str]:
        missing = []
        for p in (self.model_path, self.class_map_path, self.config_path):
            if not p.exists():
                missing.append(str(p))
        return missing

    def load(self) -> None:
        missing = self.missing_artifacts()
        if missing:
            raise ModelUnavailableError(
                f"Trained model artifacts not found: {', '.join(missing)}. "
                f"Run `python train_model.py` to train and save the model first."
            )

        import tensorflow as tf

        self._config = PreprocessingConfig.load(self.config_path)
        self._encoder = LabelEncoderMapping.load(self.class_map_path)
        self._model = tf.keras.models.load_model(self.model_path, compile=False)

        output_shape = self._model.output_shape
        num_outputs = int(output_shape[-1])
        if num_outputs != self._encoder.num_classes:
            raise RuntimeError(
                f"Model output dimension ({num_outputs}) does not match "
                f"class mapping count ({self._encoder.num_classes})."
            )

    @property
    def encoder(self) -> LabelEncoderMapping:
        if self._encoder is None:
            self.load()
        assert self._encoder is not None
        return self._encoder

    @property
    def config(self) -> PreprocessingConfig:
        if self._config is None:
            self.load()
        assert self._config is not None
        return self._config

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def predict_bytes(
        self,
        raw_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if self._model is None:
            self.load()

        t0 = time.perf_counter()
        batch_tensor, img_meta = preprocess_image_bytes(
            raw_bytes=raw_bytes,
            filename=filename,
            content_type=content_type,
            config=self.config,
            expand_batch=True,
        )

        raw_preds = self.model.predict(batch_tensor, verbose=0)
        probs = np.asarray(raw_preds[0], dtype=np.float64)

        if not np.isclose(np.sum(probs), 1.0, atol=1e-2) or np.any(probs < 0.0):
            shifted = probs - np.max(probs)
            exp_p = np.exp(shifted)
            probs = exp_p / np.sum(exp_p)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        ranked_indices = np.argsort(probs)[::-1]
        top_idx = int(ranked_indices[0])
        top_prob = float(probs[top_idx])
        top_class = self.encoder.decode(top_idx)
        parsed_top = parse_class_label(top_class)

        fresh_prob = 0.0
        rotten_prob = 0.0
        top_predictions: List[Dict[str, Any]] = []

        for rank, idx in enumerate(ranked_indices):
            cls_name = self.encoder.decode(int(idx))
            p_val = float(probs[int(idx)])
            parsed = parse_class_label(cls_name)
            if parsed["freshness"] == "fresh":
                fresh_prob += p_val
            else:
                rotten_prob += p_val

            if rank < top_k:
                top_predictions.append(
                    {
                        "class_index": int(idx),
                        "class_label": cls_name,
                        "display_name": parsed["display_name"],
                        "fruit_type": parsed["fruit_type"],
                        "freshness": parsed["freshness"],
                        "confidence": round(p_val, 6),
                    }
                )

        clipped = np.clip(probs, 1e-12, 1.0)
        normalized_entropy = float(-np.sum(clipped * np.log(clipped)) / np.log(len(probs)))

        is_low_confidence = top_prob < self.low_confidence_threshold
        if is_low_confidence:
            confidence_tier = "low"
            confidence_warning = (
                f"Low prediction confidence ({top_prob * 100:.1f}% < {self.low_confidence_threshold * 100:.0f}% threshold). "
                "The uploaded image may contain an unsupported fruit species, severe occlusion, unusual lighting, "
                "or ambiguous ripening stage. Please retake a clear, well-lit close-up of a single supported fruit."
            )
        elif top_prob < MODERATE_CONFIDENCE_THRESHOLD:
            confidence_tier = "moderate"
            confidence_warning = (
                "Moderate model confidence. Verify under even lighting and inspect multiple angles of the fruit."
            )
        else:
            confidence_tier = "high"
            confidence_warning = None

        return {
            "predicted_class": top_class,
            "predicted_index": top_idx,
            "display_name": parsed_top["display_name"],
            "fruit_type": parsed_top["fruit_type"],
            "freshness_status": parsed_top["freshness"],
            "confidence": round(top_prob, 6),
            "confidence_tier": confidence_tier,
            "is_low_confidence": is_low_confidence,
            "confidence_warning": confidence_warning,
            "aggregated_freshness": {
                "fresh_probability": round(fresh_prob, 6),
                "rotten_probability": round(rotten_prob, 6),
            },
            "normalized_entropy": round(normalized_entropy, 4),
            "top_predictions": top_predictions,
            "image_metadata": img_meta,
            "preprocessing": {
                "target_size": [self.config.input_height, self.config.input_width],
                "color_mode": self.config.color_mode,
                "normalization": self.config.normalization,
            },
            "inference_time_ms": elapsed_ms,
            "food_safety_disclaimer": FOOD_SAFETY_DISCLAIMER,
        }
