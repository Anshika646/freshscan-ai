"""
Comprehensive test suite for FreshScan AI covering:
1. Image validation, corrupted/unsupported input rejection, and RGB alpha compositing
2. Resizing and MobileNetV2 normalization output shape & exact numerical parity
3. Deterministic label encoding, decoding, and JSON serialization
4. Dataset splitting, exact SHA-256 deduplication, and perceptual dHash leakage prevention
5. Model loading, output dimension validation, and probability calibration
6. FastAPI prediction endpoint responses, invalid-input HTTP 400 handling, and missing-model HTTP 503 handling
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient
from PIL import Image

from app import create_app
from src.predict import FOOD_SAFETY_DISCLAIMER, FreshScanPredictor
from src.preprocessing import (
    DatasetRecord,
    InvalidImageError,
    LabelEncoderMapping,
    PreprocessingConfig,
    compute_balanced_class_weights,
    convert_to_rgb,
    preprocess_pil_image,
    split_dataset_leakage_safe,
    validate_and_decode_bytes,
)


def _encode_pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_validate_rejects_empty_and_corrupted_bytes():
    with pytest.raises(InvalidImageError, match="empty"):
        validate_and_decode_bytes(b"", filename="apple.jpg")

    with pytest.raises(InvalidImageError, match="Corrupted"):
        validate_and_decode_bytes(b"not-a-valid-jpeg-stream-12345", filename="apple.jpg")


def test_validate_rejects_unsupported_extensions_and_tiny_images():
    valid_img = Image.new("RGB", (64, 64), color=(180, 40, 40))
    png_bytes = _encode_pil_to_bytes(valid_img, fmt="PNG")

    with pytest.raises(InvalidImageError, match="Unsupported file extension"):
        validate_and_decode_bytes(png_bytes, filename="malware.exe")

    with pytest.raises(InvalidImageError, match="Unsupported MIME type"):
        validate_and_decode_bytes(png_bytes, filename="apple.png", content_type="application/pdf")

    tiny_img = Image.new("RGB", (8, 8), color=(100, 200, 100))
    tiny_bytes = _encode_pil_to_bytes(tiny_img, fmt="PNG")
    with pytest.raises(InvalidImageError, match="too small"):
        validate_and_decode_bytes(tiny_bytes, filename="tiny.png")


def test_convert_to_rgb_composites_transparency_onto_white():
    rgba = Image.new("RGBA", (32, 32), color=(0, 0, 0, 0))
    rgb = convert_to_rgb(rgba)
    assert rgb.mode == "RGB"
    arr = np.asarray(rgb)
    assert arr.shape == (32, 32, 3)
    assert np.all(arr == 255)

    gray = Image.new("L", (40, 30), color=128)
    rgb_from_gray = convert_to_rgb(gray)
    assert rgb_from_gray.mode == "RGB"
    assert np.asarray(rgb_from_gray).shape == (30, 40, 3)


def test_preprocessing_shape_and_mobilenetv2_parity():
    rng = np.random.default_rng(123)
    random_pixels = rng.integers(0, 256, size=(310, 470, 3), dtype=np.uint8)
    pil_img = Image.fromarray(random_pixels, mode="RGB")

    cfg = PreprocessingConfig(input_height=224, input_width=224)
    batch_tensor = preprocess_pil_image(pil_img, config=cfg, expand_batch=True)

    assert batch_tensor.shape == (1, 224, 224, 3)
    assert batch_tensor.dtype == np.float32
    assert float(batch_tensor.min()) >= -1.0 - 1e-6
    assert float(batch_tensor.max()) <= 1.0 + 1e-6

    resized_rgb = np.asarray(pil_img.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
    keras_ref = tf.keras.applications.mobilenet_v2.preprocess_input(resized_rgb.copy())
    np.testing.assert_allclose(batch_tensor[0], keras_ref, rtol=1e-5, atol=1e-6)


def test_label_encoder_mapping_roundtrip(tmp_path: Path):
    labels = ["rotten_banana", "fresh_apple", "fresh_banana", "rotten_apple", "fresh_apple"]
    encoder = LabelEncoderMapping(labels)

    assert encoder.num_classes == 4
    assert encoder.class_names == ["fresh_apple", "fresh_banana", "rotten_apple", "rotten_banana"]

    encoded = encoder.encode_many(["rotten_banana", "fresh_apple"])
    assert encoded.tolist() == [3, 0]
    assert encoder.decode(3) == "rotten_banana"
    assert encoder.decode(0) == "fresh_apple"

    map_path = tmp_path / "class_indices.json"
    encoder.save(map_path)
    reloaded = LabelEncoderMapping.load(map_path)
    assert reloaded.class_to_index == encoder.class_to_index
    assert reloaded.index_to_class == encoder.index_to_class

    with pytest.raises(KeyError):
        reloaded.encode("unknown_dragonfruit")


def test_split_dataset_prevents_exact_and_perceptual_leakage():
    records = []
    classes = ["fresh_apple", "rotten_apple", "fresh_banana", "rotten_banana"]

    for cls_idx, cls_name in enumerate(classes):
        freshness, fruit = cls_name.split("_", 1)
        for i in range(20):
            sha = f"{cls_name}_sha_{i:03d}"
            dhash = f"{cls_idx:04x}{i:012x}"
            records.append(
                DatasetRecord(
                    path=f"/fake/{cls_name}/img_{i}.jpg",
                    label=cls_name,
                    fruit_type=fruit,
                    freshness=freshness,
                    width=500,
                    height=500,
                    sha256=sha,
                    dhash=dhash,
                )
            )

        records.append(
            DatasetRecord(
                path=f"/fake/{cls_name}/exact_copy.jpg",
                label=cls_name,
                fruit_type=fruit,
                freshness=freshness,
                width=500,
                height=500,
                sha256=f"{cls_name}_sha_000",
                dhash=f"{cls_idx:04x}{0:012x}",
            )
        )

        records.append(
            DatasetRecord(
                path=f"/fake/{cls_name}/recompressed_near_dup.jpg",
                label=cls_name,
                fruit_type=fruit,
                freshness=freshness,
                width=498,
                height=498,
                sha256=f"{cls_name}_sha_recompressed",
                dhash=f"{cls_idx:04x}{1:012x}",
            )
        )

    result = split_dataset_leakage_safe(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert result["dedup_stats"]["exact_duplicates_removed"] == 4
    assert result["dedup_stats"]["leakage_verified_zero_overlap"] is True

    train_dhashes = {r.dhash for r in result["train"]}
    val_dhashes = {r.dhash for r in result["val"]}
    test_dhashes = {r.dhash for r in result["test"]}

    assert train_dhashes.isdisjoint(val_dhashes)
    assert train_dhashes.isdisjoint(test_dhashes)
    assert val_dhashes.isdisjoint(test_dhashes)

    imbalanced_y = [0] * 50 + [1] * 10
    weights = compute_balanced_class_weights(imbalanced_y, num_classes=2)
    assert weights[1] > weights[0]
    assert np.isclose(weights[1] / weights[0], 5.0)


@pytest.fixture()
def temp_model_bundle(tmp_path: Path) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    classes = ["fresh_apple", "fresh_banana", "rotten_apple", "rotten_banana"]
    encoder = LabelEncoderMapping(classes)
    encoder.save(models_dir / "class_indices.json")

    cfg = PreprocessingConfig(input_height=224, input_width=224)
    cfg.save(models_dir / "preprocessing_config.json")

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(encoder.num_classes, activation="softmax")(x)
    tiny_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    tiny_model.save(models_dir / "freshscan_mobilenetv2.keras")

    return tmp_path


def test_predictor_loads_and_matches_output_dimensions(temp_model_bundle: Path):
    m_dir = temp_model_bundle / "models"
    predictor = FreshScanPredictor(
        model_path=m_dir / "freshscan_mobilenetv2.keras",
        class_map_path=m_dir / "class_indices.json",
        config_path=m_dir / "preprocessing_config.json",
    )
    assert predictor.is_ready() is True
    predictor.load()
    assert predictor.model.output_shape == (None, 4)

    img = Image.new("RGB", (300, 300), color=(210, 60, 50))
    jpg_bytes = _encode_pil_to_bytes(img, fmt="JPEG")
    out = predictor.predict_bytes(jpg_bytes, filename="test_apple.jpg", content_type="image/jpeg")

    assert out["predicted_class"] in predictor.encoder.class_names
    assert 0.0 <= out["confidence"] <= 1.0
    assert np.isclose(
        out["aggregated_freshness"]["fresh_probability"]
        + out["aggregated_freshness"]["rotten_probability"],
        1.0,
        atol=1e-4,
    )
    assert out["food_safety_disclaimer"] == FOOD_SAFETY_DISCLAIMER


def test_api_predict_and_invalid_input_handling(temp_model_bundle: Path):
    test_app = create_app(
        models_dir=temp_model_bundle / "models",
        data_dir=temp_model_bundle / "data",
    )
    client = TestClient(test_app)

    health = client.get("/api/ml/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True

    img = Image.new("RGBA", (180, 180), color=(120, 200, 80, 255))
    png_bytes = _encode_pil_to_bytes(img, fmt="PNG")
    resp = client.post(
        "/api/ml/predict",
        files={"file": ("fresh_fruit.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_class" in body
    assert "confidence" in body
    assert "food_safety_disclaimer" in body

    bad_ext = client.post(
        "/api/ml/predict",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert bad_ext.status_code == 400
    assert "Unsupported" in bad_ext.json()["detail"]

    corrupt = client.post(
        "/api/ml/predict",
        files={"file": ("broken.jpg", b"\xff\xd8\xff\xe0corrupted", "image/jpeg")},
    )
    assert corrupt.status_code == 400


def test_api_handles_missing_model_gracefully(tmp_path: Path):
    empty_app = create_app(models_dir=tmp_path / "empty_models", data_dir=tmp_path / "empty_data")
    client = TestClient(empty_app)

    health = client.get("/api/ml/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is False

    analytics = client.get("/api/ml/analytics")
    assert analytics.status_code == 200
    assert analytics.json()["available"] is False

    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    resp = client.post(
        "/api/ml/predict",
        files={"file": ("sample.jpg", _encode_pil_to_bytes(img, "JPEG"), "image/jpeg")},
    )
    assert resp.status_code == 503
    assert "train_model.py" in resp.json()["detail"]
