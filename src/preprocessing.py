"""
Reusable, testable image preprocessing, duplicate detection, label encoding,
leakage-safe dataset splitting, and class-weight computation for FreshScan AI.

Preprocessing Decisions Documented:
1. Supported Formats: JPEG, PNG, WebP, BMP. Validated via file extension check,
   non-zero byte check, PIL header verification, and full pixel decoding to catch
   truncated or corrupted images early.
2. Consistent RGB Conversion: Images with alpha channels (RGBA, LA, or P with
   transparency) are composited onto a neutral white (#FFFFFF) canvas prior to
   converting to 3-channel RGB. Direct `.convert("RGB")` on RGBA replaces transparent
   backgrounds with black (0, 0, 0), introducing artificial dark borders that can
   mimic necrotic/rotten tissue.
3. Input Resolution: 224x224 pixels using bilinear resampling, matching the standard
   ImageNet pretraining resolution of MobileNetV2.
4. Pixel Normalization: MobileNetV2 requires pixel intensities scaled to [-1.0, 1.0]
   via `(pixel / 127.5) - 1.0` (`tf.keras.applications.mobilenet_v2.preprocess_input`).
5. Leakage Prevention: Exact byte duplicates (SHA-256) are deduplicated prior to
   splitting, while perceptual near-duplicates (64-bit difference hash, dHash) are
   grouped together so near-identical captures never span across train/val/test splits.
6. Augmentation Discipline: Applied strictly to training batches. Moderate geometric
   transforms (horizontal flip, +/-8% rotation, +/-10% zoom) and mild contrast jitter
   (+/-10%) are used. Hue shifts are intentionally excluded because surface color is
   a primary diagnostic signal for fruit oxidation, bruising, and mold.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from sklearn.utils.class_weight import compute_class_weight

ImageFile.LOAD_TRUNCATED_IMAGES = False

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024


class InvalidImageError(ValueError):
    """Raised when an image file or byte buffer is missing, unsupported, or corrupted."""


@dataclass(frozen=True)
class PreprocessingConfig:
    model_family: str = "mobilenet_v2"
    input_height: int = 224
    input_width: int = 224
    channels: int = 3
    color_mode: str = "RGB"
    alpha_composite_bg: Tuple[int, int, int] = (255, 255, 255)
    normalization: str = "mobilenet_v2_minus1_to_1"
    interpolation: str = "bilinear"
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["alpha_composite_bg"] = list(self.alpha_composite_bg)
        return data

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreprocessingConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "alpha_composite_bg" in raw:
            raw["alpha_composite_bg"] = tuple(raw["alpha_composite_bg"])
        return cls(**raw)


def validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file extension '{ext or '(none)'}'. "
            f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )


def validate_and_decode_bytes(
    raw_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Image.Image:
    if not raw_bytes or len(raw_bytes) == 0:
        raise InvalidImageError("Uploaded image file is empty (0 bytes).")

    if len(raw_bytes) > max_bytes:
        raise InvalidImageError(
            f"Image file size ({len(raw_bytes) / (1024 * 1024):.2f} MB) exceeds "
            f"maximum limit of {max_bytes / (1024 * 1024):.0f} MB."
        )

    if filename:
        validate_extension(filename)

    if content_type:
        normalized_mime = content_type.split(";")[0].strip().lower()
        if (
            normalized_mime
            and normalized_mime not in ALLOWED_MIME_TYPES
            and normalized_mime != "application/octet-stream"
        ):
            raise InvalidImageError(
                f"Unsupported MIME type '{normalized_mime}'. "
                f"Supported types: JPEG, PNG, WebP, BMP."
            )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(f"Corrupted or unreadable image data: {exc}") from exc

    if img.width < 16 or img.height < 16:
        raise InvalidImageError(
            f"Image dimensions ({img.width}x{img.height}) are too small for reliable classification (minimum 16x16)."
        )

    return img


def validate_and_load_file(path: Union[str, Path]) -> Image.Image:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise InvalidImageError(f"Image file not found: {file_path}")
    validate_extension(file_path.name)
    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        raise InvalidImageError(f"Unable to read file '{file_path}': {exc}") from exc
    return validate_and_decode_bytes(raw_bytes, filename=file_path.name)


def convert_to_rgb(
    img: Image.Image,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if img.mode == "RGB":
        return img.copy()

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, bg_color)
        canvas.paste(rgba, mask=rgba.split()[3])
        return canvas

    return img.convert("RGB")


def resize_image(
    img: Image.Image,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    height, width = target_size
    if img.mode != "RGB":
        img = convert_to_rgb(img)
    return img.resize((width, height), resample=Image.Resampling.BILINEAR)


def normalize_mobilenet_v2(pixel_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixel_array, dtype=np.float32)
    return (arr / 127.5) - 1.0


def preprocess_pil_image(
    img: Image.Image,
    config: Optional[PreprocessingConfig] = None,
    expand_batch: bool = True,
) -> np.ndarray:
    cfg = config or PreprocessingConfig()
    rgb = convert_to_rgb(img, bg_color=cfg.alpha_composite_bg)
    resized = resize_image(rgb, target_size=(cfg.input_height, cfg.input_width))
    arr = np.asarray(resized, dtype=np.float32)
    normalized = normalize_mobilenet_v2(arr)
    if expand_batch:
        return np.expand_dims(normalized, axis=0)
    return normalized


def preprocess_image_bytes(
    raw_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    config: Optional[PreprocessingConfig] = None,
    expand_batch: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = validate_and_decode_bytes(raw_bytes, filename=filename, content_type=content_type)
    metadata = {
        "original_width": img.width,
        "original_height": img.height,
        "original_mode": img.mode,
        "original_format": img.format or "UNKNOWN",
        "byte_size": len(raw_bytes),
    }
    tensor = preprocess_pil_image(img, config=config, expand_batch=expand_batch)
    return tensor, metadata


def compute_sha256_bytes(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    gray = convert_to_rgb(img).convert("L").resize(
        (hash_size + 1, hash_size),
        resample=Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


class LabelEncoderMapping:
    def __init__(self, class_names: Iterable[str]) -> None:
        unique_sorted = sorted({str(c).strip() for c in class_names if str(c).strip()})
        if len(unique_sorted) < 2:
            raise ValueError("At least 2 distinct class labels are required.")
        self.class_names: List[str] = unique_sorted
        self.class_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.index_to_class: Dict[int, str] = {
            idx: name for idx, name in enumerate(self.class_names)
        }

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def encode(self, label: str) -> int:
        if label not in self.class_to_index:
            raise KeyError(f"Unknown class label: '{label}'")
        return self.class_to_index[label]

    def encode_many(self, labels: Sequence[str]) -> np.ndarray:
        return np.array([self.encode(lbl) for lbl in labels], dtype=np.int64)

    def decode(self, index: int) -> str:
        idx = int(index)
        if idx not in self.index_to_class:
            raise KeyError(f"Unknown class index: {idx}")
        return self.index_to_class[idx]

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_to_index": self.class_to_index,
            "index_to_class": {str(k): v for k, v in self.index_to_class.items()},
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LabelEncoderMapping":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(payload["class_names"])
        instance.class_names = list(payload["class_names"])
        instance.class_to_index = {k: int(v) for k, v in payload["class_to_index"].items()}
        instance.index_to_class = {int(k): v for k, v in payload["index_to_class"].items()}
        return instance


@dataclass
class DatasetRecord:
    path: str
    label: str
    fruit_type: str
    freshness: str
    width: int
    height: int
    sha256: str
    dhash: str


def split_dataset_leakage_safe(
    records: Sequence[DatasetRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = np.random.default_rng(seed)

    sha_to_records: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for rec in records:
        sha_to_records[rec.sha256].append(rec)

    deduped_records: List[DatasetRecord] = []
    exact_duplicates_removed = 0
    conflicting_label_duplicates = 0

    for _sha, group in sorted(sha_to_records.items(), key=lambda x: x[0]):
        distinct_labels = {r.label for r in group}
        if len(distinct_labels) > 1:
            conflicting_label_duplicates += len(group)
            continue
        deduped_records.append(group[0])
        if len(group) > 1:
            exact_duplicates_removed += len(group) - 1

    dhash_groups: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for rec in deduped_records:
        dhash_groups[rec.dhash].append(rec)

    near_duplicate_clusters = sum(1 for g in dhash_groups.values() if len(g) > 1)
    near_duplicate_extra_files = sum(len(g) - 1 for g in dhash_groups.values() if len(g) > 1)

    class_to_clusters: Dict[str, List[List[DatasetRecord]]] = defaultdict(list)
    cross_class_clusters: List[List[DatasetRecord]] = []

    for _dhash_key, cluster in sorted(dhash_groups.items(), key=lambda x: x[0]):
        labels_in_cluster = {r.label for r in cluster}
        if len(labels_in_cluster) == 1:
            label = next(iter(labels_in_cluster))
            class_to_clusters[label].append(cluster)
        else:
            cross_class_clusters.append(cluster)

    train_records: List[DatasetRecord] = []
    val_records: List[DatasetRecord] = []
    test_records: List[DatasetRecord] = []

    for label in sorted(class_to_clusters.keys()):
        clusters = list(class_to_clusters[label])
        indices = rng.permutation(len(clusters))
        shuffled_clusters = [clusters[i] for i in indices]

        total_items = sum(len(c) for c in shuffled_clusters)
        target_test = max(1, int(round(total_items * test_ratio)))
        target_val = max(1, int(round(total_items * val_ratio)))

        n_clusters = len(shuffled_clusters)
        if n_clusters < 3:
            raise ValueError(
                f"Class '{label}' has only {n_clusters} unique perceptual groups; "
                f"need at least 3 to populate train, val, and test without leakage."
            )

        cls_test: List[DatasetRecord] = []
        cls_val: List[DatasetRecord] = []
        cls_train: List[DatasetRecord] = []

        cursor = 0
        while cursor < n_clusters - 2 and len(cls_test) < target_test:
            cls_test.extend(shuffled_clusters[cursor])
            cursor += 1

        while cursor < n_clusters - 1 and len(cls_val) < target_val:
            cls_val.extend(shuffled_clusters[cursor])
            cursor += 1

        while cursor < n_clusters:
            cls_train.extend(shuffled_clusters[cursor])
            cursor += 1

        test_records.extend(cls_test)
        val_records.extend(cls_val)
        train_records.extend(cls_train)

    for cluster in cross_class_clusters:
        train_records.extend(cluster)

    def _shuffle_split(items: List[DatasetRecord]) -> List[DatasetRecord]:
        if not items:
            return items
        order = rng.permutation(len(items))
        return [items[i] for i in order]

    train_records = _shuffle_split(train_records)
    val_records = _shuffle_split(val_records)
    test_records = _shuffle_split(test_records)

    train_shas = {r.sha256 for r in train_records}
    val_shas = {r.sha256 for r in val_records}
    test_shas = {r.sha256 for r in test_records}
    assert not (train_shas & val_shas), "SHA-256 leakage between train and val!"
    assert not (train_shas & test_shas), "SHA-256 leakage between train and test!"
    assert not (val_shas & test_shas), "SHA-256 leakage between val and test!"

    train_dhashes = {r.dhash for r in train_records}
    val_dhashes = {r.dhash for r in val_records}
    test_dhashes = {r.dhash for r in test_records}
    assert not (train_dhashes & val_dhashes), "Perceptual dHash leakage between train and val!"
    assert not (train_dhashes & test_dhashes), "Perceptual dHash leakage between train and test!"
    assert not (val_dhashes & test_dhashes), "Perceptual dHash leakage between val and test!"

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "dedup_stats": {
            "input_valid_images": len(records),
            "exact_duplicates_removed": exact_duplicates_removed,
            "conflicting_label_duplicates_quarantined": conflicting_label_duplicates,
            "near_duplicate_clusters": near_duplicate_clusters,
            "near_duplicate_extra_images_grouped": near_duplicate_extra_files,
            "cross_class_perceptual_clusters": len(cross_class_clusters),
            "final_clean_images": len(train_records) + len(val_records) + len(test_records),
            "leakage_verified_zero_overlap": True,
        },
    }


def compute_balanced_class_weights(
    encoded_train_labels: Sequence[int],
    num_classes: int,
) -> Dict[int, float]:
    y = np.asarray(encoded_train_labels, dtype=np.int64)
    present_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y,
    )
    weight_map: Dict[int, float] = {idx: 1.0 for idx in range(num_classes)}
    for cls_idx, w in zip(present_classes, weights):
        weight_map[int(cls_idx)] = float(round(float(w), 6))
    return weight_map


def build_training_augmentation_layer(seed: int = 42):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.08, fill_mode="reflect", seed=seed + 1),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.10, 0.10),
                width_factor=(-0.10, 0.10),
                fill_mode="reflect",
                seed=seed + 2,
            ),
            tf.keras.layers.RandomContrast(factor=0.10, seed=seed + 3),
        ],
        name="training_augmentation",
    )
