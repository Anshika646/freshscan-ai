"""
Dataset acquisition, pre-processing inspection, validation, duplicate analysis,
and leakage-safe split preparation for FreshScan AI.

Dataset Provenance:
- Title: Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality
- Standardized Mirror: Hugging Face Hub — `Project-AgML/fresh_rotten_fruit_classification` (`raw` configuration)
- Original Authors: Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin (2022)
- Publication: Data in Brief, Vol. 44, 108552 (https://doi.org/10.1016/j.dib.2022.108552)
- Primary Archive DOI: 10.17632/bdd69gyhv8.1 (Mendeley Data, V1)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Supported Fruits (8): Apple, Banana, Grape, Guava, Jujube, Orange, Pomegranate, Strawberry
- Class Labels (16):
    fresh_apple, fresh_banana, fresh_grape, fresh_guava,
    fresh_jujube, fresh_orange, fresh_pomegranate, fresh_strawberry,
    rotten_apple, rotten_banana, rotten_grape, rotten_guava,
    rotten_jujube, rotten_orange, rotten_pomegranate, rotten_strawberry
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests
from PIL import Image

from src.preprocessing import (
    ALLOWED_EXTENSIONS,
    DatasetRecord,
    InvalidImageError,
    LabelEncoderMapping,
    PreprocessingConfig,
    compute_balanced_class_weights,
    compute_dhash,
    compute_sha256_bytes,
    convert_to_rgb,
    resize_image,
    split_dataset_leakage_safe,
    validate_and_decode_bytes,
)

DATASET_METADATA: Dict[str, Any] = {
    "name": "Fresh and Rotten Fruits Dataset (Project AgML Raw Benchmark)",
    "hf_repo_id": "Project-AgML/fresh_rotten_fruit_classification",
    "hf_config": "raw",
    "hf_url": "https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification",
    "original_source": "Mendeley Data V1 (doi:10.17632/bdd69gyhv8.1) & Data in Brief 44:108552 (2022)",
    "authors": "Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin",
    "license": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
    "supported_fruits": [
        "apple",
        "banana",
        "grape",
        "guava",
        "jujube",
        "orange",
        "pomegranate",
        "strawberry",
    ],
    "freshness_states": ["fresh", "rotten"],
}

# Exact 200-row block offsets in Project-AgML/fresh_rotten_fruit_classification (raw split, 3,200 total rows)
CLASS_BLOCK_OFFSETS: List[Tuple[str, int, int]] = [
    ("fresh_apple", 0, 200),
    ("fresh_banana", 200, 200),
    ("fresh_grape", 400, 200),
    ("fresh_guava", 600, 200),
    ("fresh_jujube", 800, 200),
    ("fresh_orange", 1000, 200),
    ("fresh_pomegranate", 1200, 200),
    ("fresh_strawberry", 1400, 200),
    ("rotten_apple", 1600, 200),
    ("rotten_banana", 1800, 200),
    ("rotten_grape", 2000, 200),
    ("rotten_guava", 2200, 200),
    ("rotten_jujube", 2400, 200),
    ("rotten_orange", 2600, 200),
    ("rotten_pomegranate", 2800, 200),
    ("rotten_strawberry", 3000, 200),
]


def format_class_label(freshness: str, fruit_type: str) -> str:
    clean_fresh = str(freshness).strip().lower()
    clean_fruit = str(fruit_type).strip().lower().replace(" ", "_")
    return f"{clean_fresh}_{clean_fruit}"


def parse_class_label(class_label: str) -> Dict[str, str]:
    parts = class_label.split("_", 1)
    if len(parts) == 2 and parts[0] in ("fresh", "rotten"):
        freshness, fruit = parts[0], parts[1]
    else:
        freshness = "fresh" if "fresh" in class_label.lower() else "rotten"
        fruit = class_label
    display_fruit = fruit.replace("_", " ").title()
    display_freshness = freshness.title()
    return {
        "class_label": class_label,
        "freshness": freshness,
        "fruit_type": fruit,
        "display_name": f"{display_freshness} {display_fruit}",
    }


def _fetch_json_with_retry(url: str, timeout: int = 20, retries: int = 3) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Failed to query dataset index '{url}': {last_err}")


def _download_single_asset(task: Tuple[str, Path, int]) -> bool:
    img_url, dest_path, timeout = task
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    for attempt in range(3):
        try:
            resp = requests.get(img_url, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(1.2 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.content
            if not data:
                return False
            # Cap oversized 12-megapixel DSLR files on disk to max 1024px lossless/high-quality JPEG
            # ONLY if > 1.5 MB so disk & RAM footprint stay lean while keeping genuine photographic content.
            if len(data) > 1_500_000:
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    rgb = convert_to_rgb(im)
                    rgb.thumbnail((1024, 1024), resample=Image.Resampling.LANCZOS)
                    out_buf = io.BytesIO()
                    rgb.save(out_buf, format="JPEG", quality=93)
                    data = out_buf.getvalue()
            dest_path.write_bytes(data)
            return True
        except Exception:
            time.sleep(0.6 * (attempt + 1))
    return False


def download_agml_raw_dataset(
    raw_dir: Union[str, Path] = "data/raw",
    max_per_class: Optional[int] = 75,
    selected_fruits: Optional[Sequence[str]] = None,
    timeout: int = 25,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """
    Download genuine raw (un-augmented) fruit freshness images from
    `Project-AgML/fresh_rotten_fruit_classification` on Hugging Face Hub via
    the Hugging Face Datasets Server row index & asset CDN.

    Avoids loading monolithic ~1 GB Parquet row-groups into memory so container
    RAM stays < 100 MB throughout acquisition.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    allowed_fruits = (
        {f.strip().lower() for f in selected_fruits}
        if selected_fruits
        else set(DATASET_METADATA["supported_fruits"])
    )
    label_map_hf = {
        0: "fresh",
        1: "rotten",
        "0": "fresh",
        "1": "rotten",
        "fresh": "fresh",
        "rotten": "rotten",
    }

    target_per_class = 200 if max_per_class is None else min(200, max(10, int(max_per_class)))
    counts_per_class: Counter = Counter()
    download_tasks: List[Tuple[str, Path, int]] = []

    for expected_cls, start_offset, block_len in CLASS_BLOCK_OFFSETS:
        parsed_block = parse_class_label(expected_cls)
        if parsed_block["fruit_type"] not in allowed_fruits:
            continue

        cls_dir = raw_path / expected_cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing_files = [
            p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if len(existing_files) >= target_per_class:
            counts_per_class[expected_cls] = len(existing_files)
            continue

        needed = min(block_len, target_per_class)
        fetched_rows = 0
        while fetched_rows < needed:
            page_len = min(100, needed - fetched_rows)
            row_offset = start_offset + fetched_rows
            api_url = (
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset={DATASET_METADATA['hf_repo_id']}"
                f"&config=raw&split=train&offset={row_offset}&length={page_len}"
            )
            payload = _fetch_json_with_retry(api_url, timeout=timeout)
            rows = payload.get("rows", [])
            if not rows:
                break

            for item in rows:
                row_idx = int(item.get("row_idx", row_offset))
                row_data = item.get("row", {})
                fruit = str(row_data.get("fruit_type", "")).strip().lower()
                if fruit not in allowed_fruits:
                    continue
                freshness = label_map_hf.get(row_data.get("label"), "fresh")
                actual_cls = format_class_label(freshness, fruit)
                target_cls_dir = raw_path / actual_cls
                target_cls_dir.mkdir(parents=True, exist_ok=True)

                img_info = row_data.get("image") or {}
                src_url = img_info.get("src")
                if not src_url:
                    continue

                clean_path = src_url.split("?")[0]
                ext = Path(clean_path).suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    ext = ".jpg"

                dest_file = target_cls_dir / f"agml_raw_{row_idx:05d}{ext}"
                download_tasks.append((src_url, dest_file, timeout))

            fetched_rows += len(rows)

    print(f"[data_loader] Downloading {len(download_tasks)} raw fruit images across {len(allowed_fruits) * 2} classes...")
    succeeded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ok in pool.map(_download_single_asset, download_tasks):
            if ok:
                succeeded += 1

    # Recount actual valid files present per class directory
    counts_per_class.clear()
    for cls_dir in sorted(raw_path.iterdir()):
        if cls_dir.is_dir() and not cls_dir.name.startswith("."):
            files = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
            counts_per_class[cls_dir.name] = len(files)

    return {
        "raw_dir": str(raw_path),
        "downloaded_new_files": succeeded,
        "total_files_on_disk": sum(counts_per_class.values()),
        "counts_per_class": dict(sorted(counts_per_class.items())),
    }


def _inspect_single_file(task: Tuple[Path, str]) -> Dict[str, Any]:
    file_path, cls_name = task
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "status": "unsupported_extension",
            "path": str(file_path),
            "label": cls_name,
            "reason": f"Unsupported extension: {ext}",
        }
    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        return {
            "status": "unreadable",
            "path": str(file_path),
            "label": cls_name,
            "reason": str(exc),
        }

    if len(raw_bytes) == 0:
        return {
            "status": "empty",
            "path": str(file_path),
            "label": cls_name,
            "reason": "Zero-byte file",
        }

    try:
        img = validate_and_decode_bytes(raw_bytes, filename=file_path.name, max_bytes=40 * 1024 * 1024)
        sha256 = compute_sha256_bytes(raw_bytes)
        dhash = compute_dhash(img)
        parsed = parse_class_label(cls_name)
        return {
            "status": "valid",
            "path": str(file_path),
            "label": cls_name,
            "fruit_type": parsed["fruit_type"],
            "freshness": parsed["freshness"],
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format or ext.lstrip(".").upper(),
            "byte_size": len(raw_bytes),
            "sha256": sha256,
            "dhash": dhash,
        }
    except InvalidImageError as exc:
        return {
            "status": "corrupted",
            "path": str(file_path),
            "label": cls_name,
            "reason": str(exc),
        }


def inspect_raw_dataset(
    raw_dir: Union[str, Path] = "data/raw",
    max_workers: int = 4,
) -> Tuple[Dict[str, Any], List[DatasetRecord]]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under {root}")

    tasks: List[Tuple[Path, str]] = []
    empty_class_dirs: List[str] = []

    for cls_dir in class_dirs:
        files = sorted([f for f in cls_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
        if not files:
            empty_class_dirs.append(cls_dir.name)
        for f in files:
            tasks.append((f, cls_dir.name))

    valid_records: List[DatasetRecord] = []
    corrupted_files: List[Dict[str, str]] = []
    unsupported_files: List[Dict[str, str]] = []
    widths: List[int] = []
    heights: List[int] = []
    byte_sizes: List[int] = []
    color_modes: Counter = Counter()
    formats: Counter = Counter()
    class_counts: Counter = Counter()
    fruit_counts: Counter = Counter()
    freshness_counts: Counter = Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for res in pool.map(_inspect_single_file, tasks):
            if res["status"] == "valid":
                rec = DatasetRecord(
                    path=res["path"],
                    label=res["label"],
                    fruit_type=res["fruit_type"],
                    freshness=res["freshness"],
                    width=res["width"],
                    height=res["height"],
                    sha256=res["sha256"],
                    dhash=res["dhash"],
                )
                valid_records.append(rec)
                widths.append(res["width"])
                heights.append(res["height"])
                byte_sizes.append(res["byte_size"])
                color_modes[res["mode"]] += 1
                formats[res["format"]] += 1
                class_counts[res["label"]] += 1
                fruit_counts[res["fruit_type"]] += 1
                freshness_counts[res["freshness"]] += 1
            elif res["status"] == "unsupported_extension":
                unsupported_files.append({"path": res["path"], "reason": res["reason"]})
            else:
                corrupted_files.append({"path": res["path"], "reason": res["reason"]})

    sha_counts = Counter(r.sha256 for r in valid_records)
    dhash_counts = Counter(r.dhash for r in valid_records)
    exact_dup_groups = sum(1 for c in sha_counts.values() if c > 1)
    exact_dup_extra_files = sum(c - 1 for c in sha_counts.values() if c > 1)
    perceptual_dup_groups = sum(1 for c in dhash_counts.values() if c > 1)
    perceptual_dup_extra_files = sum(c - 1 for c in dhash_counts.values() if c > 1)

    counts_array = np.array(list(class_counts.values()), dtype=np.float64) if class_counts else np.array([0.0])
    imbalance_ratio = (
        float(round(counts_array.max() / max(counts_array.min(), 1.0), 4))
        if len(counts_array) > 0 and counts_array.min() > 0
        else 1.0
    )

    summary: Dict[str, Any] = {
        "dataset_metadata": DATASET_METADATA,
        "raw_directory": str(root),
        "discovered_classes": [d.name for d in class_dirs],
        "num_classes": len(class_counts),
        "empty_class_directories": empty_class_dirs,
        "total_files_scanned": len(tasks),
        "valid_readable_images": len(valid_records),
        "corrupted_or_unreadable_count": len(corrupted_files),
        "unsupported_file_count": len(unsupported_files),
        "corrupted_samples": corrupted_files[:20],
        "unsupported_samples": unsupported_files[:20],
        "total_size_mb": round(sum(byte_sizes) / (1024 * 1024), 2),
        "image_dimensions": {
            "width_min": int(min(widths)) if widths else 0,
            "width_max": int(max(widths)) if widths else 0,
            "width_mean": round(float(np.mean(widths)), 1) if widths else 0.0,
            "width_median": round(float(np.median(widths)), 1) if widths else 0.0,
            "height_min": int(min(heights)) if heights else 0,
            "height_max": int(max(heights)) if heights else 0,
            "height_mean": round(float(np.mean(heights)), 1) if heights else 0.0,
            "height_median": round(float(np.median(heights)), 1) if heights else 0.0,
        },
        "color_modes": dict(color_modes),
        "file_formats": dict(formats),
        "duplicates": {
            "exact_sha256_duplicate_groups": exact_dup_groups,
            "exact_sha256_duplicate_images": exact_dup_extra_files,
            "perceptual_dhash_duplicate_groups": perceptual_dup_groups,
            "perceptual_dhash_duplicate_images": perceptual_dup_extra_files,
        },
        "class_distribution": dict(sorted(class_counts.items())),
        "fruit_distribution": dict(sorted(fruit_counts.items())),
        "freshness_distribution": dict(sorted(freshness_counts.items())),
        "imbalance_ratio_max_to_min": imbalance_ratio,
    }
    return summary, valid_records


def _load_and_resize_uint8(
    record: DatasetRecord,
    config: PreprocessingConfig,
) -> np.ndarray:
    raw_bytes = Path(record.path).read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.load()
        rgb = convert_to_rgb(img, bg_color=config.alpha_composite_bg)
        resized = resize_image(rgb, target_size=(config.input_height, config.input_width))
        return np.asarray(resized, dtype=np.uint8)


def prepare_dataset_splits_and_cache(
    raw_dir: Union[str, Path] = "data/raw",
    processed_dir: Union[str, Path] = "data/processed",
    models_dir: Union[str, Path] = "models",
    sample_test_dir: Union[str, Path] = "data/sample_test_images",
    config: Optional[PreprocessingConfig] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_workers: int = 4,
) -> Dict[str, Any]:
    cfg = config or PreprocessingConfig()
    proc_path = Path(processed_dir)
    mod_path = Path(models_dir)
    sample_path = Path(sample_test_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)

    summary, valid_records = inspect_raw_dataset(raw_dir=raw_dir, max_workers=max_workers)
    encoder = LabelEncoderMapping(summary["class_distribution"].keys())
    encoder.save(mod_path / "class_indices.json")
    cfg.save(mod_path / "preprocessing_config.json")

    split_out = split_dataset_leakage_safe(
        valid_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=cfg.random_seed,
    )

    train_recs: List[DatasetRecord] = split_out["train"]
    val_recs: List[DatasetRecord] = split_out["val"]
    test_recs: List[DatasetRecord] = split_out["test"]

    y_train = encoder.encode_many([r.label for r in train_recs])
    y_val = encoder.encode_many([r.label for r in val_recs])
    y_test = encoder.encode_many([r.label for r in test_recs])

    class_weights = compute_balanced_class_weights(y_train, num_classes=encoder.num_classes)

    def _materialize_split(name: str, recs: List[DatasetRecord], labels: np.ndarray) -> str:
        print(f"[data_loader] Materializing {name} split ({len(recs)} images -> {cfg.input_height}x{cfg.input_width} RGB)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            arrays = list(pool.map(lambda r: _load_and_resize_uint8(r, cfg), recs))
        x_arr = np.stack(arrays, axis=0).astype(np.uint8)
        out_npz = proc_path / f"{name}_split.npz"
        np.savez_compressed(
            out_npz,
            images=x_arr,
            labels=labels.astype(np.int64),
            paths=np.array([r.path for r in recs]),
            sha256=np.array([r.sha256 for r in recs]),
        )
        return str(out_npz)

    train_npz = _materialize_split("train", train_recs, y_train)
    val_npz = _materialize_split("val", val_recs, y_val)
    test_npz = _materialize_split("test", test_recs, y_test)

    if sample_path.exists():
        shutil.rmtree(sample_path)
    sample_path.mkdir(parents=True, exist_ok=True)
    exported_samples: List[Dict[str, Any]] = []
    seen_sample_classes = set()

    for rec in test_recs:
        if rec.label in seen_sample_classes:
            continue
        seen_sample_classes.add(rec.label)
        src_file = Path(rec.path)
        with Image.open(src_file) as im:
            im.load()
            rgb = convert_to_rgb(im, bg_color=cfg.alpha_composite_bg)
            rgb.thumbnail((560, 560), resample=Image.Resampling.LANCZOS)
            sample_filename = f"test_{rec.label}.jpg"
            dest_file = sample_path / sample_filename
            rgb.save(dest_file, format="JPEG", quality=90)

        parsed = parse_class_label(rec.label)
        exported_samples.append(
            {
                "id": rec.label,
                "filename": sample_filename,
                "ground_truth_class": rec.label,
                "display_name": parsed["display_name"],
                "fruit_type": parsed["fruit_type"],
                "freshness": parsed["freshness"],
                "split": "test (held-out)",
                "original_dimensions": f"{rec.width}x{rec.height}",
            }
        )

    exported_samples.sort(key=lambda s: s["id"])
    (sample_path / "manifest.json").write_text(
        json.dumps(exported_samples, indent=2),
        encoding="utf-8",
    )

    def _split_class_counts(recs: List[DatasetRecord]) -> Dict[str, int]:
        c = Counter(r.label for r in recs)
        return {cls: int(c.get(cls, 0)) for cls in encoder.class_names}

    summary["splits"] = {
        "seed": cfg.random_seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
        },
        "per_class": {
            "train": _split_class_counts(train_recs),
            "val": _split_class_counts(val_recs),
            "test": _split_class_counts(test_recs),
        },
        "deduplication_and_leakage": split_out["dedup_stats"],
        "class_weights": {encoder.decode(k): v for k, v in class_weights.items()},
        "class_weights_by_index": {str(k): v for k, v in class_weights.items()},
        "artifacts": {
            "train_npz": train_npz,
            "val_npz": val_npz,
            "test_npz": test_npz,
        },
    }

    (mod_path / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
