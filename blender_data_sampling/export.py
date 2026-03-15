from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import yaml

from blender_data_sampling.common import atomic_write_json, ensure_directory
from blender_data_sampling.contracts import GenerationManifest


def _split_samples(samples: list, val_size: float, test_size: float, seed: int | None) -> dict[str, list]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    total = len(shuffled)
    val_count = int(total * val_size)
    test_count = int(total * test_size)
    train_count = total - val_count - test_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def _normalise_bbox(width: int, height: int, bbox) -> str:
    cx = (bbox.x + bbox.width / 2) / width
    cy = (bbox.y + bbox.height / 2) / height
    w = bbox.width / width
    h = bbox.height / height
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _write_label_file(path: Path, class_to_index: dict[str, int], sample) -> None:
    lines = []
    for bbox in sample.bboxes:
        class_id = class_to_index[bbox.class_name]
        lines.append(f"{class_id} {_normalise_bbox(sample.image_width, sample.image_height, bbox)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _draw_debug_image(source_path: Path, target_path: Path, sample) -> None:
    image = cv2.imread(str(source_path))
    for bbox in sample.bboxes:
        cv2.rectangle(image, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (0, 255, 0), 1)
        cv2.putText(
            image,
            bbox.class_name,
            (bbox.x, max(bbox.y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(str(target_path), image)


def export_yolo_dataset(export_dir: Path, manifest: GenerationManifest, val_size: float, test_size: float, draw_debug_bboxes: bool) -> GenerationManifest:
    raw_dir = export_dir / "raw"
    images_dir = ensure_directory(export_dir / "images")
    labels_dir = ensure_directory(export_dir / "labels")
    debug_dir = ensure_directory(export_dir / "debug") if draw_debug_bboxes else None

    split_samples = _split_samples(manifest.samples, val_size=val_size, test_size=test_size, seed=manifest.seed)
    class_to_index = {name: index for index, name in enumerate(manifest.classes)}
    split_counts: dict[str, int] = {}

    for split_name, samples in split_samples.items():
        split_counts[split_name] = len(samples)
        split_images_dir = ensure_directory(images_dir / split_name)
        split_labels_dir = ensure_directory(labels_dir / split_name)
        split_debug_dir = ensure_directory(debug_dir / split_name) if debug_dir else None

        for sample in samples:
            sample.split = split_name
            source_image = raw_dir / sample.image_filename
            target_image = split_images_dir / sample.image_filename
            shutil.move(str(source_image), str(target_image))
            _write_label_file(split_labels_dir / f"{Path(sample.image_filename).stem}.txt", class_to_index, sample)
            if split_debug_dir is not None:
                _draw_debug_image(target_image, split_debug_dir / sample.image_filename, sample)

    dataset_yaml = {
        "path": str(export_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": manifest.classes,
    }
    (export_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    manifest.split_counts = split_counts
    atomic_write_json(export_dir / "manifest.json", manifest.to_dict())
    if raw_dir.exists():
        raw_dir.rmdir()
    return manifest
