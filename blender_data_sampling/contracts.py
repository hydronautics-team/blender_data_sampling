from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    class_name: str
    object_name: str
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        return cls(
            class_name=data["class_name"],
            object_name=data["object_name"],
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass(slots=True)
class GeneratedImage:
    image_filename: str
    image_width: int
    image_height: int
    bboxes: list[BoundingBox]
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_filename": self.image_filename,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "split": self.split,
            "bboxes": [bbox.to_dict() for bbox in self.bboxes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedImage":
        return cls(
            image_filename=data["image_filename"],
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            split=data.get("split"),
            bboxes=[BoundingBox.from_dict(item) for item in data["bboxes"]],
        )


@dataclass(slots=True)
class GenerationManifest:
    competition: str
    year: str
    config_path: str
    scene_file: str
    export_dir: str
    image_prefix: str
    classes: list[str]
    seed: int | None
    samples: list[GeneratedImage]
    split_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition": self.competition,
            "year": self.year,
            "config_path": self.config_path,
            "scene_file": self.scene_file,
            "export_dir": self.export_dir,
            "image_prefix": self.image_prefix,
            "classes": self.classes,
            "seed": self.seed,
            "split_counts": self.split_counts or {},
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationManifest":
        return cls(
            competition=data["competition"],
            year=str(data["year"]),
            config_path=data["config_path"],
            scene_file=data["scene_file"],
            export_dir=data["export_dir"],
            image_prefix=data["image_prefix"],
            classes=list(data["classes"]),
            seed=data.get("seed"),
            split_counts=dict(data.get("split_counts", {})),
            samples=[GeneratedImage.from_dict(item) for item in data["samples"]],
        )
