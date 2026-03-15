from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from blender_data_sampling.common import atomic_write_json, ensure_directory, utc_timestamp_slug


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HSVRange(StrictModel):
    h: tuple[float, float]
    s: tuple[float, float]
    v: tuple[float, float]

    @model_validator(mode="after")
    def validate_order(self) -> "HSVRange":
        for name in ("h", "s", "v"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} range must be ordered low..high")
        return self


class BoundsConfig(StrictModel):
    object_name: str
    xy_padding: int = 0
    z_padding: float = 0.0


class WallColorConfig(StrictModel):
    object_name: str
    hsv: HSVRange


class NodeHSVAdjustment(StrictModel):
    node_name: str
    hue_input: str | int
    hue_range: tuple[float, float]
    saturation_input: str | int
    saturation_range: tuple[float, float]
    value_input: str | int
    value_range: tuple[float, float]
    object_name: str | None = None
    material_name: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "NodeHSVAdjustment":
        if bool(self.object_name) == bool(self.material_name):
            raise ValueError("Provide exactly one of object_name or material_name")
        return self


class LightsConfig(StrictModel):
    collection_name: str
    reference_light_names: list[str] = Field(default_factory=list)
    energy_range: tuple[float, float]
    rotation_z_deg_range: tuple[float, float] = (0.0, 360.0)


class MistConfig(StrictModel):
    material_name: str
    node_name: str
    density_input: str | int
    density_range: tuple[float, float]
    disabled_density: float
    color_input: str | int
    color_hsv: HSVRange
    classifier_path: str | None = None
    scaler_path: str | None = None


class SceneConfig(StrictModel):
    scene_file: str
    camera_name: str
    bounds: BoundsConfig
    movable_collections: list[str]
    recolor_collection: str | None = None
    recolor_material_node: str = "Principled BSDF"
    recolor_material_input: str | int = "Base Color"
    wall_color: WallColorConfig | None = None
    surface_hsv_adjustments: list[NodeHSVAdjustment] = Field(default_factory=list)
    lights: LightsConfig | None = None
    mist: MistConfig | None = None


class DatasetConfig(StrictModel):
    classes: list[str]
    object_type_by_name: dict[str, str]

    @model_validator(mode="after")
    def validate_class_mapping(self) -> "DatasetConfig":
        known = set(self.classes)
        missing = sorted(set(self.object_type_by_name.values()) - known)
        if missing:
            raise ValueError(f"Classes missing from dataset.classes: {missing}")
        return self


class RenderConfig(StrictModel):
    image_width: int
    image_height: int
    cycles_samples: int
    draw_debug_bboxes: bool = False


class SamplingConfig(StrictModel):
    num_images: int
    seed: int | None = None
    max_camera_attempts: int = 500
    max_scene_attempts: int = 20
    target_object_attempts: int = 600
    random_fallback_attempts: int = 150
    target_horizontal_distance_range: tuple[float, float] = (2.0, 8.0)
    target_vertical_offset_range: tuple[float, float] = (-0.25, 2.0)
    target_yaw_jitter_deg_range: tuple[float, float] = (-8.0, 8.0)
    target_pitch_jitter_deg_range: tuple[float, float] = (-5.0, 5.0)
    min_objects_in_view: int = 1
    min_valid_bboxes: int = 1
    min_bbox_width_px: int = 30
    min_bbox_height_px: int = 5
    min_camera_distance: float = 1.5

    @model_validator(mode="after")
    def validate_ranges(self) -> "SamplingConfig":
        ordered_ranges = (
            ("target_horizontal_distance_range", self.target_horizontal_distance_range),
            ("target_vertical_offset_range", self.target_vertical_offset_range),
            ("target_yaw_jitter_deg_range", self.target_yaw_jitter_deg_range),
            ("target_pitch_jitter_deg_range", self.target_pitch_jitter_deg_range),
        )
        for name, value in ordered_ranges:
            if value[0] > value[1]:
                raise ValueError(f"{name} must be ordered low..high")
        if self.target_object_attempts < 1 or self.random_fallback_attempts < 0:
            raise ValueError("Camera search attempts must be non-negative and target_object_attempts must be >= 1")
        return self


class RandomizationConfig(StrictModel):
    object_colors: list[tuple[float, float, float, float]] = Field(default_factory=list)


class VisibilityConfig(StrictModel):
    mode: Literal["geometric", "mist_model"] = "geometric"


class ExportConfig(StrictModel):
    format: Literal["yolo_bbox"] = "yolo_bbox"
    val_size: float = 0.15
    test_size: float = 0.15
    image_prefix: str = "img"

    @model_validator(mode="after")
    def validate_split(self) -> "ExportConfig":
        if self.val_size < 0 or self.test_size < 0:
            raise ValueError("Split sizes must be non-negative")
        if self.val_size + self.test_size >= 1:
            raise ValueError("val_size + test_size must be less than 1")
        return self


class ProjectConfig(StrictModel):
    scene: SceneConfig
    dataset: DatasetConfig
    render: RenderConfig
    sampling: SamplingConfig
    randomization: RandomizationConfig
    visibility: VisibilityConfig
    export: ExportConfig


class RunContext(StrictModel):
    competition: str
    year: str
    config_path: str
    profile_dir: str
    scene_file: str
    export_dir: str
    scene: dict[str, Any]
    dataset: dict[str, Any]
    render: dict[str, Any]
    sampling: dict[str, Any]
    randomization: dict[str, Any]
    visibility: dict[str, Any]
    export: dict[str, Any]


def load_project_config(config_path: Path) -> ProjectConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProjectConfig.model_validate(raw)


def derive_profile_metadata(config_path: Path) -> tuple[Path, str, str]:
    resolved = config_path.resolve()
    parts = resolved.parts
    if "scenes" not in parts:
        raise ValueError("Config path must be inside scenes/<competition>/<year>/")
    scenes_index = parts.index("scenes")
    if len(parts) - scenes_index < 4:
        raise ValueError("Config path must be scenes/<competition>/<year>/config.yaml")
    competition = parts[scenes_index + 1]
    year = parts[scenes_index + 2]
    profile_dir = Path(*parts[: scenes_index + 3])
    if resolved.parent != profile_dir:
        raise ValueError("Config file must live directly inside scenes/<competition>/<year>/")
    return profile_dir, competition, year


def build_export_dir(repo_root: Path, competition: str, year: str) -> Path:
    return ensure_directory(repo_root / "exports" / f"{competition}_{year}_{utc_timestamp_slug()}")


def resolve_runtime_context(config_path: Path, repo_root: Path) -> RunContext:
    resolved_config_path = config_path.resolve()
    profile_dir, competition, year = derive_profile_metadata(resolved_config_path)
    project_config = load_project_config(resolved_config_path)

    scene_file = (profile_dir / project_config.scene.scene_file).resolve()
    if scene_file.parent != profile_dir:
        raise ValueError("scene.scene_file must point to a file next to the config")
    if not scene_file.is_file():
        raise FileNotFoundError(f"Scene file not found: {scene_file}")

    export_dir = build_export_dir(repo_root, competition, year)

    scene_payload = project_config.scene.model_dump(mode="json")
    if scene_payload.get("mist"):
        mist_payload = scene_payload["mist"]
        if mist_payload.get("classifier_path"):
            mist_payload["classifier_path"] = str((profile_dir / mist_payload["classifier_path"]).resolve())
        if mist_payload.get("scaler_path"):
            mist_payload["scaler_path"] = str((profile_dir / mist_payload["scaler_path"]).resolve())

    context = RunContext(
        competition=competition,
        year=year,
        config_path=str(resolved_config_path),
        profile_dir=str(profile_dir),
        scene_file=str(scene_file),
        export_dir=str(export_dir.resolve()),
        scene=scene_payload,
        dataset=project_config.dataset.model_dump(mode="json"),
        render=project_config.render.model_dump(mode="json"),
        sampling=project_config.sampling.model_dump(mode="json"),
        randomization=project_config.randomization.model_dump(mode="json"),
        visibility=project_config.visibility.model_dump(mode="json"),
        export=project_config.export.model_dump(mode="json"),
    )
    atomic_write_json(export_dir / "resolved_config.json", context.model_dump(mode="json"))
    return context
