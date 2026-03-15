from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
from pathlib import Path
from random import Random
from typing import Any

import bpy
from joblib import load as joblib_load

from blender_data_sampling.blender_helpers import (
    apply_surface_hsv_adjustment,
    bounds_from_object,
    calculate_distance,
    camera_view_bounds_2d,
    find_max_dimension,
    generate_grid_positions,
    place_camera_near_target,
    is_object_in_front_of_camera,
    relocate_objects,
    render_image,
    rotate_camera,
    select_objects_in_camera,
    set_material_color_from_hsv,
    set_material_scalar_input,
    set_object_color_from_hsv,
    update_scene,
)
from blender_data_sampling.common import atomic_write_json, ensure_directory
from blender_data_sampling.contracts import BoundingBox, GeneratedImage, GenerationManifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = []
        if "--" in __import__("sys").argv:
            argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description="Internal Blender runtime")
    parser.add_argument("--resolved-config", required=True)
    return parser.parse_args(argv)


def _load_context(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collection_objects(collection_names: list[str]):
    objects_by_name = {}
    for collection_name in collection_names:
        collection = bpy.data.collections[collection_name]
        for obj in collection.objects:
            objects_by_name[obj.name] = obj
    return [objects_by_name[name] for name in sorted(objects_by_name)]


def _capture_light_state(light_names: list[str]) -> dict[str, dict[str, Any]]:
    state = {}
    for light_name in light_names:
        light = bpy.data.objects[light_name]
        state[light_name] = {
            "location": tuple(light.location),
            "rotation_z": light.rotation_euler[2],
            "energy": light.data.energy,
        }
    return state


def _restore_lights(light_state: dict[str, dict[str, Any]], medium_energy: float | None = None) -> None:
    for light_name, state in light_state.items():
        light = bpy.data.objects[light_name]
        light.location = state["location"]
        light.rotation_euler[2] = state["rotation_z"]
        light.data.energy = medium_energy if medium_energy is not None else state["energy"]


def _randomise_lights(lights_config: dict[str, Any], low_x: int, low_y: int, high_x: int, high_y: int, rng: Random) -> None:
    collection = bpy.data.collections[lights_config["collection_name"]]
    for light in collection.objects:
        light.location.x = rng.uniform(low_x, high_x)
        light.location.y = rng.uniform(low_y, high_y)
        light.data.energy = rng.uniform(*lights_config["energy_range"])
        light.rotation_euler[2] = __import__("math").radians(rng.uniform(*lights_config["rotation_z_deg_range"]))


def _load_visibility_tools(context: dict[str, Any]):
    if context["visibility"]["mode"] != "mist_model":
        return None, None
    mist = context["scene"].get("mist")
    if not mist or not mist.get("classifier_path") or not mist.get("scaler_path"):
        raise ValueError("visibility.mode=mist_model requires scene.mist classifier_path and scaler_path")
    with Path(mist["classifier_path"]).open("rb") as handle:
        model = pickle.load(handle)
    scaler = joblib_load(mist["scaler_path"])
    return model, scaler


def _prepare_environment(context: dict[str, Any], low_x: int, low_y: int, high_x: int, high_y: int, rng: Random, light_state):
    scene_config = context["scene"]
    visibility_mode = context["visibility"]["mode"]
    density = None

    if scene_config.get("wall_color"):
        set_object_color_from_hsv(scene_config["wall_color"]["object_name"], scene_config["wall_color"]["hsv"], rng)

    for adjustment in scene_config.get("surface_hsv_adjustments", []):
        apply_surface_hsv_adjustment(adjustment, rng)

    mist_config = scene_config.get("mist")
    lights_config = scene_config.get("lights")

    if visibility_mode == "mist_model":
        if not mist_config:
            raise ValueError("Mist visibility mode requires scene.mist config")
        density = rng.uniform(*mist_config["density_range"])
        set_material_scalar_input(mist_config, mist_config["density_input"], density)
        set_material_color_from_hsv(mist_config, mist_config["color_input"], mist_config["color_hsv"], rng)
        if lights_config and light_state:
            _restore_lights(light_state, medium_energy=None)
    else:
        if lights_config:
            _randomise_lights(lights_config, low_x, low_y, high_x, high_y, rng)
        if mist_config:
            set_material_scalar_input(mist_config, mist_config["density_input"], mist_config["disabled_density"])

    return density


def _evaluate_current_camera(context: dict[str, Any], movable_objects, camera_object, density, model, scaler, primary_objects=None):
    sampling = context["sampling"]
    dataset = context["dataset"]
    scene = bpy.context.scene
    diagnostics: Counter[str] = Counter()

    visible_names = select_objects_in_camera(scene, camera_object)
    primary_objects = primary_objects or []
    candidate_by_name = {obj.name: obj for obj in primary_objects}
    for obj in movable_objects:
        if obj.name in visible_names:
            candidate_by_name[obj.name] = obj

    objects_in_camera = [candidate_by_name[name] for name in sorted(candidate_by_name)]
    if len(objects_in_camera) < sampling["min_objects_in_view"]:
        diagnostics["too_few_objects_in_view"] += 1
        return None, diagnostics

    candidate_objects = objects_in_camera
    if context["visibility"]["mode"] == "mist_model":
        candidate_objects = []
        for obj in objects_in_camera:
            feature_vector = scaler.transform([[density, round(calculate_distance(camera_object, obj), 0)]])
            if model.predict(feature_vector)[0] == 1:
                candidate_objects.append(obj)
        if len(candidate_objects) < sampling["min_valid_bboxes"]:
            diagnostics["mist_filtered_out"] += 1
            return None, diagnostics

    valid_boxes: list[BoundingBox] = []
    for obj in candidate_objects:
        if not is_object_in_front_of_camera(camera_object, obj):
            diagnostics["behind_camera"] += 1
            continue
        view_box = camera_view_bounds_2d(scene, camera_object, obj)
        if view_box is None:
            diagnostics["invalid_bbox_projection"] += 1
            continue
        if view_box.width < sampling["min_bbox_width_px"] or view_box.height < sampling["min_bbox_height_px"]:
            diagnostics["bbox_too_small"] += 1
            continue
        if calculate_distance(camera_object, obj) < sampling["min_camera_distance"]:
            diagnostics["camera_too_close"] += 1
            continue
        valid_boxes.append(
            BoundingBox(
                class_name=dataset["object_type_by_name"][obj.name],
                object_name=obj.name,
                x=view_box.x,
                y=view_box.y,
                width=view_box.width,
                height=view_box.height,
            )
        )

    if len(valid_boxes) < sampling["min_valid_bboxes"]:
        diagnostics["too_few_valid_bboxes"] += 1
        return None, diagnostics

    return valid_boxes, diagnostics


def _sample_valid_frame(context: dict[str, Any], movable_objects, camera_object, camera_base_rotation, bounds, rng: Random, density, model, scaler):
    sampling = context["sampling"]
    diagnostics: Counter[str] = Counter()
    low_x, low_y, low_z, high_x, high_y, high_z = bounds
    x_candidates = list(range(low_x, high_x))
    y_candidates = list(range(low_y, high_y))
    random_attempts = max(sampling["random_fallback_attempts"], sampling["max_camera_attempts"])
    base_rotation = camera_base_rotation.copy()

    for _ in range(sampling["target_object_attempts"]):
        target_object = rng.choice(movable_objects)
        place_camera_near_target(
            camera_object=camera_object,
            target_object=target_object,
            bounds=bounds,
            base_rotation_euler=None,
            horizontal_distance_range=tuple(sampling["target_horizontal_distance_range"]),
            vertical_offset_range=tuple(sampling["target_vertical_offset_range"]),
            yaw_jitter_deg_range=tuple(sampling["target_yaw_jitter_deg_range"]),
            pitch_jitter_deg_range=tuple(sampling["target_pitch_jitter_deg_range"]),
            rng=rng,
        )
        update_scene()
        valid_boxes, attempt_diagnostics = _evaluate_current_camera(
            context=context,
            movable_objects=movable_objects,
            camera_object=camera_object,
            density=density,
            model=model,
            scaler=scaler,
            primary_objects=[target_object],
        )
        diagnostics.update(attempt_diagnostics)
        if valid_boxes is not None:
            return valid_boxes, diagnostics

    for _ in range(random_attempts):
        camera_object.location.x = rng.choice(x_candidates)
        camera_object.location.y = rng.choice(y_candidates)
        camera_object.location.z = rng.uniform(low_z, high_z)
        camera_object.rotation_mode = "XYZ"
        camera_object.rotation_euler = base_rotation.copy()
        rotate_camera(camera_object, rng.uniform(0.0, 360.0))
        update_scene()
        valid_boxes, attempt_diagnostics = _evaluate_current_camera(
            context=context,
            movable_objects=movable_objects,
            camera_object=camera_object,
            density=density,
            model=model,
            scaler=scaler,
        )
        diagnostics.update(attempt_diagnostics)
        if valid_boxes is not None:
            return valid_boxes, diagnostics

    return None, diagnostics


def run_generation(context: dict[str, Any]) -> GenerationManifest:
    rng = Random(context["sampling"]["seed"])
    raw_dir = ensure_directory(Path(context["export_dir"]) / "raw")

    scene_config = context["scene"]
    render_config = context["render"]
    sampling = context["sampling"]
    randomization = context["randomization"]

    movable_objects = _collection_objects(scene_config["movable_collections"])
    camera_object = bpy.data.objects[scene_config["camera_name"]]
    camera_base_rotation = camera_object.rotation_euler.copy()
    bounds_object = bpy.data.objects[scene_config["bounds"]["object_name"]]
    bounds = bounds_from_object(
        bounds_object,
        xy_padding=scene_config["bounds"]["xy_padding"],
        z_padding=scene_config["bounds"]["z_padding"],
    )
    light_state = None
    if scene_config.get("lights") and scene_config["lights"].get("reference_light_names"):
        light_state = _capture_light_state(scene_config["lights"]["reference_light_names"])

    model, scaler = _load_visibility_tools(context)
    max_dim = round(find_max_dimension(movable_objects))
    samples: list[GeneratedImage] = []

    for index in range(sampling["num_images"]):
        valid_boxes = None
        frame_diagnostics: Counter[str] = Counter()
        for _scene_attempt in range(sampling["max_scene_attempts"]):
            density = _prepare_environment(context, *bounds[:2], *bounds[3:5], rng=rng, light_state=light_state)
            positions = generate_grid_positions(
                count=len(movable_objects),
                x_low=bounds[0],
                y_low=bounds[1],
                x_high=bounds[3],
                y_high=bounds[4],
                max_dim=max(1, max_dim),
                rng=rng,
            )
            relocate_objects(
                objects=movable_objects,
                positions=positions,
                low_x=bounds[0],
                low_y=bounds[1],
                recolor_collection=scene_config.get("recolor_collection"),
                color_palette=[tuple(color) for color in randomization.get("object_colors", [])],
                material_node_name=scene_config["recolor_material_node"],
                material_input_key=scene_config["recolor_material_input"],
                rng=rng,
            )
            update_scene()
            valid_boxes, attempt_diagnostics = _sample_valid_frame(
                context,
                movable_objects,
                camera_object,
                camera_base_rotation,
                bounds,
                rng,
                density,
                model,
                scaler,
            )
            frame_diagnostics.update(attempt_diagnostics)
            if valid_boxes is not None:
                break

        if valid_boxes is None:
            diagnostic_summary = ", ".join(f"{key}={value}" for key, value in frame_diagnostics.most_common()) or "no diagnostics"
            raise RuntimeError(
                f"Unable to generate frame {index} with at least {sampling['min_valid_bboxes']} valid object(s) "
                f"after {sampling['max_scene_attempts']} scene attempts "
                f"(target_attempts={sampling['target_object_attempts']}, random_fallback_attempts={max(sampling['random_fallback_attempts'], sampling['max_camera_attempts'])}); "
                f"diagnostics: {diagnostic_summary}"
            )

        image_filename = f"{context['export']['image_prefix']}_{index:06d}.jpg"
        render_image(raw_dir / image_filename, render_config["image_width"], render_config["image_height"], render_config["cycles_samples"])
        samples.append(
            GeneratedImage(
                image_filename=image_filename,
                image_width=render_config["image_width"],
                image_height=render_config["image_height"],
                bboxes=valid_boxes,
            )
        )

    manifest = GenerationManifest(
        competition=context["competition"],
        year=context["year"],
        config_path=context["config_path"],
        scene_file=context["scene_file"],
        export_dir=context["export_dir"],
        image_prefix=context["export"]["image_prefix"],
        classes=context["dataset"]["classes"],
        seed=context["sampling"]["seed"],
        samples=samples,
    )
    atomic_write_json(Path(context["export_dir"]) / "manifest.json", manifest.to_dict())
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    context = _load_context(Path(args.resolved_config))
    run_generation(context)
    return 0
