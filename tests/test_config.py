from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from pydantic import ValidationError

from blender_data_sampling.config import derive_profile_metadata, load_project_config, resolve_runtime_context


def _base_config() -> dict:
    return {
        "scene": {
            "scene_file": "scene.blend",
            "camera_name": "Camera",
            "bounds": {"object_name": "Pool", "xy_padding": 2, "z_padding": 0.5},
            "movable_collections": ["targets"],
            "recolor_collection": "targets",
            "recolor_material_node": "Principled BSDF",
            "recolor_material_input": "Base Color",
        },
        "dataset": {
            "classes": ["gate", "marker"],
            "object_type_by_name": {"gate_main": "gate", "marker_main": "marker"},
        },
        "render": {"image_width": 640, "image_height": 480, "cycles_samples": 8},
        "sampling": {
            "num_images": 10,
            "seed": 123,
            "max_camera_attempts": 50,
            "min_objects_in_view": 2,
            "min_valid_bboxes": 1,
            "min_bbox_width_px": 10,
            "min_bbox_height_px": 10,
            "min_camera_distance": 0.5,
        },
        "randomization": {"object_colors": [[1.0, 0.0, 0.0, 1.0]]},
        "visibility": {"mode": "geometric"},
        "export": {"format": "yolo_bbox", "val_size": 0.2, "test_size": 0.1, "image_prefix": "img"},
    }


class ConfigTests(unittest.TestCase):
    def test_derive_profile_metadata(self) -> None:
        path = Path("/tmp/repo/scenes/sauvc/2026/config.yaml")
        profile_dir, competition, year = derive_profile_metadata(path)
        self.assertEqual(profile_dir, Path("/tmp/repo/scenes/sauvc/2026"))
        self.assertEqual(competition, "sauvc")
        self.assertEqual(year, "2026")

    def test_resolve_runtime_context_creates_export_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            profile_dir = repo_root / "scenes" / "demo" / "2026"
            profile_dir.mkdir(parents=True)
            (profile_dir / "scene.blend").write_bytes(b"")
            config_path = profile_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")

            context = resolve_runtime_context(config_path=config_path, repo_root=repo_root)

            self.assertEqual(context.run_mode, "run")
            self.assertEqual(context.competition, "demo")
            self.assertEqual(context.year, "2026")
            self.assertTrue(Path(context.export_dir).is_dir())
            self.assertTrue((Path(context.export_dir) / "resolved_config.json").is_file())

    def test_resolve_runtime_context_supports_debug_mode_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            profile_dir = repo_root / "scenes" / "demo" / "2026"
            profile_dir.mkdir(parents=True)
            (profile_dir / "scene.blend").write_bytes(b"")
            config_path = profile_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")

            context = resolve_runtime_context(
                config_path=config_path,
                repo_root=repo_root,
                run_mode="debug",
                num_images_override=20,
            )

            self.assertEqual(context.run_mode, "debug")
            self.assertEqual(context.sampling["num_images"], 20)
            self.assertIn("/exports/debug/", context.export_dir)

    def test_config_must_live_under_scenes_tree(self) -> None:
        with self.assertRaises(ValueError):
            derive_profile_metadata(Path("/tmp/repo/not-scenes/demo/2026/config.yaml"))

    def test_removed_draw_debug_bboxes_is_rejected(self) -> None:
        config = _base_config()
        config["render"]["draw_debug_bboxes"] = True

        with self.assertRaises(ValidationError):
            load_project_config(Path(_write_temp_config(config)))


def _write_temp_config(config: dict) -> str:
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(config_path)


if __name__ == "__main__":
    unittest.main()
