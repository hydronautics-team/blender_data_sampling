from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from blender_data_sampling.contracts import BoundingBox, GeneratedImage, GenerationManifest
from blender_data_sampling.export import export_yolo_dataset


class ExportTests(unittest.TestCase):
    def test_yolo_export_creates_split_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            raw_dir = export_dir / "raw"
            raw_dir.mkdir()

            samples = []
            for index in range(10):
                image_name = f"img_{index:06d}.jpg"
                (raw_dir / image_name).write_bytes(b"image")
                samples.append(
                    GeneratedImage(
                        image_filename=image_name,
                        image_width=100,
                        image_height=50,
                        bboxes=[
                            BoundingBox(
                                class_name="gate",
                                object_name="gate_main",
                                x=10,
                                y=5,
                                width=20,
                                height=10,
                            )
                        ],
                    )
                )

            manifest = GenerationManifest(
                competition="demo",
                year="2026",
                config_path="/tmp/demo/scenes/demo/2026/config.yaml",
                scene_file="/tmp/demo/scenes/demo/2026/scene.blend",
                export_dir=str(export_dir),
                image_prefix="img",
                classes=["gate"],
                seed=7,
                samples=samples,
            )

            exported = export_yolo_dataset(
                export_dir=export_dir,
                manifest=manifest,
                val_size=0.2,
                test_size=0.1,
                draw_debug_bboxes=False,
            )

            self.assertEqual(exported.split_counts, {"train": 7, "val": 2, "test": 1})
            self.assertTrue((export_dir / "dataset.yaml").is_file())
            self.assertTrue((export_dir / "images" / "train").is_dir())
            self.assertTrue((export_dir / "labels" / "val").is_dir())
            first_label = next((export_dir / "labels" / "train").glob("*.txt"))
            self.assertIn("0 ", first_label.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
