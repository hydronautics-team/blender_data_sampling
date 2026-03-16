from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

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
                image_name = f"img_{index:06d}.png"
                cv2.imwrite(str(raw_dir / image_name), np.zeros((50, 100, 3), dtype=np.uint8))
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
                run_mode="run",
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
                overlay_images=False,
            )

            self.assertEqual(exported.split_counts, {"train": 7, "val": 2, "test": 1})
            self.assertTrue((export_dir / "dataset.yaml").is_file())
            self.assertTrue((export_dir / "images" / "train").is_dir())
            self.assertTrue((export_dir / "labels" / "val").is_dir())
            self.assertFalse((export_dir / "debug").exists())
            first_label = next((export_dir / "labels" / "train").glob("*.txt"))
            self.assertIn("0 ", first_label.read_text(encoding="utf-8"))

    def test_debug_export_draws_bbox_into_images_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            raw_dir = export_dir / "raw"
            raw_dir.mkdir()

            image_name = "img_000000.png"
            cv2.imwrite(str(raw_dir / image_name), np.zeros((50, 100, 3), dtype=np.uint8))
            manifest = GenerationManifest(
                run_mode="debug",
                competition="demo",
                year="2026",
                config_path="/tmp/demo/scenes/demo/2026/config.yaml",
                scene_file="/tmp/demo/scenes/demo/2026/scene.blend",
                export_dir=str(export_dir),
                image_prefix="img",
                classes=["gate"],
                seed=7,
                samples=[
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
                ],
            )

            export_yolo_dataset(
                export_dir=export_dir,
                manifest=manifest,
                val_size=0.0,
                test_size=0.0,
                overlay_images=True,
            )

            exported_image = export_dir / "images" / "train" / image_name
            self.assertTrue(exported_image.is_file())
            image = cv2.imread(str(exported_image))
            self.assertGreater(int(image.sum()), 0)
            self.assertFalse((export_dir / "debug").exists())


if __name__ == "__main__":
    unittest.main()
