from __future__ import annotations

import unittest

from blender_data_sampling.cli import DEBUG_NUM_IMAGES, build_parser


class CliTests(unittest.TestCase):
    def test_run_command_parses(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["run", "--blender", "blender/blender", "--config", "scenes/demo/2026/config.yaml"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.blender, "blender/blender")

    def test_debug_command_parses(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["debug", "--blender", "blender/blender", "--config", "scenes/demo/2026/config.yaml"])

        self.assertEqual(args.command, "debug")
        self.assertEqual(args.config, "scenes/demo/2026/config.yaml")

    def test_debug_image_count_constant(self) -> None:
        self.assertEqual(DEBUG_NUM_IMAGES, 20)


if __name__ == "__main__":
    unittest.main()
