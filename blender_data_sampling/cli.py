from __future__ import annotations

import argparse
import json
from pathlib import Path

from blender_data_sampling.config import resolve_runtime_context
from blender_data_sampling.contracts import GenerationManifest
from blender_data_sampling.export import export_yolo_dataset
from blender_data_sampling.launcher import run_blender

DEBUG_NUM_IMAGES = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic dataset generation from Blender scenes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run dataset generation for one scene profile")
    run_parser.add_argument("--blender", required=True, help="Path to Blender executable")
    run_parser.add_argument("--config", required=True, help="Path to scenes/<competition>/<year>/config.yaml")
    debug_parser = subparsers.add_parser("debug", help="Generate a 20-image debug dataset with bbox overlays")
    debug_parser.add_argument("--blender", required=True, help="Path to Blender executable")
    debug_parser.add_argument("--config", required=True, help="Path to scenes/<competition>/<year>/config.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in {"run", "debug"}:
        parser.error("Unsupported command")

    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    blender_executable = Path(args.blender)
    context = resolve_runtime_context(
        config_path=config_path,
        repo_root=repo_root,
        run_mode=args.command,
        num_images_override=DEBUG_NUM_IMAGES if args.command == "debug" else None,
    )

    run_blender(
        blender_executable=blender_executable,
        scene_file=Path(context.scene_file),
        runner_script=repo_root / "blender_runner.py",
        resolved_config=Path(context.export_dir) / "resolved_config.json",
        repo_root=repo_root,
    )

    manifest_path = Path(context.export_dir) / "manifest.json"
    manifest = GenerationManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    export_yolo_dataset(
        export_dir=Path(context.export_dir),
        manifest=manifest,
        val_size=context.export["val_size"],
        test_size=context.export["test_size"],
        overlay_images=args.command == "debug",
    )
    return 0
