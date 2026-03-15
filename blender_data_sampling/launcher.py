from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _build_pythonpath(repo_root: Path) -> str:
    entries: list[str] = [str(repo_root.resolve())]
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry)
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if "site-packages" not in resolved and "dist-packages" not in resolved:
            continue
        if resolved not in entries:
            entries.append(resolved)
    return os.pathsep.join(entries)


def run_blender(blender_executable: Path, scene_file: Path, runner_script: Path, resolved_config: Path, repo_root: Path) -> subprocess.CompletedProcess[str]:
    if not blender_executable.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender_executable}")

    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(repo_root)
    command = [
        str(blender_executable),
        "-b",
        str(scene_file),
        "--python-use-system-env",
        "--python-exit-code",
        "1",
        "--python",
        str(runner_script),
        "--",
        "--resolved-config",
        str(resolved_config),
    ]
    return subprocess.run(command, check=True, text=True, env=env)
