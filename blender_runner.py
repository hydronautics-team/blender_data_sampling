from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_data_sampling.blender_runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
