"""Configure project-local paths before importing LIBERO."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))

import yaml

config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "config.yaml"
base = ROOT / "LIBERO" / "libero" / "libero"
if not config_file.exists():
    (ROOT / "datasets").mkdir(exist_ok=True)
    config_file.write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(base),
                "bddl_files": str(base / "bddl_files"),
                "init_states": str(base / "init_files"),
                "assets": str(base / "assets"),
                "datasets": str(ROOT / "datasets"),
            }
        )
    )
