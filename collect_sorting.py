"""Compatibility entry point. Use sorting_experiments.py for new collections."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts/legacy/collect_sorting.py"),
        run_name="__main__",
    )
