"""Package code and personal data for Colab, excluding virtualenvs and models."""

import argparse
from pathlib import Path
import zipfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sorting_colab.zip")
    )
    args = parser.parse_args()
    files = {p for p in Path(".").glob("*.py")}
    files.update(Path(".").glob("requirements*.txt"))
    files.update(Path(".").glob("*.md"))
    files.update(Path("docs").glob("*.md"))
    files.update(Path("configs").glob("*.json"))
    files.update(Path("scripts/legacy").glob("*.py"))
    files.update(Path("sorting").glob("*.py"))
    files.update(Path("data/demos").glob("*.json"))
    files.update(p for p in Path("data/crops").rglob("*") if p.is_file())
    files.add(Path("data/metadata.csv"))
    files.add(Path("data/README.md"))
    files.add(Path("scripts/legacy/README.md"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.as_posix())
    print(
        f"Created {args.output} with {len(files)} files; pretrained checkpoints must be supplied separately"
    )
