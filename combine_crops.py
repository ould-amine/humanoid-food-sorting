"""Combine per-video crop manifests, rebasing paths without copying images."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def combine(root: Path, output: Path, metadata: Path):
    with metadata.open(newline="") as f:
        metadata_rows = list(csv.DictReader(f))
    by_video = {r["video_id"]: r for r in metadata_rows}
    if len(by_video) != len(metadata_rows):
        raise ValueError("Duplicate video IDs in metadata")
    manifests = sorted(
        p for p in root.rglob("crops.jsonl") if p.resolve() != output.resolve()
    )
    if not manifests:
        raise ValueError(f"No crops.jsonl manifests under {root}")
    combined, seen_frames, seen_paths, videos = [], set(), set(), set()
    for manifest in manifests:
        with manifest.open() as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                video = row["video_id"]
                if video not in by_video:
                    raise ValueError(
                        f"Unknown video {video} in {manifest}:{line_number}"
                    )
                for key, value in by_video[video].items():
                    if row.get(key) != value:
                        raise ValueError(
                            f"Metadata mismatch for {video}, field {key}: "
                            f"crop={row.get(key)!r}, metadata={value!r}"
                        )
                if row["split"] not in ("train", "val", "test") or row[
                    "category"
                ] not in ("fruit", "vegetable"):
                    raise ValueError(f"Invalid split/category for {video}")
                crop = (manifest.parent / row["crop_path"]).resolve()
                if not crop.is_file():
                    raise FileNotFoundError(crop)
                identity = (video, int(row["frame_index"]))
                if identity in seen_frames or crop in seen_paths:
                    raise ValueError(
                        f"Duplicate crop/frame {identity}; select only one crop version per video"
                    )
                seen_frames.add(identity)
                seen_paths.add(crop)
                videos.add(video)
                row["crop_path"] = Path(
                    os.path.relpath(crop, output.parent.resolve())
                ).as_posix()
                row["source_manifest"] = Path(
                    os.path.relpath(manifest.resolve(), output.parent.resolve())
                ).as_posix()
                combined.append(row)
    if not combined:
        raise ValueError("All crop manifests are empty")
    combined.sort(key=lambda r: (r["video_id"], int(r["frame_index"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(combined)} crops from {len(videos)} videos to {output}")
    print("Crops per split:", dict(Counter(r["split"] for r in combined)))
    missing = sorted(by_video.keys() - videos)
    if missing:
        print("Videos without crops:", ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/crops"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/crops/manifest.jsonl")
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    args = parser.parse_args()
    combine(args.root, args.output, args.metadata)


if __name__ == "__main__":
    main()
