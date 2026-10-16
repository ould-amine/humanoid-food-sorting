"""Export classification errors or select a fixed pre-grasp decision window."""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_grasp_frame(value):
    """Accept original frame indices or extracted-frame filenames."""
    value = value.strip()
    if re.fullmatch(r"[0-9]+", value):
        return int(value)
    match = re.fullmatch(
        r"frame_([0-9]+)\.(?:jpg|jpeg|png)", value, flags=re.IGNORECASE
    )
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Invalid grasp_frame {value!r}: use 60 or frame_000060.jpg (original frame index)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("errors", "events-template", "pregrasp"):
        p = sub.add_parser(command)
        p.add_argument("predictions", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--split", choices=["train", "val", "test"], default="val")
        if command == "errors":
            p.add_argument(
                "--manifest", type=Path, default=Path("data/crops/manifest.jsonl")
            )
        if command == "events-template":
            p.add_argument("--annotations-root", type=Path, default=Path("data/frames"))
        if command == "pregrasp":
            p.add_argument(
                "--events",
                type=Path,
                required=True,
                help="CSV: video_id,grasp_frame; accepts 60 or frame_000060.jpg",
            )
            p.add_argument("--first-n", type=int, default=5)
    args = parser.parse_args()
    rows = [r for r in read_rows(args.predictions) if r["split"] == args.split]
    if not rows:
        raise ValueError("No predictions for selected split")
    if args.output.exists():
        raise FileExistsError(f"{args.output}: choose a new output")
    if args.command == "errors":
        manifest = {
            (r["video_id"], r["frame_index"]): r for r in read_rows(args.manifest)
        }
        errors = []
        for row in rows:
            predicted = max(row["probabilities"], key=row["probabilities"].get)
            if predicted != row["category"]:
                source = manifest[(row["video_id"], row["frame_index"])]
                image = args.manifest.parent / source["crop_path"]
                if not image.is_file():
                    raise FileNotFoundError(image)
                errors.append((row, predicted, image))
        args.output.mkdir(parents=True)
        with (args.output / "errors.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "video_id",
                    "frame_index",
                    "category",
                    "predicted_category",
                    "p_true",
                    "image",
                ],
            )
            writer.writeheader()
            for row, predicted, image in errors:
                name = f"{row['video_id']}_{int(row['frame_index']):06d}{image.suffix}"
                shutil.copy2(image, args.output / name)
                writer.writerow(
                    {
                        "video_id": row["video_id"],
                        "frame_index": row["frame_index"],
                        "category": row["category"],
                        "predicted_category": predicted,
                        "p_true": row["probabilities"][row["category"]],
                        "image": name,
                    }
                )
        print(f"Exported {len(errors)} incorrect crops to {args.output}")
    elif args.command == "events-template":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "grasp_frame"])
            writer.writeheader()
            for video in sorted({r["video_id"] for r in rows}):
                annotation = args.annotations_root / video / "annotated.csv"
                grasp = ""
                if annotation.is_file():
                    with annotation.open(newline="") as a:
                        frames = [
                            int(r["frame_index"])
                            for r in csv.DictReader(a)
                            if r.get("phase", "").upper() == "GRASP"
                        ]
                    if frames:
                        grasp = min(frames)
                writer.writerow({"video_id": video, "grasp_frame": grasp})
        print(
            "Review/fill grasp_frame using ORIGINAL video frame indices. Blank events are not inferred."
        )
    else:
        if args.first_n < 1:
            parser.error("--first-n must be positive")
        with args.events.open(newline="") as f:
            events = list(csv.DictReader(f))
        boundaries = {}
        event_videos = set()
        for event in events:
            if event["video_id"] in event_videos:
                raise ValueError("Duplicate video in event CSV")
            event_videos.add(event["video_id"])
            if event.get("grasp_frame", "").strip():
                boundaries[event["video_id"]] = parse_grasp_frame(event["grasp_frame"])
        selected = []
        for video in sorted({r["video_id"] for r in rows}):
            if video not in boundaries:
                raise ValueError(f"Missing reviewed grasp_frame for {video}")
            before = sorted(
                (
                    r
                    for r in rows
                    if r["video_id"] == video
                    and int(r["frame_index"]) < boundaries[video]
                ),
                key=lambda r: int(r["frame_index"]),
            )
            if len(before) < args.first_n:
                raise ValueError(
                    f"{video}: only {len(before)} pre-grasp crops, need {args.first_n}; choose a feasible fixed window on validation"
                )
            selected.extend(
                {
                    **r,
                    "decision_window": "first_pregrasp_crops",
                    "window_size": args.first_n,
                    "reviewed_grasp_frame": boundaries[video],
                }
                for r in before[: args.first_n]
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            for row in selected:
                f.write(json.dumps(row) + "\n")
        print(
            f"Saved {len(selected)} predictions, {args.first_n} per video, strictly before grasp"
        )


if __name__ == "__main__":
    main()
