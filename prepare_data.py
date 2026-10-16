"""Offline data preparation; requires actual videos/annotations, never trains."""

import argparse
import json
from pathlib import Path

import cv2

from sorting.datasets import read_metadata
from sorting.demonstrations import abstract_demo, save_demo
from sorting.vision import (
    ManualTracker,
    crop_food,
    read_annotations,
    write_tracks,
    interpolate_short_gaps,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tracks", "demo", "crops", "sam2-frames"):
        p = sub.add_parser(name)
        p.add_argument("annotations", type=Path)
        p.add_argument("--frames-dir", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--max-gap-frames",
            type=int,
            default=0,
            help="Explicit point interpolation; 0 disables",
        )
        if name in ("demo", "crops"):
            p.add_argument("--video-id", required=True)
        if name == "demo":
            p.add_argument("--grasp-frame", type=int)
            p.add_argument("--release-frame", type=int)
        if name == "crops":
            p.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
            p.add_argument(
                "--crop-size",
                type=int,
                help="Explicit fallback square size in pixels around food center when bbox is missing",
            )
            p.add_argument(
                "--skip-missing-food",
                action="store_true",
                help="Skip frames with neither food center nor bbox",
            )
    args = parser.parse_args()
    root = args.frames_dir or args.annotations.parent
    tracks = ManualTracker().track(str(root), str(args.annotations))
    if args.max_gap_frames < 0:
        parser.error("--max-gap-frames must be nonnegative")
    if args.max_gap_frames:
        tracks = interpolate_short_gaps(tracks, args.max_gap_frames)
    if args.command == "tracks":
        write_tracks(tracks, str(args.output))
    elif args.command == "demo":
        demo = abstract_demo(
            args.video_id,
            tracks,
            grasp_frame=args.grasp_frame,
            release_frame=args.release_frame,
        )
        save_demo(demo, str(args.output))
        print(
            "Demo saved with approved=false. Review event frames, shape and camera assumptions."
        )
    else:
        metadata = None
        if args.command == "crops":
            rows = read_metadata(args.metadata)
            matches = [r for r in rows if r["video_id"] == args.video_id]
            if len(matches) != 1:
                raise ValueError("video_id must occur exactly once in metadata")
            metadata = matches[0]
            if args.crop_size is not None and args.crop_size <= 0:
                parser.error("--crop-size must be positive")
            usable = []
            for track in tracks:
                if track.food.bbox_xyxy is None:
                    if track.food.center_uv is None and args.skip_missing_food:
                        continue
                    if args.crop_size is None or track.food.center_uv is None:
                        parser.error(
                            f"Frame {track.frame_index} has no usable food bbox. "
                            "Draw boxes or use --crop-size PIXELS; "
                            "add --skip-missing-food to omit frames where food is not visible."
                        )
                usable.append(track.frame_index)
            if not usable:
                parser.error("No frames with usable food annotations")
        args.output.mkdir(parents=True, exist_ok=False)
        records = []
        skipped = 0
        for index, (track, row) in enumerate(
            zip(
                tracks,
                sorted(
                    read_annotations(args.annotations),
                    key=lambda r: int(r["frame_index"]),
                ),
            )
        ):
            if metadata is not None and track.frame_index not in usable:
                skipped += 1
                continue
            path = root / (
                row.get("image_path") or f"frame_{track.frame_index:06d}.jpg"
            )
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read {path}")
            if args.command == "sam2-frames":
                destination = args.output / f"{index:06d}.jpg"
            else:
                # cv2 image is BGR; crop geometry is independent of channel order.
                image = crop_food(image, track.food, crop_size=args.crop_size)
                destination = args.output / f"{track.frame_index:06d}.jpg"
                records.append(
                    {
                        **metadata,
                        "crop_path": destination.name,
                        "frame_index": track.frame_index,
                        "time_s": track.time_s,
                        "annotation_path": str(args.annotations),
                        "crop_source": "bbox"
                        if track.food.bbox_xyxy is not None
                        else "point_square",
                        "crop_size_px": args.crop_size
                        if track.food.bbox_xyxy is None
                        else None,
                        "food_center_uv": track.food.center_uv,
                        "food_bbox_xyxy": track.food.bbox_xyxy,
                    }
                )
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"Cannot save {destination}")
        if metadata is not None:
            (args.output / "crops.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            print(
                f"Saved {len(records)} crops; skipped {skipped} frames without food annotations."
            )


if __name__ == "__main__":
    main()
