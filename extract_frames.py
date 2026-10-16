"""Extract frames and an empty annotation CSV from a personally recorded video."""

import argparse
import csv
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=int, default=5, help="Keep every Nth frame")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be positive")
    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot read {args.video}")
        args.output.mkdir(parents=True, exist_ok=False)
        fields = [
            "frame_index",
            "time_s",
            "image_path",
            "phase",
            "food_u",
            "food_v",
            "hand_u",
            "hand_v",
            "target_u",
            "target_v",
            "gripper_state",
        ]
        count = 0
        with (args.output / "annotations.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % args.every == 0:
                    name = f"frame_{index:06d}.jpg"
                    if not cv2.imwrite(str(args.output / name), frame):
                        raise RuntimeError(f"Cannot save {name}")
                    writer.writerow(
                        {
                            "frame_index": index,
                            "time_s": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                            "image_path": name,
                        }
                    )
                    count += 1
                index += 1
        if count == 0:
            raise ValueError("Video contains no decodable frames")
        print(f"Saved {count} frames and annotations.csv to {args.output}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
