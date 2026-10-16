"""Capture matched wide/close-up pre-grasp inputs, without models or manipulation."""

import argparse
import json
from pathlib import Path

import sim_setup  # noqa: F401 -- configure paths before importing LIBERO.
import cv2
import numpy as np

from sorting.scene import SortingScene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=["fruit", "vegetable"], required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap-baskets", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.texture.is_file():
        parser.error("Output must be new and texture must exist")
    args.output.mkdir(parents=True)
    scene = SortingScene(
        args.category,
        seed=args.seed,
        swap=args.swap_baskets,
        texture=args.texture,
        randomize=args.randomize,
    )
    details = {name: [] for name in ("agentview", "closeup")}
    try:
        obs = scene.reset(args.seed)
        for index in range(5):
            for view in details:
                scene.perception_view = view
                crop = scene.food_crop(obs)
                folder = args.output / view
                folder.mkdir(exist_ok=True)
                cv2.imwrite(
                    str(folder / f"classification_crop_{index:02d}.png"),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(folder / f"perception_frame_{index:02d}.png"),
                    cv2.cvtColor(scene.last_perception_frame, cv2.COLOR_RGB2BGR),
                )
                details[view].append(dict(scene.last_perception_metadata))
            if index < 4:
                for _ in range(3):
                    obs, _ = scene.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
        for view, records in details.items():
            folder = args.output / view
            (folder / "perception.json").write_text(json.dumps(records, indent=2))
            metadata = {
                "capture_only": True,
                "category": args.category,
                "seed": args.seed,
                "source_video": args.source_video,
                "texture_path": str(args.texture),
                "swap_baskets": args.swap_baskets,
                "classification_frames": 5,
                "classification_stride_control_steps": 3,
                "scene_version": "perception_camera_v3",
                "perception_camera": records[0]["camera"],
            }
            (folder / "capture.json").write_text(json.dumps(metadata, indent=2))
            print(view, "crop sizes:", [r["crop_size_wh"] for r in records], flush=True)
    finally:
        scene.close()


if __name__ == "__main__":
    main()
