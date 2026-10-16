"""Compare frozen classifiers on identical saved simulator crops; no robot rerun."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sorting.inference import DINOHead, CLIPClassifier
from sorting.contracts import CATEGORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--methods", nargs="+", choices=["dino", "clip"], default=["dino", "clip"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new report filename")
    inputs = []
    for episode in args.episodes:
        metadata_file = episode / "episode.json"
        if not metadata_file.exists():
            metadata_file = episode / "capture.json"
        metadata = json.loads(metadata_file.read_text())
        paths = sorted(episode.glob("classification_crop_*.png"))
        if len(paths) != metadata.get("classification_frames", 5):
            raise ValueError(f"Incomplete classification crops in {episode}")
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")).copy())
        inputs.append((episode, metadata, paths, images))
    reports = []
    for method in args.methods:
        classifier = (
            DINOHead(args.checkpoint, args.device)
            if method == "dino"
            else CLIPClassifier(args.device)
        )
        for episode, metadata, paths, images in inputs:
            predictions = [classifier.predict(image) for image in images]
            scores = {
                category: float(
                    np.mean([p.probabilities[category] for p in predictions])
                )
                for category in CATEGORIES
            }
            category = max(scores, key=scores.get)
            reports.append(
                {
                    "episode": str(episode),
                    "scene_version": metadata.get("scene_version"),
                    "seed": metadata["seed"],
                    "perception_camera": metadata.get("perception_camera", "agentview"),
                    "method": method,
                    "checkpoint_id": hashlib.sha256(
                        args.checkpoint.read_bytes()
                    ).hexdigest()
                    if method == "dino"
                    else "openai_clip:ViT-B/32",
                    "truth": metadata["category"],
                    "predicted_category": category,
                    "correct": category == metadata["category"],
                    "mean_probabilities": scores,
                    "crops": [
                        {
                            "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size_wh": [image.shape[1], image.shape[0]],
                            "probabilities": p.probabilities,
                        }
                        for path, image, p in zip(paths, images, predictions)
                    ],
                }
            )
        del classifier
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(
            {
                "protocol": "frozen models, mean scores over identical saved crops",
                "diagnostic_only": True,
                "results": reports,
            },
            f,
            indent=2,
        )
    for report in reports:
        print(
            report["episode"],
            report["method"],
            report["predicted_category"],
            report["mean_probabilities"],
        )
    print("Saved", args.output)


if __name__ == "__main__":
    main()
