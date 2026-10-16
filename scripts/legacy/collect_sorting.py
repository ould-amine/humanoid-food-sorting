"""Collect matched fixed/human robot demonstrations; retain unsuccessful episodes."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/crops/manifest.jsonl")
    )
    parser.add_argument(
        "--methods", nargs="+", choices=["fixed", "human"], default=["fixed", "human"]
    )
    parser.add_argument(
        "--demo", type=Path, default=Path("data/demos/banana_02_reviewed.json")
    )
    parser.add_argument(
        "--classifier", choices=["oracle", "dino", "clip"], default="oracle"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--train-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5]
    )
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[8, 9])
    args = parser.parse_args()
    groups = {"train": args.train_seeds, "val": args.val_seeds, "test": args.test_seeds}
    if any(
        set(a) & set(b)
        for i, a in enumerate(groups.values())
        for b in list(groups.values())[i + 1 :]
    ):
        parser.error("Simulation seeds must be disjoint between splits")
    if args.classifier == "dino" and not args.checkpoint:
        parser.error("DINO requires --checkpoint")
    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    # Use training-video appearances only, even for held-out simulation seeds.
    pools = {
        c: sorted(
            [r for r in rows if r["category"] == c and r["split"] == "train"],
            key=lambda r: (r["video_id"], int(r["frame_index"])),
        )
        for c in ("fruit", "vegetable")
    }
    if not all(pools.values()):
        parser.error("Need training crops of both categories")
    if "human" in args.methods:
        from sorting.demonstrations import load_demo

        if not load_demo(str(args.demo)).approved:
            parser.error("Human demo requires review")
    args.output.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for mode in args.methods:
        for split, seeds in groups.items():
            for seed in seeds:
                for category in pools:
                    rng = random.Random(seed * 17 + (category == "vegetable"))
                    crop = rng.choice(pools[category])
                    for swap in (False, True):
                        destination = (
                            args.output
                            / mode
                            / split
                            / f"{category}_{seed}_{'swap' if swap else 'normal'}"
                        )
                        command = [
                            sys.executable,
                            "run_sorting.py",
                            "--category",
                            category,
                            "--classifier",
                            args.classifier,
                            "--seed",
                            str(seed),
                            "--split",
                            split,
                            "--food-name",
                            crop["food_name"],
                            "--texture",
                            str(args.manifest.parent / crop["crop_path"]),
                            "--source-video",
                            crop["video_id"],
                            "--randomize",
                            "--device",
                            args.device,
                            "--output",
                            str(destination),
                        ]
                        if swap:
                            command.append("--swap-baskets")
                        if mode == "human":
                            command += ["--demo", str(args.demo)]
                        if args.checkpoint:
                            command += ["--checkpoint", str(args.checkpoint)]
                        result = subprocess.run(command)
                        summary_path = destination / "episode.json"
                        if not summary_path.exists():
                            raise RuntimeError(
                                f"Collection failed before recording an episode (exit {result.returncode}): {destination}"
                            )
                        summary = json.loads(summary_path.read_text())
                        outcomes.append(
                            {
                                "path": str(destination),
                                "split": split,
                                "mode": mode,
                                "success": summary["task_success"],
                                "failure": summary["failure_reason"],
                            }
                        )
                        (args.output / "collection.json").write_text(
                            json.dumps(outcomes, indent=2)
                        )
    print(
        f"Recorded {len(outcomes)} episodes, {sum(r['success'] for r in outcomes)} successful; failures retained"
    )


if __name__ == "__main__":
    main()
