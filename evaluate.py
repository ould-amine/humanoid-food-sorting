"""Aggregate recorded episodes; never starts simulation, training or inference."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def aggregate(rows):
    def mean(key):
        return sum(float(r[key]) for r in rows) / len(rows)

    picked = [r for r in rows if r["pick_success"]]
    return {
        "episodes": len(rows),
        "classification_accuracy": sum(
            r["category"] == r["predicted_category"] for r in rows
        )
        / len(rows),
        "task_success_rate": mean("task_success"),
        "pick_success_rate": mean("pick_success"),
        "correct_basket_rate": mean("correct_basket"),
        "correct_basket_rate_given_pick": (
            sum(r["correct_basket"] for r in picked) / len(picked) if picked else None
        ),
        "trajectory_length_m_mean": mean("trajectory_length_m"),
        "incomplete_episodes": sum(not r["completed"] for r in rows),
        "confusion_matrix": {
            actual: {
                predicted: sum(
                    r["category"] == actual and r["predicted_category"] == predicted
                    for r in rows
                )
                for predicted in ("fruit", "vegetable")
            }
            for actual in ("fruit", "vegetable")
        },
    }


def evaluate(root, split="test"):
    groups = defaultdict(list)
    seen_ids = set()
    for path in sorted(Path(root).glob("**/episode.json")):
        row = json.loads(path.read_text())
        if row["split"] != split:
            continue
        if row.get("schema_version") != 1:
            raise ValueError(f"Unsupported episode schema: {path}")
        identity = (row["method"], row["checkpoint_id"], row["episode_id"])
        if identity in seen_ids:
            raise ValueError(f"Duplicate evaluation episode: {identity}")
        seen_ids.add(identity)
        groups[(row["method"], row["checkpoint_id"], row["eval_group"])].append(row)
    if not groups:
        raise ValueError(
            "No recorded episodes for selected split; no metrics fabricated"
        )
    return {
        "split": split,
        "groups": [
            {
                "method": method,
                "checkpoint_id": checkpoint,
                "eval_group": group,
                **aggregate(rows),
            }
            for (method, checkpoint, group), rows in sorted(groups.items())
        ],
    }


def evaluate_classification(path, split="test"):
    """JSONL crop predictions: video_id, method, checkpoint_id, eval_group,
    category, split, probabilities={fruit:..., vegetable:...}.
    Pool scores per video so long clips do not dominate reported accuracy.
    """
    videos = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] == split:
            videos[
                (
                    row["method"],
                    row["checkpoint_id"],
                    row["eval_group"],
                    row["video_id"],
                )
            ].append(row)
    groups = defaultdict(list)
    for (method, checkpoint, group, video), rows in videos.items():
        labels = {r["category"] for r in rows}
        if len(labels) != 1 or not labels <= {"fruit", "vegetable"}:
            raise ValueError(f"Conflicting category labels for {video}")
        scores = {
            category: sum(r["probabilities"][category] for r in rows) / len(rows)
            for category in ("fruit", "vegetable")
        }
        for row in rows:
            p = row["probabilities"]
            if (
                any(not 0 <= p[c] <= 1 for c in scores)
                or abs(sum(p[c] for c in scores) - 1) > 1e-4
            ):
                raise ValueError("Invalid classifier probabilities")
        predicted = max(scores, key=scores.get)
        groups[(method, checkpoint, group)].append(predicted == rows[0]["category"])
    if not groups:
        raise ValueError("No classification predictions for selected split")
    return {
        "split": split,
        "aggregation": "mean crop scores per video",
        "groups": [
            {
                "method": m,
                "checkpoint_id": c,
                "eval_group": g,
                "videos": len(values),
                "classification_accuracy": sum(values) / len(values),
            }
            for (m, c, g), values in sorted(groups.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, nargs="?")
    parser.add_argument(
        "--classification",
        type=Path,
        help="Optional crop prediction JSONL, pooled per video",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes is None and args.classification is None:
        parser.error("Provide episodes and/or --classification")
    results = {}
    if args.episodes is not None:
        results["robot_episodes"] = evaluate(args.episodes, args.split)
    if args.classification is not None:
        results["human_video_classification"] = evaluate_classification(
            args.classification, args.split
        )
    report = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            f.write(report + "\n")
    else:
        print(report)


if __name__ == "__main__":
    main()
