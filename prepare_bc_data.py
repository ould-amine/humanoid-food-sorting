"""Select equal fixed/human successful scene pairs without copying robot images."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument(
        "--classifier", choices=["oracle", "dino", "clip"], default="oracle"
    )
    parser.add_argument("--budget-per-category", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.budget_per_category is not None and args.budget_per_category < 1:
        parser.error("Budget must be positive")
    root = args.experiment.resolve()
    plan = json.loads((root / "plan.json").read_text())
    status = json.loads((root / "deterministic/status.json").read_text())
    scenes = defaultdict(dict)
    counts = defaultdict(int)
    for job in plan["jobs"]:
        if job["classifier"] != args.classifier or job["split"] not in ("train", "val"):
            continue
        counts["planned_" + job["split"]] += 1
        entry = status.get(job["id"])
        if not entry:
            continue
        path = root / Path(entry["episode"])
        if not (path / "episode.json").exists():
            continue
        metadata = json.loads((path / "episode.json").read_text())
        if metadata["task_success"]:
            scenes[job["scene_id"]][job["motion"]] = (job, path)
    grouped = defaultdict(list)
    for scene_id, modes in scenes.items():
        if set(modes) != {"fixed", "human"}:
            continue
        job = modes["fixed"][0]
        grouped[(job["split"], job["category"])].append((scene_id, modes))
    selected = {"fixed": [], "human": []}
    selection_counts = {}
    for split in ("train", "val"):
        quota = min(len(grouped[(split, c)]) for c in ("fruit", "vegetable"))
        if args.budget_per_category is not None:
            quota = min(quota, args.budget_per_category)
        if quota < 1:
            raise ValueError(
                f"Need successful fixed/human pairs of BOTH categories for {split}"
            )
        selection_counts[split + "_episodes_per_category_per_method"] = quota
        for category in ("fruit", "vegetable"):
            for _, modes in sorted(grouped[(split, category)])[:quota]:
                for mode in selected:
                    selected[mode].append(modes[mode][1])
    import os

    args.output.mkdir(parents=True)
    for mode, folders in selected.items():
        payload = {
            "schema_version": 1,
            "classifier": args.classifier,
            "motion": mode,
            "episode_dirs": [
                Path(os.path.relpath(p, args.output.resolve())).as_posix()
                for p in folders
            ],
            "selection_counts": selection_counts,
            "planned_counts": dict(counts),
            "selection_rule": "same successful scene pairs; equal category/episode budgets; excludes all test data",
        }
        (args.output / f"bc_{mode}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(selection_counts, indent=2))
    print("Saved portable episode lists; original failures remain in experiment report")


if __name__ == "__main__":
    main()
