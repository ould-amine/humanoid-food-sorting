"""Plan, resume and report matched sorting experiments. Planning runs no models."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(value):
    return ROOT / value  # Absolute external paths remain absolute.


def portable_source(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def plan(args):
    if args.output.exists():
        raise FileExistsError("Use a new experiment directory")
    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    seeds = {"train": args.train_seeds, "val": args.val_seeds, "test": args.test_seeds}
    for a in args.splits:
        for b in args.splits:
            if a != b and set(seeds[a]) & set(seeds[b]):
                raise ValueError("Seeds must be disjoint across splits")
    demo = json.loads(args.demo.read_text()) if "human" in args.motions else None
    if demo and not demo["approved"]:
        raise ValueError("Review human demo before planning")
    if demo:
        training_ids = {r["video_id"] for r in rows if r["split"] == "train"}
        if demo["video_id"] not in training_ids:
            raise ValueError("Motion template must come from a training video")
    if "dino" in args.classifiers and args.checkpoint is None:
        raise ValueError("DINO requires --checkpoint")
    jobs, inventory = [], {}
    # Keep texture source videos in their original split. Pick each video's first
    # available crop, then choose among videos of the same type deterministically.
    first = {}
    for row in rows:
        key = row["video_id"]
        if key not in first or int(row["frame_index"]) < int(first[key]["frame_index"]):
            first[key] = row
    for split in args.splits:
        candidates = defaultdict(list)
        for row in first.values():
            if row["split"] == split and (
                not args.food_names or row["food_name"] in args.food_names
            ):
                candidates[row["food_name"]].append(row)
        if not candidates:
            raise ValueError(f"No source videos for {split}")
        inventory[split] = sorted(candidates)
        for food, videos in sorted(candidates.items()):
            videos.sort(key=lambda r: r["video_id"])
            for seed in seeds[split]:
                rng = random.Random(f"{split}:{food}:{seed}")
                source = rng.choice(videos)
                texture = (args.manifest.parent / source["crop_path"]).resolve()
                for swap in (False, True):
                    scene_id = f"{split}_{food}_{seed}_{'swap' if swap else 'normal'}"
                    for classifier in args.classifiers:
                        for motion in args.motions:
                            job_id = f"{classifier}_{motion}_{scene_id}"
                            jobs.append(
                                {
                                    "id": job_id,
                                    "scene_id": scene_id,
                                    "classifier": classifier,
                                    "motion": motion,
                                    "split": split,
                                    "seed": seed,
                                    "swap": swap,
                                    "food_name": food,
                                    "category": source["category"],
                                    "eval_group": source["eval_group"],
                                    "source_video": source["video_id"],
                                    "source_session": source["session_id"],
                                    "texture": portable_source(texture),
                                    "texture_sha256": digest(texture),
                                }
                            )
    configuration = {
        "schema_version": 1,
        "manifest_sha256": digest(args.manifest),
        "checkpoint": portable_source(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
        "demo": portable_source(args.demo) if demo else None,
        "demo_sha256": digest(args.demo) if demo else None,
        "scene_version": "perception_camera_v3",
        "perception_view": args.perception_view,
        "randomize": True,
        "inventory": inventory,
        "code_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [ROOT / "run_sorting.py", *sorted((ROOT / "sorting").glob("*.py"))]
        },
        "jobs": jobs,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", configuration)
    print(f"Planned {len(jobs)} episodes; no simulation or inference started")
    print("Inventory:", inventory)


def execute(args):
    root = args.experiment.resolve()
    configuration = json.loads((root / "plan.json").read_text())
    for relative, expected in configuration["code_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(
                f"Code changed since planning: {relative}; create a new plan"
            )
    for kind in ("checkpoint", "demo"):
        if (
            configuration[kind]
            and digest(source_path(configuration[kind]))
            != configuration[f"{kind}_sha256"]
        ):
            raise ValueError(f"{kind} changed since planning")
    policy_path = getattr(args, "policy_checkpoint", None)
    policy_hash = digest(policy_path) if policy_path else None
    if policy_path:
        import torch

        policy = torch.load(policy_path, map_location="cpu", weights_only=True)
        if policy.get("training_motion") and policy["training_motion"] != args.motion:
            raise ValueError("BC checkpoint training motion does not match --motion")
        training_seeds = set(policy.get("train_seeds", [])) | set(
            policy.get("val_seeds", [])
        )
        if any(
            j["split"] == "test" and j["seed"] in training_seeds
            for j in configuration["jobs"]
        ):
            raise ValueError("BC evaluation seeds overlap training/validation")
    branch = root / ("bc_" + policy_hash[:12] if policy_path else "deterministic")
    branch.mkdir(exist_ok=True)
    status_path = branch / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    launched = 0
    for job in configuration["jobs"]:
        if policy_path and (job["split"] != "test" or job["motion"] != args.motion):
            continue
        # Resume terminal successes AND failures. Never overwrite partial data.
        if job["id"] in status:
            continue
        if args.limit is not None and launched >= args.limit:
            break
        if digest(source_path(job["texture"])) != job["texture_sha256"]:
            raise ValueError("Source texture changed since planning")
        output = branch / "episodes" / job["id"]
        if output.exists():
            raise FileExistsError(
                f"Unindexed partial episode: {output}; inspect it before resuming"
            )
        command = [
            sys.executable,
            str(ROOT / "run_sorting.py"),
            "--classifier",
            job["classifier"],
            "--category",
            job["category"],
            "--food-name",
            job["food_name"],
            "--source-video",
            job["source_video"],
            "--texture",
            job["texture"],
            "--seed",
            str(job["seed"]),
            "--split",
            job["split"],
            "--eval-group",
            job["eval_group"],
            "--randomize",
            "--perception-view",
            configuration["perception_view"],
            "--device",
            args.device,
            "--output",
            str(output),
        ]
        if job["swap"]:
            command.append("--swap-baskets")
        if job["classifier"] == "dino":
            command += ["--checkpoint", configuration["checkpoint"]]
        if policy_path:
            command += ["--policy-checkpoint", str(policy_path.resolve())]
        elif job["motion"] == "human":
            command += ["--demo", configuration["demo"]]
        logs = branch / "logs"
        logs.mkdir(exist_ok=True)
        print("Running", job["id"], flush=True)
        with (logs / (job["id"] + ".txt")).open("w") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        metadata_file = output / "episode.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            metadata["scene_id"] = job["scene_id"]
            metadata["source_session"] = job["source_session"]
            metadata["planned_motion"] = job["motion"]
            write_json(metadata_file, metadata)
        status[job["id"]] = {
            "exit_code": process.returncode,
            "episode": output.relative_to(root).as_posix(),
            "recorded": metadata_file.exists(),
            "policy_sha256": policy_hash,
        }
        write_json(status_path, status)
        launched += 1
        if not metadata_file.exists():
            print(
                "Infrastructure failure; stopping. Inspect log and use a new plan after fixing the cause.",
                flush=True,
            )
            break
    print("Recorded status for", len(status), "jobs in", branch)


def summarize(args):
    root = args.experiment.resolve()
    configuration = json.loads((root / "plan.json").read_text())
    jobs = {j["id"]: j for j in configuration["jobs"]}
    groups = defaultdict(list)
    for status_file in sorted(root.glob("*/status.json")):
        branch = status_file.parent.name
        for identity, status in json.loads(status_file.read_text()).items():
            job = jobs[identity]
            path = root / Path(status["episode"]) / "episode.json"
            metadata = json.loads(path.read_text()) if path.exists() else None
            key = (
                branch,
                job["classifier"],
                job["motion"],
                job["split"],
                job["eval_group"],
            )
            groups[key].append((job, status, metadata))
    reports = []
    for key, values in sorted(groups.items()):
        recorded = [m for _, _, m in values if m]
        total = len(values)
        failures = Counter(
            "infrastructure_failure"
            if m is None
            else (
                m["failure_reason"]
                or ("success" if m["task_success"] else "unspecified_failure")
            )
            for _, _, m in values
        )
        reports.append(
            {
                "branch": key[0],
                "classifier": key[1],
                "motion": key[2],
                "split": key[3],
                "eval_group": key[4],
                "attempted": total,
                "recorded": len(recorded),
                "infrastructure_failures": total - len(recorded),
                "classification_accuracy_recorded": sum(
                    m["category"] == m["predicted_category"] for m in recorded
                )
                / len(recorded)
                if recorded
                else None,
                "classification_errors_recorded": sum(
                    m["category"] != m["predicted_category"] for m in recorded
                ),
                "task_success_rate_all_attempted": sum(
                    m["task_success"] for m in recorded
                )
                / total,
                "pick_success_rate_all_attempted": sum(
                    m["pick_success"] for m in recorded
                )
                / total,
                "correct_basket_rate_all_attempted": sum(
                    m["correct_basket"] for m in recorded
                )
                / total,
                "trajectory_length_m_mean_recorded": sum(
                    m["trajectory_length_m"] for m in recorded
                )
                / len(recorded)
                if recorded
                else None,
                "failures": dict(failures),
                "scene_ids": [j["scene_id"] for j, _, _ in values],
            }
        )
    paired = []
    from itertools import combinations

    for (key_a, values_a), (key_b, values_b) in combinations(sorted(groups.items()), 2):
        if key_a[3:] != key_b[3:]:
            continue
        a = {j["scene_id"]: m for j, _, m in values_a}
        b = {j["scene_id"]: m for j, _, m in values_b}
        shared = sorted(a.keys() & b.keys())
        if not shared:
            continue

        def success(metadata):
            return bool(metadata and metadata["task_success"])

        wins_a = sum(success(a[s]) and not success(b[s]) for s in shared)
        wins_b = sum(success(b[s]) and not success(a[s]) for s in shared)
        paired.append(
            {
                "a": list(key_a[:3]),
                "b": list(key_b[:3]),
                "split": key_a[3],
                "eval_group": key_a[4],
                "shared_scenes": len(shared),
                "a_only_success": wins_a,
                "b_only_success": wins_b,
                "both_success": sum(success(a[s]) and success(b[s]) for s in shared),
                "neither_success": sum(
                    not success(a[s]) and not success(b[s]) for s in shared
                ),
                "a_minus_b_success_rate": (wins_a - wins_b) / len(shared),
            }
        )
    deterministic_status = root / "deterministic/status.json"
    attempted = (
        len(json.loads(deterministic_status.read_text()))
        if deterministic_status.exists()
        else 0
    )
    result = {
        "planned_deterministic": len(jobs),
        "pending_deterministic": len(jobs) - attempted,
        "groups": reports,
        "paired_comparisons": paired,
        "oracle_is_control_diagnostic": True,
        "note": "Pending jobs excluded; infrastructure failures count against all-attempted success. Classification denominator is recorded episodes.",
    }
    write_json(root / "report.json", result)
    lines = [
        "# Sorting experiment report",
        "",
        "Oracle-category groups are control diagnostics.",
        "",
        "| Branch | Classifier | Motion | Split | Group | Attempted | Class acc. | Task success |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in reports:
        acc = (
            "n/a"
            if r["classification_accuracy_recorded"] is None
            else f"{100 * r['classification_accuracy_recorded']:.1f}%"
        )
        lines.append(
            f"| {r['branch']} | {r['classifier']} | {r['motion']} | {r['split']} | {r['eval_group']} | {r['attempted']} | {acc} | {100 * r['task_success_rate_all_attempted']:.1f}% |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("Saved", root / "report.json", "and report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/crops/manifest.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--demo", type=Path, default=ROOT / "data/demos/banana_02_reviewed.json"
    )
    p.add_argument(
        "--classifiers",
        nargs="+",
        choices=["oracle", "dino", "clip"],
        default=["oracle", "dino", "clip"],
    )
    p.add_argument(
        "--motions", nargs="+", choices=["fixed", "human"], default=["fixed", "human"]
    )
    p.add_argument("--food-names", nargs="+")
    p.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"], default=["val"]
    )
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--val-seeds", type=int, nargs="+", default=[6])
    p.add_argument("--test-seeds", type=int, nargs="+", default=[8])
    p.add_argument(
        "--perception-view", choices=["closeup", "agentview"], default="closeup"
    )
    for command in ("run", "evaluate-bc"):
        p = sub.add_parser(command)
        p.add_argument("experiment", type=Path)
        p.add_argument("--device", default="cpu")
        p.add_argument("--limit", type=int)
        if command == "evaluate-bc":
            p.add_argument("--policy-checkpoint", type=Path, required=True)
            p.add_argument("--motion", choices=["fixed", "human"], required=True)
    p = sub.add_parser("report")
    p.add_argument("experiment", type=Path)
    args = parser.parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    {"plan": plan, "run": execute, "report": summarize, "evaluate-bc": execute}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
