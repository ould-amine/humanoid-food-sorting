"""Run deterministic two-bin Panda sorting and record complete robot episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import sim_setup  # noqa: F401 -- configure paths before importing LIBERO.
from sorting.scene import SortingScene
from sorting.inference import OracleCategory, DINOHead, CLIPClassifier
from sorting.demonstrations import load_demo
from sorting.runner import collect_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=["fruit", "vegetable"], required=True)
    parser.add_argument(
        "--classifier", choices=["oracle", "dino", "clip"], default="oracle"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--demo", type=Path)
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        help="Optional trained SmallBCPolicy checkpoint",
    )
    parser.add_argument("--execute-chunk-steps", type=int, default=1)
    parser.add_argument(
        "--oracle-target",
        choices=["fruit", "vegetable"],
        help="Oracle-only negative control: deliberately route to this category",
    )
    parser.add_argument(
        "--texture",
        type=Path,
        help="Real RGB food crop for the cuboid proxy; not a realistic food mesh",
    )
    parser.add_argument("--source-video", help="Provenance of texture, if provided")
    parser.add_argument("--food-name", default="food_proxy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--perception-view", choices=["closeup", "agentview"], default="closeup"
    )
    parser.add_argument("--perception-size", type=int, default=512)
    parser.add_argument("--perception-fovy", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap-baskets", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--eval-group", choices=["seen", "unseen"], default="seen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.perception_size < 64 or not 1 <= args.perception_fovy < 120:
        parser.error("Perception size must be >=64; fovy must be in [1,120)")
    if args.output.exists():
        parser.error("Choose a new --output directory")
    if args.classifier == "dino" and args.checkpoint is None:
        parser.error("--classifier dino requires --checkpoint")
    if args.demo and args.policy_checkpoint:
        parser.error("Choose a deterministic demo or a learned policy, not both")
    if args.oracle_target and args.classifier != "oracle":
        parser.error(
            "--oracle-target is a control diagnostic, only for --classifier oracle"
        )
    if args.texture and (not args.texture.is_file() or not args.source_video):
        parser.error("Texture must exist and have --source-video provenance")
    demo = load_demo(str(args.demo)) if args.demo else None
    if demo and not demo.approved:
        parser.error("Review and approve the abstract demo first")
    if args.classifier == "oracle":
        classifier = OracleCategory(args.oracle_target or args.category)
    elif args.classifier == "dino":
        classifier = DINOHead(args.checkpoint, args.device)
    else:
        classifier = CLIPClassifier(args.device)
    policy = None
    policy_data = None
    if args.policy_checkpoint:
        import torch
        from sorting.models import SmallBCPolicy

        policy_data = torch.load(
            args.policy_checkpoint, map_location="cpu", weights_only=True
        )
        policy = SmallBCPolicy(horizon=policy_data["horizon"]).to(args.device)
        policy.load_state_dict(policy_data["state_dict"])
    scene = SortingScene(
        args.category,
        seed=args.seed,
        swap=args.swap_baskets,
        texture=args.texture,
        randomize=args.randomize,
        perception_view=args.perception_view,
        perception_size=args.perception_size,
        perception_fovy=args.perception_fovy,
    )
    metadata = {
        "episode_id": args.output.name,
        "method": f"{args.classifier}_{'bc' if policy else 'human' if demo else 'fixed'}",
        "food_name": args.food_name,
        "object_id": args.food_name,
        "session_id": f"sim_{args.seed}",
        "split": args.split,
        "eval_group": args.eval_group,
        "category": args.category,
        "instruction": "Sort the food into its category basket.",
        "seed": args.seed,
        "human_video_ids": [args.source_video] if args.source_video else [],
        "checkpoint_id": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if args.checkpoint
        else args.classifier,
        "scene_type": "custom_robosuite_food_proxy",
        "scene_version": "perception_camera_v3",
        "oracle_category": args.classifier == "oracle",
        "crop_source": "simulator_segmentation",
        "swap_baskets": args.swap_baskets,
        "texture_path": str(args.texture) if args.texture else None,
    }
    if policy_data:
        metadata["classifier_checkpoint_id"] = metadata["checkpoint_id"]
        metadata["checkpoint_id"] = hashlib.sha256(
            args.policy_checkpoint.read_bytes()
        ).hexdigest()
        metadata["human_video_ids"] = sorted(
            set(metadata["human_video_ids"] + policy_data["human_video_ids"])
        )
    summary = collect_episode(
        scene,
        classifier,
        metadata,
        str(args.output),
        demo=demo,
        policy=policy,
        max_steps=800,
        execute_chunk_steps=args.execute_chunk_steps,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "predicted_category",
                    "pick_success",
                    "placed_basket",
                    "task_success",
                    "trajectory_length_m",
                    "failure_reason",
                )
            },
            indent=2,
        )
    )
    if not summary["task_success"]:
        raise SystemExit(
            "Episode recorded, but sorting failed; inspect rollout and events"
        )


if __name__ == "__main__":
    main()
