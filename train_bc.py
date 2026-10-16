"""Train a small action-chunk policy on successful recorded robot episodes."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sorting.datasets import RobotChunkDataset
from sorting.models import SmallBCPolicy, bc_loss
from sorting.learning import train_bc_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=100,
        help="Fixed optimizer budget; sample episodes uniformly then timesteps",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/robot_data"),
        help="Local disk cache for memory-mapped RGB episodes",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new checkpoint filename")
    if min(args.epochs, args.horizon, args.batch_size, args.steps_per_epoch) < 1:
        parser.error("Epochs, horizon and batch size must be positive")
    torch.manual_seed(args.seed)
    if args.episodes.is_file():
        selection = json.loads(args.episodes.read_text())
        folders = [
            (args.episodes.parent / p).resolve() for p in selection["episode_dirs"]
        ]
    else:
        selection = {}
        folders = sorted(p.parent for p in args.episodes.glob("**/episode.json"))
    train = RobotChunkDataset(
        folders, split="train", horizon=args.horizon, cache_dir=args.cache_dir
    )
    val = RobotChunkDataset(
        folders, split="val", horizon=args.horizon, cache_dir=args.cache_dir
    )
    if not len(train) or not len(val):
        parser.error(
            "Need successful train AND val robot episodes, with independent seeds"
        )
    train_seeds = {m["seed"] for m, _ in train.episodes}
    val_seeds = {m["seed"] for m, _ in val.episodes}
    if train_seeds & val_seeds:
        parser.error("Train and val reuse simulation seeds")
    if {m["category"] for m, _ in train.episodes} != {"fruit", "vegetable"}:
        parser.error("Include both categories in successful training episodes")
    model = SmallBCPolicy(horizon=args.horizon).to(args.device)
    states = torch.cat(
        [
            torch.as_tensor(d["state"][:-1].copy(), dtype=torch.float32)
            for _, d in train.episodes
        ]
    ).to(args.device)
    model.set_state_statistics(states)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weights = [1 / len(train.episodes[ep][1]["actions"]) for ep, _ in train.index]
    sampler = WeightedRandomSampler(
        weights, args.steps_per_epoch * args.batch_size, replacement=True
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    best, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        metrics = train_bc_epoch(model, train_loader, optimizer, args.device)
        model.eval()
        total, valid = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                predicted = model(batch["image"], batch["state"], batch["category"])
                count = int(batch["mask"].sum())
                total += (
                    float(bc_loss(predicted, batch["actions"], batch["mask"])) * count
                )
                valid += count
        loss = total / valid
        history.append({"epoch": epoch, **metrics, "val_action_mse": loss})
        print(json.dumps(history[-1]), flush=True)
        if loss < best:
            best = loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "horizon": args.horizon,
            "seed": args.seed,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "categories": ["fruit", "vegetable"],
            "train_seeds": sorted(train_seeds),
            "val_seeds": sorted(val_seeds),
            "training_motion": selection.get("motion"),
            "source_classifier": selection.get("classifier"),
            "val_action_mse": best,
            "history": history,
            "train_episode_ids": [m["episode_id"] for m, _ in train.episodes],
            "val_episode_ids": [m["episode_id"] for m, _ in val.episodes],
            "human_video_ids": sorted(
                {v for m, _ in train.episodes for v in m["human_video_ids"]}
            ),
            "action_convention": "OSC_POSE 0.05m/0.5rad normalized, -1=open +1=close",
        },
        args.output,
    )
    print(f"Saved BC checkpoint to {args.output}; action MSE is not task success")


if __name__ == "__main__":
    main()
