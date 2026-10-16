"""Explicit classification CLI: CLIP, frozen DINO features, linear head.

Model commands download pretrained weights on first execution. Nothing runs on
import. Test data is excluded unless --splits test / --split test is requested.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from sorting.contracts import CATEGORIES


def read_manifest(path, splits):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["split"] in splits]
    if not rows:
        raise ValueError(f"No crops for splits {splits}")
    seen = set()
    video_labels = {}
    for row in rows:
        identity = (row["video_id"], row["frame_index"])
        if identity in seen:
            raise ValueError(f"Duplicate crop: {identity}")
        seen.add(identity)
        if row["category"] not in CATEGORIES:
            raise ValueError("Unknown category")
        labels = (row["category"], row["split"], row["session_id"], row["eval_group"])
        if row["video_id"] in video_labels and video_labels[row["video_id"]] != labels:
            raise ValueError("Inconsistent metadata within a video")
        video_labels[row["video_id"]] = labels
        if not (path.parent / row["crop_path"]).is_file():
            raise FileNotFoundError(row["crop_path"])
    return rows


def images_for(rows, manifest):
    images = []
    for row in rows:
        with Image.open(manifest.parent / row["crop_path"]) as image:
            images.append(image.convert("RGB"))
    return images


def reserve_output(path):
    if path.exists():
        raise FileExistsError(f"{path}: choose a new output name")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_predictions(path, rows, probabilities, method, checkpoint_id):
    reserve_output(path)
    with path.open("x") as f:
        for row, scores in zip(rows, probabilities.tolist()):
            result = {
                **row,
                "method": method,
                "checkpoint_id": checkpoint_id,
                "probabilities": dict(zip(CATEGORIES, scores)),
            }
            f.write(json.dumps(result) + "\n")


@torch.inference_mode()
def clip_predictions(args):
    import clip

    reserve_output(args.output)
    rows = read_manifest(args.manifest, args.splits)
    model, preprocess = clip.load(
        args.model, device=args.device, download_root=str(args.cache / "clip")
    )
    model.eval()
    text = model.encode_text(
        clip.tokenize(["a fruit", "a vegetable"]).to(args.device)
    ).float()
    text = F.normalize(text, dim=-1)
    probabilities = []
    for start in range(0, len(rows), args.batch_size):
        images = images_for(rows[start : start + args.batch_size], args.manifest)
        batch = torch.stack([preprocess(im) for im in images]).to(args.device)
        features = F.normalize(model.encode_image(batch).float(), dim=-1)
        probabilities.append(
            (model.logit_scale.exp() * features @ text.T).softmax(-1).cpu()
        )
        print(
            f"CLIP: {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True
        )
    write_predictions(
        args.output,
        rows,
        torch.cat(probabilities),
        "clip_zero_shot",
        f"openai_clip:{args.model}",
    )


@torch.inference_mode()
def dino_features(args):
    from transformers import AutoImageProcessor, AutoModel

    reserve_output(args.output)
    if args.reference_checkpoint is not None:
        checkpoint = torch.load(
            args.reference_checkpoint, map_location="cpu", weights_only=True
        )
        args.model, args.revision = checkpoint["model"], checkpoint["revision"]
    rows = read_manifest(args.manifest, args.splits)
    processor = AutoImageProcessor.from_pretrained(
        args.model, revision=args.revision, cache_dir=str(args.cache)
    )
    model = (
        AutoModel.from_pretrained(
            args.model, revision=args.revision, cache_dir=str(args.cache)
        )
        .to(args.device)
        .eval()
    )
    model.requires_grad_(False)
    embeddings = []
    for start in range(0, len(rows), args.batch_size):
        images = images_for(rows[start : start + args.batch_size], args.manifest)
        batch = processor(images=images, return_tensors="pt").to(args.device)
        # CLS embedding from the frozen encoder; no image labels enter the model.
        features = model(**batch).last_hidden_state[:, 0].float()
        embeddings.append(F.normalize(features, dim=-1).cpu())
        print(
            f"DINOv2: {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True
        )
    payload = {
        "features": torch.cat(embeddings),
        "rows": rows,
        "model": args.model,
        "revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "normalization": "l2",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    torch.save(payload, args.output)
    print(f"Saved embeddings to {args.output}")


def pooled_metrics(probabilities, rows):
    videos = defaultdict(list)
    labels = {}
    for scores, row in zip(probabilities, rows):
        video = row["video_id"]
        label = CATEGORIES.index(row["category"])
        if video in labels and labels[video] != label:
            raise ValueError("Conflicting labels within a video")
        labels[video] = label
        videos[video].append(scores)
    correct, loss = 0, 0.0
    for video, scores in videos.items():
        probs = torch.stack(scores).mean(0)
        correct += int(probs.argmax().item() == labels[video])
        loss -= float(probs[labels[video]].clamp_min(1e-8).log())
    return {
        "video_accuracy": correct / len(videos),
        "video_cross_entropy": loss / len(videos),
        "videos": len(videos),
    }


def train_head(args):
    reserve_output(args.output)
    torch.manual_seed(args.seed)
    data = torch.load(args.features, map_location="cpu", weights_only=True)
    rows, features = data["rows"], data["features"]
    if (
        features.ndim != 2
        or len(features) != len(rows)
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Invalid cached embeddings")
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val = [i for i, r in enumerate(rows) if r["split"] == "val"]
    if not train or not val:
        raise ValueError("Feature cache must contain train and val crops")
    for key in ("video_id", "session_id"):
        if {rows[i][key] for i in train} & {rows[i][key] for i in val}:
            raise ValueError(f"Train/val share {key}")
    if {rows[i]["category"] for i in train} != set(CATEGORIES):
        raise ValueError("Training requires both categories")
    x = features[train].to(args.device)
    y = torch.tensor(
        [CATEGORIES.index(rows[i]["category"]) for i in train], device=args.device
    )
    val_x = features[val].to(args.device)
    val_rows = [rows[i] for i in val]
    # Give each training video equal total weight, regardless of duration.
    counts = Counter(rows[i]["video_id"] for i in train)
    weights = torch.tensor(
        [1 / counts[rows[i]["video_id"]] for i in train], device=args.device
    )
    head = nn.Linear(features.shape[1], 2).to(args.device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_score, best_loss, best_state, best_epoch = -1.0, float("inf"), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        losses = F.cross_entropy(head(x), y, reduction="none")
        loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            metrics = pooled_metrics(head(val_x).softmax(-1).cpu(), val_rows)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), **metrics})
        print(json.dumps(history[-1]), flush=True)
        score, val_loss = metrics["video_accuracy"], metrics["video_cross_entropy"]
        if score > best_score or (score == best_score and val_loss < best_loss):
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
    checkpoint = {
        "head": best_state,
        "embedding_dim": features.shape[1],
        "categories": list(CATEGORIES),
        "model": data["model"],
        "revision": data["revision"],
        "normalization": data["normalization"],
        "manifest_sha256": data["manifest_sha256"],
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_video_accuracy": best_score,
        "train_video_ids": sorted({rows[i]["video_id"] for i in train}),
        "val_video_ids": sorted({r["video_id"] for r in val_rows}),
        "history": history,
    }
    torch.save(checkpoint, args.output)
    print(f"Saved best head (epoch {best_epoch}) to {args.output}")


@torch.inference_mode()
def predict_head(args):
    reserve_output(args.output)
    data = torch.load(args.features, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["categories"] != list(CATEGORIES):
        raise ValueError("Checkpoint category order mismatch")
    for key in ("model", "revision", "normalization"):
        if data[key] != checkpoint[key]:
            raise ValueError(f"Feature/checkpoint {key} mismatch")
    indices = [i for i, r in enumerate(data["rows"]) if r["split"] == args.split]
    if not indices:
        raise ValueError("No features for selected split")
    head = nn.Linear(checkpoint["embedding_dim"], 2).to(args.device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    probs = head(data["features"][indices].to(args.device)).softmax(-1).cpu()
    checkpoint_id = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    write_predictions(
        args.output,
        [data["rows"][i] for i in indices],
        probs,
        "dinov2_linear",
        checkpoint_id,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("clip", "embed", "train-head", "predict-head"):
        p = sub.add_parser(command)
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
        p.add_argument("--output", type=Path, required=True)
        if command in ("clip", "embed"):
            p.add_argument(
                "--manifest", type=Path, default=Path("data/crops/manifest.jsonl")
            )
            p.add_argument(
                "--splits",
                nargs="+",
                choices=["train", "val", "test"],
                default=["val"] if command == "clip" else ["train", "val"],
            )
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--cache", type=Path, default=Path(".cache/models"))
            p.add_argument(
                "--model",
                default="ViT-B/32" if command == "clip" else "facebook/dinov2-small",
            )
        if command == "embed":
            p.add_argument(
                "--revision",
                default="main",
                help="Use the saved model commit for later test extraction",
            )
            p.add_argument(
                "--reference-checkpoint",
                type=Path,
                help="Use the exact backbone model/revision recorded in a trained head",
            )
        if command in ("train-head", "predict-head"):
            p.add_argument("--features", type=Path, required=True)
        if command == "train-head":
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--lr", type=float, default=0.001)
            p.add_argument("--weight-decay", type=float, default=0.01)
            p.add_argument("--seed", type=int, default=0)
        if command == "predict-head":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--split", choices=["train", "val", "test"], default="val")
    args = parser.parse_args()
    if getattr(args, "batch_size", 1) < 1 or getattr(args, "epochs", 1) < 1:
        parser.error("Batch size and epochs must be positive")
    {
        "clip": clip_predictions,
        "embed": dino_features,
        "train-head": train_head,
        "predict-head": predict_head,
    }[args.command](args)


if __name__ == "__main__":
    main()
