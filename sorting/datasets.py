"""Metadata/crop manifests and episode-level action-chunk datasets."""

import csv
import json
from pathlib import Path

import numpy as np

from .contracts import CATEGORIES


def read_metadata(path):
    """Keep sessions disjoint; object_id is a food type and may cross splits.

    Physical item identity is not available in this schema. eval_group refers to
    exact food_name membership in train, not semantic food-family novelty.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    session_splits = {}
    identities = {}
    for row in rows:
        required = (
            "video_id",
            "video_path",
            "food_name",
            "category",
            "session_id",
            "object_id",
            "split",
            "eval_group",
        )
        if any(not row.get(k) for k in required):
            raise ValueError(f"Incomplete metadata for {row.get('video_id')}")
        if row["video_id"] in seen:
            raise ValueError("Duplicate video_id")
        seen.add(row["video_id"])
        if row["category"] not in CATEGORIES or row["split"] not in (
            "train",
            "val",
            "test",
        ):
            raise ValueError("Unknown category or split")
        if row["eval_group"] not in ("seen", "unseen"):
            raise ValueError("Unknown eval_group")
        session_splits.setdefault(row["session_id"], set()).add(row["split"])
        identities.setdefault(row["object_id"], set()).add(
            (row["food_name"], row["category"])
        )
    if any(len(splits) > 1 for splits in session_splits.values()):
        raise ValueError("A recording session occurs in multiple splits")
    if any(len(labels) > 1 for labels in identities.values()):
        raise ValueError("An object_id maps to inconsistent food names/categories")
    train_foods = {r["food_name"] for r in rows if r["split"] == "train"}
    for row in rows:
        if row["eval_group"] == "unseen" and row["food_name"] in train_foods:
            raise ValueError("Unseen food_name also appears in training")
        if (
            row["split"] != "train"
            and row["eval_group"] == "seen"
            and row["food_name"] not in train_foods
        ):
            raise ValueError("Seen evaluation food_name is absent from training")
    return rows


class CropDataset:
    """JSONL manifest: crop_path, category, video_id, object_id, session_id, split.

    Construct manifests from validated video metadata; never randomly split crops.
    """

    def __init__(self, manifest, split, preprocess):
        self.root = Path(manifest).parent
        self.rows = [
            json.loads(line)
            for line in Path(manifest).read_text().splitlines()
            if line.strip()
        ]
        self.rows = [r for r in self.rows if r["split"] == split]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        with Image.open(self.root / row["crop_path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        return self.preprocess(rgb), CATEGORIES.index(row["category"])


class RobotChunkDataset:
    """Never cross episode boundaries. Padding is masked out in BC loss.

    Images are float32 RGB C,H,W in [0,1]; category is ground truth for training.
    Include failed episodes only explicitly; default is successful demonstrations.
    """

    def __init__(
        self,
        episode_dirs,
        *,
        split="train",
        horizon=8,
        successful_only=True,
        cache_dir=None,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.episodes = []
        self.index = []
        for folder in map(Path, episode_dirs):
            meta = json.loads((folder / "episode.json").read_text())
            if meta["split"] != split or (successful_only and not meta["task_success"]):
                continue
            if cache_dir is None:
                with np.load(folder / "episode.npz") as f:
                    data = {k: f[k].copy() for k in f.files}
            else:
                # Decode each compressed episode once, then memory-map arrays.
                # This avoids keeping all robot RGB videos in Colab RAM.
                import hashlib
                import tempfile
                import shutil

                source = folder / "episode.npz"
                hasher = hashlib.sha256()
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                cache = Path(cache_dir)
                cache.mkdir(parents=True, exist_ok=True)
                destination = cache / hasher.hexdigest()
                keys = ("images", "state", "actions")
                if not destination.exists():
                    temporary = Path(tempfile.mkdtemp(dir=cache))
                    try:
                        with np.load(source) as f:
                            for key in keys:
                                np.save(temporary / f"{key}.npy", f[key])
                        temporary.rename(destination)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                data = {
                    key: np.load(destination / f"{key}.npy", mmap_mode="r")
                    for key in keys
                }
            if len(data["images"]) != len(data["actions"]) + 1:
                raise ValueError("Broken observation/action alignment")
            episode_id = len(self.episodes)
            self.episodes.append((meta, data))
            self.index.extend((episode_id, t) for t in range(len(data["actions"])))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        import torch

        ep, t = self.index[index]
        meta, data = self.episodes[ep]
        chunk = data["actions"][t : t + self.horizon]
        target = np.zeros((self.horizon, 7), dtype=np.float32)
        target[: len(chunk)] = chunk
        mask = np.arange(self.horizon) < len(chunk)
        return {
            "image": torch.from_numpy(data["images"][t].copy()).permute(2, 0, 1).float()
            / 255,
            "state": torch.tensor(data["state"][t], dtype=torch.float32),
            "category": torch.tensor(CATEGORIES.index(meta["category"])),
            "actions": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
        }


class WorldTransitionDataset:
    """Adjacent robot observations for action-conditioned world-model training.

    The caller applies the same frozen visual encoder to image and next_image.
    Multi-step held-out evaluation must use autoregressive predictions separately.
    """

    def __init__(self, episode_dirs, *, split="train"):
        self.base = RobotChunkDataset(
            episode_dirs, split=split, horizon=1, successful_only=False
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        import torch

        ep, t = self.base.index[index]
        _, data = self.base.episodes[ep]
        sample = self.base[index]
        sample["action"] = sample.pop("actions")[0]
        sample.pop("mask")
        sample["next_image"] = (
            torch.from_numpy(data["images"][t + 1].copy()).permute(2, 0, 1).float()
            / 255
        )
        sample["next_state"] = torch.as_tensor(
            data["state"][t + 1], dtype=torch.float32
        )
        return sample
