"""Explicit training building blocks. Nothing runs on import; no model downloads."""

import torch

from .models import bc_loss


def train_classifier_epoch(model, loader, optimizer, device="cpu"):
    """Caller must construct optimizer from model.head_parameters() only."""
    allowed = {id(p) for p in model.head_parameters()}
    if any(
        id(p) not in allowed
        for group in optimizer.param_groups
        for p in group["params"]
    ):
        raise ValueError(
            "Classifier optimizer must contain only linear-head parameters"
        )
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    if not count:
        raise ValueError("No training crops")
    return {"cross_entropy": total / count, "samples": count}


def train_bc_epoch(model, loader, optimizer, device="cpu"):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        predicted = model(batch["image"], batch["state"], batch["category"])
        loss = bc_loss(predicted, batch["actions"], batch["mask"])
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch["image"])
        count += len(batch["image"])
    if not count:
        raise ValueError("No successful training demonstrations")
    return {"masked_action_mse": total / count, "samples": count}
