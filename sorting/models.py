"""Small BC policy, optional SmolVLA bridge and action-conditioned latent WM."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import CATEGORIES


class SmallBCPolicy(nn.Module):
    def __init__(self, state_dim=9, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.category = nn.Embedding(2, 8)
        self.head = nn.Sequential(
            nn.Linear(64 + state_dim + 8, 128),
            nn.ReLU(),
            nn.Linear(128, horizon * 7),
            nn.Tanh(),
        )
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def set_state_statistics(self, training_states):
        """Fit exclusively on training episodes; serialized in state_dict."""
        with torch.no_grad():
            self.state_mean.copy_(training_states.mean(0))
            self.state_std.copy_(training_states.std(0, unbiased=False).clamp_min(1e-6))

    def forward(self, image, state, category):
        image = F.interpolate(
            image, size=(96, 96), mode="bilinear", align_corners=False
        )
        features = torch.cat(
            [
                self.encoder(image),
                (state - self.state_mean) / self.state_std,
                self.category(category),
            ],
            dim=-1,
        )
        return self.head(features).reshape(-1, self.horizon, 7)

    def reset(self):
        pass  # Stateless; runner controls chunk execution/replanning.

    @torch.inference_mode()
    def act(self, observation, category, instruction=""):
        self.eval()
        device = self.state_mean.device
        image = (
            torch.from_numpy(observation.image.copy())
            .permute(2, 0, 1)
            .float()[None]
            .to(device)
            / 255
        )
        state = torch.as_tensor(observation.state, dtype=torch.float32, device=device)[
            None
        ]
        label = torch.tensor([CATEGORIES.index(category)], device=device)
        return self(image, state, label)[0].cpu().numpy()


def bc_loss(prediction, target, valid_mask):
    """Padded timesteps contribute no loss. Gripper remains a 7th regression target."""
    mask = valid_mask.unsqueeze(-1).to(prediction.dtype)
    return (((prediction - target) ** 2) * mask).sum() / (mask.sum() * 7).clamp_min(1)


class SmolVLAAdapter:
    """Inject an already loaded policy and its checkpoint-specific processors.

    prepare(obs, category, instruction) -> LeRobot batch
    infer(policy, batch) -> model action/chunk
    decode(output) -> H,7 *denormalized to this project's normalized OSC space*

    Mapping cameras, state dimensions, gripper sign and action normalization is
    checkpoint-specific. A pretrained checkpoint is not assumed Panda-compatible.
    Fine-tuning/export to LeRobot is a separate integration milestone.
    """

    def __init__(self, policy, prepare, infer, decode):
        self.policy = policy
        self.prepare, self.infer, self.decode = prepare, infer, decode

    def reset(self):
        self.policy.reset()

    @torch.inference_mode()
    def act(self, observation, category, instruction):
        output = self.infer(
            self.policy, self.prepare(observation, category, instruction)
        )
        actions = np.asarray(self.decode(output), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("SmolVLA decode must produce finite H,7 OSC actions")
        if np.any(np.abs(actions) > 1):
            raise ValueError(
                "SmolVLA actions exceed this controller's normalized bounds"
            )
        return actions


class LatentWorldModel(nn.Module):
    """Predict next visual embedding + robot state, conditioned on action/category.

    Embeddings are targets from a frozen encoder; no pixel/video prediction claim.
    A trainable image decoder can later support image prediction with its own loss.
    """

    def __init__(self, embedding_dim, state_dim=9, hidden_dim=128):
        super().__init__()
        self.embedding_dim, self.state_dim = embedding_dim, state_dim
        self.category = nn.Embedding(2, 8)
        total = embedding_dim + state_dim
        self.dynamics = nn.Sequential(
            nn.Linear(total + 7 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total),
        )

    def forward(self, embedding, state, action, category):
        current = torch.cat([embedding, state], -1)
        inputs = torch.cat([current, action, self.category(category)], -1)
        predicted = current + self.dynamics(inputs)
        return predicted[..., : self.embedding_dim], predicted[
            ..., self.embedding_dim :
        ]

    def rollout(self, embedding, state, actions, category):
        """actions: B,H,7; autoregressive rather than teacher-forced prediction."""
        embeddings, states = [], []
        for t in range(actions.shape[1]):
            embedding, state = self(embedding, state, actions[:, t], category)
            embeddings.append(embedding)
            states.append(state)
        return torch.stack(embeddings, 1), torch.stack(states, 1)


def world_model_loss(
    pred_embedding, pred_state, target_embedding, target_state, state_mean, state_std
):
    """Training-only normalization statistics; embedding/state losses reported separately."""
    visual = F.mse_loss(pred_embedding, target_embedding.detach())
    state = F.mse_loss(
        (pred_state - state_mean) / state_std.clamp_min(1e-6),
        (target_state - state_mean) / state_std.clamp_min(1e-6),
    )
    return {
        "loss": visual + state,
        "embedding_mse": visual,
        "normalized_state_mse": state,
    }
