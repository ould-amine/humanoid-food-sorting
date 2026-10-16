"""Optional frozen DINOv2 + linear head and CLIP zero-shot adapters.

No torch.hub / from_pretrained calls: callers explicitly load local models and
their matching preprocessing. Importing the module downloads nothing.
"""

import numpy as np
import torch
from torch import nn

from .contracts import CATEGORIES, Prediction


class ConstantFruitBaseline:
    """No semantic classifier: always route to the fruit basket."""

    def predict(self, rgb_crop):
        return Prediction("fruit", {"fruit": 1.0, "vegetable": 0.0}, "constant_fruit")


class FrozenDINOClassifier(nn.Module):
    def __init__(
        self, backbone: nn.Module, embedding_dim: int, preprocess, device="cpu"
    ):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Linear(embedding_dim, len(CATEGORIES))
        self.preprocess = preprocess  # RGB ndarray -> C,H,W tensor.
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Frozen includes dropout/batchnorm behaviour.
        return self

    def forward(self, images):
        with torch.no_grad():
            embeddings = self.backbone(images)
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError("DINO adapter must return B,d pooled embeddings")
        return self.head(embeddings)

    def head_parameters(self):
        return self.head.parameters()

    @torch.inference_mode()
    def predict(self, rgb_crop: np.ndarray) -> Prediction:
        self.eval()
        device = self.head.weight.device
        probs = (
            self(self.preprocess(rgb_crop).unsqueeze(0).to(device))
            .softmax(-1)[0]
            .cpu()
            .tolist()
        )
        return Prediction(
            CATEGORIES[int(np.argmax(probs))],
            dict(zip(CATEGORIES, probs)),
            "dinov2_linear",
        )


class CLIPZeroShot:
    """OpenAI CLIP model + injected preprocessing and tokenizer.

    These two-prompt softmax scores are relative scores, not calibrated confidence.
    """

    def __init__(self, model, preprocess, tokenize, device="cpu"):
        self.model = model.to(device).eval().requires_grad_(False)
        self.preprocess, self.device = preprocess, device
        self.prompts = ["a fruit", "a vegetable"]
        with torch.inference_mode():
            text = self.model.encode_text(tokenize(self.prompts).to(device)).float()
            self.text = nn.functional.normalize(text, dim=-1)

    @torch.inference_mode()
    def predict(self, rgb_crop):
        image = self.preprocess(rgb_crop).unsqueeze(0).to(self.device)
        image = nn.functional.normalize(self.model.encode_image(image).float(), dim=-1)
        logits = self.model.logit_scale.exp() * image @ self.text.T
        probs = logits.softmax(-1)[0].cpu().tolist()
        return Prediction(
            CATEGORIES[int(np.argmax(probs))],
            dict(zip(CATEGORIES, probs)),
            "clip_zero_shot",
        )
