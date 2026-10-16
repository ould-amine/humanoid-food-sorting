"""Classifiers for rollout integration; weights load only on explicit construction."""

import numpy as np

from .contracts import CATEGORIES, Prediction


class OracleCategory:
    """Control diagnostic only; intentionally exposes supplied category."""

    def __init__(self, category):
        self.category = category

    def predict(self, image):
        return Prediction(
            self.category,
            {c: float(c == self.category) for c in CATEGORIES},
            "oracle_category_diagnostic",
        )


class DINOHead:
    def __init__(self, checkpoint, device="cpu", cache=".cache/models"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch, self.device = torch, device
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["categories"] != list(CATEGORIES) or data["normalization"] != "l2":
            raise ValueError("Unsupported classifier checkpoint conventions")
        self.processor = AutoImageProcessor.from_pretrained(
            data["model"], revision=data["revision"], cache_dir=cache
        )
        self.model = (
            AutoModel.from_pretrained(
                data["model"], revision=data["revision"], cache_dir=cache
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.head = torch.nn.Linear(data["embedding_dim"], 2).to(device)
        self.head.load_state_dict(data["head"])
        self.head.eval()

    def predict(self, image):
        with self.torch.inference_mode():
            batch = self.processor(images=image, return_tensors="pt").to(self.device)
            embedding = self.model(**batch).last_hidden_state[:, 0].float()
            embedding = self.torch.nn.functional.normalize(embedding, dim=-1)
            scores = self.head(embedding).softmax(-1)[0].cpu().tolist()
        return Prediction(
            CATEGORIES[int(np.argmax(scores))],
            dict(zip(CATEGORIES, scores)),
            "dinov2_simulator_crop",
        )


class CLIPClassifier:
    def __init__(self, device="cpu"):
        import clip
        from PIL import Image
        from .classification import CLIPZeroShot

        model, preprocess = clip.load(
            "ViT-B/32", device=device, download_root=".cache/models/clip"
        )
        self.classifier = CLIPZeroShot(
            model, lambda rgb: preprocess(Image.fromarray(rgb)), clip.tokenize, device
        )

    def predict(self, image):
        return self.classifier.predict(image)
