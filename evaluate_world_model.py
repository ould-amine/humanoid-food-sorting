"""Offline WM errors from exported predictions; no inference is run.

NPZ: predicted_embedding/target_embedding (N,H,d), predicted_state/target_state
(N,H,9). States in original robot units. Optional predicted_rgb/target_rgb in
[0,1] with shape N,H,height,width,3, produced by a separately trained decoder.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(path):
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    result = {}
    for kind in ("embedding", "state"):
        predicted, target = arrays[f"predicted_{kind}"], arrays[f"target_{kind}"]
        if (
            predicted.shape != target.shape
            or predicted.ndim != 3
            or min(predicted.shape) == 0
        ):
            raise ValueError(f"Expected matching nonempty N,H,d {kind} arrays")
        if not np.isfinite(predicted).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite predictions/targets")
        if kind == "embedding":
            result["embedding_mse_by_horizon"] = (
                ((predicted - target) ** 2).mean(axis=(0, 2)).tolist()
            )
        else:
            if predicted.shape[-1] != 9:
                raise ValueError("Expected the project's 9D robot state")
            result["eef_error_m_by_horizon"] = (
                np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1)
                .mean(0)
                .tolist()
            )
    if arrays["predicted_embedding"].shape[:2] != arrays["predicted_state"].shape[:2]:
        raise ValueError("Embedding/state horizons must agree")
    if "predicted_rgb" in arrays or "target_rgb" in arrays:
        p, t = arrays["predicted_rgb"], arrays["target_rgb"]
        if (
            p.shape != t.shape
            or p.ndim != 5
            or p.shape[-1] != 3
            or p.shape[:2] != arrays["predicted_state"].shape[:2]
        ):
            raise ValueError("Expected matching N,H,height,width,3 RGB arrays")
        if not (
            np.isfinite(p).all()
            and np.isfinite(t).all()
            and 0 <= p.min() <= p.max() <= 1
            and 0 <= t.min() <= t.max() <= 1
        ):
            raise ValueError("RGB predictions must be finite in [0,1]")
        result["rgb_mse_by_horizon"] = ((p - t) ** 2).mean(axis=(0, 2, 3, 4)).tolist()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions), indent=2))
