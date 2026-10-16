"""Interchangeable manual/SAM2 tracking, crops, and lossless track exchange."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import Detection, TrackFrame


def read_annotations(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _point(row, name):
    values = (row.get(f"{name}_u", ""), row.get(f"{name}_v", ""))
    if not any(values):
        return None
    if not all(values):
        raise ValueError(f"Incomplete {name} point at frame {row['frame_index']}")
    return tuple(float(v) for v in values)


def _box(row, name):
    values = [row.get(f"{name}_{key}", "") for key in ("x1", "y1", "x2", "y2")]
    if not any(values):
        return None
    if not all(values):
        raise ValueError(f"Incomplete {name} bounding box")
    box = tuple(float(v) for v in values)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Bounding box must have positive width and height")
    return box


class ManualTracker:
    """Read points/bboxes; missing observations remain missing (no silent filling)."""

    def track(self, frames_dir: str, annotations_csv: str) -> list[TrackFrame]:
        result = []
        for row in read_annotations(annotations_csv):
            detections = {
                name: Detection(_point(row, name), _box(row, name))
                for name in ("food", "hand", "target")
            }
            result.append(
                TrackFrame(
                    int(row["frame_index"]),
                    float(row["time_s"]),
                    **detections,
                    phase=(row.get("phase") or None),
                )
            )
        result.sort(key=lambda f: f.frame_index)
        if len({f.frame_index for f in result}) != len(result):
            raise ValueError("Duplicate frame indices")
        if any(b.time_s <= a.time_s for a, b in zip(result, result[1:])):
            raise ValueError("Frame times must increase strictly")
        return result


def write_tracks(tracks: list[TrackFrame], path: str):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        json.dump(
            {
                "schema_version": 1,
                "coordinate_space": "image_pixels",
                "frames": [asdict(t) for t in tracks],
            },
            f,
            indent=2,
        )


def interpolate_short_gaps(tracks, max_gap_frames: int):
    """Explicit opt-in linear interpolation between labelled points only.

    Gap limit counts original video frames. No extrapolation, bbox invention or
    automatic static-target assumption. Review interpolated contact events.
    """
    from copy import deepcopy

    tracks = deepcopy(tracks)
    for name in ("food", "hand", "target"):
        known = [
            i for i, f in enumerate(tracks) if getattr(f, name).center_uv is not None
        ]
        for a, b in zip(known, known[1:]):
            if (
                b == a + 1
                or tracks[b].frame_index - tracks[a].frame_index > max_gap_frames
            ):
                continue
            start = np.array(getattr(tracks[a], name).center_uv)
            end = np.array(getattr(tracks[b], name).center_uv)
            for i in range(a + 1, b):
                alpha = (tracks[i].time_s - tracks[a].time_s) / (
                    tracks[b].time_s - tracks[a].time_s
                )
                getattr(tracks[i], name).center_uv = tuple(
                    (1 - alpha) * start + alpha * end
                )
                getattr(tracks[i], name).source = "manual_interpolated"
    return tracks


def crop_food(
    rgb: np.ndarray, detection: Detection, *, crop_size: int | None = None
) -> np.ndarray:
    """Prefer a bbox; optionally use an explicitly sized square around a point.

    crop_size is in original-image pixels. This fallback is not segmentation and
    does not estimate object extent. The result is clipped at image boundaries.
    """
    box = detection.bbox_xyxy
    if detection.bbox_xyxy is None:
        if crop_size is None:
            raise ValueError(
                "Missing food bbox: draw food_x1/y1/x2/y2 or explicitly pass --crop-size PIXELS for point-centred crops"
            )
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if detection.center_uv is None:
            raise ValueError("Cannot crop a frame without a food box or food center")
        u, v = detection.center_uv
        h, w = rgb.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            raise ValueError("Food center is outside image")
        radius = crop_size / 2
        box = (u - radius, v - radius, u + radius, v + radius)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Food box is outside image")
    return rgb[y1:y2, x1:x2].copy()


class SAM2Tracker:
    """Inject a SAM2VideoPredictor constructed externally, with local weights.

    Run this optional backend in a separate environment. SAM2 frame indices refer
    to positions in the extracted sequence, not original-video frame indices.
    The frame directory must contain only SAM2-compatible numbered JPEGs.
    """

    def __init__(self, predictor, device="cuda", masks_dir: str | None = None):
        self.predictor = predictor
        self.device = device
        self.masks_dir = Path(masks_dir) if masks_dir else None

    def track(self, frames_dir: str, annotations_csv: str) -> list[TrackFrame]:
        import torch

        tracks = ManualTracker().track(frames_dir, annotations_csv)
        paths = sorted(Path(frames_dir).glob("*.jpg"), key=lambda p: int(p.stem))
        if len(paths) != len(tracks):
            raise ValueError(
                "SAM2 expects one numbered JPEG per annotation row, in frame order"
            )
        names = {1: "food", 2: "hand", 3: "target"}
        if self.masks_dir:
            self.masks_dir.mkdir(parents=True, exist_ok=False)
        with torch.inference_mode():
            state = self.predictor.init_state(video_path=frames_dir)
            for index, frame in enumerate(tracks):
                for obj_id, name in names.items():
                    det = getattr(frame, name)
                    if det.center_uv is not None:
                        self.predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=index,
                            obj_id=obj_id,
                            points=np.asarray([det.center_uv], dtype=np.float32),
                            labels=np.array([1], dtype=np.int32),
                        )
            for index, ids, logits in self.predictor.propagate_in_video(state):
                for i, obj_id in enumerate(ids):
                    name = names[int(obj_id)]
                    mask = (logits[i] > 0).detach().cpu().numpy().squeeze()
                    ys, xs = np.where(mask)
                    detection = Detection(None, source="sam2_mask_centroid")
                    if len(xs):
                        detection.center_uv = (float(xs.mean()), float(ys.mean()))
                        detection.bbox_xyxy = (
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()) + 1,
                            int(ys.max()) + 1,
                        )
                        if self.masks_dir:
                            path = (
                                self.masks_dir
                                / f"{tracks[index].frame_index:06d}_{name}.npy"
                            )
                            np.save(path, mask)
                            detection.mask_path = str(path)
                    # Preserve explicit human corrections and the hand grasp point.
                    existing = getattr(tracks[index], name)
                    if existing.center_uv is None:
                        setattr(tracks[index], name, detection)
                    elif existing.bbox_xyxy is None and detection.bbox_xyxy is not None:
                        existing.bbox_xyxy = detection.bbox_xyxy
                        existing.mask_path = detection.mask_path
        return tracks
