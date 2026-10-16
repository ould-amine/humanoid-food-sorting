"""Heuristic phase extraction; outputs require review before retargeting."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .contracts import AbstractDemo, TrackFrame, Phase


def abstract_demo(
    video_id: str,
    tracks: list[TrackFrame],
    *,
    grasp_frame=None,
    release_frame=None,
    near_threshold=0.15,
    coupling_threshold=0.12,
    min_object_speed=0.03,
    consecutive=3,
) -> AbstractDemo:
    """Distances/speeds are normalized by initial object-to-target separation.

    Manual event indices override estimates. We never infer RELEASE merely from
    the end of a clip; ambiguous/missing events raise instead of fabricating labels.
    Fixed camera required. SAM2 hand centroid is only a rough proximity cue.
    """
    # Entry/exit occlusions do not require invented hand positions. Only retain
    # the interval where a hand is actually observed; internal gaps still require
    # explicit reviewed interpolation upstream.
    visible_hand = [i for i, t in enumerate(tracks) if t.hand.center_uv is not None]
    trimmed = False
    if visible_hand:
        trimmed = visible_hand[0] != 0 or visible_hand[-1] != len(tracks) - 1
        tracks = tracks[visible_hand[0] : visible_hand[-1] + 1]
    if len(tracks) < max(3, consecutive + 1):
        raise ValueError("Too few tracked frames")
    if any(
        t.food.center_uv is None
        or t.hand.center_uv is None
        or t.target.center_uv is None
        for t in tracks
    ):
        raise ValueError(
            "Review missing food/hand/target points before phase extraction"
        )
    times = np.array([t.time_s for t in tracks])
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Non-increasing timestamps")
    food = np.array([t.food.center_uv for t in tracks], dtype=float)
    hand = np.array([t.hand.center_uv for t in tracks], dtype=float)
    target = np.array([t.target.center_uv for t in tracks], dtype=float)
    scale = np.linalg.norm(target[0] - food[0])
    if scale < 1:
        raise ValueError("Object and target are too close to normalize")
    near = np.linalg.norm(hand - food, axis=1) / scale < near_threshold
    object_velocity = np.diff(food, axis=0) / dt[:, None] / scale
    hand_velocity = np.diff(hand, axis=0) / dt[:, None] / scale
    coupled = (
        np.linalg.norm(hand_velocity - object_velocity, axis=1) < coupling_threshold
    )
    moving = np.linalg.norm(object_velocity, axis=1) > min_object_speed
    frame_map = {t.frame_index: i for i, t in enumerate(tracks)}

    def manual_index(frame, phase):
        if frame is not None:
            if frame not in frame_map:
                raise ValueError(f"Event frame {frame} is not present in tracks")
            return frame_map[frame]
        return next(
            (i for i, t in enumerate(tracks) if (t.phase or "").upper() == phase), None
        )

    grasp = manual_index(grasp_frame, Phase.GRASP.value)
    release = manual_index(release_frame, Phase.RELEASE.value)
    source = (
        "manual" if grasp is not None and release is not None else "heuristic_or_mixed"
    )
    if grasp is None:
        grasp = next(
            (
                i
                for i in range(len(tracks) - consecutive)
                if near[i : i + consecutive].all()
                and coupled[i : i + consecutive].all()
                and moving[i : i + consecutive].all()
            ),
            None,
        )
    if grasp is None:
        raise ValueError("No reliable grasp candidate; annotate GRASP manually")
    if release is None:
        at_target = np.linalg.norm(food - target, axis=1) / scale < near_threshold
        stationary = np.r_[
            False, np.linalg.norm(object_velocity, axis=1) < min_object_speed
        ]
        release = next(
            (
                i
                for i in range(grasp + 1, len(tracks) - consecutive + 1)
                if at_target[i : i + consecutive].all()
                and stationary[i : i + consecutive].all()
                and (~near[i : i + consecutive]).all()
            ),
            None,
        )
    if release is None or release <= grasp:
        raise ValueError("No reliable release after grasp; annotate RELEASE manually")
    return AbstractDemo(
        video_id=video_id,
        object_start=food[0].tolist(),
        target=target[0].tolist(),
        approach_trajectory=((hand[: grasp + 1] - food[0]) / scale).tolist(),
        transport_trajectory=((food[grasp : release + 1] - food[0]) / scale).tolist(),
        hand_object_offsets=((hand - food) / scale).tolist(),
        target_object_offsets=((target - food) / scale).tolist(),
        grasp_frame=tracks[grasp].frame_index,
        release_frame=tracks[release].frame_index,
        grasp_time_s=tracks[grasp].time_s,
        release_time_s=tracks[release].time_s,
        phase_source=source,
        assumptions=["fixed_camera", "2d_shape_only"]
        + (["trimmed_hand_entry_exit_occlusions"] if trimmed else []),
    )


def save_demo(demo: AbstractDemo, path: str):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as f:
        json.dump(asdict(demo), f, indent=2)


def load_demo(path: str) -> AbstractDemo:
    return AbstractDemo(**json.loads(Path(path).read_text()))
