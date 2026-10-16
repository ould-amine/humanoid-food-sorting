"""Versioned episode format for imitation, evaluation and world modelling."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .contracts import CATEGORIES, Events, RobotObservation


class EpisodeRecorder:
    def __init__(self, output: str, metadata: dict, initial: RobotObservation):
        required = {
            "episode_id",
            "method",
            "food_name",
            "object_id",
            "session_id",
            "split",
            "eval_group",
            "category",
            "predicted_category",
            "instruction",
            "seed",
            "human_video_ids",
            "checkpoint_id",
            "position_source",
        }
        missing = required - metadata.keys()
        if missing:
            raise ValueError(f"Missing episode metadata: {sorted(missing)}")
        if (
            metadata["category"] not in CATEGORIES
            or metadata["predicted_category"] not in CATEGORIES
        ):
            raise ValueError("Invalid category")
        if metadata["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        if metadata["eval_group"] not in ("seen", "unseen"):
            raise ValueError("Invalid seen/unseen group")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.observations = [initial]
        self.actions = []
        self.events = []
        self.saved = False

    def append(self, action, next_observation: RobotObservation, events: Events):
        if self.saved:
            raise RuntimeError("Episode already finalized")
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape != (7,)
            or not np.isfinite(action).all()
            or np.any(np.abs(action) > 1)
        ):
            raise ValueError("Expected finite normalized action with shape (7,)")
        if next_observation.time_s <= self.observations[-1].time_s:
            raise ValueError("Observation times must increase")
        self.actions.append(action.copy())
        self.observations.append(next_observation)
        self.events.append(events)

    def finish(self, *, completed: bool, failure_reason: str | None = None):
        if self.saved or not self.actions:
            raise ValueError("Episode empty or already finalized")
        images = np.stack([o.image for o in self.observations])
        state = np.stack([o.state for o in self.observations])
        np.savez_compressed(
            self.output / "episode.npz",
            images=images,
            state=state,
            time_s=[o.time_s for o in self.observations],
            actions=np.stack(self.actions),
        )
        import cv2

        h, w = images.shape[1:3]
        fps = 1 / np.median(np.diff([o.time_s for o in self.observations]))
        writer = cv2.VideoWriter(
            str(self.output / "rollout.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot encode rollout.mp4")
        try:
            for image in images:
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        picked = any(e.pick_success for e in self.events)
        placed = self.events[-1].placed_basket
        # Ground-truth category, never the classifier's selected category.
        correct = placed == f"{self.metadata['category']}_basket"
        summary = {
            **self.metadata,
            "schema_version": 1,
            "state_layout": [
                "eef_x",
                "eef_y",
                "eef_z",
                "quat_x",
                "quat_y",
                "quat_z",
                "quat_w",
                "finger_1",
                "finger_2",
            ],
            "action_layout": [
                "dx",
                "dy",
                "dz",
                "drot_x",
                "drot_y",
                "drot_z",
                "gripper",
            ],
            "observation_alignment": "N+1 observations for N actions",
            "completed": completed,
            "failure_reason": failure_reason,
            "pick_success": picked,
            "placed_basket": placed,
            "correct_basket": correct,
            "task_success": bool(self.events[-1].task_success and correct),
            "trajectory_length_m": float(
                np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1).sum()
            ),
            "events": [asdict(e) for e in self.events],
        }
        (self.output / "episode.json").write_text(json.dumps(summary, indent=2) + "\n")
        self.saved = True
        return summary
