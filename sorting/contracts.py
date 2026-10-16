"""Shared, serializable contracts; pixel coordinates never imply robot metres."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Literal

import numpy as np


Category = Literal["fruit", "vegetable"]
CATEGORIES = ("fruit", "vegetable")


class Phase(str, Enum):
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    TRANSPORT = "TRANSPORT"
    RELEASE = "RELEASE"


@dataclass
class Detection:
    center_uv: tuple[float, float] | None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    confidence: float | None = None  # None for human labels, not invented certainty.
    source: str = "manual"
    mask_path: str | None = None


@dataclass
class TrackFrame:
    frame_index: int
    time_s: float
    food: Detection
    hand: Detection
    target: Detection
    phase: str | None = None


class Tracker(Protocol):
    def track(self, frames_dir: str, annotations_csv: str) -> list[TrackFrame]: ...


@dataclass
class Prediction:
    category: Category
    probabilities: dict[str, float]
    method: str


class Classifier(Protocol):
    def predict(self, rgb_crop: np.ndarray) -> Prediction: ...


@dataclass
class AbstractDemo:
    video_id: str
    object_start: list[float]  # Original pixels, for provenance only.
    target: list[float]
    approach_trajectory: list[
        list[float]
    ]  # (hand - initial object) / start-target distance.
    transport_trajectory: list[list[float]]  # (object - initial object) / distance.
    hand_object_offsets: list[list[float]]
    target_object_offsets: list[list[float]]
    grasp_frame: int
    release_frame: int
    grasp_time_s: float
    release_time_s: float
    coordinate_space: str = "image_relative_normalized"
    schema_version: int = 1
    phase_source: str = "heuristic"
    approved: bool = False
    assumptions: list[str] = field(
        default_factory=lambda: ["fixed_camera", "2d_shape_only"]
    )
    review_notes: str = ""


@dataclass
class RobotObservation:
    image: np.ndarray  # H,W,3 RGB uint8, top-left origin.
    state: np.ndarray  # eef xyz (metres), quaternion xyzw, two finger qpos.
    time_s: float


@dataclass
class Scene:
    object_position: np.ndarray  # xyz, robot world frame, metres.
    baskets: dict[str, np.ndarray]  # Desired object support positions, metres.
    grasp_offset: np.ndarray  # Offset from object centre to gripper grasp site.
    placement_offset: np.ndarray  # Offset from desired object position to gripper site.
    eef_down_quat_xyzw: np.ndarray
    position_source: str = "simulator_oracle"


@dataclass
class Events:
    pick_success: bool
    placed_basket: str | None
    task_success: bool
    terminated: bool = False


class SortingEnvironment(Protocol):
    """Adapter boundary for the future two-basket LIBERO task.

    Events must be computed from physical scene state, independent of classifier
    output. task_success requires release + stable placement in ground-truth basket.
    """

    def reset(self, seed: int) -> RobotObservation: ...
    def scene(self) -> Scene: ...
    def food_crop(self, observation: RobotObservation) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[RobotObservation, Events]: ...
    def close(self) -> None: ...


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(
        self, observation: RobotObservation, category: Category, instruction: str
    ) -> np.ndarray:
        """Return H,7 normalized OSC_POSE actions (H may be 1)."""
        ...
