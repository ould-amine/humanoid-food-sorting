"""Deterministic object-relative waypoint planning and closed-loop OSC control."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import AbstractDemo, CATEGORIES, RobotObservation, Scene


@dataclass
class Waypoint:
    name: str
    xyz: np.ndarray
    gripper: float
    hold_steps: int = 1


def _human_shape(points, demo, start, end):
    """Transfer a 2D curve's shape into table XY; height is supplied by planner.

    Remove image translation/scale and align source-target direction with robot
    source-target direction. Camera perspective is not recovered by this mapping.
    Endpoint anchoring preserves shape residuals, not absolute human positions.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Human trajectory needs at least two finite 2D waypoints")
    if not np.isfinite(points).all():
        raise ValueError("Non-finite human trajectory")
    axis = np.asarray(demo.target) - np.asarray(demo.object_start)
    axis = axis / np.linalg.norm(axis)
    basis = np.array([axis, [-axis[1], axis[0]]])
    local = points @ basis.T
    alpha = np.linspace(0, 1, len(points))[:, None]
    residual = local - ((1 - alpha) * local[0] + alpha * local[-1])
    displacement = end[:2] - start[:2]
    length = np.linalg.norm(displacement)
    direction = displacement / length if length > 1e-6 else np.array([1.0, 0.0])
    robot_basis = np.array([direction, [-direction[1], direction[0]]])
    # Bound large/noisy excursions; scale relative to source-target robot distance.
    residual = np.clip(residual, -0.25, 0.25) @ robot_basis * max(length, 0.05)
    path = (1 - alpha) * start + alpha * end
    path[:, :2] += residual
    return path


def make_plan(
    scene: Scene,
    category: str,
    demo: AbstractDemo | None = None,
    clearance=0.12,
    max_curve_points=16,
) -> list[Waypoint]:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category}")
    if max_curve_points < 2:
        raise ValueError("At least two curve points are required")
    grasp = np.asarray(scene.object_position) + scene.grasp_offset
    place = np.asarray(scene.baskets[f"{category}_basket"]) + scene.placement_offset
    if not np.isfinite(np.r_[grasp, place]).all() or clearance <= 0:
        raise ValueError("Invalid scene coordinates or clearance")
    height = max(grasp[2], place[2]) + clearance
    above_object, above_target = grasp.copy(), place.copy()
    above_object[2] = above_target[2] = height
    approach = [above_object]
    transport = [above_target]
    close_hold = 20
    if demo is not None:
        if not demo.approved or demo.coordinate_space != "image_relative_normalized":
            raise ValueError("Review and approve the abstract demo before retargeting")
        # Start approach behind the object along the object-target direction.
        start = above_object.copy()
        delta = above_target[:2] - above_object[:2]
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            raise ValueError("Object and basket coincide")
        start[:2] -= 0.05 * delta / norm
        approach = _human_shape(demo.approach_trajectory, demo, start, above_object)
        transport = _human_shape(
            demo.transport_trajectory, demo, above_object, above_target
        )
        approach = approach[
            np.linspace(
                0, len(approach) - 1, min(len(approach), max_curve_points)
            ).astype(int)
        ]
        transport = transport[
            np.linspace(
                0, len(transport) - 1, min(len(transport), max_curve_points)
            ).astype(int)
        ]
    return (
        [Waypoint("APPROACH", p, -1) for p in approach]
        + [
            Waypoint("descend_to_grasp", grasp, -1),
            Waypoint("GRASP", grasp, 1, close_hold),
            Waypoint("lift", above_object, 1),
        ]
        + [Waypoint("TRANSPORT", p, 1) for p in transport]
        + [
            Waypoint("descend_to_release", place, 1),
            Waypoint("RELEASE", place, -1, 20),
            Waypoint("retreat", above_target, -1, 10),
        ]
    )


class WaypointController:
    """For the pinned OSC_POSE config: +/-1 maps to 5 cm / 0.5 rad deltas."""

    def __init__(
        self,
        plan,
        downward_quat,
        tolerance_m=0.008,
        timeout_steps=200,
        position_scale=0.05,
        rotation_scale=0.5,
    ):
        self.plan = plan
        self.rotation = Rotation.from_quat(downward_quat)
        self.tolerance = tolerance_m
        self.timeout = timeout_steps
        self.position_scale, self.rotation_scale = position_scale, rotation_scale
        self.reset()

    def reset(self):
        self.index = self.age = self.dwell = 0

    @property
    def finished(self):
        return self.index >= len(self.plan)

    def act(self, observation: RobotObservation, category=None, instruction=""):
        if self.finished:
            raise StopIteration("Waypoint plan complete")
        waypoint = self.plan[self.index]
        error = waypoint.xyz - observation.state[:3]
        current = Rotation.from_quat(observation.state[3:7])
        rotation_error = (self.rotation * current.inv()).as_rotvec()
        self.age += 1
        if self.age > self.timeout:
            raise TimeoutError(f"Waypoint {waypoint.name} was not reached")
        reached = (
            np.linalg.norm(error) < self.tolerance
            and np.linalg.norm(rotation_error) < 0.1
        )
        self.dwell = self.dwell + 1 if reached else 0
        action = np.r_[
            np.clip(error / self.position_scale, -0.3, 0.3),
            np.clip(rotation_error / self.rotation_scale, -0.3, 0.3),
            waypoint.gripper,
        ]
        if self.dwell >= waypoint.hold_steps:
            self.index += 1
            self.age = self.dwell = 0
        return action[None, :]
