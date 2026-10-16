"""Environment bridge and independent physical success checks for two-basket sorting.

This does not create food assets or silently reuse the bowl task as sorting.
The task builder supplies scene geometry, food crop and physical-state readers.
"""

from dataclasses import dataclass

import numpy as np

from .contracts import CATEGORIES, Events, RobotObservation


@dataclass
class BasketRegion:
    lower_xyz: np.ndarray
    upper_xyz: np.ndarray

    def contains(self, object_center, object_half_extents):
        # Contact solvers allow tiny support-surface penetration.
        tolerance = 0.001
        return bool(
            np.all(object_center - object_half_extents >= self.lower_xyz - tolerance)
            and np.all(
                object_center + object_half_extents <= self.upper_xyz + tolerance
            )
        )


@dataclass
class PhysicalState:
    object_center: np.ndarray
    object_half_extents: np.ndarray
    object_speed_m_s: float
    held_by_gripper: bool  # Contact/grasp check supplied by simulator adapter.
    gripper_open: bool  # Actual finger state, not requested action.


class SuccessMonitor:
    def __init__(
        self,
        ground_truth_category,
        regions,
        *,
        lift_height_m=0.04,
        stable_steps=10,
        speed_threshold_m_s=0.02,
    ):
        if ground_truth_category not in CATEGORIES or stable_steps < 1:
            raise ValueError("Invalid success monitor configuration")
        if set(regions) != {"fruit_basket", "vegetable_basket"}:
            raise ValueError("Both named basket regions are required")
        if any(np.any(r.upper_xyz <= r.lower_xyz) for r in regions.values()):
            raise ValueError("Invalid basket region bounds")
        self.expected = f"{ground_truth_category}_basket"
        self.regions = regions
        self.lift_height = lift_height_m
        self.stable_steps, self.speed_threshold = stable_steps, speed_threshold_m_s

    def reset(self, initial: PhysicalState):
        self.start_z = initial.object_center[2]
        self.picked = False
        self.dwell = {name: 0 for name in self.regions}

    def update(self, state: PhysicalState) -> Events:
        self.picked |= bool(
            state.held_by_gripper
            and state.object_center[2] - self.start_z >= self.lift_height
        )
        placed = None
        for name, region in self.regions.items():
            stable = (
                state.gripper_open
                and not state.held_by_gripper
                and state.object_speed_m_s < self.speed_threshold
                and region.contains(state.object_center, state.object_half_extents)
            )
            self.dwell[name] = self.dwell[name] + 1 if stable else 0
            if self.dwell[name] >= self.stable_steps:
                placed = name
        success = self.picked and placed == self.expected
        return Events(
            self.picked,
            placed,
            success,
            terminated=bool(self.picked and placed is not None),
        )


class LiberoSortingAdapter:
    """Inject a configured two-basket ControlEnv plus explicit perception hooks.

    scene_reader(env) -> Scene
    crop_reader(env, observation) -> RGB food crop
    physics_reader(env) -> PhysicalState
    None of these hooks may provide category via policy observation/state.
    Use the same position/crop source across compared methods and record it.
    """

    def __init__(
        self,
        env,
        scene_reader,
        crop_reader,
        physics_reader,
        monitor,
        control_hz=20,
        image_origin="opengl",
    ):
        self.env = env
        self.scene_reader, self.crop_reader, self.physics_reader = (
            scene_reader,
            crop_reader,
            physics_reader,
        )
        self.monitor, self.control_hz, self.image_origin = (
            monitor,
            control_hz,
            image_origin,
        )
        self.steps = 0

    def _observation(self, raw):
        image = raw["agentview_image"]
        if self.image_origin == "opengl":
            image = image[::-1]
        return RobotObservation(
            np.ascontiguousarray(image),
            np.r_[
                raw["robot0_eef_pos"],
                raw["robot0_eef_quat"],
                raw["robot0_gripper_qpos"],
            ],
            self.steps / self.control_hz,
        )

    def reset(self, seed):
        self.env.seed(seed)
        raw = self.env.reset()
        self.steps = 0
        self.monitor.reset(self.physics_reader(self.env))
        return self._observation(raw)

    def scene(self):
        return self.scene_reader(self.env)

    def food_crop(self, observation):
        return self.crop_reader(self.env, observation)

    def step(self, action):
        raw, _, done, _ = self.env.step(action)
        self.steps += 1
        events = self.monitor.update(self.physics_reader(self.env))
        events.terminated |= bool(done)
        return self._observation(raw), events

    def close(self):
        self.env.close()
