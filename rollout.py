"""Verify Panda movement and gripper control; save video and aligned observations."""

import argparse
import json
from pathlib import Path

import sim_setup  # noqa: F401 -- configure paths before importing LIBERO.
import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite import macros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/panda_control"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in ("rollout.mp4", "observations.npz", "summary.json"):
        if (args.output / filename).exists():
            raise FileExistsError(
                f"{args.output / filename}: choose a new --output directory"
            )
    task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )
    writer = None
    actions, images, positions, grippers, phases = [], [], [], [], []
    # Small movements above the table; this checks control, not task completion.
    sequence = [
        ("open", 30, [0, 0, 0, 0, 0, 0, -1]),
        ("raise", 20, [0, 0, 0.15, 0, 0, 0, -1]),
        ("move_y", 15, [0, 0.12, 0, 0, 0, 0, -1]),
        ("close", 30, [0, 0, 0, 0, 0, 0, 1]),
        ("reopen", 30, [0, 0, 0, 0, 0, 0, -1]),
    ]
    endpoints = {}
    try:
        env.seed(args.seed)
        obs = env.reset()
        writer = cv2.VideoWriter(
            str(args.output / "rollout.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (256, 256),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open MP4 encoder")

        def record(observation):
            rgb = observation["agentview_image"]
            if macros.IMAGE_CONVENTION == "opengl":
                rgb = rgb[::-1]
            rgb = np.ascontiguousarray(rgb)
            images.append(rgb.copy())
            positions.append(observation["robot0_eef_pos"].copy())
            grippers.append(observation["robot0_gripper_qpos"].copy())
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        record(obs)
        for name, steps, command in sequence:
            for _ in range(steps):
                action = np.asarray(command, dtype=np.float64)
                obs, _, _, _ = env.step(action)
                actions.append(action)
                phases.append(name)
                record(obs)
            endpoints[name] = {
                "eef_pos": positions[-1].tolist(),
                "gripper_qpos": grippers[-1].tolist(),
            }
    finally:
        if writer is not None:
            writer.release()
        env.close()

    movement = float(
        np.linalg.norm(
            np.array(endpoints["raise"]["eef_pos"]) - endpoints["open"]["eef_pos"]
        )
    )
    opening = lambda name: float(np.sum(np.abs(endpoints[name]["gripper_qpos"])))
    checks = {
        "panda_moved": movement > 0.02,
        "gripper_closed": opening("open") - opening("close") > 0.01,
        "gripper_reopened": opening("reopen") - opening("close") > 0.01,
    }
    np.savez_compressed(
        args.output / "observations.npz",
        agentview_image=np.stack(images),
        robot0_eef_pos=np.stack(positions),
        robot0_gripper_qpos=np.stack(grippers),
        actions=np.stack(actions),
        phase=np.asarray(phases),
        observation_time_s=np.arange(len(images)) / 20,
    )
    summary = {
        "task": task.language,
        "seed": args.seed,
        "control_hz": 20,
        "action_layout": ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"],
        "gripper_command": {"open": -1, "close": 1},
        "alignment": "N actions, N+1 observations; action[t] connects obs[t] to obs[t+1]",
        "movement_m": movement,
        "checks": checks,
        "phase_endpoints": endpoints,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise RuntimeError("Control verification failed; inspect summary and rollout")
    print(f"PASS: video and observations saved in {args.output}")


if __name__ == "__main__":
    main()
