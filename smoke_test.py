"""Run a LIBERO task and render a camera frame."""

from pathlib import Path

import sim_setup  # noqa: F401 -- configure paths before importing LIBERO.


def main():
    import mujoco
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
    print(f"MuJoCo {mujoco.__version__}; task: {task.language}")
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=128,
        camera_widths=128,
    )
    try:
        env.seed(0)
        obs = env.reset()
        for _ in range(10):
            obs, reward, done, info = env.step(np.zeros(7))
        print(
            f"PASS: reset, 10 simulation steps, camera frame {obs['agentview_image'].shape}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
