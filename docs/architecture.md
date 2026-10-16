# Architecture

## Data flow

```text
personal video → frames → annotations → food crops → category
                              ↓
                       reviewed human demo
                              ↓
category + robot observation → controller or BC policy → Panda actions
                                                         ↓
                                            episodes and evaluation
```

The classifier chooses the basket category. It does not predict robot motion. Human motion is represented relative to the food and target; human finger configurations are not copied.

## Modules

| Module | Responsibility |
| --- | --- |
| `sorting/contracts.py` | Shared observations, tracks, demos, and categories |
| `sorting/vision.py` | Manual tracking, optional tracking interfaces, interpolation, crops |
| `sorting/demonstrations.py` | APPROACH, GRASP, TRANSPORT, RELEASE abstraction |
| `sorting/classification.py`, `sorting/inference.py` | Classifier interfaces and frozen model inference |
| `sorting/scene.py` | Panda, textured food proxy, labeled baskets, cameras |
| `sorting/environment.py` | Environment bridge and physical success checks |
| `sorting/retargeting.py`, `sorting/runner.py` | Relative trajectory retargeting and execution |
| `sorting/recording.py` | Episode arrays, metadata, and video |
| `sorting/datasets.py`, `sorting/learning.py` | Dataset loading and training utilities |
| `sorting/models.py` | BC network and experimental model interfaces |

## Contracts

Human annotations use image coordinates and original video frame indices. Missing points remain missing unless interpolation is explicitly requested. A reviewed demo records object start, target, approach/transport trajectories, and grasp/release events. Approval requires checking those events and the camera assumptions.

The robot observation contains RGB, end-effector position, orientation, and finger state. The current state vector has nine values: position (3), quaternion (4), and finger positions (2). Actions have seven values: translation (3), rotation (3), and gripper command (1), using the simulator's OSC controller scaling.

An episode stores N actions and N+1 observations. Classification accuracy, physical pick success, basket correctness, and task success are separate measurements. The true category is used for scoring; only the selected classifier's prediction routes semantic runs.

## Scope

Manual annotation, CLIP zero-shot, frozen DINOv2 with a linear head, deterministic sorting, reviewed human retargeting, matched collection, and BC training/evaluation entry points are implemented. SAM2 is optional behind a tracking interface. SmolVLA and world-model components require further integration and experiments; their presence does not imply a trained policy or validated predictor.

The scene uses a food-textured cuboid. Object position and segmentation come from simulator state. The close-up camera also uses object localization. Human motion comes from 2D video and a reviewed mapping into the robot workspace, not recovered metric 3D trajectories. Treat these assumptions as part of every reported result.

See [sorting](sorting.md) for single episodes and [experiments](experiments.md) for controlled comparisons.
