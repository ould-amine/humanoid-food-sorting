# Sorting simulation

The custom robosuite/MuJoCo scene contains a Panda, a food-textured cuboid, and two baskets. The blue basket is labeled FRUIT; the orange basket is labeled VEGETABLE. `--swap-baskets` swaps their positions while preserving their identities.

## Single episode

First check physical control with the oracle category:

```sh
python run_sorting.py --category fruit --food-name banana \
  --texture data/crops/banana_02_points/000000.jpg --source-video banana_02 \
  --output outputs/banana_oracle_new
```

Add `--demo data/demos/banana_02_reviewed.json` to use the reviewed human motion template. Without a demo, the controller follows its fixed approach, grasp, lift, transport, and release sequence.

For semantic routing with the frozen DINO classifier:

```sh
python run_sorting.py --category fruit --classifier dino --device cuda \
  --checkpoint results/classification_run_01/dino_head.pt \
  --food-name banana --texture data/crops/banana_02_points/000000.jpg \
  --source-video banana_02 --demo data/demos/banana_02_reviewed.json \
  --swap-baskets --output outputs/banana_dino_new
```

`--category` supplies ground truth for scoring. In DINO/CLIP runs, the prediction chooses the basket. Use `--classifier clip` without a DINO checkpoint for CLIP. Defaults use a separate close-up camera for classification and the wide agent camera for robot observations. Both object localization and crop segmentation are oracle.

## Outputs and review

Each completed episode contains `episode.json`, `episode.npz`, `rollout.mp4`, and perception artifacts. The arrays pair N actions with N+1 observations. Metadata records classification, physical events, provenance, basket layout, and success.

Inspect the video and metadata together. Check whether the object is lifted, released, and remains in the correct basket. A failed task can still produce a complete episode and a nonzero exit code; inspect recorded outputs before treating that code as an infrastructure error.

For perception diagnostics, inspect `capture_perception.py --help` and `compare_sim_crops.py --help`. Capture wide and close-up views of the same scene, then compare both frozen models on the saved crops. These are diagnostic observations, not evidence of held-out task success.

## Next experiment

Use [matched experiments](experiments.md) to compare classifiers, fixed/human motion, layouts, and BC under identical scenes. Use [Colab setup](colab.md) for GPU inference and collection. Full matrix collection and BC training require new runs; existing single-episode results do not establish generalization.

The environment is a controlled textured-cuboid prototype. Human 2D motion shapes the controller trajectory; simulator state supplies metric positions. Realistic food geometry, learned localization, SmolVLA fine-tuning, and world-model validation remain separate work.
