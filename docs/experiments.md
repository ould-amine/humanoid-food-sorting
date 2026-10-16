# Matched sorting experiments and BC

All orchestration is implemented. The full GPU matrix and BC training have not
been executed here. Start with the 12-rollout development matrix below, inspect
failures, then generate train/val/test data with a fixed protocol.

## Plan, execute, resume, report

`sorting_experiments.py plan` creates an immutable plan without loading models.
It chooses the first available crop of a source video, keeping that video's
train/val/test split and seen/unseen label. For each food/seed, classifier and
motion variants use the same texture, initial randomization and normal/swapped
layout. The human template must come from a training video.

```sh
python sorting_experiments.py plan --output outputs/dev_matrix \
  --checkpoint results/classification_run_01/dino_head.pt \
  --splits val --val-seeds 6 --food-names banana carrot yellow_pepper \
  --classifiers dino clip --motions human
python sorting_experiments.py run outputs/dev_matrix --device cuda
python sorting_experiments.py report outputs/dev_matrix
```

This plans 12 episodes (3 foods x 2 layouts x 2 classifiers). Each full episode
uses five close-up crops before grasp. Add oracle for control diagnostics and
fixed for the motion baseline. Keep camera and classifiers frozen during testing.

Running the same `run` command resumes finished jobs, including recorded failures,
without overwriting them. `--limit N` launches at most N new jobs. Per-job console
logs, status.json and original episodes are retained. An infrastructure failure
is recorded and stops collection. Interrupted, unindexed partial episode folders
require inspection; they are never overwritten automatically. Changed source
code, texture, demo or DINO checkpoint is rejected: create a fresh plan instead.

Reports distinguish infrastructure errors, classification errors, pick success,
correct basket and task success. Pending jobs are shown separately. Infrastructure
failures count against all-attempted task success; classification accuracy uses
recorded episodes only. Oracle groups are never semantic end-to-end scores.
The plan's scene IDs allow checking exactly which method comparisons are paired.

## Full collection

```sh
python sorting_experiments.py plan --output outputs/full_matrix \
  --checkpoint results/classification_run_01/dino_head.pt \
  --splits train val test --train-seeds 0 1 2 3 --val-seeds 6 7 --test-seeds 8 9 \
  --classifiers oracle dino clip --motions fixed human
python sorting_experiments.py run outputs/full_matrix --device cuda
python sorting_experiments.py report outputs/full_matrix
```

Planning prints the count and food inventory; all available types are included
unless --food-names is specified. This can be a long collection. Subsets must
include both categories for BC. Crop textures inherit source splits; the shared
human trajectory is a training template applied across types, not unseen motion
learning. Simulator geometry, camera targeting and segmentation remain oracle.
Cuboid-texture success is not a realistic-food or standard-LIBERO benchmark score.

## Equal-budget BC data and training

Use successful oracle-category episodes first to isolate motion learning from
classifier errors. This selection uses the intersection of successful fixed/human
scene pairs, with equal fruit/vegetable episode budgets in train and validation.
Original generation failures stay in the experiment report. No test data is used.

```sh
python prepare_bc_data.py outputs/full_matrix --classifier oracle \
  --output outputs/full_matrix/bc_data
python train_bc.py outputs/full_matrix/bc_data/bc_fixed.json --device cuda \
  --epochs 50 --steps-per-epoch 100 --output outputs/bc_fixed.pt
python train_bc.py outputs/full_matrix/bc_data/bc_human.json --device cuda \
  --epochs 50 --steps-per-epoch 100 --output outputs/bc_human.pt
```

Selection files reference episode folders relative to their location, without
duplicating RGB arrays. Set --budget-per-category to cap both methods identically.
The same epochs/batch size/steps give equal optimizer budgets. Sampling gives each
episode equal weight regardless of trajectory length. Padding is masked, state
statistics use train only, and validation action MSE selects the checkpoint.
Closed-loop sorting success must still be evaluated independently.
Robot images are decoded once to a local memory-mapped cache (`.cache/robot_data`,
or --cache-dir). Keep this cache on Colab local disk, not Drive. It uses disk space
but avoids holding the full demonstration video dataset in RAM.

```sh
python sorting_experiments.py evaluate-bc outputs/full_matrix --device cuda \
  --motion fixed --policy-checkpoint outputs/bc_fixed.pt
python sorting_experiments.py evaluate-bc outputs/full_matrix --device cuda \
  --motion human --policy-checkpoint outputs/bc_human.pt
python sorting_experiments.py report outputs/full_matrix
```

Each policy runs only the matching motion branch's held-out test scenes, under
each planned classifier. Checkpoint-hashed branches prevent mixing different
policies; training/validation seed reuse and motion-label mismatches are rejected.
Method scene lists in report.json support paired comparisons with deterministic
control. Compare complete groups with identical scene coverage, not partial runs.

## Colab launcher

Build `artifacts/sorting_colab.zip` with `python make_sorting_bundle.py`, then upload it to MyDrive/Humanoid. Reuse the isolated
Python 3.11 setup from [colab.md](colab.md). In an already configured notebook, update code
and run the development matrix:

```python
from pathlib import Path
from datetime import datetime
import shutil
shutil.copy2(drive_root / 'sorting_colab.zip', '/content/sorting_colab.zip')
shutil.unpack_archive('/content/sorting_colab.zip', project)
suite = drive_root / 'results' / ('sorting_suite_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
checkpoint = drive_root / 'results/classification_run_01/dino_head.pt'
run(python, 'sorting_experiments.py', 'plan', '--output', suite,
    '--checkpoint', checkpoint, '--splits', 'val', '--val-seeds', '6',
    '--food-names', 'banana', 'carrot', 'yellow_pepper',
    '--classifiers', 'dino', 'clip', '--motions', 'human')
run(python, 'sorting_experiments.py', 'run', suite, '--device', 'cuda')
run(python, 'sorting_experiments.py', 'report', suite)
print((suite / 'report.md').read_text())
```

For the full matrix, replace plan arguments with the full collection example,
using another suite directory. Results are written directly to Drive here so
completed episodes survive runtime resets. Code/data source paths inside the
project are relative, while the Drive checkpoint path remains absolute.

After a full collection, train in the same notebook:

```python
selection = suite / 'bc_data'
run(python, 'prepare_bc_data.py', suite, '--classifier', 'oracle', '--output', selection)
for motion in ('fixed', 'human'):
    policy = suite / ('bc_' + motion + '.pt')
    run(python, 'train_bc.py', selection / ('bc_' + motion + '.json'),
        '--device', 'cuda', '--epochs', '50', '--steps-per-epoch', '100', '--output', policy)
    run(python, 'sorting_experiments.py', 'evaluate-bc', suite,
        '--device', 'cuda', '--motion', motion, '--policy-checkpoint', policy)
run(python, 'sorting_experiments.py', 'report', suite)
```

Do not run the BC cell after the val-only development matrix: it intentionally
requires train and val pairs. Keep SmolVLA and world-model experiments separate
until deterministic and BC closed-loop reports are available. Existing adapters
remain available; this suite does not claim to implement their fine-tuning.
