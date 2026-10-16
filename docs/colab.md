# Colab

Use a GPU runtime for classifier inference, embedding extraction, or BC training. Simulation also requires the pinned Python 3.11 environment below. Run from local Colab storage; save results to Drive.

## Upload current code and data

On your computer, from the project root:

```sh
python make_sorting_bundle.py
```

Upload `artifacts/sorting_colab.zip` to `MyDrive/Humanoid/`. Keep the trained DINO head at `MyDrive/Humanoid/results/classification_run_01/dino_head.pt`, or adjust the path below. The bundle excludes checkpoints and the external LIBERO checkout. Archived bundles contain historical code and should not be used for new runs.

In Colab, select a GPU runtime, then run:

```python
from google.colab import drive
from pathlib import Path
from datetime import datetime
import os
import shutil
import subprocess
import sys

drive.mount('/content/drive')
drive_root = Path('/content/drive/MyDrive/Humanoid')
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
project = Path('/content') / f'HumanoidSorting_{run_id}'
project.mkdir()
archive = Path('/content') / f'sorting_{run_id}.zip'
shutil.copy2(drive_root / 'sorting_colab.zip', archive)
shutil.unpack_archive(archive, project)

def run(*command):
    return subprocess.run([str(arg) for arg in command], cwd=project, check=True)
```

## Simulation and classification environment

Create an isolated environment so the runtime's preinstalled Python and packages do not change the simulation dependencies:

```python
run(sys.executable, '-m', 'pip', 'install', 'uv')
run('uv', 'python', 'install', '3.11')
run('uv', 'venv', '--python', '3.11', project / '.venv-colab')
python = project / '.venv-colab/bin/python'
run('git', 'clone', 'https://github.com/Lifelong-Robot-Learning/LIBERO.git', 'LIBERO')
subprocess.run(['git', 'checkout', '8f1084e3132a39270c3a13ebe37270a43ece2a01'],
               cwd=project / 'LIBERO', check=True)
run('uv', 'pip', 'install', '--python', python, '-r', 'requirements-sim.txt')
run('uv', 'pip', 'install', '--python', python, '-r', 'requirements-classification.txt')
run('apt-get', 'update')
run('apt-get', 'install', '-y', 'libegl1', 'libgl1')
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['USE_TF'] = '0'
run(python, '-c', 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())')
```

If CUDA is false, check the runtime GPU selection before launching GPU commands. Use this environment's `python`, not the notebook's `sys.executable`, for all subsequent simulation commands.

## First sorting run

```python
checkpoint = drive_root / 'results/classification_run_01/dino_head.pt'
output = drive_root / 'results' / f'banana_dino_{run_id}'
run(python, 'run_sorting.py',
    '--category', 'fruit', '--classifier', 'dino', '--device', 'cuda',
    '--checkpoint', checkpoint,
    '--food-name', 'banana', '--source-video', 'banana_02',
    '--texture', 'data/crops/banana_02_points/000000.jpg',
    '--demo', 'data/demos/banana_02_reviewed.json',
    '--swap-baskets', '--output', output)
```

A physical task failure may return exit code 1 while saving an episode. Read the subprocess output and `episode.json` to distinguish a failed task from an installation or inference error. Do not rerun into an existing output directory.

Continue with the [matched experiment launcher](experiments.md#colab-launcher). Keep its code, data, and model revision unchanged when resuming a plan. After updating source code, create a new plan.

## Classification-only alternative

For classification without simulation, keep Colab's preinstalled CUDA PyTorch and use its native Python runtime:

```python
os.environ['USE_TF'] = '0'
run(sys.executable, '-m', 'pip', 'install', '-r', 'requirements-classification-colab.txt')
python = Path(sys.executable)
```

Use this alternative instead of the isolated simulation setup. Follow [classification](classification.md), replacing shell commands with `run(python, ...)` and storing output paths under `drive_root / 'results'`. Do not install the older simulation requirements into the native Colab runtime.
