# Food sorting from personal videos

Use personally recorded manipulation videos to classify food and guide a Panda robot into the fruit or vegetable basket.

The pipeline combines manual annotations, frozen visual backbones, relative human trajectories, and a deterministic controller. Behavioral cloning and matched experiment collection are implemented; full BC training and closed-loop evaluation remain to be run. SmolVLA and world-model components are experimental interfaces.

## Start here

- [Data and annotation conventions](data/README.md)
- [Classification](docs/classification.md)
- [Sorting simulation](docs/sorting.md)
- [Matched experiments and behavioral cloning](docs/experiments.md)
- [Colab setup](docs/colab.md)
- [Architecture and limitations](docs/architecture.md)

Run commands from the project root. Keep the existing environment if it works:

```sh
source .venv/bin/activate
python run_sorting.py --help
python sorting_experiments.py --help
```

For a new local setup, use Python 3.11 and the pinned dependencies:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sim.txt
python -m pip install -r requirements-classification.txt
```

`requirements-sim.txt` expects the LIBERO checkout at `LIBERO/`. The Colab guide includes its installation. On Intel macOS, `fix_mujoco_macos.py` provides the project-specific MuJoCo library fix if needed.

## Project layout

| Path | Purpose |
| --- | --- |
| Root Python scripts | Command-line entry points |
| `sorting/` | Tracking, classification, simulation, control, datasets, and models |
| `configs/` | Reference experiment configuration |
| `docs/` | Workflow guides |
| `data/` | Personal videos, metadata, annotations, crops, and demos |
| `outputs/` | Local generated outputs and review files |
| `results/` | Saved experiment results and checkpoints |
| `artifacts/` | Current Colab bundle and archived upload bundles |
| `scripts/legacy/` | Earlier collection implementation; root compatibility entry point retained |
| `LIBERO/` | External simulator repository |

`data/metadata.csv` defines the dataset split. Recorded outputs describe completed runs; configuration files do not establish performance. Simulation uses textured cuboids, oracle object localization, and labeled baskets. It is a controlled prototype, not a standard LIBERO benchmark or realistic-food manipulation evaluation.

Create a current Colab upload bundle with:

```sh
python make_sorting_bundle.py
```

The bundle is written to `artifacts/sorting_colab.zip`. It includes crops and demos, but excludes videos, environments, external LIBERO code, results, and pretrained checkpoints. Output paths generally refuse overwrite; choose a fresh name for each run.
