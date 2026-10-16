# Classification

Compare CLIP zero-shot with a frozen DINOv2 backbone and a trained two-class linear head. Class labels follow the project's culinary fruit/vegetable convention.

## Prepare the inputs

Review food crops first, then combine their manifests:

```sh
python combine_crops.py
```

`data/metadata.csv` defines train/val/test and seen/unseen membership. Keep every crop from one video in the same split. The current split contains 28 train, 15 validation, and 15 test videos. Only one test video belongs to an unseen food type, so unseen generalization evidence is limited.

## Train and validate

Use `--device cpu` locally or `--device cuda` in a GPU environment. Install dependencies using [Colab setup](colab.md) or the root README. Create the output directory before running:

```sh
mkdir -p results/classification_new
python classify.py clip --device cuda --splits val \
  --output results/classification_new/clip_val.jsonl
python classify.py embed --device cuda --splits train val \
  --output results/classification_new/dino_features.pt
python classify.py train-head --device cuda \
  --features results/classification_new/dino_features.pt \
  --output results/classification_new/dino_head.pt
python classify.py predict-head \
  --features results/classification_new/dino_features.pt \
  --checkpoint results/classification_new/dino_head.pt --split val \
  --output results/classification_new/dino_val.jsonl
python evaluate.py --classification results/classification_new/dino_val.jsonl \
  --split val --output results/classification_new/dino_val_report.json
```

## Frozen test evaluation

After selecting the model using validation, keep weights and preprocessing frozen. Extract test embeddings with the backbone revision saved in the head:

```sh
python classify.py embed --device cuda --splits test \
  --reference-checkpoint results/classification_new/dino_head.pt \
  --output results/classification_new/dino_test_features.pt
python classify.py predict-head \
  --features results/classification_new/dino_test_features.pt \
  --checkpoint results/classification_new/dino_head.pt --split test \
  --output results/classification_new/dino_test.jsonl
python classify.py clip --device cuda --splits test \
  --output results/classification_new/clip_test.jsonl
python evaluate.py --classification results/classification_new/dino_test.jsonl \
  --split test --output results/classification_new/dino_test_report.json
```

If `--reference-checkpoint` is unrecognized, the runtime has an older copy of `classify.py`; upload the current bundle. Existing output files cause an intentional error. Reuse a valid completed artifact or select a new output path.

## Pre-grasp evaluation

Review each video's grasp boundary in a CSV with `video_id,grasp_frame`. Frame values accept an integer or a filename such as `frame_000060.jpg`. Evaluate only crops before this boundary to reduce cues from the destination basket:

```sh
python review_classification.py pregrasp results/classification_new/dino_test.jsonl \
  --events outputs/review/test_grasp_events.csv --split test \
  --output outputs/review/dino_test_pregrasp_new.jsonl
python evaluate.py --classification outputs/review/dino_test_pregrasp_new.jsonl \
  --split test --output outputs/review/dino_test_pregrasp_report_new.json
```

Use the same boundaries and pooling protocol for both classifiers. Human-video accuracy and simulator-crop accuracy measure different input distributions. Diagnose simulator crops with the paired capture workflow in [sorting](sorting.md) before collecting a large matrix.
