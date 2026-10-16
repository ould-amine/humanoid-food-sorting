# Personal dataset

`metadata.csv` is the source of truth for video identity, category, split, session, and evaluation group. Do not infer those fields again from crop filenames when preparing experiments.

| Field | Meaning |
| --- | --- |
| `split` | `train`, `val`, or `test`; all frames from a video share it |
| `session_id` | Recording session; keep sessions separate across splits |
| `object_id` | Food type, such as `red_apple` or `carrot`; not a physical specimen ID |
| `eval_group` | `seen` if the food type appears in training; otherwise `unseen` |
| `instruction` | Food-specific sorting instruction |

The current CSV has 58 videos: 28 train, 15 validation, and 15 test. Test contains 14 seen videos and one unseen video (`red_pepper`). White courgette follows the recording-session split recorded in the CSV.

## Layout

```text
data/
  metadata.csv
  videos/                    original personal recordings
  frames/<video_id>/         extracted images and annotation CSVs
  crops/<video_id>_points/    reviewed food crops and per-video manifest
  crops/manifest.jsonl       combined manifest
  demos/                     candidate and reviewed abstract demonstrations
```

## Annotation conventions

Mark the food center, a consistent hand contact/reference point, and the intended release point inside the basket. Use a stable reachable point within the basket rather than its rim. When the final placement point is uncertain, use the basket interior center consistently.

Leave an invisible point blank. Do not invent coordinates or reuse the last visible position. Interpolation is disabled unless explicitly requested. Food bounding boxes use `food_x1`, `food_y1`, `food_x2`, `food_y2` in image pixels; point coordinates use the existing annotation CSV columns. Frame indices always refer to the original video.

```sh
python extract_frames.py data/videos/banana_02.mov --every 5 \
  --output data/frames/banana_02_new
python annotate.py data/frames/banana_02_new/annotations.csv \
  --output data/frames/banana_02_new/annotated.csv
```

Use the actual filename and extension from your recordings. Left clicks cycle food → hand → target on the same frame. Keys `1/2/3` select a point; Tab clears that point and advances, including when a mistaken point already exists. Right-drag draws a food box. Keys `n/p` navigate, `s` saves, and `q` saves and exits. Keys `a/g/t/r` label the four manipulation phases.

## Crops and demos

```sh
python prepare_data.py crops data/frames/banana_02/annotated.csv \
  --video-id banana_02 --crop-size 160 --skip-missing-food \
  --output data/crops/banana_02_new
python combine_crops.py
python prepare_data.py demo data/frames/banana_02/annotated.csv \
  --video-id banana_02 --output data/demos/banana_02_candidate_new.json
```

`--crop-size` explicitly enables square crops around annotated food centers when boxes are absent. Choose a size that contains the food and review all resulting crops. Without boxes, mask-derived bounds, or this fallback, crop extraction rejects the annotations.

Review candidate grasp/release frames and trajectories before approving a demo. Use explicit `--grasp-frame` and `--release-frame` overrides when necessary. Keep candidate and reviewed files separate. Never use test videos to select model settings or the human motion template.

See [classification](../docs/classification.md) and [sorting](../docs/sorting.md) for the next stages.
