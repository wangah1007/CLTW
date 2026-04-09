## Overview

This directory provides a batch evaluation toolkit for **image quality / disturbance-factor metrics**, designed for datasets like CLTW that are organized by category folders. For each image folder, it outputs a score summary as `result.json`.

## Contents

- **Automation entry**
  - `metric_runner.py`: Recursively scans the dataset, selects the corresponding metric script based on the folder category, and writes `result.json` into each image folder.

- **Per-category metrics (under `metrics/`)**
  - `angle.py`: distortion / slanted angle (**input is `Label.txt`**, different from others)
  - `background.py`: complex background
  - `blur.py`: small/blurred image (blur)
  - `lowlight.py`: low light
  - `material.py`: special material
  - `occlusion.py`: occlusion
  - `reflection.py`: reflection/glare
  - `screen.py`: screen texture (moire / pattern)

## Data & annotation format

### Folder structure (CLTW convention)

Starting from `--root`, the runner searches for folders like:

- `<root>/<split>/<category>/<category>/`

Inside each "image folder":

- Images: `*.jpg` (also supports `png/bmp/tif`, etc.)
- `Label.txt`: used **only** by `angle.py`
- JSON annotations: one `*.json` per image with the same stem (e.g. `xxx.jpg` → `xxx.json`)

### JSON format (example)

The JSON top-level contains `characters`. Each character contains `transcription` and `points` (4-point box):

```json
{
  "filename": "xxx.jpg",
  "characters": [
    {"transcription": "字", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
  ]
}
```

## Workflow (high-level)

1. `metric_runner.py` scans the dataset recursively and finds each category image folder.
2. It selects the metric script by folder name:
   - `扭曲斜角度`: reads `Label.txt` in the same folder and scores images
   - Other categories: reads the per-image JSON file in the same folder and scores images
3. It writes `result.json` under that image folder (key = image filename, value = score or `null`).

## How to run

Run from this directory on your local machine or server:

```bash
python metric_runner.py --root "/path/to/CLTW/CLTW"
```

After finishing, each image folder will contain:

- `result.json`

