# Cat Detection — YOLO26 Training Assessment

## Overview

This project trains a **YOLO26** object detector on a custom cat-detection dataset using COCO-pretrained weights. The notebook covers dataset inspection, train/val/test splitting, model training, evaluation, and prediction visualisation.

## Repository Structure

```
m6-04-assessment/
├── m6-04-assessment.ipynb   # Main notebook (all 6 tasks)
├── data.yaml                # YOLO dataset config
├── data/                    # Dataset (not committed — see below)
│   └── DATA_CLEAN/
│       ├── images/
│       ├── labels/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── runs/
│   └── detect/
│       ├── cat_detection
|       |   └── yolo26m_run
|       |       └── weights/
│       |           └── best.pt      # Best checkpoint 
│       └── val
├── yolo26m.pt
├── yolo26n.pt
└── .gitignore
```

## Setup

**1. Install dependencies**

```bash
pip install ultralytics torch torchvision matplotlib pandas pillow opencv-python
```

**2. Download the dataset**

Download the cat-detection dataset from Google Drive and place it at:

```
data/DATA_CLEAN/
```

> The `data/` folder is excluded from version control via `.gitignore`.

**3. Run the notebook**

Open `m6-04-assessment.ipynb` and run all cells top-to-bottom. The notebook will:

- Inspect and visualise the dataset (Task 1)
- Generate `train.txt`, `val.txt`, `test.txt` splits and `data.yaml` (Task 2)
- Download the COCO-pretrained `yolo26m.pt` weights and train for 30 epochs (Task 3)
- Evaluate on the test split and report mAP, Precision, Recall (Task 4)
- Visualise predictions including failure cases (Task 5)

## Model

| Setting | Value |
|---|---|
| Architecture | YOLO26m |
| Pretrained weights | COCO (`yolo26m.pt`) |
| Input size | 640 × 640 |
| Epochs | 30 |
| Seed | 42 |

**Why `yolo26m`?** The dataset is mid-sized (~1–2k images) and training was done on a GPU. The `m` variant offers a good accuracy/speed trade-off and produces a checkpoint under 100 MB that is practical to export to ONNX and ship in a Docker container next week.

## Results (Test Set)

| Metric | Value |
|---|---|
| mAP@0.5 | 0.6891 |
| mAP@0.5:0.95 | 0.8924 |
| Mean Precision | 0.8645 |
| Mean Recall | 0.8550 |

## Reproducing the Training Run

```python
from ultralytics import YOLO

model = YOLO("yolo26m.pt")
results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    project="runs",
    name="cats_v1",
    seed=42,
)
```

## Best Checkpoint

The best checkpoint is saved at `runs/cats_v1/weights/best.pt`.  
If the file exceeds 50 MB it will not be committed — a Google Drive link will be provided here instead.

## .gitignore

```
data/
runs/
*.pt
__pycache__/
.ipynb_checkpoints/
```