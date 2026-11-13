# Road Marking Detection -- YOLO11n

A computer vision project exploring road marking detection using traditional OpenCV techniques and YOLO11n.

## Overview

This projects fine-tunes a pretrained YOLO11n model to detect road markings using bounding box annotations.

## Tech Stack

- Python 3.11
- PyTorch
- Ultralytics YOLO11n
- OpenCV
- NVIDIA CUDA

## Dataset

- 50 annotated road images
- 40 training images
- 10 validation images
- One class: road_line
- Pascal VOC XML converted to YOLO format

## Repository layout

```text
src/lane_detector.py   # OpenCV classical CV lane baseline
src/run.py             # Run classical CV on a video
src/train_yolo.py      # Fine-tune YOLO11n on XML->YOLO annotations
src/predict_yolo.py    # Predict on images, folder or video with best.pt
tests/                # Unit tests that don't require training weights
models/yolo_v1/        # Copy your completed best.pt here
prepare_yolo.py        # Convert Pascal VOC XML archive to YOLO labels
generate_sample.py     # Generate a synthetic video for OpenCV smoke tests
```

## Installation (Windows PowerShell)

Requires Python 3.11. In a new clone, create a virtual environment and install a compatible PyTorch CUDA build via the official selector before installing other requirements. Existing environment users can reuse their working CUDA PyTorch.

```powershell
cd D:\Projects\lane-line-detection-v1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Classical CV (OpenCV baseline)

```powershell
python generate_sample.py
python -m src.run --input data/sample_road.mp4 --output outputs/lane_overlay.mp4
```

This pipeline is grayscale -> Gaussian blur -> Canny -> ROI -> Hough lines -> overlay. Synthetic smoke tests do not establish road accuracy.

## YOLO detection

1. Copy your *real* trained checkpoint into `models/yolo_v1/best.pt` (see the README in that folder).
2. Put your own test pictures in `data/external_test/` (excluded from Git).
3. Run:

```powershell
python -m src.predict_yolo --source data/external_test --device 0
```

## Baseline Results

| Metric | Score |
|---|---|
| Precision | 0.741 |
| Recall | 0.321 |
| mAP@50 | 0.345 |
| mAP@50-95 | 0.219 |

## Limitations

- Small validation dataset
- Missed detections on continuous road markings
- False positives in non-road regions
- Bounding boxes do not represent continuous lane geometry

## Next Research Direction

Explore pixel-level lane segmentation using a lightweight U-Net.

## Disclaimer

Research and portfolio demonstration only.
Not intended for real-world vehicle safety systems.