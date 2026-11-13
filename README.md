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