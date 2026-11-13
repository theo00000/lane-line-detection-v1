# Architecture

## Research objective

Identify rectangular image regions that likely contain road markings. The training class is `road_line`. This is an **object detection** task, not lane geometry estimation.

## Two distinct pipelines

**Classical baseline**

```text
Image/video frame -> grayscale -> Gaussian blur -> Canny edges
                  -> region-of-interest mask -> Hough segments
                  -> line grouping/averaging -> line overlay
```

Purpose: show how geometric heuristics behave without learned weights. It may fail under shadows, curved markings, clutter, varying camera angles, and camera-position changes. The provided example is synthetic.

**YOLO11n detector**

```text
Pascal VOC XML + image ZIP
  -> EXIF-normalized image / YOLO xywh annotations
  -> train/validation split
  -> pretrained YOLO11n fine-tuning
  -> best.pt checkpoint
  -> image / video prediction
  -> class + confidence + bounding box visualization
```

The YOLO output does not specify which markings belong to the ego vehicle's left/right boundaries, nor their continuous curves. A segmentation/keypoint/polyline annotation and a distinct estimator would be needed for that application.

## Modules

| Module | Responsibility |
|---|---|
| `prepare_yolo.py` | Read original ZIP; apply EXIF orientation; convert labeled boxes to YOLO normalized coordinates |
| `src/train_yolo.py` | Baseline training using Ultralytics API |
| `src/predict_yolo.py` | CLI inference for files, folders, videos or camera |
| `src/lane_detector.py` | Image-processing helper functions for classical baseline |
| `src/run.py` | Process and render video with classical baseline |
| `scripts/evaluate_models.py` | Run validation consistently and export metrics |
| `tests/` | Unit tests for pure logic and CLI argument handling |

## System boundaries

No tracking across frames, steering recommendation, lane-departure alert, road-scene understanding, sensor fusion, or deployment-grade FPS guarantee is provided in V1.
