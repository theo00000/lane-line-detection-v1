# Training and reproducibility

## Hardware / environment from completed V1 log

- Windows, Python 3.11.9, Ultralytics 8.4.211
- NVIDIA GeForce MX450, 2,048 MiB VRAM, driver 581.83
- Torch build reported as 2.14.1+cu126 in the V1 log
- AMP requested (`amp=True`), but automatic checks failed, so Ultralytics disabled it during training

## V1: recorded experiment (completed)

Training dataset 40 images, validation 10, 53 annotated validation objects. Completed 30 epochs with YOLO11n pretrained initialization, `imgsz=416`, `batch=1`, `workers=0`, seed reported 0. Auto optimizer selected AdamW LR 0.002. Run ended normally and saved `best.pt` / `last.pt`. Training took about 0.078 hours (4.68 minutes) on that machine.

To **reproduce a new baseline** using the repository script:

```powershell
python -m src.train_yolo --data yolo_road_line/data.yaml --epochs 30 --imgsz 416 --batch 1 --device 0 --name road_line_reproduction
```

Note: `src/train_yolo.py` explicitly sets `amp=False` for numeric stability and is therefore not a bit-for-bit reproduction of the original command (whose AMP was auto-disabled). Package/library updates and nondeterministic GPU operations can also change outcomes.

## V2: proposed continuation (NOT confirmed trained)

This is a controlled *candidate* experiment, not a reported result. Load V1 checkpoint, reduce learning rate, turn off mosaic, and use a higher resolution; because multiple settings change simultaneously, a positive result cannot be attributed to one adjustment.

```powershell
# Run from D:\Projects\lane-line-detection-v1, reusing the original converted dataset.
$data = "D:/Projects/lane-line-detection/yolo_road_line/data.yaml"
yolo detect train model=models/yolo_v1/best.pt data=$data epochs=40 imgsz=512 batch=1 device=0 workers=0 amp=False optimizer=AdamW lr0=0.0005 patience=12 mosaic=0.0 degrees=2 translate=0.05 scale=0.15 hsv_h=0.01 hsv_s=0.3 hsv_v=0.25 fliplr=0.5 seed=42 project=runs name=road_line_v2
```

For CUDA Out Of Memory on MX450 2GB, reduce `imgsz` to 416 and record the change as a **different experiment**. The command creates `runs/road_line_v2/weights/best.pt` if successful.

## Validate on same configuration

Do not compare V1 original `imgsz=416` metrics directly with a new `imgsz=512` evaluation and call all differences a training effect. Run **both** checkpoints using identical dataset/split, `imgsz`, confidence policy, and library version.

```powershell
python scripts/evaluate_models.py --data $data --imgsz 512 --device 0 --v2 runs/road_line_v2/weights/best.pt
```

Outputs: `reviews/comparison/metrics.json`, `reviews/comparison/summary.md`, plus each model's Ultralytics plots in `reviews/comparison/v1/` and `reviews/comparison/v2/`. Do not upload raw training caches or huge `runs/` directories; curate a few plots after confirming you have redistribution rights.

## Interpret carefully

Small and noisy validation sets are fragile. Use image/box-level failure inspection, verify labels first, and create a genuinely held-out test set before claiming improved generalization. Also test inference end-to-end on video before asserting real-time throughput.
