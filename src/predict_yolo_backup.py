"""Fine-tune YOLO11n on a converted road-line bounding-box dataset.

Requires yolo_road_line/data.yaml; see prepare_yolo.py.
This script is optional; the original V1 training run is already complete.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune road marking detector")
    parser.add_argument("--data", default="yolo_road_line/data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="road_line_v1")
    args = parser.parse_args()
    config = Path(args.data).resolve()
    if not config.is_file():
        raise SystemExit(f"Dataset YAML missing: {config}. Run prepare_yolo.py first.")
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        amp=False,  # AMP was automatically disabled in the original MX450 run.
        cache=False,
        project=str((Path("runs") / "detect").resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
