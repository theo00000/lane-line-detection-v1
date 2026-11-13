"""Run YOLO11n road-marking detection on images, folders or video.

Example:
    python -m src.predict_yolo --source data/external_test --device 0

Requires models/yolo_v1/best.pt, copied from the user's completed local run.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict road marking bounding boxes")
    parser.add_argument("--source", required=True, help="Image, image folder, video, or webcam index 0")
    parser.add_argument("--weights", default="models/yolo_v1/best.pt")
    parser.add_argument("--device", default="0", help="0 for NVIDIA CUDA, or cpu")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--name", default="road_line_demo")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise SystemExit(
            f"Model checkpoint not found: {weights}\n"
            "Copy the real best.pt from your training run to models/yolo_v1/best.pt."
        )
    source = 0 if args.source == "0" else args.source
    if isinstance(source, str) and not Path(source).exists() and not source.startswith(("http://", "https://")):
        raise SystemExit(f"Input image/folder/video does not exist: {source}")
    if not (0.0 <= args.conf <= 1.0):
        raise SystemExit("--conf must be between 0 and 1")

    # Delayed import allows CLI tests without requiring the model package to load.
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result_dir = (Path("runs") / "detect").resolve()
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(result_dir),
        name=args.name,
        exist_ok=True,
        save=True,
        stream=True,
    )
    count = 0
    for _ in results:
        count += 1
    print(f"Processed {count} frame(s) / image(s). Saved under: {result_dir / args.name}")


if __name__ == "__main__":
    main()
