"""Evaluate YOLO V1 and optionally V2 with identical settings.

Examples:
    python scripts/evaluate_models.py --data yolo_road_line/data.yaml --imgsz 512
    python scripts/evaluate_models.py --data yolo_road_line/data.yaml --imgsz 512 --v2 models/yolo_v2/best.pt

Writes reviews/comparison/metrics.json and reviews/comparison/summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Matched YOLO box evaluation")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml for SAME validation split")
    parser.add_argument("--v1", type=Path, default=Path("models/yolo_v1/best.pt"))
    parser.add_argument("--v2", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=Path("reviews/comparison"))
    return parser.parse_args(argv)


def extract_box_metrics(result):
    """Extract box-detection metrics from an Ultralytics metrics object."""
    box = result.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def make_table(records):
    header = "| Model | Precision | Recall | mAP@50 | mAP@50:95 |\n|---|---:|---:|---:|---:|"
    rows = [header]
    for name, value in records.items():
        m = value["metrics"]
        rows.append(
            f'| {name} | {m["precision"]:.3f} | {m["recall"]:.3f} | '
            f'{m["map50"]:.3f} | {m["map50_95"]:.3f} |'
        )
    return "\n".join(rows)


def main():
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset config missing: {args.data}")
    models = {"v1": args.v1}
    if args.v2:
        models["v2"] = args.v2
    for label, path in models.items():
        if not path.is_file():
            raise SystemExit(f"{label.upper()} checkpoint missing: {path}")

    from ultralytics import YOLO  # Import only when actually running validation

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, checkpoint in models.items():
        metrics = YOLO(str(checkpoint.resolve())).val(
            data=str(args.data.resolve()),
            imgsz=args.imgsz,
            batch=1,
            device=args.device,
            workers=0,
            plots=True,
            project=str(out),
            name=name,
            exist_ok=True,
        )
        records[name] = {
            "checkpoint": str(checkpoint.resolve()),
            "metrics": extract_box_metrics(metrics),
        }

    report = {
        "dataset_config": str(args.data.resolve()),
        "image_size": args.imgsz,
        "device": args.device,
        "comparison_scope": "same data.yaml and validation resolution; validation, not held-out test",
        "models": records,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = (
        "# Validation comparison (not independent test)\n\n"
        f"Dataset: `{args.data.resolve()}`  \n"
        f"Image size: `{args.imgsz}`  \n"
        f"Device: `{args.device}`\n\n"
        + make_table(records)
        + "\n\nInterpret results cautiously: the validation split contains only 10 images "
        "in the original snapshot. Independent test results are not available.\n"
    )
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nSaved JSON and Markdown to {out}")


if __name__ == "__main__":
    main()
