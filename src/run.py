"""Run classical CV lane detection on a video or webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .lane_detector import detect_lanes


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV lane detection baseline")
    parser.add_argument("--input", default="data/sample_road.mp4", help="Input video path, or 0 for webcam")
    parser.add_argument("--output", default="outputs/lane_overlay.mp4", help="Output video path")
    parser.add_argument("--show", action="store_true", help="Preview output frame by frame (press Q to exit)")
    parser.add_argument("--debug", action="store_true", help="Display ROI-masked Canny edges (requires --show)")
    args = parser.parse_args()

    source: int | str = 0 if args.input == "0" else args.input
    if isinstance(source, str) and not Path(source).is_file():
        raise SystemExit(f"Input not found: {source}. Run `python generate_sample.py` first.")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open input: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("Input does not report a valid frame size")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Cannot create output video: {out_path}")

    frames, total_seconds = 0, 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            result, edges = detect_lanes(frame)
            duration = time.perf_counter() - start
            frames += 1
            total_seconds += duration
            current_fps = 1.0 / max(duration, 1e-8)
            cv2.putText(result, f"Processing: {current_fps:.1f} FPS", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(result)
            if args.show:
                cv2.imshow("Lane detection baseline", result)
                if args.debug:
                    cv2.imshow("Masked Canny edges", edges)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if not frames:
        raise SystemExit("No readable frames in input")
    print(f"Processed {frames} frames. Average processing FPS: {frames / total_seconds:.2f}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
