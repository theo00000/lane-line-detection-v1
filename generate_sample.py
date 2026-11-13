"""Create a synthetic driving video to check the pipeline without external data."""

import math
from pathlib import Path

import cv2
import numpy as np


def generate(path: str = "data/sample_road.mp4", width: int = 960, height: int = 540, n: int = 90) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = 25
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write {destination}")
    for frame_no in range(n):
        frame = np.full((height, width, 3), (183, 168, 130), dtype=np.uint8)
        y_horizon = int(height * 0.47)
        cv2.rectangle(frame, (0, y_horizon), (width, height), (69, 119, 70), -1)
        shift = int(width * 0.016 * math.sin(frame_no * 0.045))
        tl = (int(.43 * width) + shift, int(.56 * height))
        tr = (int(.57 * width) + shift, int(.56 * height))
        bl = (int(.15 * width) + shift, height - 1)
        br = (int(.85 * width) + shift, height - 1)
        road = np.array([tl, tr, br, bl], dtype=np.int32)
        cv2.fillPoly(frame, [road], (78, 78, 78))
        cv2.line(frame, tl, bl, (240, 240, 240), 11, cv2.LINE_AA)
        cv2.line(frame, tr, br, (20, 210, 250), 11, cv2.LINE_AA)
        cv2.putText(frame, "SYNTHETIC TEST - NOT A REAL BENCHMARK", (15, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (210, 210, 210), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"Created {destination} ({n} frames, {width}x{height}, {fps} FPS)")


if __name__ == "__main__":
    generate()
