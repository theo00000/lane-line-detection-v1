"""Classical CV lane detector (deliberately CPU-based baseline)."""

from __future__ import annotations

import cv2
import numpy as np


def region_of_interest(edges: np.ndarray) -> np.ndarray:
    """Keep a camera-relative trapezoid around the road ahead."""
    height, width = edges.shape[:2]
    vertices = np.array([
        [int(0.08 * width), height - 1],
        [int(0.43 * width), int(0.59 * height)],
        [int(0.57 * width), int(0.59 * height)],
        [int(0.92 * width), height - 1],
    ], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(edges, mask)


def _fit_lane(
    segments: list[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Fit x = a*y + b from detected segment endpoints."""
    if not segments:
        return None

    x_coords: list[float] = []
    y_coords: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        x_coords.extend((x1, x2))
        y_coords.extend((y1, y2))
        weights.extend((np.sqrt(length), np.sqrt(length)))

    try:
        # x is a function of y, preventing problems for steep lane lines.
        a, b = np.polyfit(y_coords, x_coords, deg=1, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None

    y_bottom = height - 1
    y_top = int(0.60 * height)
    x_bottom = int(np.clip(a * y_bottom + b, 0, width - 1))
    x_top = int(np.clip(a * y_top + b, 0, width - 1))
    return (x_bottom, y_bottom), (x_top, y_top)


def detect_lanes(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rendered BGR frame, masked Canny edges)."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    masked = region_of_interest(edges)

    raw_lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(20, int(0.035 * width)),
        maxLineGap=max(15, int(0.045 * width)),
    )

    left: list[tuple[int, int, int, int]] = []
    right: list[tuple[int, int, int, int]] = []
    if raw_lines is not None:
        for line in raw_lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 2:
                continue
            slope = dy / dx
            if abs(slope) < 0.50 or abs(slope) > 10:
                continue  # Reject almost-horizontal or extreme noise.
            midx = (x1 + x2) / 2
            if slope < 0 and midx < width * 0.55:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and midx > width * 0.45:
                right.append((x1, y1, x2, y2))

    line_layer = np.zeros_like(frame)
    for candidates, color in ((left, (0, 255, 0)), (right, (0, 255, 255))):
        points = _fit_lane(candidates, height, width)
        if points is not None:
            cv2.line(line_layer, points[0], points[1], color, 10, cv2.LINE_AA)

    overlay = cv2.addWeighted(frame, 0.82, line_layer, 0.95, 0)
    return overlay, masked
