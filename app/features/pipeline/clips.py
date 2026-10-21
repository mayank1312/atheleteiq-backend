"""
Cuts one student's footage out of the group video: keeps only the frames where they were tracked, follows them with a
smoothed, zoomed square crop, and writes a single clip. These clips are internal (analysis + audit) - the app never
shows them.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from app.features.pipeline.tracking import Track
from app.features.pipeline.video_utils import VideoInfo


@dataclass
class ClipResult:
    path: str
    n_frames: int
    seconds: float
    segments: list[tuple[int, int]]


def build_segments(frames: list[int], stride: int, fps: float, max_gap_s: float = 1.5, min_len_s: float = 1.5) -> list[tuple[int, int]]:
    """Group tracked frame indices into continuous segments; bridge tiny gaps, drop blips."""
    if not frames:
        return []
    frames = sorted(set(frames))
    max_gap = max(stride, int(max_gap_s * fps))
    segs, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > max_gap:
            segs.append((start, prev))
            start = f
        prev = f
    segs.append((start, prev))
    return [(a, b) for a, b in segs if (b - a + stride) / fps >= min_len_s]


def _per_frame_boxes(track: Track, first: int, last: int) -> np.ndarray:
    """Linearly interpolate the box for every frame between processed frames."""
    fr = np.array(track.frames, dtype=np.float64)
    bx = np.stack(track.boxes).astype(np.float64)
    idx = np.arange(first, last + 1, dtype=np.float64)
    return np.stack([np.interp(idx, fr, bx[:, c]) for c in range(4)], axis=1)


def _smooth_square_crops(boxes: np.ndarray, W: int, H: int, pad: float = 1.4, alpha: float = 0.12) -> np.ndarray:
    """Turn boxes into smoothed square crops (cx, cy, side) so the virtual camera doesn't jitter."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * pad
    out = np.empty((len(boxes), 3))
    s = np.array([cx[0], cy[0], side[0]])
    for i in range(len(boxes)):
        s = (1 - alpha) * s + alpha * np.array([cx[i], cy[i], side[i]])
        out[i] = s
    out[:, 2] = np.clip(out[:, 2], 64, min(W, H))
    return out


def write_student_clip(
    video_path: str, out_path: str, track: Track, segments: list[tuple[int, int]], info: VideoInfo, out_size: int = 720
) -> ClipResult:
    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (out_size, out_size))
    n_written = 0
    try:
        for (a, b) in segments:
            boxes = _per_frame_boxes(track, a, b)
            crops = _smooth_square_crops(boxes, info.width, info.height)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            for k in range(b - a + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                cx, cy, side = crops[k]
                x1 = int(round(min(max(cx - side / 2, 0), info.width - side)))
                y1 = int(round(min(max(cy - side / 2, 0), info.height - side)))
                s = int(round(side))
                patch = frame[y1 : y1 + s, x1 : x1 + s]
                if patch.size == 0:
                    continue
                writer.write(cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA))
                n_written += 1
    finally:
        cap.release()
        writer.release()
    return ClipResult(path=str(out_path), n_frames=n_written, seconds=n_written / info.fps, segments=segments)
