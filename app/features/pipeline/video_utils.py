from dataclasses import dataclass

import cv2
import numpy as np


class VideoError(Exception):
    pass


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("Video could not be opened (unsupported or corrupted file).")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0 or n <= 0:
            raise VideoError("Video has no readable frames.")
        if fps <= 0 or fps > 240:
            fps = 30.0  # some containers report nonsense; assume 30
        return VideoInfo(width=w, height=h, fps=fps, frame_count=n)
    finally:
        cap.release()


def sample_frames(path: str, n: int = 6, max_side: int = 768) -> list[np.ndarray]:
    """n frames evenly spread across the video (skipping the very first/last frame)."""
    info = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for k in range(n):
            idx = int((k + 0.5) / n * (info.frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(_shrink(frame, max_side))
    finally:
        cap.release()
    return frames


def _shrink(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, max_side: int = 768, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", _shrink(frame, max_side), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("Could not encode frame.")
    return buf.tobytes()
