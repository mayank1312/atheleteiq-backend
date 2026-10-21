"""
One YOLO-pose pass gives, per frame: every person's box, a persistent track id (ByteTrack) and 17 body keypoints.
Those three things feed everything downstream (identity, clip cropping, biomechanics), so the video is decoded
through the network only once.

While streaming we also keep the few best "face visible" head crops per track - face recognition later runs on
those crops only, never on whole frames.
"""
import heapq
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from app.core.config import settings
from app.features.pipeline.video_utils import VideoInfo

# Let PyTorch quietly fall back to CPU for the few ops Apple's MPS backend doesn't implement yet.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

FACE_KPTS = (0, 1, 2, 3, 4)  # nose, eyes, ears (COCO order)
MAX_FACE_CANDIDATES = 6
MIN_FACE_CROP_SIDE = 160


@dataclass(order=True)
class FaceCandidate:
    score: float
    frame_idx: int = field(compare=False)
    crop: np.ndarray = field(compare=False, repr=False)


@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)  # original video frame indices
    boxes: list[np.ndarray] = field(default_factory=list)  # (4,) x1,y1,x2,y2
    kpts: list[np.ndarray] = field(default_factory=list)  # (17,3) x,y,conf
    face_heap: list[FaceCandidate] = field(default_factory=list)
    appearance_samples: list[np.ndarray] = field(default_factory=list)

    @property
    def face_candidates(self) -> list[FaceCandidate]:
        return sorted(self.face_heap, reverse=True)

    def add_face(self, cand: FaceCandidate) -> None:
        if len(self.face_heap) < MAX_FACE_CANDIDATES:
            heapq.heappush(self.face_heap, cand)
        elif cand.score > self.face_heap[0].score:
            heapq.heapreplace(self.face_heap, cand)

    def add_appearance(self, descriptor: np.ndarray) -> None:
        """Keep a small appearance memory for recovery after an occlusion or turn."""
        if len(self.appearance_samples) < 12:
            self.appearance_samples.append(descriptor)


def resolve_device(pref: str | None = None) -> str:
    pref = (pref or settings.DEVICE).lower()
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@lru_cache
def get_model():
    from ultralytics import YOLO  # heavy import, only when analysis actually runs

    return YOLO(settings.POSE_MODEL)


def head_crop(frame: np.ndarray, box: np.ndarray, kp: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Score how usable this person's face is and cut a head-region crop. None if the face isn't visible."""
    nose, leye, reye = kp[0], kp[1], kp[2]
    if min(nose[2], leye[2], reye[2]) < 0.5:
        return None
    pts = np.array([kp[i][:2] for i in FACE_KPTS if kp[i][2] >= 0.4])
    cx, cy = pts.mean(axis=0)
    eye_dist = float(np.linalg.norm(leye[:2] - reye[:2]))
    if eye_dist < 6:  # face too small / too far away to recognise
        return None
    side = eye_dist * 4.2  # eyes are ~1/4 of a head width apart
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    score = float(np.mean([nose[2], leye[2], reye[2]])) * min(1.0, eye_dist / 30.0)
    if min(crop.shape[:2]) < MIN_FACE_CROP_SIDE:  # upscale so the face detector has something to work with
        import cv2

        f = MIN_FACE_CROP_SIDE / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return score, crop


def appearance_descriptor(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Describe jersey/body appearance without using a face.

    The descriptor is deliberately coarse: HSV colour distribution plus colour
    moments. It can reconnect a ByteTrack fragment, but is not treated as proof
    of identity on its own.
    """
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    crop = frame[y1 + (y2 - y1) // 5 : y1 + (y2 - y1) * 4 // 5, x1 : x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (32, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 4], [0, 180, 0, 256]).flatten()
    hist /= np.linalg.norm(hist) or 1.0
    pixels = crop.astype(np.float32) / 255.0
    moments = np.concatenate((pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))))
    descriptor = np.concatenate((hist, moments)).astype(np.float32)
    return descriptor / (np.linalg.norm(descriptor) or 1.0)


def track_people(
    video_path: str,
    info: VideoInfo,
    stride: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> dict[int, Track]:
    stride = max(1, stride or settings.FRAME_STRIDE)
    model = get_model()
    results = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=settings.TRACKER,
        classes=[0],  # people only
        conf=0.25,
        imgsz=settings.DETECT_IMGSZ,
        device=resolve_device(),
        vid_stride=stride,
        verbose=False,
    )

    tracks: dict[int, Track] = {}
    total = max(1, info.frame_count // stride)
    for i, r in enumerate(results):
        frame_idx = i * stride
        if on_progress and i % 25 == 0:
            on_progress(min(1.0, i / total))
        boxes = r.boxes
        if boxes is None or boxes.id is None or r.keypoints is None:
            continue
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy()
        frame = r.orig_img
        for tid, box, kp in zip(ids, xyxy, kps):
            tr = tracks.setdefault(tid, Track(track_id=tid))
            tr.frames.append(frame_idx)
            tr.boxes.append(np.asarray(box, dtype=np.float32))
            tr.kpts.append(np.asarray(kp, dtype=np.float32))
            appearance = appearance_descriptor(frame, box)
            if appearance is not None:
                tr.add_appearance(appearance)
            hc = head_crop(frame, box, kp)
            if hc is not None:
                tr.add_face(FaceCandidate(score=hc[0], frame_idx=frame_idx, crop=hc[1]))
    if on_progress:
        on_progress(1.0)
    return tracks


def merge_tracks(parts: list[Track]) -> Track:
    """Join the fragments of one person (ByteTrack can split a track after occlusion) into a single timeline."""
    merged = Track(track_id=parts[0].track_id)
    rows = []
    for t in parts:
        rows.extend(zip(t.frames, t.boxes, t.kpts))
    rows.sort(key=lambda r: r[0])
    seen = set()
    for f, b, k in rows:
        if f in seen:
            continue
        seen.add(f)
        merged.frames.append(f)
        merged.boxes.append(b)
        merged.kpts.append(k)
    for t in parts:
        for c in t.face_heap:
            merged.add_face(c)
        for appearance in t.appearance_samples:
            merged.add_appearance(appearance)
    return merged
