"""Cheap sanity checks run BEFORE a self-upload is allowed to consume a quota slot."""
import numpy as np

from app.core.config import settings
from app.features.pipeline.tracking import get_model, resolve_device

MIN_PERSON_HEIGHT_FRAC = 0.15  # a player must be at least 15% of the frame height to be analysable


def find_player_frames(frames: list[np.ndarray]) -> tuple[int, int | None]:
    """Returns (how many sampled frames contain a clearly visible person, index of the frame with the biggest person)."""
    if not frames:
        return 0, None
    model = get_model()
    visible, best_i, best_h = 0, None, 0.0
    for i, f in enumerate(frames):
        res = model.predict(f, classes=[0], conf=0.35, imgsz=settings.DETECT_IMGSZ, device=resolve_device(), verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        h = float(np.max(xyxy[:, 3] - xyxy[:, 1])) / f.shape[0]
        if h >= MIN_PERSON_HEIGHT_FRAC:
            visible += 1
            if h > best_h:
                best_h, best_i = h, i
    return visible, best_i
