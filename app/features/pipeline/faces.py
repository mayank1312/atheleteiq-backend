"""Face embeddings (ArcFace via InsightFace 'buffalo_l'). FACE_BACKEND=stub swaps in fake vectors for wiring tests."""
import hashlib
from functools import lru_cache

import cv2
import numpy as np

from app.core.config import settings

EMBEDDING_DIM = 512
EMBEDDING_MODEL_VERSION = "arcface-buffalo_l-v1"
STUB_MODEL_VERSION = "stub-hash-v1"
MIN_DET_SCORE = 0.5


class FaceError(Exception):
    """The image doesn't contain a usable face."""


def current_model_version() -> str:
    return STUB_MODEL_VERSION if settings.FACE_BACKEND == "stub" else EMBEDDING_MODEL_VERSION


@lru_cache
def _face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_face_bgr(img: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Embedding of the most prominent face in the image, plus detector confidence. None if no face."""
    if settings.FACE_BACKEND == "stub":
        digest = hashlib.sha256(img.tobytes()[:4096]).digest()
        raw = np.frombuffer((digest * (EMBEDDING_DIM // len(digest) + 1))[:EMBEDDING_DIM], dtype=np.uint8)
        return _normalize(raw.astype(np.float32) - 127.5), 1.0
    faces = _face_app().get(img)
    faces = [f for f in faces if float(f.det_score) >= MIN_DET_SCORE]
    if not faces:
        return None
    best = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) * float(f.det_score))
    return _normalize(np.asarray(best.normed_embedding, dtype=np.float32)), float(best.det_score)


def embed_from_bytes(image_bytes: bytes) -> tuple[list[float], int]:
    """Registration path. Returns (embedding, number_of_faces_found). Raises FaceError if unusable."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError("That file isn't a readable image.")
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        s = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if settings.FACE_BACKEND == "stub":
        emb, _ = embed_face_bgr(img)
        return emb.tolist(), 1
    faces = [f for f in _face_app().get(img) if float(f.det_score) >= MIN_DET_SCORE]
    if not faces:
        raise FaceError("No clear face found. Face the camera in good light and try again.")
    best = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
    return _normalize(np.asarray(best.normed_embedding, dtype=np.float32)).tolist(), len(faces)
