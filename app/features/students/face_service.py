"""
Face embedding generation.

PROTOTYPE STUB: returns a deterministic fake vector so the registration
flow, DB writes, and API contract can all be built/tested before the real
InsightFace model is wired in. Swap generate_embedding() for a real
InsightFace (ArcFace) call — same signature, same return shape — and
nothing else in the app needs to change.
"""

import hashlib

EMBEDDING_DIM = 512


def generate_embedding(image_bytes: bytes) -> list[float]:
    """
    Real implementation (next step) will:
      1. Detect the face in image_bytes (RetinaFace / YOLO-face)
      2. Align + crop it
      3. Run InsightFace's ArcFace model to get a 512-d embedding
      4. Return it as a plain list[float] for JSON storage

    For now: hash the image bytes into a fixed-length pseudo-vector so the
    rest of the system has something real to store and compare against.
    """
    digest = hashlib.sha256(image_bytes).digest()
    # Repeat/trim the digest bytes out to EMBEDDING_DIM floats in [0, 1)
    raw = (digest * (EMBEDDING_DIM // len(digest) + 1))[:EMBEDDING_DIM]
    return [b / 255.0 for b in raw]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
