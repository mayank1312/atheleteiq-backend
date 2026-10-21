"""
Decides which tracked person is which enrolled student.

Faces are only visible in a fraction of frames, so we do NOT match frame by frame. Instead each track collects its
best few head crops, those are averaged into one embedding, and that single embedding is compared with the academy
roster. The track's identity then applies to every frame the track covers - that is what lets us cut a student out
of group footage even when they're turned away from the camera.

Safety rules:
  * the best match must clear an absolute threshold AND beat the runner-up by a margin (avoids look-alike mixups)
  * one student can own several tracks (tracker fragments after occlusion), but two tracks that are on screen at the
    same time can't both be the same student - the weaker match is dropped
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.core.config import settings
from app.features.pipeline.faces import embed_face_bgr
from app.features.pipeline.tracking import Track


@dataclass
class RosterEntry:
    student_id: int
    embedding: np.ndarray  # L2-normalised


@dataclass
class TrackMatch:
    track_id: int
    student_id: int | None
    similarity: float
    margin: float
    n_faces: int
    source: str = "face"


def track_embedding(
    track: Track, embed: Callable = embed_face_bgr
) -> tuple[np.ndarray | None, int]:
    vecs = []
    for cand in track.face_candidates:
        res = embed(cand.crop)
        if res is not None:
            vecs.append(res[0])
    if not vecs:
        return None, 0
    mean = np.mean(np.stack(vecs), axis=0)
    n = np.linalg.norm(mean)
    return (mean / n if n > 0 else mean), len(vecs)


def match_tracks(
    tracks: dict[int, Track],
    roster: list[RosterEntry],
    threshold: float | None = None,
    margin: float | None = None,
    embed: Callable = embed_face_bgr,
) -> dict[int, TrackMatch]:
    threshold = settings.FACE_MATCH_THRESHOLD if threshold is None else threshold
    margin = settings.FACE_MATCH_MARGIN if margin is None else margin
    matches: dict[int, TrackMatch] = {}
    if not roster:
        return matches
    mat = np.stack([r.embedding for r in roster])  # (R, D)

    for tid, tr in tracks.items():
        emb, n_faces = track_embedding(tr, embed)
        if emb is None:
            matches[tid] = TrackMatch(tid, None, 0.0, 0.0, 0, "unmatched")
            continue
        sims = mat @ emb
        order = np.argsort(-sims)
        best = float(sims[order[0]])
        second = float(sims[order[1]]) if len(roster) > 1 else -1.0
        gap = best - second if len(roster) > 1 else best
        ok = best >= threshold and (len(roster) == 1 or gap >= margin)
        matches[tid] = TrackMatch(
            tid, roster[order[0]].student_id if ok else None, best, gap, n_faces,
            "face" if ok else "unmatched",
        )

    _recover_occluded_fragments(tracks, matches)
    _resolve_time_conflicts(tracks, matches)
    return matches


def _appearance_similarity(a: Track, b: Track) -> float | None:
    if not a.appearance_samples or not b.appearance_samples:
        return None
    left = np.mean(np.stack(a.appearance_samples), axis=0)
    right = np.mean(np.stack(b.appearance_samples), axis=0)
    left /= np.linalg.norm(left) or 1.0
    right /= np.linalg.norm(right) or 1.0
    return float(np.dot(left, right))


def _recover_occluded_fragments(tracks: dict[int, Track], matches: dict[int, TrackMatch]) -> None:
    """Propagate a high-confidence face match to a nearby visual track fragment.

    A face match remains the anchor. Appearance is only used for a short,
    non-overlapping gap, which prevents a jersey colour from identifying an
    unrelated player across the whole video.
    """
    anchors = [m for m in matches.values() if m.student_id is not None and m.source == "face"]
    for candidate in matches.values():
        if candidate.student_id is not None:
            continue
        tr = tracks[candidate.track_id]
        best: tuple[float, TrackMatch] | None = None
        for anchor in anchors:
            anchor_track = tracks[anchor.track_id]
            a_end, b_start = anchor_track.frames[-1], tr.frames[0]
            c_end, d_start = tr.frames[-1], anchor_track.frames[0]
            gap = min(abs(b_start - a_end), abs(d_start - c_end))
            if gap > 90:  # at most about three seconds at 30 fps
                continue
            if set(anchor_track.frames) & set(tr.frames):
                continue
            similarity = _appearance_similarity(anchor_track, tr)
            if similarity is None or similarity < 0.78:
                continue
            score = similarity - gap / 9000
            if best is None or score > best[0]:
                best = (score, anchor)
        if best is not None:
            candidate.student_id = best[1].student_id
            candidate.similarity = best[0]
            candidate.margin = 0.0
            candidate.source = "appearance_linked"


def _resolve_time_conflicts(tracks: dict[int, Track], matches: dict[int, TrackMatch]) -> None:
    by_student: dict[int, list[TrackMatch]] = {}
    for m in matches.values():
        if m.student_id is not None:
            by_student.setdefault(m.student_id, []).append(m)
    for ms in by_student.values():
        ms.sort(key=lambda m: m.similarity, reverse=True)
        kept: list[TrackMatch] = []
        for m in ms:
            f = set(tracks[m.track_id].frames)
            clash = False
            for k in kept:
                other = set(tracks[k.track_id].frames)
                inter = len(f & other)
                if inter and inter / max(1, min(len(f), len(other))) > 0.1:
                    clash = True
                    break
            if clash:
                m.student_id = None  # same person can't be in two places at once
            else:
                kept.append(m)


def dominant_track(tracks: dict[int, Track]) -> int | None:
    """Fallback for self-uploads when no face could be read: the person who is on screen longest and largest."""
    best_id, best_score = None, 0.0
    for tid, tr in tracks.items():
        if not tr.frames:
            continue
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in tr.boxes]
        score = len(tr.frames) * float(np.mean(areas))
        if score > best_score:
            best_id, best_score = tid, score
    return best_id
