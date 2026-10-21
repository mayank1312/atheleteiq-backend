"""Video in -> per-student analysis out. No database here, so it can run from a script or from a Celery task."""
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from app.core.config import settings
from app.features.pipeline.clips import ClipResult, build_segments, write_student_clip
from app.features.pipeline.identity import RosterEntry, TrackMatch, dominant_track, match_tracks
from app.features.pipeline.metrics import compute_metrics
from app.features.pipeline.tracking import Track, merge_tracks, track_people
from app.features.pipeline.video_utils import VideoInfo, probe_video

log = logging.getLogger("athleteiq.pipeline")
MIN_TRACK_SECONDS = 1.0


class AnalysisFailed(Exception):
    """Raised when a video can't yield an analysis; the message is safe to show the student."""


@dataclass
class StudentAnalysis:
    student_id: int
    track_ids: list[int]
    identity_confidence: str  # face_matched | appearance_linked | assumed
    similarity: float
    metrics: dict
    clip: ClipResult | None


@dataclass
class VideoAnalysis:
    info: VideoInfo
    students: list[StudentAnalysis] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def analyze_video(
    video_path: str,
    roster: list[RosterEntry],
    out_dir: str,
    heights: dict[int, float | None] | None = None,
    fixed_camera: bool = True,
    only_student: int | None = None,
    on_progress: Callable[[float], None] | None = None,
    embed: Callable | None = None,
) -> VideoAnalysis:
    """
    roster:        students who may appear (academy footage) - or just the uploader (self-upload).
    only_student:  self-upload mode. If set, only that student's face-matched tracks are accepted.
    """
    heights = heights or {}
    info = probe_video(video_path)
    stride = max(1, settings.FRAME_STRIDE)
    os.makedirs(out_dir, exist_ok=True)

    tracks = track_people(video_path, info, stride, on_progress)
    min_frames = MIN_TRACK_SECONDS * info.fps / stride
    tracks = {tid: t for tid, t in tracks.items() if len(t.frames) >= min_frames}
    result = VideoAnalysis(info=info, debug={"tracks_found": len(tracks)})
    if not tracks:
        return result

    bypass_identity = settings.TEST_PHASE_BYPASS_FACE_MATCHING and only_student is not None
    if bypass_identity:
        selected = dominant_track(tracks)
        matches = {
            tid: TrackMatch(tid, only_student if tid == selected else None, 0.0, 0.0, 0)
            for tid in tracks
        }
    else:
        kwargs = {"embed": embed} if embed else {}
        matches = match_tracks(tracks, roster, **kwargs)
    result.debug["matches"] = {tid: {"student": m.student_id, "sim": round(m.similarity, 3), "margin": round(m.margin, 3), "faces": m.n_faces} for tid, m in matches.items()}

    groups: dict[int, list[int]] = {}
    for tid, m in matches.items():
        if m.student_id is not None and (only_student is None or m.student_id == only_student):
            groups.setdefault(m.student_id, []).append(tid)

    confidence = {
        sid: (
            "assumed"
            if bypass_identity
            else "appearance_linked"
            if any(matches[t].source == "appearance_linked" for t in tids)
            else "face_matched"
        )
        for sid in groups
    }
    for sid, tids in groups.items():
        parts = [tracks[t] for t in tids]
        merged: Track = merge_tracks(parts)
        segs = build_segments(merged.frames, stride, info.fps)
        if not segs:
            continue
        metrics = compute_metrics(merged, stride, info.fps, heights.get(sid), fixed_camera)
        # Keep the tracked-window duration for biomechanics, but also expose the
        # original upload duration so the report can explain coverage honestly.
        metrics.setdefault("meta", {})["source_duration_s"] = info.duration_s
        clip = write_student_clip(video_path, os.path.join(out_dir, f"student_{sid}.mp4"), merged, segs, info)
        sim = max((matches[t].similarity for t in tids), default=0.0)
        result.students.append(StudentAnalysis(sid, tids, confidence[sid], sim, metrics, clip))
    return result
