"""DB-aware glue: loads rosters/history, runs the pipeline, stores clips + reports."""
import logging
import os
import tempfile
from datetime import date

from sqlalchemy.orm import Session

from app.core import storage
from app.core.config import settings
from app.core.timeutils import utcnow
from app.db.session import SessionLocal
from app.features.pipeline.faces import current_model_version
from app.features.pipeline.identity import RosterEntry
from app.features.pipeline.metrics import history_summary
from app.features.pipeline.orchestrator import AnalysisFailed, StudentAnalysis, analyze_video
from app.features.pipeline.report_llm import flatten_for_columns, generate_report
from app.features.reports.models import Report
from app.features.sessions.models import StudentClip, TrainingSession
from app.features.students.models import Student
from app.features.uploads.models import SelfUpload

import numpy as np

log = logging.getLogger("athleteiq.service")


def load_roster(db: Session, academy_id: int | None = None, student_id: int | None = None):
    q = db.query(Student)
    q = q.filter(Student.id == student_id) if student_id else q.filter(Student.academy_id == academy_id)
    roster, heights, students = [], {}, {}
    ver = current_model_version()
    for s in q.all():
        students[s.id] = s
        heights[s.id] = s.height_cm
        fe = s.face_embedding
        if fe is None or fe.model_version != ver:
            log.warning("student %s has no usable face embedding (version %s); needs re-enrolment", s.id, fe.model_version if fe else None)
            continue
        v = np.asarray(fe.embedding_vector, dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            roster.append(RosterEntry(s.id, v / n))
    return roster, heights, students


def athlete_context(s: Student) -> dict:
    ctx = {"height_cm": s.height_cm, "weight_kg": s.weight_kg}
    if s.date_of_birth:
        t = date.today()
        ctx["age"] = t.year - s.date_of_birth.year - ((t.month, t.day) < (s.date_of_birth.month, s.date_of_birth.day))
    return ctx


def previous_summaries(db: Session, student_id: int, sport_hint: str, exclude_clip_id: int | None, limit: int = 3) -> list[dict]:
    rows = (
        db.query(Report, StudentClip)
        .join(StudentClip, Report.student_clip_id == StudentClip.id)
        .filter(StudentClip.student_id == student_id, StudentClip.status == "ready")
        .order_by(Report.generated_at.desc())
        .limit(limit + 1)
        .all()
    )
    out = []
    for rep, clip in rows:
        if clip.id == exclude_clip_id:
            continue
        out.append({"date": rep.generated_at.date().isoformat(), **history_summary(rep.motion_metrics_json)})
    return out[:limit]


def save_student_result(db: Session, student: Student, sa: StudentAnalysis, sport: str, source: str,
                        session: TrainingSession | None = None, clip: StudentClip | None = None) -> StudentClip:
    if clip is None:
        clip = StudentClip(session_id=session.id if session else None, student_id=student.id, source=source, status="processing")
        db.add(clip)
        db.flush()
    if sa.clip and os.path.exists(sa.clip.path):
        try:
            clip.clip_path = storage.upload_file(sa.clip.path, storage.new_key(f"clips/{student.id}", "clip.mp4"))
        except Exception as exc:  # the analysis is still valuable without the stored clip
            log.error("clip upload failed for student %s: %s", student.id, exc)
    report = generate_report(
        sport, athlete_context(student), sa.metrics, source, sa.identity_confidence,
        previous_summaries(db, student.id, sport, clip.id),
    )
    motion, tech, scope = flatten_for_columns(report)
    db.add(Report(
        student_clip_id=clip.id, motion_metrics_json=sa.metrics, headline=report.headline,
        key_metrics_json=[k.model_dump() for k in report.key_metrics], structured_json=report.model_dump(),
        motion_tracking_analysis=motion, technique_analysis=tech, scope_of_improvement=scope,
    ))
    clip.status, clip.identity_confidence, clip.ready_at, clip.error = "ready", sa.identity_confidence, utcnow(), None
    db.commit()
    return clip


def _ext(key: str) -> str:
    return key.rsplit(".", 1)[-1] if "." in key else "mp4"


def process_session(session_id: int) -> None:
    db = SessionLocal()
    try:
        session = db.get(TrainingSession, session_id)
        if session is None:
            return
        session.status = "processing"
        db.commit()
        with tempfile.TemporaryDirectory() as tmp:
            local = storage.download_to(session.raw_footage_path, os.path.join(tmp, f"raw.{_ext(session.raw_footage_path)}"))
            roster, heights, students = load_roster(db, academy_id=session.academy_id)
            if not roster:
                log.warning("session %s: academy %s has no enrolled faces; nothing to match", session_id, session.academy_id)
            result = analyze_video(str(local), roster, os.path.join(tmp, "out"), heights, fixed_camera=True)
            log.info("session %s: %s tracks, %s students identified", session_id, result.debug.get("tracks_found"), len(result.students))
            for sa in result.students:
                save_student_result(db, students[sa.student_id], sa, session.sport, "academy", session=session)
        session.status = "done"
        db.commit()
    except Exception:
        db.rollback()
        s = db.get(TrainingSession, session_id)
        if s:
            s.status = "failed"
            db.commit()
        log.exception("session %s failed", session_id)
        raise
    finally:
        db.close()


def process_self_upload(upload_id: int) -> None:
    from app.features.uploads.quota import refund_quota

    db = SessionLocal()
    try:
        upload = db.get(SelfUpload, upload_id)
        if upload is None:
            return
        clip = upload.clip
        student = db.get(Student, upload.student_id)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local = storage.download_to(upload.video_path, os.path.join(tmp, f"in.{_ext(upload.video_path)}"))
                roster, heights, _ = load_roster(db, student_id=student.id)
                result = analyze_video(str(local), roster, os.path.join(tmp, "out"), heights, fixed_camera=False, only_student=student.id)
                if not result.students:
                    if settings.TEST_PHASE_BYPASS_FACE_MATCHING:
                        raise AnalysisFailed("We couldn't detect a person in this video to analyse.")
                    raise AnalysisFailed(
                        "We couldn't detect a face matching your profile in this video. "
                        "Only videos containing you can be analysed; record yourself with your face and whole body clearly visible."
                    )
                save_student_result(db, student, result.students[0], upload.sport, "self_upload", clip=clip)
            upload.status, upload.completed_at = "done", utcnow()
            db.commit()
        except Exception as exc:
            db.rollback()
            upload = db.get(SelfUpload, upload_id)
            clip = upload.clip
            reason = str(exc) if isinstance(exc, AnalysisFailed) else "Something went wrong while analysing this video."
            if not isinstance(exc, AnalysisFailed):
                log.exception("self-upload %s failed", upload_id)
            upload.status, upload.rejection_reason, upload.completed_at = "failed", reason, utcnow()
            if clip:
                clip.status, clip.error = "failed", reason
            refund_quota(db, upload)  # a failed analysis never costs the student a credit
            db.commit()
    finally:
        db.close()
