import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core import storage
from app.core.config import settings
from app.core.deps import require_active_subscription
from app.core.timeutils import utcnow
from app.db.session import get_db
from app.features.sessions.models import StudentClip
from app.features.sessions.router import stream_to_temp
from app.features.students.models import Student
from app.features.uploads.models import SelfUpload
from app.features.uploads.quota import PACKS, QuotaExceeded, add_topup, consume_quota, quota_status
from app.features.uploads.schemas import PackOut, QuotaOut, SelfUploadOut, TopUpRequest
from app.features.uploads.tasks import process_self_upload, validate_self_upload
from app.features.uploads.validation import checks_as_list, initial_checks

router = APIRouter(prefix="/uploads", tags=["self-uploads"])


def _quota_out(db: Session, student: Student) -> QuotaOut:
    q = quota_status(db, student)
    return QuotaOut(
        unlimited=q.unlimited,
        has_subscription=q.has_subscription, included_total=q.included_total, included_used=q.included_used,
        included_remaining=q.included_remaining, topup_remaining=q.topup_remaining, remaining=q.remaining, resets_on=q.resets_on,
    )


def _out(u: SelfUpload) -> SelfUploadOut:
    return SelfUploadOut(
        id=u.id, sport=u.sport, status=u.status, checks=checks_as_list(u), can_submit=u.status == "validated",
        rejection_reason=u.rejection_reason, duration_s=u.duration_s, created_at=u.created_at,
    )


def _owned(db: Session, student: Student, upload_id: int) -> SelfUpload:
    u = db.get(SelfUpload, upload_id)
    if u is None or u.student_id != student.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return u


def _exhausted() -> HTTPException:
    return HTTPException(status.HTTP_402_PAYMENT_REQUIRED, {"code": "quota_exhausted", "message": "You've used all your uploads for this cycle."})


@router.get("/quota", response_model=QuotaOut)
def get_quota(student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    return _quota_out(db, student)


@router.get("/packs", response_model=list[PackOut])
def list_packs(student: Student = Depends(require_active_subscription)):
    return list(PACKS.values())


@router.post("/topup", response_model=QuotaOut)
def buy_topup(payload: TopUpRequest, student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    """PROTOTYPE: grants credits without payment. In production only the verified payment webhook may call add_topup()."""
    if payload.pack_id not in PACKS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown pack")
    add_topup(db, student, payload.pack_id)
    db.commit()
    return _quota_out(db, student)


@router.post("/validate", response_model=SelfUploadOut, status_code=status.HTTP_201_CREATED)
def start_validation(
    sport: str = Form(...),
    video: UploadFile = File(...),
    student: Student = Depends(require_active_subscription),
    db: Session = Depends(get_db),
):
    """Step 1: upload the video and start the checks. Uses no quota - that only happens at /submit."""
    match = next((s for s in student.sports if s.lower() == sport.strip().lower()), None)
    if match is None and not settings.TEST_PHASE_BYPASS_FACE_MATCHING:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"'{sport}' isn't one of your registered sports ({', '.join(student.sports)}).")
    match = match or sport.strip() or "Unknown"
    if not settings.TEST_PHASE_UNLIMITED_UPLOADS and quota_status(db, student).remaining <= 0:
        raise _exhausted()  # don't spend compute on a video they can't submit
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Please upload a video file.")

    tmp = stream_to_temp(video, settings.MAX_UPLOAD_MB)
    try:
        key = storage.new_key(f"self_uploads/{student.id}", video.filename or "video.mp4")
        storage.upload_file(tmp, key, content_type=video.content_type or "video/mp4")
    finally:
        os.unlink(tmp)

    upload = SelfUpload(student_id=student.id, sport=match, video_path=key, status="validating", checks_json=initial_checks())
    db.add(upload)
    db.commit()
    db.refresh(upload)
    validate_self_upload.delay(upload.id)
    return _out(upload)


@router.get("/{upload_id}", response_model=SelfUploadOut)
def get_upload(upload_id: int, student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    """Poll this to animate the checklist."""
    return _out(_owned(db, student, upload_id))


@router.post("/{upload_id}/submit", response_model=SelfUploadOut)
def submit_upload(upload_id: int, student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    """Step 2: the student confirms. This is the moment a credit is spent."""
    u = _owned(db, student, upload_id)
    if u.status != "validated":
        raise HTTPException(status.HTTP_409_CONFLICT, "This upload hasn't passed validation (or was already submitted).")
    try:
        consume_quota(db, student, u)
    except QuotaExceeded:
        raise _exhausted()
    u.status, u.submitted_at = "submitted", utcnow()
    db.add(StudentClip(student_id=student.id, self_upload_id=u.id, source="self_upload", status="processing"))
    db.commit()
    db.refresh(u)
    process_self_upload.delay(u.id)
    return _out(u)


@router.delete("/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
def discard_upload(upload_id: int, student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    """Cancel a draft (validating / validated / rejected). Submitted uploads can't be discarded."""
    u = _owned(db, student, upload_id)
    if u.status in ("submitted", "done", "failed"):
        raise HTTPException(status.HTTP_409_CONFLICT, "This upload has already been submitted.")
    storage.delete_object(u.video_path)
    if u.thumbnail_path:
        storage.delete_object(u.thumbnail_path)
    db.delete(u)
    db.commit()
