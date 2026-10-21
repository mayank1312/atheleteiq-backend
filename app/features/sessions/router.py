import os
import shutil
import tempfile
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core import storage
from app.core.config import settings
from app.core.deps import get_academy_from_key
from app.db.session import get_db
from app.features.academies.models import Academy
from app.features.sessions.models import TrainingSession
from app.features.sessions.schemas import TrainingSessionOut
from app.features.sessions.tasks import process_training_session

router = APIRouter(prefix="/sessions", tags=["sessions"])


def stream_to_temp(upload: UploadFile, max_mb: int) -> str:
    """Copy an upload to disk in chunks so a big video never sits in memory. Raises 413 past the limit."""
    limit = max_mb * 1024 * 1024
    suffix = os.path.splitext(upload.filename or "")[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File is larger than {max_mb} MB.")
                out.write(chunk)
    except Exception:
        os.unlink(path)
        raise
    return path


@router.post("/upload", response_model=TrainingSessionOut, status_code=status.HTTP_201_CREATED)
def upload_training_session(
    sport: str = Form(...),
    recorded_at: datetime = Form(...),
    footage: UploadFile = File(...),
    academy: Academy = Depends(get_academy_from_key),
    db: Session = Depends(get_db),
):
    """
    Called by an academy's camera system / staff tooling, authenticated with that academy's X-Academy-Key.
    Students never see this footage; it only feeds the analysis pipeline.
    """
    tmp = stream_to_temp(footage, settings.MAX_ACADEMY_UPLOAD_MB)
    try:
        key = storage.new_key(f"raw_sessions/{academy.id}", footage.filename or "session.mp4")
        storage.upload_file(tmp, key)
    finally:
        os.unlink(tmp)
    session = TrainingSession(academy_id=academy.id, sport=sport, raw_footage_path=key, recorded_at=recorded_at)
    db.add(session)
    db.commit()
    db.refresh(session)
    process_training_session.delay(session.id)
    return session
