import hashlib

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decode_access_token
from app.core.timeutils import as_utc, utcnow
from app.db.session import get_db
from app.features.academies.models import Academy
from app.features.students.models import Student
from app.features.subscriptions.models import Subscription

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_current_student(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Student:
    student_id = decode_access_token(token)
    if student_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    student = db.get(Student, int(student_id))
    if student is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Student not found")
    return student


def latest_active_subscription(db: Session, student_id: int) -> Subscription | None:
    subs = (
        db.query(Subscription)
        .filter(Subscription.student_id == student_id, Subscription.status == "active")
        .order_by(Subscription.expires_at.desc())
        .all()
    )
    now = utcnow()
    return next((s for s in subs if as_utc(s.expires_at) > now), None)


def require_active_subscription(student: Student = Depends(get_current_student), db: Session = Depends(get_db)) -> Student:
    """The paywall. Login works without a subscription; every content route depends on this instead."""
    if latest_active_subscription(db, student.id) is None:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, {"code": "subscription_required", "message": "Active subscription required. Please subscribe to view this content."})
    return student


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    if not settings.ADMIN_API_KEY:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Admin API disabled: set ADMIN_API_KEY in .env")
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid admin key")


def get_academy_from_key(x_academy_key: str | None = Header(default=None), db: Session = Depends(get_db)) -> Academy:
    """Cameras / academy tooling authenticate with the key issued when the academy was created."""
    if not x_academy_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-Academy-Key header")
    academy = db.query(Academy).filter(Academy.api_key_hash == hash_api_key(x_academy_key), Academy.is_partner_active.is_(True)).first()
    if academy is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid academy key")
    return academy
