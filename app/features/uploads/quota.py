"""
Self-upload quota.
  * Every active subscription cycle includes SELF_UPLOAD_INCLUDED_PER_CYCLE uploads (default 2).
  * The cycle is the current subscription's window, so it resets on the renewal date.
  * Top-up credits are only spent after the included uploads are used.
  * A slot is consumed when the student SUBMITS (not when they validate) and is refunded if analysis fails.
"""
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import latest_active_subscription
from app.core.timeutils import as_utc, utcnow
from app.features.students.models import Student
from app.features.uploads.models import SelfUpload, UploadTopUp

# Placeholder pricing - decide real numbers before launch.
PACKS = {
    "pack_2": {"id": "pack_2", "uploads": 2, "price_inr": 149},
    "pack_5": {"id": "pack_5", "uploads": 5, "price_inr": 299},
}


class QuotaExceeded(Exception):
    pass


@dataclass
class QuotaStatus:
    unlimited: bool
    has_subscription: bool
    included_total: int
    included_used: int
    topup_remaining: int
    resets_on: object  # datetime | None

    @property
    def included_remaining(self) -> int:
        if self.unlimited:
            return 2**31 - 1
        return max(0, self.included_total - self.included_used)

    @property
    def remaining(self) -> int:
        if self.unlimited:
            return 2**31 - 1
        return self.included_remaining + self.topup_remaining


def quota_status(db: Session, student: Student) -> QuotaStatus:
    sub = latest_active_subscription(db, student.id)
    if sub is None:
        return QuotaStatus(settings.TEST_PHASE_UNLIMITED_UPLOADS, False, 0, 0, 0, None)
    start = as_utc(sub.started_at)
    used = sum(
        1
        for u in db.query(SelfUpload).filter(SelfUpload.student_id == student.id, SelfUpload.quota_source == "included", SelfUpload.quota_refunded.is_(False)).all()
        if u.submitted_at and as_utc(u.submitted_at) >= start
    )
    topups = db.query(UploadTopUp).filter(UploadTopUp.student_id == student.id).all()
    return QuotaStatus(
        settings.TEST_PHASE_UNLIMITED_UPLOADS,
        True,
        settings.SELF_UPLOAD_INCLUDED_PER_CYCLE,
        used,
        sum(t.credits_remaining for t in topups),
        as_utc(sub.expires_at),
    )


def consume_quota(db: Session, student: Student, upload: SelfUpload) -> None:
    if settings.TEST_PHASE_UNLIMITED_UPLOADS:
        return
    q = quota_status(db, student)
    if q.included_remaining > 0:
        upload.quota_source = "included"
        return
    topup = (
        db.query(UploadTopUp)
        .filter(UploadTopUp.student_id == student.id, UploadTopUp.credits_remaining > 0)
        .order_by(UploadTopUp.created_at)
        .first()
    )
    if topup is None:
        raise QuotaExceeded()
    topup.credits_remaining -= 1
    upload.quota_source, upload.topup_id = "topup", topup.id


def refund_quota(db: Session, upload: SelfUpload) -> None:
    if upload.quota_refunded or upload.quota_source is None:
        return
    if upload.quota_source == "topup" and upload.topup_id:
        t = db.get(UploadTopUp, upload.topup_id)
        if t:
            t.credits_remaining += 1
    upload.quota_refunded = True


def add_topup(db: Session, student: Student, pack_id: str) -> UploadTopUp:
    pack = PACKS[pack_id]
    t = UploadTopUp(student_id=student.id, pack_id=pack_id, credits_total=pack["uploads"], credits_remaining=pack["uploads"], created_at=utcnow())
    db.add(t)
    return t
