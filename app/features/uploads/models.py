from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class SelfUpload(Base):
    """
    A student-submitted video.
    validating -> validated -> submitted -> done            (happy path; quota is consumed at 'submitted')
    validating -> rejected                                   (failed pre-checks; quota NOT consumed)
    submitted  -> failed                                     (failed during analysis; quota refunded)
    """
    __tablename__ = "self_uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True, nullable=False)

    sport: Mapped[str] = mapped_column(String(100), nullable=False)
    video_path: Mapped[str] = mapped_column(String(500), nullable=False)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(20), default="validating")
    checks_json: Mapped[dict] = mapped_column(JSON, default=dict)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    quota_source: Mapped[str | None] = mapped_column(String(10), nullable=True)  # included | topup
    topup_id: Mapped[int | None] = mapped_column(ForeignKey("upload_topups.id"), nullable=True)
    quota_refunded: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    clip = relationship("StudentClip", back_populates="self_upload", foreign_keys="StudentClip.self_upload_id", uselist=False)


class UploadTopUp(Base):
    """Purchased extra uploads. Used only after the monthly included uploads run out."""
    __tablename__ = "upload_topups"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True, nullable=False)
    pack_id: Mapped[str] = mapped_column(String(30), nullable=False)
    credits_total: Mapped[int] = mapped_column(Integer, nullable=False)
    credits_remaining: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
