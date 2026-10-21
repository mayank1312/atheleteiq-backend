from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class TrainingSession(Base):
    """One raw footage upload from an academy camera - the shared pool several students appear in."""
    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    academy_id: Mapped[int] = mapped_column(ForeignKey("academies.id"), nullable=False)

    sport: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_footage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="uploaded")  # uploaded/processing/done/failed

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    academy = relationship("Academy", back_populates="sessions")
    clips = relationship("StudentClip", back_populates="session")


class StudentClip(Base):
    """A student-specific analysis unit: cropped clip + report. Comes from an academy session OR a self-upload."""
    __tablename__ = "student_clips"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("training_sessions.id"), nullable=True)
    self_upload_id: Mapped[int | None] = mapped_column(ForeignKey("self_uploads.id"), nullable=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False)

    source: Mapped[str] = mapped_column(String(20), default="academy")  # academy | self_upload
    clip_path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # internal only, never sent to the app
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending/processing/ready/failed
    identity_confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)  # face_matched | assumed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session = relationship("TrainingSession", back_populates="clips")
    student = relationship("Student", back_populates="clips")
    report = relationship("Report", back_populates="clip", uselist=False)
    self_upload = relationship("SelfUpload", back_populates="clip", foreign_keys=[self_upload_id])
