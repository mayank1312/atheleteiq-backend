from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_clip_id: Mapped[int] = mapped_column(ForeignKey("student_clips.id"), unique=True, nullable=False)

    # Raw numbers measured from the pose data - the ground truth the written analysis is based on.
    motion_metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    headline: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_metrics_json: Mapped[list | None] = mapped_column(JSON, nullable=True)  # stat chips for the app
    structured_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # full structured report content

    # Flattened text copies of the three segments.
    motion_tracking_analysis: Mapped[str] = mapped_column(Text, nullable=False)
    technique_analysis: Mapped[str] = mapped_column(Text, nullable=False)
    scope_of_improvement: Mapped[str] = mapped_column(Text, nullable=False)

    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    clip = relationship("StudentClip", back_populates="report")
