from datetime import date, datetime, timezone

from sqlalchemy import DateTime, Date, ForeignKey, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class DietPlan(Base):
    """One plan per student per week — not regenerated per video (see product spec)."""
    __tablename__ = "diet_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False)

    week_start_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Snapshot of the measurements this plan was generated from, so later
    # monthly/yearly rollups can show trend lines even if the student profile changes.
    height_cm_snapshot: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg_snapshot: Mapped[float] = mapped_column(Float, nullable=False)

    plan_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # meals, macros, notes — LLM-generated

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    student = relationship("Student", back_populates="diet_plans")
