from datetime import date, datetime, timezone

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    academy_id: Mapped[int] = mapped_column(ForeignKey("academies.id"), nullable=False)

    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # One or more sports, comma separated ("Soccer,Badminton"). The first is the primary sport.
    sport: Mapped[str] = mapped_column(String(100), nullable=False)
    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)

    # Needed for a safe diet plan (age matters a lot for minors).
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)  # male | female
    diet_preference: Mapped[str | None] = mapped_column(String(20), nullable=True)  # veg | non_veg | egg | vegan
    allergies: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    academy = relationship("Academy", back_populates="students")
    face_embedding = relationship("FaceEmbedding", back_populates="student", uselist=False)
    subscriptions = relationship("Subscription", back_populates="student")
    clips = relationship("StudentClip", back_populates="student")
    diet_plans = relationship("DietPlan", back_populates="student")

    @property
    def sports(self) -> list[str]:
        return [s.strip() for s in self.sport.split(",") if s.strip()]

    @property
    def academy_name(self) -> str | None:
        return self.academy.name if self.academy else None


class FaceEmbedding(Base):
    """
    Numeric face vector only - the selfie itself is never stored (biometric data minimisation).
    model_version lets us ignore vectors from an older/fake model when matching.
    """
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), unique=True, nullable=False)
    embedding_vector: Mapped[list] = mapped_column(JSON, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="arcface-buffalo_l-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    student = relationship("Student", back_populates="face_embedding")


class MeasurementLog(Base):
    """History of height/weight so the app can draw trends and the diet planner can see change over time."""
    __tablename__ = "measurement_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True, nullable=False)
    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
