from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base


class Academy(Base):
    __tablename__ = "academies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_partner_active: Mapped[bool] = mapped_column(default=True)
    # sha256 of the key that cameras / academy tooling send as X-Academy-Key. The raw key is shown once at creation.
    api_key_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    students = relationship("Student", back_populates="academy")
    sessions = relationship("TrainingSession", back_populates="academy")
