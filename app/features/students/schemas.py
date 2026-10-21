from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str
    academy_id: int
    academy_name: str | None = None
    sport: str
    sports: list[str] = []
    height_cm: float
    weight_kg: float
    date_of_birth: date | None = None
    gender: str | None = None
    diet_preference: str | None = None
    allergies: str | None = None


class StudentMeasurementUpdate(BaseModel):
    height_cm: float = Field(gt=0, lt=300)
    weight_kg: float = Field(gt=0, lt=400)


class StudentProfileUpdate(BaseModel):
    """Editable account and athlete profile fields."""
    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    email: str | None = Field(default=None, min_length=3, max_length=255)
    academy_name: str | None = Field(default=None, min_length=1, max_length=255)
    sport: str | None = Field(default=None, min_length=1, max_length=100)
    sports: list[str] | None = Field(default=None, min_length=1, max_length=10)
    height_cm: float | None = Field(default=None, gt=0, lt=300)
    weight_kg: float | None = Field(default=None, gt=0, lt=400)
    date_of_birth: date | None = None
    gender: str | None = Field(default=None, pattern="^(male|female)$")
    diet_preference: str | None = Field(default=None, pattern="^(veg|non_veg|egg|vegan)$")
    allergies: str | None = Field(default=None, max_length=500)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class MeasurementPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    height_cm: float
    weight_kg: float
    recorded_at: datetime
