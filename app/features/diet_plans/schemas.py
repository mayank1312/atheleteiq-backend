from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class DietPlanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    week_start_date: date
    height_cm_snapshot: float
    weight_kg_snapshot: float
    plan_json: dict
    created_at: datetime


class TrendPoint(BaseModel):
    week_start: date
    value: float


class QuarterSummary(BaseModel):
    label: str
    plans: int
    avg_calories: float | None
    avg_weight_kg: float | None


class DietRollup(BaseModel):
    period: str  # monthly | yearly
    plans: list[DietPlanOut]
    calories_trend: list[TrendPoint]
    weight_trend: list[TrendPoint]
    avg_calories: float | None
    weight_change_kg: float | None
    quarters: list[QuarterSummary] = []  # yearly only


class DietPending(BaseModel):
    status: str  # generating
    detail: str
