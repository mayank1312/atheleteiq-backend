from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import latest_active_subscription, require_active_subscription
from app.core.timeutils import week_start
from app.db.session import get_db
from app.features.diet_plans.models import DietPlan
from app.features.reports.router import to_item
from app.features.reports.schemas import AnalysisItem
from app.features.sessions.models import StudentClip
from app.features.students.models import Student
from app.features.uploads.router import _quota_out
from app.features.uploads.schemas import QuotaOut

router = APIRouter(prefix="/home", tags=["home"])


class DietSummary(BaseModel):
    status: str  # ready | pending | profile_incomplete
    week_start: str | None = None
    calories: int | None = None
    focus_note: str | None = None


class SubscriptionSummary(BaseModel):
    plan_type: str
    renews_on: datetime


class HomeOut(BaseModel):
    name: str
    academy_name: str | None
    sports: list[str]
    height_cm: float
    weight_kg: float
    latest_analysis: AnalysisItem | None
    diet: DietSummary
    upload_quota: QuotaOut
    subscription: SubscriptionSummary


@router.get("", response_model=HomeOut)
def home(student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    """One call for the dashboard. Never triggers slow work - it only reports what already exists."""
    latest = db.query(StudentClip).filter(StudentClip.student_id == student.id).order_by(StudentClip.created_at.desc()).first()
    plan = db.query(DietPlan).filter(DietPlan.student_id == student.id, DietPlan.week_start_date == week_start()).first()
    if plan:
        notes = plan.plan_json.get("focus_notes") or []
        diet = DietSummary(status="ready", week_start=str(plan.week_start_date), calories=plan.plan_json.get("calorie_target"), focus_note=notes[0] if notes else None)
    elif not student.date_of_birth or not student.gender:
        diet = DietSummary(status="profile_incomplete")
    else:
        diet = DietSummary(status="pending")
    sub = latest_active_subscription(db, student.id)
    return HomeOut(
        name=student.full_name, academy_name=student.academy_name, sports=student.sports, height_cm=student.height_cm, weight_kg=student.weight_kg,
        latest_analysis=to_item(latest) if latest else None, diet=diet, upload_quota=_quota_out(db, student),
        subscription=SubscriptionSummary(plan_type=sub.plan_type, renews_on=sub.expires_at),
    )
