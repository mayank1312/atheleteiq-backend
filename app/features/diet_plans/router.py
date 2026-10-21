import logging
from collections import defaultdict
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.timeutils import today_local, week_start
from app.db.session import get_db
from app.core.deps import require_active_subscription
from app.features.diet_plans.models import DietPlan
from app.features.diet_plans.nutrition import AgeOutOfRange, ProfileIncomplete, compute_targets
from app.features.diet_plans.schemas import DietPending, DietPlanOut, DietRollup, QuarterSummary, TrendPoint
from app.features.diet_plans.tasks import generate_diet_plan
from app.features.students.models import Student

router = APIRouter(prefix="/diet-plans", tags=["diet_plans"])
log = logging.getLogger("athleteiq.diet")


def _claim(student_id: int, wk) -> bool:
    """Stops a polling app from queueing the same generation over and over."""
    try:
        import redis

        return bool(redis.Redis.from_url(settings.REDIS_URL).set(f"diet:gen:{student_id}:{wk}", "1", nx=True, ex=600))
    except Exception:
        return True


def _profile_check(s: Student) -> None:
    try:
        compute_targets(s.date_of_birth, s.gender, s.height_cm, s.weight_kg, today_local())
    except ProfileIncomplete as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, {"code": "profile_incomplete", "missing": exc.missing, "message": "Add these details to get your diet plan."})
    except AgeOutOfRange as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, {"code": "age_out_of_range", "message": str(exc)})


@router.get("/current", response_model=DietPlanOut, responses={202: {"model": DietPending}})
def current_week_plan(student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    """
    200 -> this week's plan. 202 -> it's being prepared (poll again in a few seconds).
    409 -> profile is missing date_of_birth/gender.
    """
    wk = week_start()
    plan = db.query(DietPlan).filter(DietPlan.student_id == student.id, DietPlan.week_start_date == wk).first()
    if plan:
        return plan
    _profile_check(student)
    if _claim(student.id, wk):
        generate_diet_plan.delay(student.id)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=DietPending(status="generating", detail="Your plan for this week is being prepared.").model_dump(),
    )


def _rollup(period: str, days: int, student: Student, db: Session) -> DietRollup:
    cutoff = today_local() - timedelta(days=days)
    plans = db.query(DietPlan).filter(DietPlan.student_id == student.id, DietPlan.week_start_date >= cutoff).order_by(DietPlan.week_start_date).all()
    cal = [TrendPoint(week_start=p.week_start_date, value=p.plan_json["calorie_target"]) for p in plans if p.plan_json.get("calorie_target")]
    wt = [TrendPoint(week_start=p.week_start_date, value=p.weight_kg_snapshot) for p in plans]
    quarters = []
    if period == "yearly":
        groups = defaultdict(list)
        for p in plans:
            groups[f"{p.week_start_date.year}-Q{(p.week_start_date.month - 1) // 3 + 1}"].append(p)
        for label, ps in sorted(groups.items()):
            cs = [p.plan_json.get("calorie_target") for p in ps if p.plan_json.get("calorie_target")]
            quarters.append(QuarterSummary(label=label, plans=len(ps), avg_calories=round(sum(cs) / len(cs)) if cs else None,
                                           avg_weight_kg=round(sum(p.weight_kg_snapshot for p in ps) / len(ps), 1)))
    return DietRollup(
        period=period, plans=plans, calories_trend=cal, weight_trend=wt,
        avg_calories=round(sum(c.value for c in cal) / len(cal)) if cal else None,
        weight_change_kg=round(wt[-1].value - wt[0].value, 1) if len(wt) >= 2 else None, quarters=quarters,
    )


@router.get("/monthly", response_model=DietRollup)
def monthly(student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    return _rollup("monthly", 31, student, db)


@router.get("/yearly", response_model=DietRollup)
def yearly(student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    return _rollup("yearly", 366, student, db)
