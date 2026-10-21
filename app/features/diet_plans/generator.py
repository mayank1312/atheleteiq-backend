"""Weekly diet plan: deterministic targets (nutrition.py) + LLM-written meals, with an offline template fallback."""
import json
import logging
from datetime import date

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.timeutils import today_local, week_start
from app.features.diet_plans.models import DietPlan
from app.features.diet_plans.nutrition import Targets, compute_targets
from app.features.reports.models import Report
from app.features.sessions.models import StudentClip
from app.features.students.models import MeasurementLog, Student

log = logging.getLogger("athleteiq.diet")

DISCLAIMER = "General nutrition guidance for active young people, not medical advice. For medical conditions or concerns, consult a doctor or registered dietitian."


class MealItem(BaseModel):
    name: str
    portion: str = ""
    notes: str = ""


class PlanContent(BaseModel):
    summary: str
    meals: dict[str, list[MealItem]]
    focus_notes: list[str] = Field(default_factory=list)
    changes_from_last_week: str = ""
    training_day_tips: str = ""


SYSTEM_PROMPT = """You are a sports nutrition assistant writing a one-week eating guide for a young athlete.

The calorie and macro targets are FIXED and given to you. Do not change them and do not propose weight loss, calorie cutting, fasting or 'cutting' phases.

Rules:
- Strictly respect diet_preference (veg = no meat/fish/egg; egg = vegetarian plus eggs; non_veg = anything; vegan = no animal products) and NEVER include anything matching the allergies field.
- If is_minor is true: focus on regular balanced meals, growth, energy for training, hydration. Do NOT recommend supplements, protein powders, energy drinks, or restrictive eating.
- Use everyday {cuisine} foods that are easy to find and afford, with realistic portions (cups, pieces, grams).
- Meals must add up roughly to the target across the day (snacks included). Include pre- and post-training snack ideas.
- 'changes_from_last_week' must explain what changed versus the previous plan and why (weight trend, training focus), or say it is the first plan.
- Keep advice specific and practical. No medical claims.

Return ONLY JSON, no markdown:
{"summary": "2 sentences", "meals": {"breakfast": [{"name":"","portion":"","notes":""}], "lunch": [...], "dinner": [...], "snacks": [...]},
 "focus_notes": ["3 to 4 short, specific tips for this week"], "changes_from_last_week": "", "training_day_tips": ""}"""


def _template(t: Targets, pref: str | None) -> PlanContent:
    pref = pref or "veg"
    protein = {"veg": "paneer or dal", "vegan": "tofu or chickpeas", "egg": "eggs or paneer", "non_veg": "chicken, fish or eggs"}[pref]
    return PlanContent(
        summary=f"A balanced week around {t.calories} kcal with about {t.protein_g} g protein. (Automatic template - AI-written plan was unavailable.)",
        meals={
            "breakfast": [MealItem(name="Oats or poha with fruit", portion="1 large bowl"), MealItem(name=f"{protein.split(' or ')[0].title()} or nuts", portion="1 serving")],
            "lunch": [MealItem(name="Roti/rice, dal or curry, vegetables, curd", portion="balanced plate"), MealItem(name=protein.title(), portion="1 palm-sized serving")],
            "dinner": [MealItem(name="Roti/rice with vegetables and a protein", portion="balanced plate")],
            "snacks": [MealItem(name="Banana + handful of nuts", notes="about 45 min before training"), MealItem(name="Milk or lassi with fruit", notes="within an hour after training")],
        },
        focus_notes=["Eat at regular times; don't skip breakfast.", f"Aim for about {t.hydration_l} L of water on training days.", "Add a fruit and a vegetable to every main meal."],
        changes_from_last_week="", training_day_tips="Eat a carbohydrate-rich snack before training and a protein + carb snack afterwards.",
    )


def _context(db: Session, s: Student, t: Targets, prev: DietPlan | None) -> dict:
    logs = db.query(MeasurementLog).filter(MeasurementLog.student_id == s.id).order_by(MeasurementLog.recorded_at.desc()).limit(4).all()
    reps = (
        db.query(Report).join(StudentClip, Report.student_clip_id == StudentClip.id)
        .filter(StudentClip.student_id == s.id).order_by(Report.generated_at.desc()).limit(2).all()
    )
    focus = []
    for r in reps:
        items = (r.structured_json or {}).get("scope_of_improvement") or []
        focus += [i.get("area") for i in items[:2] if i.get("area")]
    return {
        "athlete": {"age": t.age, "is_minor": t.is_minor, "gender": s.gender, "height_cm": s.height_cm, "weight_kg": s.weight_kg, "sports": s.sports},
        "diet_preference": s.diet_preference, "allergies": s.allergies or "none",
        "fixed_targets": {"calories": t.calories, "protein_g": t.protein_g, "carbs_g": t.carbs_g, "fat_g": t.fat_g, "hydration_l": t.hydration_l},
        "recent_weights_kg_newest_first": [round(m.weight_kg, 1) for m in logs],
        "recent_training_focus_areas": focus,
        "previous_plan_summary": (prev.plan_json.get("summary") if prev else None),
        "previous_plan_calories": (prev.plan_json.get("calorie_target") if prev else None),
    }


def _llm_plan(ctx: dict) -> PlanContent | None:
    if not settings.ANTHROPIC_API_KEY:
        return None
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=settings.DIET_MODEL, max_tokens=3500, system=SYSTEM_PROMPT.format(cuisine=settings.DIET_CUISINE),
                messages=[{"role": "user", "content": json.dumps(ctx, ensure_ascii=False)}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return PlanContent(**json.loads(text[text.find("{") : text.rfind("}") + 1]))
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            log.warning("diet plan parse failed (attempt %s): %s", attempt + 1, exc)
        except Exception as exc:
            log.error("diet LLM call failed: %s", exc)
            break
    return None


def generate_plan(db: Session, student: Student, for_week: date | None = None) -> DietPlan:
    """Idempotent: returns the existing plan if this week's is already there. Raises ProfileIncomplete / AgeOutOfRange."""
    wk = for_week or week_start()
    existing = db.query(DietPlan).filter(DietPlan.student_id == student.id, DietPlan.week_start_date == wk).first()
    if existing:
        return existing
    t = compute_targets(student.date_of_birth, student.gender, student.height_cm, student.weight_kg, today_local())
    prev = db.query(DietPlan).filter(DietPlan.student_id == student.id, DietPlan.week_start_date < wk).order_by(DietPlan.week_start_date.desc()).first()
    content = _llm_plan(_context(db, student, t, prev)) or _template(t, student.diet_preference)
    if not content.changes_from_last_week:
        content.changes_from_last_week = "This is your first plan." if prev is None else "Targets refreshed for this week."
    plan = content.model_dump()
    # numbers always come from the deterministic calculation, whatever the model wrote
    plan.update({
        "calorie_target": t.calories, "macros": {"protein_g": t.protein_g, "carbs_g": t.carbs_g, "fat_g": t.fat_g},
        "hydration_l": t.hydration_l, "target_method": t.method, "disclaimer": DISCLAIMER,
    })
    row = DietPlan(student_id=student.id, week_start_date=wk, height_cm_snapshot=student.height_cm, weight_kg_snapshot=student.weight_kg, plan_json=plan)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
