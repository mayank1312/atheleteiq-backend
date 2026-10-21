import logging

from app.core.celery_app import celery_app

log = logging.getLogger("athleteiq.diet")


@celery_app.task(name="diet_plans.generate_diet_plan")
def generate_diet_plan(student_id: int):
    from app.db.session import SessionLocal
    from app.features.diet_plans.generator import generate_plan
    from app.features.diet_plans.nutrition import AgeOutOfRange, ProfileIncomplete
    from app.features.students.models import Student

    db = SessionLocal()
    try:
        s = db.get(Student, student_id)
        if s:
            try:
                generate_plan(db, s)
            except (ProfileIncomplete, AgeOutOfRange) as exc:
                log.info("no diet plan for student %s: %s", student_id, exc)
    finally:
        db.close()


@celery_app.task(name="diet_plans.generate_due_diet_plans")
def generate_due_diet_plans():
    """Nightly: make sure every subscribed student with a complete profile has this week's plan waiting."""
    from app.core.deps import latest_active_subscription
    from app.db.session import SessionLocal
    from app.features.diet_plans.generator import generate_plan
    from app.features.diet_plans.nutrition import AgeOutOfRange, ProfileIncomplete
    from app.features.students.models import Student

    db = SessionLocal()
    try:
        for s in db.query(Student).all():
            if latest_active_subscription(db, s.id) is None:
                continue
            try:
                generate_plan(db, s)
            except (ProfileIncomplete, AgeOutOfRange):
                continue
            except Exception:
                db.rollback()
                log.exception("diet plan failed for student %s", s.id)
    finally:
        db.close()
