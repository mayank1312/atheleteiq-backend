from celery import Celery
from celery.schedules import crontab

import app.db.base  # noqa: F401 registers every SQLAlchemy model for worker tasks
from app.core.config import settings

celery_app = Celery("athleteiq", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.autodiscover_tasks(
    ["app.features.sessions", "app.features.uploads", "app.features.diet_plans"], related_name="tasks"
)

celery_app.conf.update(
    timezone=settings.APP_TIMEZONE,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=6 * 3600,
    # Keep upload tasks on the local development worker queue.
    task_routes={
        "sessions.process_training_session": {"queue": "pipeline"},
        "uploads.process_self_upload": {"queue": "pipeline"},
        "uploads.validate_self_upload": {"queue": "pipeline"},
        "diet_plans.*": {"queue": "celery"},
    },
    beat_schedule={
        # Weekly plans are created ahead of time so the student never waits on an LLM call.
        "generate-due-diet-plans": {"task": "diet_plans.generate_due_diet_plans", "schedule": crontab(hour=3, minute=0)},
    },
)
