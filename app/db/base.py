"""Importing this module registers every model on Base.metadata (used by create_all in main.py)."""
from app.db.base_class import Base  # noqa: F401
from app.features.academies.models import Academy  # noqa: F401
from app.features.diet_plans.models import DietPlan  # noqa: F401
from app.features.reports.models import Report  # noqa: F401
from app.features.sessions.models import StudentClip, TrainingSession  # noqa: F401
from app.features.students.models import FaceEmbedding, MeasurementLog, Student  # noqa: F401
from app.features.subscriptions.models import Subscription  # noqa: F401
from app.features.uploads.models import SelfUpload, UploadTopUp  # noqa: F401
