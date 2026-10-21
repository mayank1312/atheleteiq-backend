import logging

from fastapi import FastAPI

import app.db.base  # noqa: F401  registers every model on Base.metadata
from app.db.base_class import Base
from app.db.migrations import apply_light_migrations
from app.db.session import engine
from app.features.academies.router import router as academies_router
from app.features.admin.router import router as admin_router
from app.features.auth.router import router as auth_router
from app.features.diet_plans.router import router as diet_plans_router
from app.features.home.router import router as home_router
from app.features.reports.router import router as analyses_router
from app.features.sessions.router import router as sessions_router
from app.features.students.router import router as students_router
from app.features.subscriptions.router import router as subscriptions_router
from app.features.uploads.router import router as uploads_router

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="AthleteIQ API", version="0.2.0")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)  # creates NEW tables
    apply_light_migrations(engine)         # alters existing ones (Postgres only)


for r in (auth_router, academies_router, students_router, subscriptions_router, home_router, analyses_router,
          uploads_router, diet_plans_router, sessions_router, admin_router):
    app.include_router(r)


@app.get("/health")
def health_check():
    return {"status": "ok"}
