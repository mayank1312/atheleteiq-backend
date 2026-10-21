"""
Lightweight, idempotent schema upgrades for Postgres (Supabase).

Base.metadata.create_all() creates NEW tables but never alters existing ones, so columns
added after the first release are applied here. Every statement is safe to re-run.
Replace with Alembic before real academy data goes in.
"""
import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger("athleteiq.migrations")

STATEMENTS = [
    # students: profile fields needed for diet planning
    "ALTER TABLE students ADD COLUMN IF NOT EXISTS date_of_birth DATE",
    "ALTER TABLE students ADD COLUMN IF NOT EXISTS gender VARCHAR(20)",
    "ALTER TABLE students ADD COLUMN IF NOT EXISTS diet_preference VARCHAR(20)",
    "ALTER TABLE students ADD COLUMN IF NOT EXISTS allergies TEXT",
    # academies: hashed API key used by cameras / academy tooling to push footage
    "ALTER TABLE academies ADD COLUMN IF NOT EXISTS api_key_hash VARCHAR(64)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_academies_api_key_hash ON academies (api_key_hash)",
    # clips can now come from a self-upload (no training session)
    "ALTER TABLE student_clips ALTER COLUMN session_id DROP NOT NULL",
    "ALTER TABLE student_clips ADD COLUMN IF NOT EXISTS source VARCHAR(20) NOT NULL DEFAULT 'academy'",
    "ALTER TABLE student_clips ADD COLUMN IF NOT EXISTS self_upload_id INTEGER REFERENCES self_uploads(id)",
    "ALTER TABLE student_clips ADD COLUMN IF NOT EXISTS identity_confidence VARCHAR(20)",
    "ALTER TABLE student_clips ADD COLUMN IF NOT EXISTS error TEXT",
    # richer reports
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS headline TEXT",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS key_metrics_json JSON",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS structured_json JSON",
    # keep new tables closed to Supabase's public REST API (backend connects as owner, unaffected)
    "ALTER TABLE self_uploads ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE upload_topups ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE measurement_logs ENABLE ROW LEVEL SECURITY",
]


def apply_light_migrations(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        return
    for stmt in STATEMENTS:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:  # keep booting; surface the problem in logs
            log.warning("migration skipped: %s -> %s", stmt, exc)
