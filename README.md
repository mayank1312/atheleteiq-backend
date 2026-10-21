# AthleteIQ Backend (Phase 1)

FastAPI backend: auth, academy lookup, student registration + face capture,
subscription gating, and the async session-processing pipeline.

For the complete request, database, Celery, upload, and computer-vision flow,
see [BACKEND_FLOW.md](BACKEND_FLOW.md).

## Folder structure (feature-based)

```
app/
  core/            # config, security (JWT/hashing), deps (auth + subscription gate), storage, celery app
  db/               # SQLAlchemy engine/session, Base + model registry
  features/
    auth/           # login -> JWT
    academies/      # partner academy lookup (used at registration)
    students/       # registration (incl. face capture), profile
    subscriptions/   # paywall status + prototype activation
    sessions/       # raw footage upload, per-student clips, the Celery pipeline task
    reports/        # the gated analysis output (motion/technique/improvement)
    diet_plans/     # weekly plan + monthly/yearly rollups
  main.py           # wires all routers
```

Each feature folder is self-contained: `models.py` (DB tables), `schemas.py`
(Pydantic request/response shapes), `router.py` (endpoints). Add new features
by copying this pattern — then import its models in `app/db/base.py` and its
router in `app/main.py`.

## Run locally (MacBook Air M4 prototype)

```bash
# 1. Start Postgres + Redis + MinIO
docker compose up -d

# 2. Python env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Env vars
cp .env.example .env
# edit .env if needed — defaults match docker-compose.yml

# 4. Create the MinIO bucket (one-time)
#    open http://localhost:9001 (minioadmin/minioadmin) and create a bucket
#    named "athleteiq-media", or use the mc CLI.

# 5. Run the API
# Bind to all interfaces so a phone on the same Wi-Fi can reach the API.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. In a second terminal, run the Celery worker (handles upload validation and session processing)
#    The solo pool avoids macOS prefork issues during local development.
venv/bin/celery -A app.core.celery_app worker --loglevel=info --pool=solo -Q pipeline
```

API docs (Swagger UI): http://localhost:8000/docs

For a physical phone, replace `localhost` with this Mac's LAN address in the
phone-accessible URL (for example, `http://192.168.1.34:8000/docs`). The phone
and Mac must be on the same Wi-Fi network, and macOS Firewall must allow
incoming connections for Python.

## Try the flow

1. `GET /academies` — empty at first; insert a test academy row directly in
   Postgres, or add a quick seed script (next step).
2. `POST /students/register` (multipart: form fields + `face_image` file)
3. `POST /auth/login` (OAuth2 form: `username`=email, `password`) → returns JWT
4. `POST /subscriptions/activate` (Bearer token) — prototype-only, no real payment
5. `POST /sessions/upload` (multipart footage file) — kicks off the Celery task
6. `GET /sessions/my-clips`, `GET /reports/by-clip/{id}` — subscription-gated

`GET /students/me` returns the current profile. Use
`PATCH /students/me/profile` to edit the name, email, academy, sports,
measurements, date of birth, gender, diet preferences, allergies, or password.
Academy changes must use an active partner academy, and email addresses must be
unique.

### Temporary test phase

Set `TEST_PHASE_BYPASS_FACE_MATCHING=true` and
`TEST_PHASE_UNLIMITED_UPLOADS=true` to test self-uploads without face
verification or upload quota limits. The pipeline still tracks people and
uses the largest/longest tracked person for the analysis, recording the
identity confidence as `assumed`. Academy footage continues to use face
matching. Disable both settings before production use.

### Customer-friendly analysis response

`GET /analyses/{clip_id}` keeps the existing written report fields and also
returns presentation-ready fields for the mobile app:

- `plain_language_summary`: short points with a title, simple explanation, and
  tone (`positive`, `attention`, or `info`) for insight cards.
- `data_quality`: whether the session has good, limited, or insufficient data,
  including analyzable time, full uploaded-video duration, confidence, camera
  stability, and human-readable reasons. This lets the frontend show coverage
  honestly when only part of a video contains reliable tracked movement.
- `chart_data`: five chart-safe series (movement speed, knee range, body
  control, detected actions, and left/right difference). Each series contains
  a title, unit, availability flag, optional note, and `{label, value}` points.

The frontend should use `available` and `note` instead of displaying missing
values as zero. When `data_quality.status` is `insufficient`, show the
recording guidance and avoid presenting performance conclusions as facts.

### Identity through turned-away footage

Face recognition is used as the identity anchor, not as a requirement on every
frame. The pose tracker keeps a person track after they turn away, and the
pipeline stores a coarse torso/leg appearance descriptor (HSV colour
distribution plus colour moments). If ByteTrack splits that person into a
nearby, non-overlapping fragment, the fragment can inherit the face-confirmed
identity when its appearance is sufficiently similar and the gap is no more
than about three seconds. These links are labelled `appearance_linked`;
colour alone is never allowed to identify a person across a long gap, and
ambiguous matches remain unlabelled rather than being silently assigned.

## What's stubbed vs real

- **Real now:** auth, subscription gating, DB schema, file upload to
  S3-compatible storage, async task wiring.
- **Stubbed (Section 4 of the project spec, not yet built):** face embedding
  generation (`app/features/students/face_service.py`), and the whole
  detect→track→match→crop→pose→LLM chain (`app/features/sessions/tasks.py`).
  Swap those function bodies for the real models — the API/DB contracts
  around them are already in place so nothing else needs to change.
