# AthleteIQ Backend Flow

This document explains how a request moves through the AthleteIQ backend, where
data is stored, and which work is synchronous versus asynchronous. It is
intended for anyone joining the backend project or integrating a mobile app,
academy camera system, or operations tooling.

## 1. What the backend does

AthleteIQ is a FastAPI application with four main responsibilities:

1. Create student accounts and store a face embedding for identity matching.
2. Authenticate students with JWT access tokens and protect paid content with
   an active-subscription dependency.
3. Accept academy footage or student self-uploaded videos and turn them into
   per-student clips, movement metrics, and written reports.
4. Generate weekly nutrition plans and monthly/yearly nutrition rollups.

The API process handles validation, database writes, and storage uploads. Redis
and Celery handle video validation, computer-vision analysis, report creation,
and diet-plan generation outside the request that initiated them.

## 2. Runtime architecture

```text
Mobile app ───────────────┐
                          │ HTTPS/HTTP
Academy camera/tool ──────┼──> FastAPI (app.main:app)
                          │       │
Admin tooling ────────────┘       ├── PostgreSQL (SQLAlchemy)
                                  ├── Supabase Storage (videos/clips/images)
                                  └── Redis broker/backend
                                          │
                                          └── Celery workers
                                              ├── upload validation
                                              ├── video analysis pipeline
                                              └── diet-plan generation
```

### Application startup

`app/main.py`:

1. Imports `app.db.base`, which imports every SQLAlchemy model and registers it
   on the shared `Base.metadata`.
2. Creates a `FastAPI` application.
3. On startup, runs `Base.metadata.create_all(bind=engine)` and then the
   lightweight Postgres migrations in `app/db/migrations.py`.
4. Includes all feature routers.
5. Exposes `GET /health`, returning `{"status": "ok"}`.

The project currently uses `create_all` plus lightweight migrations rather than
an Alembic migration command during application startup. Production deployment
should run schema migrations as an explicit release step.

## 3. Authentication and access control

### Student JWT flow

1. A student registers through `POST /students/register`.
2. The password is stored as a bcrypt hash.
3. The face image is converted to an embedding and discarded; the original
   selfie is not stored.
4. The student calls `POST /auth/login` using OAuth2 form fields:
   `username=<email>` and `password=<password>`.
5. The API returns a bearer JWT whose subject is the student ID.
6. Student requests send `Authorization: Bearer <token>`.
7. `get_current_student` decodes the token and loads the student from the
   database.

Login does not require a subscription. This lets an expired subscriber log in
and see the subscription screen instead of receiving an authentication error.

### Subscription gate

Routes that expose analysis, uploads, or diet content use
`require_active_subscription`. It finds the student's latest non-expired
subscription and returns HTTP 402 when none exists.

### Academy camera authentication

Academy footage uses `X-Academy-Key`, not a student JWT:

1. An admin creates an academy and receives its API key.
2. The database stores only a SHA-256 hash of that key.
3. `get_academy_from_key` hashes the request header and finds an active partner
   academy.
4. `POST /sessions/upload` accepts footage only after this check.

### Admin authentication

Admin routes use `X-Admin-Key`. They are disabled with HTTP 503 when
`ADMIN_API_KEY` is empty and return HTTP 403 for an invalid key.

## 4. End-to-end student onboarding

```text
GET /academies or GET /academies/lookup
             │
             ▼
POST /students/register (multipart form + face_image)
             │
             ├── validate academy, email, measurements, sport, image
             ├── create face embedding
             ├── insert Student
             ├── insert FaceEmbedding
             └── insert initial MeasurementLog
             │
             ▼
POST /auth/login
             │
             ▼
POST /subscriptions/activate (prototype only)
```

Registration requires an active partner academy. Multiple sports are accepted
as a comma-separated value; the first is the primary sport. Valid diet
preferences are `veg`, `non_veg`, `egg`, and `vegan`.

The face backend is controlled by `FACE_BACKEND`. The real backend uses
InsightFace/ArcFace (`buffalo_l`) and stores a normalized vector. `stub` is
only for wiring tests and must not be used for production identity matching.

## 5. Academy footage flow

Academy footage is shared footage that may contain multiple enrolled students.

```text
Academy system
    │
    ├── POST /sessions/upload
    │       ├── authenticate X-Academy-Key
    │       ├── stream video to a temporary file
    │       ├── enforce MAX_ACADEMY_UPLOAD_MB
    │       ├── upload raw video to storage/raw_sessions/{academy_id}/
    │       ├── insert TrainingSession(status="uploaded")
    │       └── enqueue sessions.process_training_session
    │
    └── Student later polls GET /analyses and GET /analyses/{clip_id}
```

The HTTP response returns the newly created `TrainingSession` immediately.
The analysis itself runs in a Celery worker.

### Academy worker flow

`sessions.process_training_session` calls
`app.features.pipeline.service.process_session`:

1. Load the `TrainingSession`.
2. Set its status to `processing`.
3. Download raw footage to a temporary worker directory.
4. Load the academy roster and usable face embeddings.
5. Decode, detect, track, and pose-estimate people.
6. Match tracks to the roster.
7. Merge fragments belonging to the same student.
8. Compute movement metrics and write a student-specific clip.
9. Upload the clip to `clips/{student_id}/`.
10. Generate and store a `Report`.
11. Mark each `StudentClip` as `ready`.
12. Mark the `TrainingSession` as `done`.

If anything fails, the session is marked `failed`, the transaction is rolled
back, and the exception is logged for the worker.

## 6. Student self-upload flow

Self-upload is deliberately a two-step process so rejected videos do not spend
an upload credit.

```text
POST /uploads/validate
    │
    ├── require active subscription
    ├── verify sport is registered
    ├── check remaining quota
    ├── stream and store video
    ├── insert SelfUpload(status="validating")
    └── enqueue uploads.validate_self_upload
             │
             ▼
GET /uploads/{upload_id}  (poll checklist)
             │
       validating
        /       \
   rejected   validated
                  │
                  ▼
       POST /uploads/{upload_id}/submit
                  │
                  ├── consume quota
                  ├── create StudentClip(status="processing")
                  ├── set upload status to submitted
                  └── enqueue uploads.process_self_upload
                           │
                           ▼
                    done or failed
```

### Validation checklist

`uploads.validate_self_upload` runs these checks and saves each result in
`SelfUpload.checks_json`:

1. Video can be read and is between `MIN_VIDEO_SECONDS` and
   `MAX_VIDEO_SECONDS`.
2. Resolution and frame rate meet the configured minimum.
3. Sampled frames match the declared sport using the configured vision model.
4. A player is visible in enough sampled frames.

A thumbnail is generated when possible. Any failed check makes the upload
`rejected`; no quota is consumed. A validation failure is safe to display to
the student through `rejection_reason` and the checklist messages.

### Self-upload analysis

`uploads.process_self_upload`:

1. Downloads the stored video to temporary worker storage.
2. Loads only the uploader's current face embedding.
3. Runs the same video analysis pipeline in self-upload mode.
4. Requires the result to identify the uploader, unless the test bypass is
   enabled.
5. Stores the cropped clip and report.
6. Marks the upload `done` and the clip `ready`.

If analysis fails, the upload and clip become `failed`, a user-safe reason is
stored, and the quota is refunded. Validation rejection and analysis failure
therefore have different states and different credit behavior.

## 7. Computer-vision pipeline

The central entry point is `analyze_video` in
`app/features/pipeline/orchestrator.py`.

```text
Video
  │
  ▼
probe_video
  │
  ▼
YOLO pose detection + ByteTrack tracking
  │  (17 keypoints, boxes, persistent track IDs)
  ▼
discard tracks shorter than MIN_TRACK_SECONDS
  │
  ├── face crops from the best head/keypoint candidates
  ├── coarse body/torso appearance descriptors
  └── track timeline
          │
          ▼
track identity matching
  │
  ├── face embedding similarity
  ├── absolute threshold
  ├── runner-up margin
  ├── short-gap appearance linking
  └── overlapping-track conflict resolution
          │
          ▼
merge fragments for each student
          │
          ├── build student clip segments
          ├── compute movement/angle/symmetry/event metrics
          └── attach data-quality metadata
          │
          ▼
LLM report generation + structured report persistence
```

Identity labels are:

- `face_matched`: a track passed face similarity and margin checks.
- `appearance_linked`: a nearby, non-overlapping tracker fragment inherited a
  face-confirmed identity using a coarse appearance descriptor.
- `assumed`: test-only self-upload bypass selected the dominant track.

Appearance is not allowed to identify a person across a long gap. Ambiguous
matches remain unlabelled.

The pipeline records both the tracked/analyzable window and the full uploaded
video duration. The reports API uses these values to produce a `data_quality`
object rather than presenting missing measurements as zero.

## 8. Reports and analysis responses

`GET /analyses` returns the student's newest clips, mixing academy and
self-upload results. Each item includes its source, sport, status, headline,
creation time, ready time, and expected ready time.

`GET /analyses/{clip_id}`:

- verifies the clip belongs to the authenticated student;
- returns processing, ready, or failed state;
- returns failure reason when analysis failed;
- returns the stored written report when ready;
- adds `plain_language_summary`, `data_quality`, and `chart_data`.

The frontend should use `chart_data[*].available` and `note`. It should not
render unavailable values as zero. When `data_quality.status` is
`insufficient`, the UI should show recording guidance and avoid presenting
performance conclusions as facts.

## 9. Diet-plan flow

```text
GET /diet-plans/current
       │
       ├── existing current-week plan -> 200
       ├── incomplete profile -> 409
       ├── age outside supported range -> 422
       └── no plan:
             ├── claim Redis key to prevent duplicate jobs
             ├── enqueue diet_plans.generate_diet_plan
             └── return 202 (poll again)
```

The worker:

1. Computes deterministic calorie, macro, and hydration targets.
2. Loads recent measurements and recent report improvement areas.
3. Uses the configured diet LLM to write meals when available.
4. Falls back to a safe deterministic template if the LLM is unavailable.
5. Overwrites numeric targets with the deterministic values.
6. Stores one idempotent `DietPlan` for the week.

The nightly Celery Beat task `diet_plans.generate_due_diet_plans` prepares plans
for subscribed students with complete profiles. `GET /diet-plans/monthly` and
`GET /diet-plans/yearly` aggregate saved plans; they do not generate new ones.

## 10. API surface

All routes are registered in `app/main.py`.

| Area | Method and path | Auth | Purpose |
|---|---|---|---|
| Health | `GET /health` | None | Liveness check |
| Academies | `GET /academies` | None | List active academies |
| Academies | `GET /academies/lookup` | None | Look up a partner academy |
| Admin | `POST /admin/academies` | `X-Admin-Key` | Create academy and camera key |
| Admin | `POST /admin/academies/{academy_id}/rotate-key` | `X-Admin-Key` | Rotate camera key |
| Auth | `POST /auth/login` | None | Issue JWT |
| Auth | `POST /auth/reset-password` | None | Replace password |
| Students | `POST /students/register` | None | Create student and face embedding |
| Students | `GET /students/me` | JWT | Read profile |
| Students | `PATCH /students/me/profile` | JWT | Update profile |
| Students | `PATCH /students/me/measurements` | JWT | Update and log measurements |
| Students | `GET /students/me/measurements/history` | JWT | Read measurement history |
| Subscriptions | `GET /subscriptions/me` | JWT | Read subscription state |
| Subscriptions | `POST /subscriptions/activate` | JWT | Prototype direct activation |
| Academy sessions | `POST /sessions/upload` | `X-Academy-Key` | Upload academy footage |
| Self uploads | `GET /uploads/quota` | JWT + subscription | Read upload credits |
| Self uploads | `GET /uploads/packs` | JWT + subscription | List top-up packs |
| Self uploads | `POST /uploads/topup` | JWT + subscription | Prototype grant top-up |
| Self uploads | `POST /uploads/validate` | JWT + subscription | Start video checks |
| Self uploads | `GET /uploads/{upload_id}` | JWT + subscription | Poll checks/status |
| Self uploads | `POST /uploads/{upload_id}/submit` | JWT + subscription | Spend credit and analyze |
| Self uploads | `DELETE /uploads/{upload_id}` | JWT + subscription | Delete unsubmitted upload |
| Analyses | `GET /analyses` | JWT + subscription | List own analyses |
| Analyses | `GET /analyses/{clip_id}` | JWT + subscription | Read one report |
| Diet | `GET /diet-plans/current` | JWT + subscription | Get or queue weekly plan |
| Diet | `GET /diet-plans/monthly` | JWT + subscription | Monthly rollup |
| Diet | `GET /diet-plans/yearly` | JWT + subscription | Yearly rollup |
| Home | `GET /home` | JWT + subscription | Home dashboard payload |

Resource ownership is enforced for student uploads and analysis clips. A
missing clip and another student's clip intentionally return the same 404
response.

## 11. Database entities

The model registry in `app/db/base.py` currently includes:

| Table/model | Role |
|---|---|
| `academies` / `Academy` | Partner academy and hashed camera API key |
| `students` / `Student` | Account, academy, sports, body, and diet profile |
| `face_embeddings` / `FaceEmbedding` | Current biometric vector and model version |
| `measurement_logs` / `MeasurementLog` | Height/weight history |
| `subscriptions` / `Subscription` | Active and expired access periods |
| `training_sessions` / `TrainingSession` | Academy raw footage job |
| `student_clips` / `StudentClip` | One student's analysis unit |
| `reports` / `Report` | Metrics and written analysis for one clip |
| `self_uploads` / `SelfUpload` | Student upload validation and processing state |
| `upload_topups` / `UploadTopUp` | Purchased extra upload credits |
| `diet_plans` / `DietPlan` | Weekly generated plan and snapshots |

Video, generated clips, and thumbnails are stored in Supabase-compatible
object storage. Database path fields point to those objects; the API does not
return internal clip paths directly.

## 12. Celery queues and task states

The Celery app uses Redis for both broker and result backend:

| Task | Queue | Trigger |
|---|---|---|
| `sessions.process_training_session` | `pipeline` | Academy session upload |
| `uploads.validate_self_upload` | `pipeline` | Self-upload validation |
| `uploads.process_self_upload` | `pipeline` | Student submits validated upload |
| `diet_plans.generate_diet_plan` | `celery` | Current-plan request |
| `diet_plans.generate_due_diet_plans` | Beat/`celery` | Nightly schedule |

For local macOS development, run the pipeline worker with the `solo` pool:

```bash
venv/bin/celery -A app.core.celery_app worker \
  --loglevel=info --pool=solo -Q pipeline
```

The API process is started with:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 13. Configuration that changes behavior

Important settings are defined in `app/core/config.py` and loaded from `.env`:

- `DATABASE_URL`, `REDIS_URL`: database and Celery infrastructure.
- `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`: auth.
- `ADMIN_API_KEY`: enables admin routes.
- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_BUCKET`: object storage.
- `OPENAI_API_KEY`, `REPORT_MODEL`, `DIET_MODEL`, `SPORT_CHECK_MODEL`: LLM work.
- `POSE_MODEL`, `TRACKER`, `DEVICE`, `FRAME_STRIDE`, `DETECT_IMGSZ`: CV runtime.
- `FACE_BACKEND`, `FACE_MATCH_THRESHOLD`, `FACE_MATCH_MARGIN`: identity.
- `MAX_UPLOAD_MB`, `MAX_ACADEMY_UPLOAD_MB`, video duration and resolution limits.
- `SELF_UPLOAD_INCLUDED_PER_CYCLE`: included upload quota.
- `TEST_PHASE_BYPASS_FACE_MATCHING`, `TEST_PHASE_UNLIMITED_UPLOADS`: test-only
  behavior; disable both before production.
- `ANALYSIS_SLA_HOURS`, `APP_TIMEZONE`: expected readiness and date handling.

Secrets must be supplied through environment configuration and must not be
committed to the repository.

## 14. Local development and verification

1. Start Postgres, Redis, and Supabase-compatible storage as configured by the
   local environment.
2. Create a virtual environment and install `requirements.txt`.
3. Configure `.env`.
4. Start the API and the `pipeline` Celery worker.
5. Open `/docs` for the generated OpenAPI/Swagger UI.
6. Run the test suite with `pytest`.

Useful test areas include API flow, identity matching, tracking, metrics,
clips/reports, and nutrition. Heavy model-backed checks may require the model
weights and configured external services.

## 15. Important current limitations

- `/subscriptions/activate` directly grants access and is prototype-only.
  Production should activate subscriptions from a verified payment webhook.
- `/uploads/topup` also grants credits without payment verification.
- Startup schema management uses `create_all` and lightweight migrations.
- LLM report and diet generation depend on configured providers; diet plans have
  an offline template fallback.
- The test bypass settings intentionally weaken identity and quota rules and
  must not be enabled in production.
