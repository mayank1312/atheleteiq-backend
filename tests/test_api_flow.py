import os
from datetime import datetime, timezone
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import storage
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.features.diet_plans import tasks as diet_tasks
from app.features.pipeline.clips import ClipResult
from app.features.pipeline.metrics import compute_metrics
from app.features.pipeline.orchestrator import StudentAnalysis, VideoAnalysis
from app.features.pipeline.video_utils import VideoInfo
from app.features.sessions import tasks as session_tasks
from app.features.uploads import tasks as upload_tasks
from app.main import app
from tests.synthetic import stick_track
from tests.test_clips_and_report import make_video

BLOB: dict[str, bytes] = {}
QUEUED: list = []


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    BLOB.clear(); QUEUED.clear()
    monkeypatch.setattr(storage, "upload_file", lambda p, k, content_type="video/mp4": BLOB.__setitem__(k, open(p, "rb").read()) or k)
    monkeypatch.setattr(storage, "upload_bytes", lambda d, k, content_type="image/jpeg": BLOB.__setitem__(k, d) or k)
    monkeypatch.setattr(storage, "download_to", lambda k, p: (open(p, "wb").write(BLOB[k]), __import__("pathlib").Path(p))[1])
    monkeypatch.setattr(storage, "delete_object", lambda k: BLOB.pop(k, None))
    monkeypatch.setattr(storage, "get_signed_url", lambda k, e=3600: f"https://signed.example/{k}")
    for name, task in (("validate", upload_tasks.validate_self_upload), ("process_up", upload_tasks.process_self_upload),
                       ("process_sess", session_tasks.process_training_session), ("diet", diet_tasks.generate_diet_plan)):
        monkeypatch.setattr(task, "delay", lambda *a, _n=name, **k: QUEUED.append((_n, a)))
    monkeypatch.setattr("app.features.diet_plans.router._claim", lambda *a: True)
    monkeypatch.setattr("app.features.pipeline.precheck.find_player_frames", lambda frames: (len(frames), 0))


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def face_jpg():
    img = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def video_bytes(tmp_path, seconds=6, w=854, h=480):
    p = tmp_path / f"v{seconds}{w}.mp4"; make_video(p, w=w, h=h, seconds=seconds)
    return p.read_bytes()


def make_academy(client, name="Test Academy"):
    r = client.post("/admin/academies", json={"name": name}, headers={"X-Admin-Key": "test-admin-key"})
    assert r.status_code == 201, r.text
    return r.json()


def make_student(client, email="s@example.com", sport="Soccer,Badminton", academy="Test Academy", **extra):
    data = {"full_name": "Stu Dent", "email": email, "password": "password123", "academy_name": academy, "sport": sport, "height_cm": "170", "weight_kg": "62", **extra}
    r = client.post("/students/register", data=data, files={"face_image": ("f.jpg", face_jpg(), "image/jpeg")})
    assert r.status_code == 201, r.text
    tok = client.post("/auth/login", data={"username": email, "password": "password123"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}, r.json()


def test_reset_password_updates_login_credentials(client):
    make_academy(client)
    make_student(client, email="reset@example.com")

    reset = client.post(
        "/auth/reset-password",
        json={"email": "RESET@example.com", "new_password": "newpassword123"},
    )
    assert reset.status_code == 200, reset.text

    old_login = client.post("/auth/login", data={"username": "reset@example.com", "password": "password123"})
    assert old_login.status_code == 401

    new_login = client.post("/auth/login", data={"username": "reset@example.com", "password": "newpassword123"})
    assert new_login.status_code == 200, new_login.text


def subscribe(client, h):
    assert client.post("/subscriptions/activate", headers=h, json={}).status_code == 200


def upload(client, h, tmp_path, sport="Soccer", seconds=6, w=854, h_px=480):
    return client.post("/uploads/validate", headers=h, data={"sport": sport},
                       files={"video": ("clip.mp4", video_bytes(tmp_path, seconds, w, h_px), "video/mp4")})


def run_validation(uid):
    from app.features.uploads.validation import run_validation as rv
    rv(uid)


# ------------------------------------------------------------------ access / security
def test_admin_and_academy_key_security(client, tmp_path):
    assert client.post("/admin/academies", json={"name": "X"}).status_code == 403
    a = make_academy(client)
    assert a["api_key"].startswith("aiq_")
    files = {"footage": ("s.mp4", video_bytes(tmp_path), "video/mp4")}
    form = {"sport": "Soccer", "recorded_at": "2026-09-28T10:00:00Z"}
    assert client.post("/sessions/upload", data=form, files=files).status_code == 401
    assert client.post("/sessions/upload", data=form, files=files, headers={"X-Academy-Key": "wrong"}).status_code == 401
    r = client.post("/sessions/upload", data=form, files=files, headers={"X-Academy-Key": a["api_key"]})
    assert r.status_code == 201 and r.json()["academy_id"] == a["id"]
    assert QUEUED[-1][0] == "process_sess"
    old = a["api_key"]
    new = client.post(f"/admin/academies/{a['id']}/rotate-key", headers={"X-Admin-Key": "test-admin-key"}).json()["api_key"]
    assert client.post("/sessions/upload", data=form, files={"footage": ("s.mp4", video_bytes(tmp_path), "video/mp4")}, headers={"X-Academy-Key": old}).status_code == 401
    assert new != old


def test_paywall_returns_machine_readable_code(client):
    make_academy(client); h, _ = make_student(client)
    r = client.get("/analyses", headers=h)
    assert r.status_code == 402 and r.json()["detail"]["code"] == "subscription_required"
    subscribe(client, h)
    assert client.get("/analyses", headers=h).json() == []


def test_students_cannot_reach_video_of_any_kind(client):
    paths = {r.path for r in app.routes}
    assert not any("video" in p or "my-clips" in p for p in paths)


# ------------------------------------------------------------------ self-upload flow
def test_full_self_upload_happy_path_and_quota(client, tmp_path):
    make_academy(client); h, _ = make_student(client); subscribe(client, h)
    q = client.get("/uploads/quota", headers=h).json()
    assert q["remaining"] == 2 and q["included_total"] == 2 and q["resets_on"]

    assert upload(client, h, tmp_path, sport="Tennis").status_code == 400          # not one of their sports
    r = upload(client, h, tmp_path)
    assert r.status_code == 201 and r.json()["status"] == "validating" and not r.json()["can_submit"]
    uid = r.json()["id"]
    assert client.post(f"/uploads/{uid}/submit", headers=h).status_code == 409     # can't submit before validation passes

    run_validation(uid)
    v = client.get(f"/uploads/{uid}", headers=h).json()
    assert v["status"] == "validated" and v["can_submit"]
    st = {c["key"]: c["status"] for c in v["checks"]}
    assert st == {"length": "passed", "quality": "passed", "sport_match": "skipped", "person_visible": "passed"}
    assert client.get("/uploads/quota", headers=h).json()["remaining"] == 2        # validation cost nothing

    s = client.post(f"/uploads/{uid}/submit", headers=h)
    assert s.status_code == 200 and s.json()["status"] == "submitted"
    assert client.get("/uploads/quota", headers=h).json()["remaining"] == 1
    assert client.post(f"/uploads/{uid}/submit", headers=h).status_code == 409     # no double-spend
    listing = client.get("/analyses", headers=h).json()
    assert len(listing) == 1 and listing[0]["status"] == "processing" and listing[0]["source"] == "self_upload"
    assert listing[0]["expected_ready_at"]


def test_rejected_video_costs_nothing(client, tmp_path):
    make_academy(client); h, _ = make_student(client); subscribe(client, h)
    uid = upload(client, h, tmp_path, seconds=2).json()["id"]                        # too short
    run_validation(uid)
    v = client.get(f"/uploads/{uid}", headers=h).json()
    assert v["status"] == "rejected" and not v["can_submit"] and "at least" in v["rejection_reason"]
    assert next(c for c in v["checks"] if c["key"] == "length")["status"] == "failed"
    assert client.post(f"/uploads/{uid}/submit", headers=h).status_code == 409
    assert client.get("/uploads/quota", headers=h).json()["remaining"] == 2

    lowres = upload(client, h, tmp_path, w=320, h_px=240).json()["id"]
    run_validation(lowres)
    assert next(c for c in client.get(f"/uploads/{lowres}", headers=h).json()["checks"] if c["key"] == "quality")["status"] == "failed"


def test_no_player_visible_is_rejected(client, tmp_path, monkeypatch):
    monkeypatch.setattr("app.features.pipeline.precheck.find_player_frames", lambda frames: (0, None))
    make_academy(client); h, _ = make_student(client); subscribe(client, h)
    uid = upload(client, h, tmp_path).json()["id"]; run_validation(uid)
    v = client.get(f"/uploads/{uid}", headers=h).json()
    assert v["status"] == "rejected" and "couldn't see a player" in v["rejection_reason"]


def test_quota_exhaustion_topup_and_refund_on_failed_analysis(client, tmp_path, monkeypatch):
    make_academy(client); h, _ = make_student(client); subscribe(client, h)
    ids = []
    for _ in range(2):
        uid = upload(client, h, tmp_path).json()["id"]; run_validation(uid)
        assert client.post(f"/uploads/{uid}/submit", headers=h).status_code == 200
        ids.append(uid)
    assert client.get("/uploads/quota", headers=h).json()["remaining"] == 0
    r = upload(client, h, tmp_path)
    assert r.status_code == 402 and r.json()["detail"]["code"] == "quota_exhausted"

    assert client.get("/uploads/packs", headers=h).json()[0]["uploads"] == 2
    assert client.post("/uploads/topup", headers=h, json={"pack_id": "nope"}).status_code == 400
    q = client.post("/uploads/topup", headers=h, json={"pack_id": "pack_2"}).json()
    assert q["topup_remaining"] == 2 and q["remaining"] == 2

    uid = upload(client, h, tmp_path).json()["id"]; run_validation(uid)
    client.post(f"/uploads/{uid}/submit", headers=h)
    q = client.get("/uploads/quota", headers=h).json()
    assert q["topup_remaining"] == 1 and q["included_remaining"] == 0

    # analysis fails -> credit comes back and the app can show 'upload rejected'
    monkeypatch.setattr("app.features.pipeline.service.analyze_video",
                        lambda *a, **k: VideoAnalysis(info=VideoInfo(640, 360, 30, 100), students=[], debug={}))
    from app.features.pipeline.service import process_self_upload
    process_self_upload(uid)
    q = client.get("/uploads/quota", headers=h).json()
    assert q["topup_remaining"] == 2
    item = next(i for i in client.get("/analyses", headers=h).json() if i["status"] == "failed")
    d = client.get(f"/analyses/{item['id']}", headers=h).json()
    assert "couldn't find you" in d["failure_reason"] and d["motion_tracking_analysis"] is None


def test_included_credit_refunded_when_analysis_fails(client, tmp_path, monkeypatch):
    make_academy(client); h, _ = make_student(client); subscribe(client, h)
    uid = upload(client, h, tmp_path).json()["id"]; run_validation(uid); client.post(f"/uploads/{uid}/submit", headers=h)
    assert client.get("/uploads/quota", headers=h).json()["included_remaining"] == 1
    monkeypatch.setattr("app.features.pipeline.service.analyze_video", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    from app.features.pipeline.service import process_self_upload
    process_self_upload(uid)
    assert client.get("/uploads/quota", headers=h).json()["included_remaining"] == 2


def fake_analysis(student_id, tmp_path, confidence="face_matched", stride=2):
    m = compute_metrics(stick_track(seconds=10), stride, 30, 170, fixed_camera=True)
    clip = tmp_path / f"c{student_id}.mp4"; make_video(clip, seconds=2)
    return StudentAnalysis(student_id, [1], confidence, 0.8, m, ClipResult(str(clip), 60, 2.0, [(0, 60)]))


def test_self_upload_analysis_becomes_a_report_with_thumbnail_but_no_video(client, tmp_path, monkeypatch):
    make_academy(client); h, me = make_student(client); subscribe(client, h)
    uid = upload(client, h, tmp_path).json()["id"]; run_validation(uid); client.post(f"/uploads/{uid}/submit", headers=h)
    monkeypatch.setattr("app.features.pipeline.service.analyze_video",
                        lambda *a, **k: VideoAnalysis(info=VideoInfo(640, 360, 30, 100), students=[fake_analysis(me["id"], tmp_path)], debug={}))
    from app.features.pipeline.service import process_self_upload
    process_self_upload(uid)
    item = client.get("/analyses", headers=h).json()[0]
    assert item["status"] == "ready" and item["headline"]
    d = client.get(f"/analyses/{item['id']}", headers=h).json()
    assert d["source"] == "self_upload" and d["identity_confidence"] == "face_matched"
    assert d["thumbnail_url"].startswith("https://signed.example/thumbnails/")
    assert d["key_metrics"] and d["technique_analysis"]["summary"] and d["scope_of_improvement"]
    assert "clip_path" not in d and "video" not in str(d.keys())
    assert client.get("/uploads/quota", headers=h).json()["included_remaining"] == 1   # credit stays spent on success
    assert client.get(f"/uploads/{uid}", headers=h).json()["status"] == "done"


# ------------------------------------------------------------------ academy sessions
def test_academy_session_results_reach_only_the_right_student(client, tmp_path, monkeypatch):
    a = make_academy(client)
    h1, s1 = make_student(client, "a@example.com"); h2, s2 = make_student(client, "b@example.com")
    subscribe(client, h1); subscribe(client, h2)
    r = client.post("/sessions/upload", data={"sport": "Soccer", "recorded_at": "2026-09-28T10:00:00Z"},
                    files={"footage": ("s.mp4", video_bytes(tmp_path), "video/mp4")}, headers={"X-Academy-Key": a["api_key"]})
    sid = r.json()["id"]
    seen = {}
    def fake(video, roster, out, heights, **k):
        seen["roster"] = sorted(x.student_id for x in roster); seen["heights"] = heights
        return VideoAnalysis(info=VideoInfo(640, 360, 30, 100), students=[fake_analysis(s1["id"], tmp_path)], debug={"tracks_found": 3})
    monkeypatch.setattr("app.features.pipeline.service.analyze_video", fake)
    monkeypatch.setattr("app.features.pipeline.service.current_model_version", lambda: "stub-hash-v1")
    from app.features.pipeline.service import process_session
    process_session(sid)
    assert seen["roster"] == [s1["id"], s2["id"]] and seen["heights"][s1["id"]] == 170
    mine = client.get("/analyses", headers=h1).json()
    assert len(mine) == 1 and mine[0]["source"] == "academy" and mine[0]["sport"] == "Soccer" and mine[0]["status"] == "ready"
    assert client.get("/analyses", headers=h2).json() == []
    assert client.get(f"/analyses/{mine[0]['id']}", headers=h2).status_code == 404   # someone else's report
    d = client.get(f"/analyses/{mine[0]['id']}", headers=h1).json()
    assert d["thumbnail_url"] is None                                               # academy footage is never exposed
    from app.features.sessions.models import StudentClip, TrainingSession
    db = SessionLocal()
    assert db.get(TrainingSession, sid).status == "done"
    assert db.query(StudentClip).one().clip_path.startswith("clips/")                # clip kept internally
    db.close()


def test_student_with_old_fake_embedding_is_excluded_from_roster(client, tmp_path):
    make_academy(client); make_student(client, "a@example.com")
    from app.features.pipeline.service import load_roster
    db = SessionLocal()
    from app.features.students.models import FaceEmbedding
    fe = db.query(FaceEmbedding).one(); fe.model_version = "insightface-buffalo_l"; db.commit()   # legacy stub label
    monkey_ver = "arcface-buffalo_l-v1"
    with patch("app.features.pipeline.service.current_model_version", return_value=monkey_ver):
        roster, _, students = load_roster(db, academy_id=1)
    assert roster == [] and len(students) == 1
    db.close()


# ------------------------------------------------------------------ registration & profile
def test_registration_validation_and_measurement_history(client):
    make_academy(client)
    files = {"face_image": ("f.jpg", face_jpg(), "image/jpeg")}
    base = {"full_name": "A", "email": "a@x.com", "password": "password123", "academy_name": "Nope", "sport": "Soccer", "height_cm": "170", "weight_kg": "60"}
    assert client.post("/students/register", data=base, files=files).status_code == 400
    base["academy_name"] = "Test Academy"
    assert client.post("/students/register", data={**base, "gender": "x"}, files=files).status_code == 422
    assert client.post("/students/register", data=base, files={"face_image": ("f.txt", b"hi", "text/plain")}).status_code == 400
    h, me = make_student(client, "ok@x.com")
    assert me["sports"] == ["Soccer", "Badminton"]
    assert client.post("/students/register", data={**base, "email": "ok@x.com"}, files=files).status_code == 409
    client.patch("/students/me/measurements", headers=h, json={"height_cm": 171, "weight_kg": 63})
    hist = client.get("/students/me/measurements/history", headers=h).json()
    assert [p["weight_kg"] for p in hist] == [62, 63]


# ------------------------------------------------------------------ diet + home
def test_diet_plan_flow_profile_gate_generation_and_rollups(client, monkeypatch):
    make_academy(client); h, me = make_student(client); subscribe(client, h)
    r = client.get("/diet-plans/current", headers=h)
    assert r.status_code == 409 and set(r.json()["detail"]["missing"]) == {"date_of_birth", "gender"}
    assert client.get("/home", headers=h).json()["diet"]["status"] == "profile_incomplete"

    p = client.patch("/students/me/profile", headers=h, json={"date_of_birth": "2010-03-01", "gender": "male", "diet_preference": "veg", "allergies": "peanuts"})
    assert p.status_code == 200 and p.json()["diet_preference"] == "veg"
    r = client.get("/diet-plans/current", headers=h)
    assert r.status_code == 202 and r.json()["status"] == "generating" and QUEUED[-1][0] == "diet"
    assert client.get("/home", headers=h).json()["diet"]["status"] == "pending"

    diet_tasks.generate_diet_plan.run(me["id"])        # what the worker would do
    r = client.get("/diet-plans/current", headers=h)
    assert r.status_code == 200
    plan = r.json()["plan_json"]
    assert plan["calorie_target"] % 50 == 0 and plan["macros"]["protein_g"] > 60
    assert set(plan["meals"]) >= {"breakfast", "lunch", "dinner", "snacks"} and plan["disclaimer"] and plan["changes_from_last_week"]
    home = client.get("/home", headers=h).json()
    assert home["diet"]["status"] == "ready" and home["diet"]["calories"] == plan["calorie_target"]
    diet_tasks.generate_diet_plan.run(me["id"])        # idempotent: still one plan this week
    m = client.get("/diet-plans/monthly", headers=h).json()
    assert len(m["plans"]) == 1 and m["avg_calories"] == plan["calorie_target"]
    y = client.get("/diet-plans/yearly", headers=h).json()
    assert len(y["quarters"]) == 1 and y["quarters"][0]["plans"] == 1


def test_home_summary_shape(client):
    make_academy(client); h, _ = make_student(client); subscribe(client, h)
    home = client.get("/home", headers=h).json()
    assert home["name"] == "Stu Dent" and home["academy_name"] == "Test Academy" and home["sports"][0] == "Soccer"
    assert home["latest_analysis"] is None and home["upload_quota"]["remaining"] == 2 and home["subscription"]["plan_type"] == "standard"
