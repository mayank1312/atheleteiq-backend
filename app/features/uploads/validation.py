"""
The checklist behind screen 6B. Each check is saved as soon as it finishes so the app can poll and animate
spinner -> tick/cross. Nothing here touches quota: a rejected video costs the student nothing.
"""
import logging
import os
import tempfile

from sqlalchemy.orm.attributes import flag_modified

from app.core import storage
from app.core.config import settings
from app.db.session import SessionLocal
from app.features.pipeline.video_utils import VideoError, encode_jpeg, probe_video, sample_frames
from app.features.uploads.models import SelfUpload

log = logging.getLogger("athleteiq.validation")

CHECKS = [
    ("length", "Checking video length"),
    ("quality", "Checking video quality/resolution"),
    ("sport_match", "Confirming this looks like {sport} footage"),
    ("person_visible", "Checking for a visible player"),
]


def initial_checks() -> dict:
    return {k: {"status": "pending", "message": ""} for k, _ in CHECKS}


def checks_as_list(upload: SelfUpload) -> list[dict]:
    out = []
    for key, label in CHECKS:
        c = (upload.checks_json or {}).get(key, {"status": "pending", "message": ""})
        out.append({"key": key, "label": label.format(sport=upload.sport), "status": c["status"], "message": c.get("message", "")})
    return out


def _set(db, upload, key, status, message=""):
    checks = dict(upload.checks_json or initial_checks())
    checks[key] = {"status": status, "message": message}
    upload.checks_json = checks
    flag_modified(upload, "checks_json")
    db.commit()


def run_validation(upload_id: int) -> None:
    from app.features.pipeline.precheck import find_player_frames
    from app.features.pipeline.sport_check import check_sport

    db = SessionLocal()
    try:
        upload = db.get(SelfUpload, upload_id)
        if upload is None:
            return
        with tempfile.TemporaryDirectory() as tmp:
            local = storage.download_to(upload.video_path, os.path.join(tmp, "v." + upload.video_path.rsplit(".", 1)[-1]))
            try:
                info = probe_video(str(local))
            except VideoError as exc:
                _set(db, upload, "length", "failed", str(exc))
                for k in ("quality", "sport_match", "person_visible"):
                    _set(db, upload, k, "skipped", "Not checked because the video couldn't be read.")
                _finish(db, upload)
                return

            upload.duration_s, upload.width, upload.height = round(info.duration_s, 1), info.width, info.height
            db.commit()

            if info.duration_s < settings.MIN_VIDEO_SECONDS:
                _set(db, upload, "length", "failed", f"Video is {info.duration_s:.0f}s; it must be at least {settings.MIN_VIDEO_SECONDS:.0f}s.")
            elif info.duration_s > settings.MAX_VIDEO_SECONDS:
                _set(db, upload, "length", "failed", f"Video is {info.duration_s:.0f}s; trim it to {settings.MAX_VIDEO_SECONDS:.0f}s or less.")
            else:
                _set(db, upload, "length", "passed", f"{info.duration_s:.0f}s")

            if info.short_side < settings.MIN_SHORT_SIDE_PX:
                _set(db, upload, "quality", "failed", f"Resolution is too low ({info.short_side}p). Record at {settings.MIN_SHORT_SIDE_PX}p or higher.")
            elif info.fps < 15:
                _set(db, upload, "quality", "failed", f"Frame rate is too low ({info.fps:.0f} fps). Record at 24 fps or higher.")
            else:
                _set(db, upload, "quality", "passed", f"{info.width}x{info.height} at {info.fps:.0f} fps")

            frames = sample_frames(str(local), n=6)
            if settings.TEST_PHASE_BYPASS_FACE_MATCHING:
                _set(db, upload, "sport_match", "passed", "Skipped in test phase.")
                _set(db, upload, "person_visible", "passed", "Person selection is handled during analysis.")
                best = None
            else:
                _set(db, upload, "sport_match", **_kw(check_sport(frames, upload.sport)))

                visible, best = find_player_frames(frames)
                if visible >= 2:
                    _set(db, upload, "person_visible", "passed")
                else:
                    _set(db, upload, "person_visible", "failed", "We couldn't see a player clearly. Keep your whole body in frame, closer to the camera.")

            pick = best if best is not None else (len(frames) // 2 if frames else None)
            if pick is not None:
                try:
                    upload.thumbnail_path = storage.upload_bytes(encode_jpeg(frames[pick], 480), storage.new_key(f"thumbnails/{upload.student_id}", "t.jpg"))
                except Exception as exc:
                    log.warning("thumbnail upload failed: %s", exc)
            _finish(db, upload)
    except Exception:
        log.exception("validation crashed for upload %s", upload_id)
        db.rollback()
        u = db.get(SelfUpload, upload_id)
        if u:
            u.status, u.rejection_reason = "rejected", "We couldn't check this video right now. Please try again."
            db.commit()
        raise
    finally:
        db.close()


def _kw(result: dict) -> dict:
    return {"status": result["status"], "message": result.get("message", "")}


def _finish(db, upload: SelfUpload) -> None:
    failed = [c for c in (upload.checks_json or {}).values() if c["status"] == "failed"]
    if failed:
        upload.status = "rejected"
        upload.rejection_reason = failed[0]["message"]
    else:
        upload.status = "validated"
        upload.rejection_reason = None
    db.commit()
