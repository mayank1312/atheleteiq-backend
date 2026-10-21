"""Supabase Storage wrapper. All media lives in one PRIVATE bucket; the API only ever hands out short-lived signed URLs."""
import uuid
from functools import lru_cache
from pathlib import Path

from app.core.config import settings


@lru_cache
def _bucket():
    from supabase import create_client  # lazy: API boots even if storage isn't configured yet

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY missing in .env")
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    return client.storage.from_(settings.SUPABASE_BUCKET)


def new_key(prefix: str, filename: str, default_ext: str = "mp4") -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else default_ext
    return f"{prefix}/{uuid.uuid4()}.{ext}"


def upload_file(local_path: str | Path, key: str, content_type: str = "video/mp4") -> str:
    """Streams a local file up (no need to hold it in memory) and returns the key."""
    _bucket().upload(path=key, file=str(local_path), file_options={"content-type": content_type, "upsert": "true"})
    return key


def upload_bytes(data: bytes, key: str, content_type: str = "image/jpeg") -> str:
    _bucket().upload(path=key, file=data, file_options={"content-type": content_type, "upsert": "true"})
    return key


def download_to(key: str, local_path: str | Path) -> Path:
    data = _bucket().download(key)
    Path(local_path).write_bytes(data)
    return Path(local_path)


def delete_object(key: str) -> None:
    try:
        _bucket().remove([key])
    except Exception:
        pass  # best-effort cleanup


def get_signed_url(key: str, expires_in: int = 3600) -> str:
    res = _bucket().create_signed_url(key, expires_in)
    return res.get("signedURL") or res.get("signedUrl") or ""
