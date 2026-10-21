from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.core.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime) -> datetime:
    """Some drivers hand back naive datetimes; treat those as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def today_local() -> date:
    return datetime.now(ZoneInfo(settings.APP_TIMEZONE)).date()


def week_start(d: date | None = None) -> date:
    """Monday of the week containing d (default: today, app timezone)."""
    d = d or today_local()
    return d - timedelta(days=d.weekday())
