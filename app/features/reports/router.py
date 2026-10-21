from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core import storage
from app.core.config import settings
from app.core.deps import require_active_subscription
from app.core.timeutils import as_utc
from app.db.session import get_db
from app.features.reports.schemas import (
    AnalysisDetail,
    AnalysisItem,
    ChartPointOut,
    ChartSeriesOut,
    DataQualityOut,
    ReportPointOut,
)
from app.features.sessions.models import StudentClip
from app.features.students.models import Student

router = APIRouter(prefix="/analyses", tags=["analyses"])

STATUS_MAP = {"ready": "ready", "failed": "failed"}


def _sport(clip: StudentClip) -> str | None:
    if clip.source == "self_upload" and clip.self_upload:
        return clip.self_upload.sport
    return clip.session.sport if clip.session else None


def _expected(clip: StudentClip):
    if clip.status in ("ready", "failed"):
        return None
    start = clip.self_upload.submitted_at if clip.source == "self_upload" and clip.self_upload and clip.self_upload.submitted_at else clip.created_at
    return as_utc(start) + timedelta(hours=settings.ANALYSIS_SLA_HOURS)


def to_item(clip: StudentClip) -> AnalysisItem:
    return AnalysisItem(
        id=clip.id, source=clip.source or "academy", sport=_sport(clip), status=STATUS_MAP.get(clip.status, "processing"),
        headline=clip.report.headline if clip.report else None, created_at=clip.created_at, ready_at=clip.ready_at,
        expected_ready_at=_expected(clip),
    )


@router.get("", response_model=list[AnalysisItem])
def list_analyses(
    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
    student: Student = Depends(require_active_subscription), db: Session = Depends(get_db),
):
    """The Analysis tab. Newest first; mixes academy sessions and self-uploads, tagged by `source`."""
    clips = db.query(StudentClip).filter(StudentClip.student_id == student.id).order_by(StudentClip.created_at.desc()).offset(offset).limit(limit).all()
    return [to_item(c) for c in clips]


def _chart_series(metrics: dict) -> list[ChartSeriesOut]:
    """Turn measured values into safe, directly plottable frontend series."""
    movement = metrics.get("movement", {})
    angles = metrics.get("joint_angles", {})
    symmetry = metrics.get("symmetry", {})
    scores = metrics.get("scores", {})
    events = metrics.get("events", {})
    series: list[ChartSeriesOut] = []

    def add(id: str, title: str, unit: str, values: list[tuple[str, object]], note: str | None = None):
        points = [ChartPointOut(label=label, value=float(value)) for label, value in values if value is not None]
        series.append(ChartSeriesOut(id=id, title=title, unit=unit, data=points, available=bool(points), note=note))

    add("movement", "Movement speed", "body-heights/second", [
        ("Average", movement.get("avg_speed_body_heights_per_s")),
        ("Moving average", movement.get("moving_avg_speed_body_heights_per_s")),
        ("Top speed", movement.get("top_speed_body_heights_per_s")),
    ], None if movement.get("speed_reliable", True) else "Relative estimate because the camera was moving.")
    add("knee-range", "Knee movement range", "degrees", [
        ("Left knee", (angles.get("knee_left") or {}).get("range")),
        ("Right knee", (angles.get("knee_right") or {}).get("range")),
    ])
    add("body-control", "Body control score", "out of 100", [
        ("Posture stability", (scores.get("posture_stability") or {}).get("value")),
    ], "An indicative movement-consistency score, not a medical measure.")
    add("actions", "Detected actions", "count", [
        ("Overhead actions", (events.get("overhead_reach") or {}).get("count")),
        ("Deep knee bends", (events.get("deep_knee_flexion") or {}).get("count")),
        ("Jumps", (events.get("jumps") or {}).get("count")),
        ("Fast leg swings", (events.get("fast_leg_swings") or {}).get("count")),
    ])
    add("symmetry", "Left/right difference", "percent", [
        ("Knee range", symmetry.get("knee_rom_asymmetry_pct")),
        ("Hip range", symmetry.get("hip_rom_asymmetry_pct")),
        ("Shoulder range", symmetry.get("shoulder_rom_asymmetry_pct")),
    ], "Lower percentages generally mean more similar movement between sides.")
    return series


def _data_quality(metrics: dict) -> DataQualityOut:
    meta = metrics.get("meta", {})
    usable = bool(meta.get("usable"))
    analyzable = meta.get("analyzable_seconds")
    # duration_s is the usable tracked window; source_duration_s is the full
    # uploaded video. Prefer the latter so coverage is not presented as clip length.
    duration = meta.get("source_duration_s", meta.get("duration_s"))
    confidence = meta.get("avg_keypoint_confidence")
    reasons = list(metrics.get("notes") or [])
    if not usable and not reasons:
        reasons.append("There was not enough clear body movement to make reliable measurements.")
    status = "good" if usable and (confidence is None or confidence >= 0.6) else "limited" if usable else "insufficient"
    label = {"good": "Good quality data", "limited": "Some measurements are limited", "insufficient": "More video is needed"}[status]
    return DataQualityOut(
        status=status,
        label=label,
        analyzable_seconds=analyzable,
        clip_duration_seconds=duration,
        confidence=confidence,
        fixed_camera=meta.get("fixed_camera"),
        reasons=reasons,
    )


def _plain_language_summary(metrics: dict) -> list[ReportPointOut]:
    meta = metrics.get("meta", {})
    movement = metrics.get("movement", {})
    symmetry = metrics.get("symmetry", {})
    scores = metrics.get("scores", {})
    points: list[ReportPointOut] = []
    if not meta.get("usable"):
        return [ReportPointOut(
            title="This video needs a clearer view",
            message="The athlete was visible for too little time to make dependable movement or technique measurements.",
            tone="attention",
        )]
    speed = movement.get("avg_speed_body_heights_per_s")
    if speed is not None:
        points.append(ReportPointOut(title="Movement baseline", message=f"Average movement speed was {speed:.2f} body-heights per second.", tone="info"))
    asym = symmetry.get("knee_rom_asymmetry_pct")
    if asym is not None:
        tone = "attention" if asym > 20 else "positive"
        points.append(ReportPointOut(title="Left/right balance", message=f"Left and right knee movement differed by about {asym:.0f}%.", tone=tone))
    stability = (scores.get("posture_stability") or {}).get("value")
    if stability is not None:
        tone = "positive" if stability >= 70 else "attention"
        points.append(ReportPointOut(title="Body control", message=f"Posture stability was {stability:.0f} out of 100 for this session.", tone=tone))
    if not points:
        points.append(ReportPointOut(title="What was measured", message="The session produced a usable movement baseline for comparison with future uploads.", tone="info"))
    return points


def _as_report_point(value: ReportPointOut | dict) -> ReportPointOut:
    return value if isinstance(value, ReportPointOut) else ReportPointOut.model_validate(value)


def _as_chart_series(value: ChartSeriesOut | dict) -> ChartSeriesOut:
    return value if isinstance(value, ChartSeriesOut) else ChartSeriesOut.model_validate(value)


@router.get("/{clip_id}", response_model=AnalysisDetail)
def analysis_detail(clip_id: int, student: Student = Depends(require_active_subscription), db: Session = Depends(get_db)):
    clip = db.get(StudentClip, clip_id)
    if clip is None or clip.student_id != student.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")  # same answer for "missing" and "someone else's"
    base = to_item(clip).model_dump()
    detail = AnalysisDetail(**base, academy_name=student.academy_name, identity_confidence=clip.identity_confidence, failure_reason=clip.error)
    rep = clip.report
    if clip.status == "ready" and rep:
        s = rep.structured_json or {}
        metrics = rep.motion_metrics_json or {}
        detail.key_metrics = s.get("key_metrics") or rep.key_metrics_json or []
        detail.plain_language_summary = [
            _as_report_point(item) for item in (s.get("plain_language_summary") or _plain_language_summary(metrics))
        ]
        detail.chart_data = [_as_chart_series(item) for item in (s.get("chart_data") or _chart_series(metrics))]
        detail.data_quality = DataQualityOut(**(s.get("data_quality") or _data_quality(metrics).model_dump()))
        detail.motion_tracking_analysis = s.get("motion_tracking_analysis") or rep.motion_tracking_analysis
        detail.technique_analysis = s.get("technique_analysis") or {"summary": rep.technique_analysis, "observations": []}
        detail.scope_of_improvement = s.get("scope_of_improvement") or [
            {"priority": "medium", "area": "Recommendation", "why": line, "drill": ""} for line in rep.scope_of_improvement.splitlines() if line.strip()
        ]
        detail.data_quality_note = s.get("data_quality_note")
    if clip.source == "self_upload" and clip.self_upload and clip.self_upload.thumbnail_path:
        try:
            detail.thumbnail_url = storage.get_signed_url(clip.self_upload.thumbnail_path)
        except Exception:
            detail.thumbnail_url = None
    return detail
