from datetime import datetime

from pydantic import BaseModel


class AnalysisItem(BaseModel):
    id: int
    source: str  # academy | self_upload
    sport: str | None
    status: str  # processing | ready | failed
    headline: str | None = None
    created_at: datetime
    ready_at: datetime | None = None
    expected_ready_at: datetime | None = None  # only while processing


class TechniqueOut(BaseModel):
    summary: str
    observations: list[str] = []


class ImprovementOut(BaseModel):
    priority: str
    area: str
    why: str
    drill: str = ""


class KeyMetricOut(BaseModel):
    label: str
    value: str


class ReportPointOut(BaseModel):
    title: str
    message: str
    tone: str = "info"  # positive | attention | info


class ChartPointOut(BaseModel):
    label: str
    value: float


class ChartSeriesOut(BaseModel):
    id: str
    title: str
    unit: str
    data: list[ChartPointOut] = []
    available: bool = True
    note: str | None = None


class DataQualityOut(BaseModel):
    status: str  # good | limited | insufficient
    label: str
    analyzable_seconds: float | None = None
    clip_duration_seconds: float | None = None
    confidence: float | None = None
    fixed_camera: bool | None = None
    reasons: list[str] = []


class AnalysisDetail(AnalysisItem):
    academy_name: str | None = None
    key_metrics: list[KeyMetricOut] = []
    plain_language_summary: list[ReportPointOut] = []
    chart_data: list[ChartSeriesOut] = []
    data_quality: DataQualityOut | None = None
    motion_tracking_analysis: str | None = None
    technique_analysis: TechniqueOut | None = None
    scope_of_improvement: list[ImprovementOut] = []
    data_quality_note: str | None = None
    identity_confidence: str | None = None  # face_matched | appearance_linked | assumed
    thumbnail_url: str | None = None  # self-upload reports only; academy footage is never exposed
    failure_reason: str | None = None
