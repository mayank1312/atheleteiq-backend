from datetime import datetime

from pydantic import BaseModel, ConfigDict


class QuotaOut(BaseModel):
    unlimited: bool = False
    has_subscription: bool
    included_total: int
    included_used: int
    included_remaining: int
    topup_remaining: int
    remaining: int
    resets_on: datetime | None


class PackOut(BaseModel):
    id: str
    uploads: int
    price_inr: int


class TopUpRequest(BaseModel):
    pack_id: str


class CheckOut(BaseModel):
    key: str
    label: str
    status: str  # pending | passed | failed | skipped
    message: str = ""


class SelfUploadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sport: str
    status: str  # validating | validated | rejected | submitted | done | failed
    checks: list[CheckOut]
    can_submit: bool
    rejection_reason: str | None = None
    duration_s: float | None = None
    created_at: datetime
