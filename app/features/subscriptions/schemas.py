from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SubscriptionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    plan_type: str
    status: str
    started_at: datetime
    expires_at: datetime


class SubscriptionActivateRequest(BaseModel):
    plan_type: str = "standard"
    duration_days: int = 30
