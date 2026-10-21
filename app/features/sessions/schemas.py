from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TrainingSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    academy_id: int
    sport: str
    status: str
    recorded_at: datetime
