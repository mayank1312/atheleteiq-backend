from pydantic import BaseModel, ConfigDict


class AcademyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_partner_active: bool


class AcademyLookup(BaseModel):
    """Used by the app to validate an academy name exists before registration."""
    exists: bool
    academy: AcademyOut | None = None
