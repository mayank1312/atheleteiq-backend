import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.core.deps import hash_api_key, require_admin
from app.db.session import get_db
from app.features.academies.models import Academy

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class AcademyCreate(BaseModel):
    name: str
    contact_email: str | None = None


class AcademyCreated(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    api_key: str  # shown ONCE - only its hash is stored


def _issue_key(academy: Academy) -> str:
    key = "aiq_" + secrets.token_urlsafe(32)
    academy.api_key_hash = hash_api_key(key)
    return key


@router.post("/academies", response_model=AcademyCreated, status_code=status.HTTP_201_CREATED)
def create_academy(payload: AcademyCreate, db: Session = Depends(get_db)):
    if db.query(Academy).filter(Academy.name.ilike(payload.name.strip())).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Academy already exists")
    academy = Academy(name=payload.name.strip(), contact_email=payload.contact_email)
    key = _issue_key(academy)
    db.add(academy)
    db.commit()
    db.refresh(academy)
    return AcademyCreated(id=academy.id, name=academy.name, api_key=key)


@router.post("/academies/{academy_id}/rotate-key", response_model=AcademyCreated)
def rotate_key(academy_id: int, db: Session = Depends(get_db)):
    academy = db.get(Academy, academy_id)
    if academy is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Academy not found")
    key = _issue_key(academy)
    db.commit()
    return AcademyCreated(id=academy.id, name=academy.name, api_key=key)
