from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.features.academies.models import Academy
from app.features.academies.schemas import AcademyLookup, AcademyOut

router = APIRouter(prefix="/academies", tags=["academies"])


@router.get("", response_model=list[AcademyOut])
def list_academies(db: Session = Depends(get_db)):
    """Powers an autocomplete/dropdown in the app's registration screen."""
    return db.query(Academy).filter(Academy.is_partner_active.is_(True)).order_by(Academy.name).all()


@router.get("/lookup", response_model=AcademyLookup)
def lookup_academy(name: str, db: Session = Depends(get_db)):
    """
    Called as the student types their academy name during registration.
    Registration itself re-validates this server-side too — never trust the client.
    """
    academy = (
        db.query(Academy)
        .filter(Academy.name.ilike(name.strip()), Academy.is_partner_active.is_(True))
        .first()
    )
    return AcademyLookup(exists=academy is not None, academy=academy)
