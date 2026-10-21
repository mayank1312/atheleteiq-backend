from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.features.auth.schemas import PasswordResetRequest, Token
from app.features.students.models import Student

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Standard OAuth2 password flow (username field = email).
    Login itself always succeeds on valid credentials — it does NOT check
    subscription status. The subscription paywall is enforced per-route via
    require_active_subscription, so a lapsed-sub user can still log in and
    see the "please subscribe" screen instead of a confusing auth failure.
    """
    student = db.query(Student).filter(Student.email == form_data.username.lower()).first()

    if student is None or not verify_password(form_data.password, student.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=str(student.id))
    return Token(access_token=access_token)


@router.post("/reset-password")
def reset_password(payload: PasswordResetRequest, db: Session = Depends(get_db)):
    if len(payload.new_password) < 8:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Password must be at least 8 characters.")

    student = db.query(Student).filter(Student.email == payload.email.strip().lower()).first()
    if student is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No account found for this email.")

    student.hashed_password = hash_password(payload.new_password)
    db.commit()
    return {"message": "Password updated successfully."}
