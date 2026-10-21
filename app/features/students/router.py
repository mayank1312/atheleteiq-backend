from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_student
from app.core.security import hash_password
from app.db.session import get_db
from app.features.academies.models import Academy
from app.features.pipeline.faces import FaceError, current_model_version, embed_from_bytes
from app.features.students.models import FaceEmbedding, MeasurementLog, Student
from app.features.students.schemas import MeasurementPoint, StudentMeasurementUpdate, StudentOut, StudentProfileUpdate

router = APIRouter(prefix="/students", tags=["students"])

MAX_FACE_IMAGE_BYTES = 8 * 1024 * 1024
KNOWN_SPORTS_HINT = "comma-separated for several, e.g. 'Soccer,Badminton'; first is the primary sport"


@router.post("/register", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
async def register_student(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(..., min_length=8),
    academy_name: str = Form(...),
    sport: str = Form(..., description=KNOWN_SPORTS_HINT),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    date_of_birth: date | None = Form(None),
    gender: str | None = Form(None),
    diet_preference: str | None = Form(None),
    allergies: str | None = Form(None),
    face_image: UploadFile = File(..., description="Selfie captured during onboarding"),
    db: Session = Depends(get_db),
):
    academy = db.query(Academy).filter(Academy.name.ilike(academy_name.strip()), Academy.is_partner_active.is_(True)).first()
    if academy is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Academy not found. Check the spelling, or this academy is not yet a partner.")
    if db.query(Student).filter(Student.email == email.lower()).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")
    if not 0 < height_cm < 300 or not 0 < weight_kg < 400:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Height/weight out of range.")
    if gender and gender not in ("male", "female"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "gender must be 'male' or 'female'.")
    if diet_preference and diet_preference not in ("veg", "non_veg", "egg", "vegan"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "diet_preference must be veg, non_veg, egg or vegan.")
    if face_image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Face image must be JPEG or PNG.")

    image_bytes = await face_image.read()
    if len(image_bytes) > MAX_FACE_IMAGE_BYTES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Image too large.")
    try:
        embedding, n_faces = embed_from_bytes(image_bytes)  # the image itself is discarded after this
    except FaceError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    if n_faces > 1:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "More than one face in the photo. Take a selfie with only you in frame.")

    student = Student(
        academy_id=academy.id, full_name=full_name, email=email.lower(), hashed_password=hash_password(password),
        sport=",".join(s.strip() for s in sport.split(",") if s.strip()), height_cm=height_cm, weight_kg=weight_kg,
        date_of_birth=date_of_birth, gender=gender, diet_preference=diet_preference, allergies=allergies,
    )
    db.add(student)
    db.flush()
    db.add(FaceEmbedding(student_id=student.id, embedding_vector=embedding, model_version=current_model_version()))
    db.add(MeasurementLog(student_id=student.id, height_cm=height_cm, weight_kg=weight_kg))
    db.commit()
    db.refresh(student)
    return student


@router.get("/me", response_model=StudentOut)
def get_my_profile(current_student: Student = Depends(get_current_student)):
    return current_student


@router.patch("/me/measurements", response_model=StudentOut)
def update_measurements(payload: StudentMeasurementUpdate, current_student: Student = Depends(get_current_student), db: Session = Depends(get_db)):
    """Height/weight feed the diet planner, so every update is also logged for trend charts."""
    current_student.height_cm, current_student.weight_kg = payload.height_cm, payload.weight_kg
    db.add(MeasurementLog(student_id=current_student.id, height_cm=payload.height_cm, weight_kg=payload.weight_kg))
    db.commit()
    db.refresh(current_student)
    return current_student


@router.get("/me/measurements/history", response_model=list[MeasurementPoint])
def measurement_history(current_student: Student = Depends(get_current_student), db: Session = Depends(get_db)):
    return db.query(MeasurementLog).filter(MeasurementLog.student_id == current_student.id).order_by(MeasurementLog.recorded_at).all()


@router.patch("/me/profile", response_model=StudentOut)
def update_profile(payload: StudentProfileUpdate, current_student: Student = Depends(get_current_student), db: Session = Depends(get_db)):
    values = payload.model_dump(exclude_unset=True)
    academy_name = values.pop("academy_name", None)
    sports = values.pop("sports", None)
    password = values.pop("password", None)
    height_cm = values.get("height_cm")
    weight_kg = values.get("weight_kg")

    if "email" in values:
        values["email"] = values["email"].strip().lower()
        existing = (
            db.query(Student)
            .filter(Student.email == values["email"], Student.id != current_student.id)
            .first()
        )
        if existing:
            raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")

    if "full_name" in values:
        values["full_name"] = values["full_name"].strip()
        if not values["full_name"]:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Name cannot be blank.")

    if sports is not None and "sport" in values:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Send either sport or sports, not both.")
    if sports is not None:
        cleaned_sports = [sport.strip() for sport in sports if sport.strip()]
        if not cleaned_sports:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "At least one sport is required.")
        values["sport"] = ",".join(cleaned_sports)
    elif "sport" in values:
        cleaned_sport = values["sport"].strip()
        if not cleaned_sport:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "At least one sport is required.")
        values["sport"] = cleaned_sport

    if academy_name is not None:
        academy = (
            db.query(Academy)
            .filter(Academy.name.ilike(academy_name.strip()), Academy.is_partner_active.is_(True))
            .first()
        )
        if academy is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Academy not found or is not an active partner.")
        current_student.academy_id = academy.id

    values.pop("height_cm", None)
    values.pop("weight_kg", None)
    for field, value in values.items():
        setattr(current_student, field, value)
    if password is not None:
        current_student.hashed_password = hash_password(password)
    if height_cm is not None or weight_kg is not None:
        new_height = height_cm if height_cm is not None else current_student.height_cm
        new_weight = weight_kg if weight_kg is not None else current_student.weight_kg
        current_student.height_cm, current_student.weight_kg = new_height, new_weight
        db.add(MeasurementLog(student_id=current_student.id, height_cm=new_height, weight_kg=new_weight))
    db.commit()
    db.refresh(current_student)
    return current_student
