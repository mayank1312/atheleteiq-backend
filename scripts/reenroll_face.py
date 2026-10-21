"""
Replace a student's stored face vector with a real one (support-assisted re-enrolment).
Needed once for students registered before the real face model was added.

  python -m scripts.reenroll_face --email test@example.com --image ~/Desktop/face.jpg
"""
import argparse
import os

from app.db.session import SessionLocal
from app.features.pipeline.faces import current_model_version, embed_from_bytes
from app.features.students.models import FaceEmbedding, Student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    emb, n = embed_from_bytes(open(os.path.expanduser(a.image), "rb").read())
    if n > 1:
        raise SystemExit("More than one face in the photo - use a solo selfie.")
    db = SessionLocal()
    s = db.query(Student).filter(Student.email == a.email.lower()).first()
    if s is None:
        raise SystemExit(f"No student with email {a.email}")
    fe = db.query(FaceEmbedding).filter(FaceEmbedding.student_id == s.id).first()
    if fe:
        fe.embedding_vector, fe.model_version = emb, current_model_version()
    else:
        db.add(FaceEmbedding(student_id=s.id, embedding_vector=emb, model_version=current_model_version()))
    db.commit()
    print(f"Face re-enrolled for {s.full_name} ({current_model_version()}). The photo itself was not stored.")


if __name__ == "__main__":
    main()
