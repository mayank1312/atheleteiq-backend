import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_athleteiq.db")
os.environ.setdefault("FACE_BACKEND", "stub")
os.environ.setdefault("ANTHROPIC_API_KEY", "")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("FRAME_STRIDE", "2")
