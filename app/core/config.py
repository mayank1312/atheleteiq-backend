from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database / queue ---
    DATABASE_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Auth ---
    JWT_SECRET_KEY: str = "jnsjdnsajxnasjhcbsin"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ADMIN_API_KEY: str = ""  # protects /admin/* (creating academies + their camera keys)

    # --- Supabase storage ---
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("SUPABASE_SERVICE_KEY", "SUPABASE_SECRET_KEY"),
    )
    SUPABASE_BUCKET: str = "athleteiq-media"

    # --- LLM ---
    OPENAI_API_KEY: str = ""
    REPORT_MODEL: str = "gpt-5.2"
    DIET_MODEL: str = "gpt-5.2"
    SPORT_CHECK_MODEL: str = "gpt-5-mini"
    LLM_REASONING_EFFORT: str = ""  # optional: low | medium | high

    # --- Computer vision ---
    POSE_MODEL: str = "yolov8s-pose.pt"  # detection + tracking + 17-keypoint pose in one pass
    TRACKER: str = "bytetrack.yaml"
    DEVICE: str = "auto"  # auto | mps | cpu | cuda
    FRAME_STRIDE: int = 2  # analyse every Nth frame (raise for long footage / slower machines)
    DETECT_IMGSZ: int = 960
    FACE_BACKEND: str = "insightface"  # insightface | stub (stub = fake vectors, for wiring tests only)
    FACE_MATCH_THRESHOLD: float = 0.38  # cosine similarity needed to call a track "this student"
    FACE_MATCH_MARGIN: float = 0.05  # best match must beat the runner-up by this much

    # --- Self-upload rules ---
    TEST_PHASE_BYPASS_FACE_MATCHING: bool = False
    TEST_PHASE_UNLIMITED_UPLOADS: bool = False
    SELF_UPLOAD_INCLUDED_PER_CYCLE: int = 2
    MAX_UPLOAD_MB: int = 50  # Supabase free tier caps a single file at 50 MB
    MAX_ACADEMY_UPLOAD_MB: int = 50
    MIN_VIDEO_SECONDS: float = 5
    MAX_VIDEO_SECONDS: float = 180
    MIN_SHORT_SIDE_PX: int = 480
    STRICT_SPORT_CHECK: bool = False  # if True, a missing API key fails the sport check instead of skipping it

    # --- Misc ---
    DIET_CUISINE: str = "Indian"
    ANALYSIS_SLA_HOURS: int = 2
    APP_TIMEZONE: str = "Asia/Kolkata"


settings = Settings()
