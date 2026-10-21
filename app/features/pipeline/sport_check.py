"""Does this footage actually show the sport the student says it does? Vision check on a few sampled frames."""
import logging

import numpy as np

from app.core.config import settings
from app.core.llm import llm_available, vision_json
from app.features.pipeline.video_utils import encode_jpeg

log = logging.getLogger("athleteiq.sportcheck")

PROMPT = """These are frames sampled from one video a student uploaded for analysis.
The student says the sport is: {sport}.
Decide whether the footage shows someone playing or practising {sport} (drills, warm-up specific to it, or match play all count).
Reply with ONLY a JSON object: {{"matches": true or false, "detected_sport": "what you think it shows, or 'none'", "confidence": a number from 0 to 1, "reason": "one short sentence"}}"""


def check_sport(frames: list[np.ndarray], sport: str) -> dict:
    """Returns {'status': passed|failed|skipped, 'message': str}."""
    if not llm_available():
        if settings.STRICT_SPORT_CHECK:
            return {"status": "failed", "message": "Sport verification is unavailable right now. Please try again later."}
        return {"status": "skipped", "message": "Sport verification skipped (not configured)."}
    if not frames:
        return {"status": "failed", "message": "No frames could be read for sport verification."}
    try:
        data = vision_json([encode_jpeg(f, 640) for f in frames[:5]], PROMPT.format(sport=sport), settings.SPORT_CHECK_MODEL)
    except Exception as exc:
        log.error("sport check failed: %s", exc)
        return {"status": "skipped", "message": "Sport verification was unavailable; continuing."}
    matches, conf = bool(data.get("matches")), float(data.get("confidence", 0) or 0)
    if matches and conf >= 0.5:
        return {"status": "passed", "message": f"Looks like {sport} footage."}
    detected = data.get("detected_sport", "another activity")
    return {"status": "failed", "message": f"This looks like {detected}, not {sport}. Pick the right sport or a different video."}
