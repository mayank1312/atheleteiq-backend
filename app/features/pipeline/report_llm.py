"""
Turns measured metrics into the written report. The model never sees video - only the numbers computed in metrics.py,
plus the student's context and their last few sessions for trend comparison.

If OPENAI_API_KEY is missing (or the model output is unusable) we fall back to a rule-based report built directly
from the numbers, so the pipeline still completes and is testable offline.
"""
import json
import logging

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.llm import chat_json, llm_available

log = logging.getLogger("athleteiq.report")


class KeyMetric(BaseModel):
    label: str
    value: str


class Technique(BaseModel):
    summary: str
    observations: list[str] = Field(default_factory=list)


class ImprovementItem(BaseModel):
    priority: str = "medium"  # high | medium | low
    area: str
    why: str
    drill: str


class ReportContent(BaseModel):
    headline: str
    key_metrics: list[KeyMetric] = Field(default_factory=list)
    motion_tracking_analysis: str
    technique_analysis: Technique
    scope_of_improvement: list[ImprovementItem] = Field(default_factory=list)
    data_quality_note: str = ""
    generated_by: str = "ai"  # ai | rule_based


SYSTEM_PROMPT = """You are a sports performance analyst writing a report for a young athlete at a training academy.

You are given MEASURED DATA extracted from pose tracking of the athlete's footage. You do not see the video.

Hard rules:
- Use only numbers that appear in the data. Never invent a measurement, a drill result, or something you cannot see in the data.
- If a metric is null/unavailable, say it could not be measured; do not guess.
- Movement figures marked approximate, or speed with speed_reliable=false, must be described as approximate/relative.
- Do not diagnose injuries or medical conditions. If a pattern could relate to injury risk, suggest the athlete speak with their coach or a physiotherapist.
- The athlete may be a minor: keep advice age-appropriate, encouraging and safe (no extreme training loads).
- Be specific to the sport given. Explain what each finding means for performance in plain language a teenager can follow.
- If a "history" is provided, mention concrete change versus previous sessions only where the numbers actually differ.
- If data quality is low or the athlete was barely visible, say so plainly in data_quality_note and keep conclusions modest.

Return ONLY a JSON object, no prose, no markdown fences, with exactly this shape:
{
 "headline": "one sentence summary, max 14 words",
 "key_metrics": [{"label": "...", "value": "..."}],          // 3 to 6 chips, values include units
 "motion_tracking_analysis": "2-4 short paragraphs on movement: speed, cadence, coverage, jumps, reach, balance",
 "technique_analysis": {"summary": "1-2 paragraphs", "observations": ["specific observation tied to a number", "..."]},
 "scope_of_improvement": [{"priority": "high|medium|low", "area": "...", "why": "tied to the data", "drill": "one concrete drill or cue"}],
 "data_quality_note": "one or two sentences"
}
Give 3 to 5 improvement items, ordered by priority."""


def _compact(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def _extract_json(text: str) -> dict:
    """Parse a JSON object returned by a model, including accidental code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("report response must be a JSON object")
    return value


def build_user_prompt(sport: str, athlete: dict, metrics: dict, source: str, identity_confidence: str, history: list[dict]) -> str:
    ctx = {
        "sport": sport,
        "athlete": athlete,
        "footage_source": source,
        "identity_confidence": identity_confidence,
    }
    parts = [f"CONTEXT:\n{_compact(ctx)}", f"MEASURED DATA:\n{_compact(metrics)}"]
    if history:
        parts.append(f"HISTORY (previous sessions, newest first):\n{_compact(history)}")
    return "\n\n".join(parts)


def generate_report(
    sport: str, athlete: dict, metrics: dict, source: str, identity_confidence: str, history: list[dict] | None = None
) -> ReportContent:
    if not llm_available():
        return rule_based_report(sport, metrics)
    prompt = build_user_prompt(sport, athlete, metrics, source, identity_confidence, history or [])
    for attempt in range(2):
        try:
            data = chat_json(SYSTEM_PROMPT, prompt, settings.REPORT_MODEL, max_tokens=10000)
            return ReportContent(**data, generated_by="ai")
        except (ValueError, ValidationError) as exc:  # bad/empty output: try once more
            log.warning("report output unusable (attempt %s): %s", attempt + 1, exc)
        except Exception as exc:  # network / auth / unknown model: retrying won't help
            log.error("report LLM call failed: %s", exc)
            break
    return rule_based_report(sport, metrics)


# ----------------------------------------------------------------------------- offline fallback
def _fmt(v, unit="", nd=0):
    return "n/a" if v is None else f"{round(v, nd) if nd else int(round(v))}{unit}"


def rule_based_report(sport: str, metrics: dict) -> ReportContent:
    meta, mv = metrics.get("meta", {}), metrics.get("movement", {})
    ja, sym, ev, sc = metrics.get("joint_angles", {}), metrics.get("symmetry", {}), metrics.get("events", {}), metrics.get("scores", {})

    if not meta.get("usable"):
        return ReportContent(
            headline="Not enough clear footage to analyse this session",
            motion_tracking_analysis="You were visible with a readable body pose for too little time to measure movement reliably.",
            technique_analysis=Technique(summary="No technique conclusions could be drawn from this footage.", observations=[]),
            scope_of_improvement=[ImprovementItem(priority="high", area="Footage quality", why="Too little usable pose data.", drill="Record from a steady position with your whole body in frame and good light.")],
            data_quality_note="Analysis was skipped for reliability.", generated_by="rule_based",
        )

    chips, obs, todo = [], [], []
    spd = mv.get("avg_speed_m_s_approx")
    if spd is not None:
        chips.append(KeyMetric(label="Avg speed (approx)", value=f"{spd:.1f} m/s"))
    if mv.get("cadence_steps_per_min") is not None:
        chips.append(KeyMetric(label="Cadence", value=f"{_fmt(mv['cadence_steps_per_min'])} steps/min"))
    if (sc.get("posture_stability") or {}).get("value") is not None:
        chips.append(KeyMetric(label="Posture stability", value=f"{_fmt(sc['posture_stability']['value'])}/100"))
    for name in ("knee_left", "knee_right"):
        if ja.get(name, {}).get("available") and len(chips) < 6:
            chips.append(KeyMetric(label=f"{name.replace('_', ' ').title()} range", value=f"{_fmt(ja[name]['range'])}°"))
    if ev.get("jumps", {}).get("count"):
        chips.append(KeyMetric(label="Jumps", value=str(ev["jumps"]["count"])))
    if ev.get("overhead_reach", {}).get("count"):
        chips.append(KeyMetric(label="Overhead actions", value=str(ev["overhead_reach"]["count"])))

    asym = sym.get("knee_rom_asymmetry_pct")
    if asym is not None:
        obs.append(f"Left and right knee range of motion differ by about {asym:.0f}%.")
        if asym > 20:
            todo.append(ImprovementItem(priority="high", area="Left/right balance", why=f"Knee range differs by ~{asym:.0f}% between legs.", drill="Single-leg squats and lunges, matching reps and depth on both sides."))
    lean = ja.get("trunk_lean_from_vertical", {})
    if lean.get("available"):
        obs.append(f"Average trunk lean was {lean['mean']:.0f}° from vertical (variation ±{lean['std']:.0f}°).")
    deep = ev.get("deep_knee_flexion", {})
    if deep.get("count"):
        obs.append(f"{deep['count']} deep knee-bend moments (minimum knee angle {_fmt(deep.get('min_knee_angle_deg'))}°).")
    if not todo:
        todo.append(ImprovementItem(priority="medium", area="Movement consistency", why="Use these measurements as a baseline.", drill="Repeat similar drills next session and compare the numbers."))

    move_txt = (
        f"You moved at an average of {_fmt(mv.get('avg_speed_body_heights_per_s'), nd=2)} body-heights per second"
        f" and covered about {_fmt(mv.get('distance_covered_body_heights'), nd=1)} body-heights of distance while tracked."
    ) if mv.get("avg_speed_body_heights_per_s") is not None else "Movement speed could not be measured from this footage."
    return ReportContent(
        headline=f"{sport}: baseline measurements recorded",
        key_metrics=chips[:6],
        motion_tracking_analysis=move_txt + " (Automatic summary - AI-written analysis was unavailable for this report.)",
        technique_analysis=Technique(summary="Joint-angle measurements from this session are listed below.", observations=obs),
        scope_of_improvement=todo,
        data_quality_note="; ".join(metrics.get("notes", [])) or "Data quality looks adequate.",
        generated_by="rule_based",
    )


def flatten_for_columns(rep: ReportContent) -> tuple[str, str, str]:
    """Plain-text copies for the three Text columns (also handy for search/export)."""
    tech = rep.technique_analysis.summary + "".join(f"\n- {o}" for o in rep.technique_analysis.observations)
    scope = "\n".join(f"[{i.priority}] {i.area}: {i.why} Drill: {i.drill}" for i in rep.scope_of_improvement)
    return rep.motion_tracking_analysis, tech, scope
