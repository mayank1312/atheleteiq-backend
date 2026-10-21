"""
Biomechanics from pose keypoints. Pure numpy/scipy - no ML models here, so this is deterministic and unit-testable.

Everything is measured relative to the athlete's own body size (body-heights) because the camera has no metric
calibration. Real-world units (m/s, cm) are derived from the student's registered height and are flagged
approximate; speed is only marked reliable for fixed cameras (academy rigs), not hand-held phone footage.
The LLM only ever sees these numbers - it never watches video.
"""
import math

import numpy as np
from scipy.signal import find_peaks

from app.features.pipeline.tracking import Track

CONF_MIN = 0.4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
BODY_KPTS = list(range(5, 17))


# ----------------------------------------------------------------------------- helpers
def _fill_short_gaps(x: np.ndarray, max_gap: int = 3) -> np.ndarray:
    x = x.copy()
    isn = np.isnan(x)
    if not isn.any() or isn.all():
        return x
    idx = np.arange(len(x))
    interp = np.interp(idx, idx[~isn], x[~isn])
    i, n = 0, len(x)
    while i < n:
        if isn[i]:
            j = i
            while j < n and isn[j]:
                j += 1
            if (j - i) <= max_gap and i > 0 and j < n:
                x[i:j] = interp[i:j]
            i = j
        else:
            i += 1
    return x


def _smooth(x: np.ndarray, w: int = 3) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w)
    valid = ~np.isnan(x)
    num = np.convolve(np.where(valid, x, 0.0), k, mode="same")
    den = np.convolve(valid.astype(float), k, mode="same")
    out = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)
    out[~valid] = np.nan
    return out


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(axis=1) / np.where(n == 0, np.nan, n)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _stats(x: np.ndarray, min_cov: float = 0.2) -> dict:
    valid = x[~np.isnan(x)]
    cov = len(valid) / max(1, len(x))
    if len(valid) < 5 or cov < min_cov:
        return {"available": False, "coverage": round(cov, 2)}
    p5, p10, p90, p95 = np.percentile(valid, [5, 10, 90, 95])
    return {
        "available": True,
        "coverage": round(cov, 2),
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p10": float(p10),
        "p90": float(p90),
        "range": float(p95 - p5),  # robust range of motion
        "std": float(np.std(valid)),
    }


def _runs(mask: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 3)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack([a, b]), axis=0)


def _asym(l: dict, r: dict) -> float | None:
    if not (l.get("available") and r.get("available")):
        return None
    m = max(l["range"], r["range"])
    return abs(l["range"] - r["range"]) / m * 100 if m > 0 else None


# ----------------------------------------------------------------------------- main
def compute_metrics(track: Track, stride: int, fps: float, height_cm: float | None = None, fixed_camera: bool = True) -> dict:
    frames = np.array(track.frames)
    idx = (frames - frames[0]) // stride
    T = int(idx[-1]) + 1
    eff_fps = fps / stride

    K = np.full((T, 17, 3), np.nan)
    B = np.full((T, 4), np.nan)
    K[idx] = np.stack(track.kpts)
    B[idx] = np.stack(track.boxes)
    xy, conf = K[:, :, :2].copy(), K[:, :, 2]
    xy[conf < CONF_MIN] = np.nan
    for j in range(17):
        for c in range(2):
            xy[:, j, c] = _smooth(_fill_short_gaps(xy[:, j, c]))

    heights = B[:, 3] - B[:, 1]
    body_h = float(np.nanpercentile(heights, 75)) if np.isfinite(heights).any() else float("nan")

    body_ok = (~np.isnan(xy[:, BODY_KPTS, 0])).sum(axis=1) >= 8
    analyzable_s = float(body_ok.sum() / eff_fps)
    body_conf = conf[:, BODY_KPTS]
    avg_conf = float(np.nanmean(np.where(body_conf > 0, body_conf, np.nan))) if np.isfinite(body_conf).any() else float("nan")
    usable = analyzable_s >= 3.0 and np.isfinite(body_h)

    meta = {
        "duration_s": T / eff_fps,
        "analyzable_seconds": analyzable_s,
        "effective_fps": eff_fps,
        "frames_used": int(len(frames)),
        "avg_keypoint_confidence": avg_conf,
        "usable": bool(usable),
        "fixed_camera": fixed_camera,
        "body_height_px": body_h,
        "scale_source": "registered_height" if height_cm else "none",
    }
    if not usable:
        return _clean({
            "meta": meta, "joint_angles": {}, "symmetry": {}, "movement": {}, "events": {}, "scores": {},
            "notes": ["Athlete was visible with a readable body pose for too little time to measure reliably."],
        })

    P = lambda j: xy[:, j, :]  # noqa: E731

    # ---- joint angles
    ang = {
        "knee_left": _angle(P(L_HIP), P(L_KN), P(L_AN)),
        "knee_right": _angle(P(R_HIP), P(R_KN), P(R_AN)),
        "hip_left": _angle(P(L_SH), P(L_HIP), P(L_KN)),
        "hip_right": _angle(P(R_SH), P(R_HIP), P(R_KN)),
        "elbow_left": _angle(P(L_SH), P(L_EL), P(L_WR)),
        "elbow_right": _angle(P(R_SH), P(R_EL), P(R_WR)),
        "shoulder_left": _angle(P(L_HIP), P(L_SH), P(L_EL)),
        "shoulder_right": _angle(P(R_HIP), P(R_SH), P(R_EL)),
    }
    sh_mid, hip_mid = _mid(P(L_SH), P(R_SH)), _mid(P(L_HIP), P(R_HIP))
    trunk_vec = sh_mid - hip_mid
    with np.errstate(invalid="ignore"):
        lean = np.degrees(np.arctan2(np.abs(trunk_vec[:, 0]), -trunk_vec[:, 1]))
    ang["trunk_lean_from_vertical"] = lean
    joint_stats = {k: _stats(v) for k, v in ang.items()}

    symmetry = {
        "knee_rom_asymmetry_pct": _asym(joint_stats["knee_left"], joint_stats["knee_right"]),
        "hip_rom_asymmetry_pct": _asym(joint_stats["hip_left"], joint_stats["hip_right"]),
        "elbow_rom_asymmetry_pct": _asym(joint_stats["elbow_left"], joint_stats["elbow_right"]),
        "shoulder_rom_asymmetry_pct": _asym(joint_stats["shoulder_left"], joint_stats["shoulder_right"]),
    }

    # ---- movement
    torso_len = np.nanmedian(np.linalg.norm(sh_mid - hip_mid, axis=1))
    torso_len = float(torso_len) if np.isfinite(torso_len) and torso_len > 0 else body_h * 0.3
    step = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1)  # NaN where either sample missing
    speed_bhs = step * eff_fps / body_h
    finite = speed_bhs[np.isfinite(speed_bhs)]
    movement: dict = {"speed_reliable": fixed_camera}
    body_m = (height_cm / 100.0) if height_cm else None
    if len(finite) >= 5:
        cap = float(np.percentile(finite, 99))
        f = np.clip(finite, 0, cap)
        moving = f[f > 0.15]
        movement.update({
            "avg_speed_body_heights_per_s": float(np.mean(f)),
            "moving_avg_speed_body_heights_per_s": float(np.mean(moving)) if len(moving) else 0.0,
            "top_speed_body_heights_per_s": float(np.percentile(f, 95)),
            "moving_fraction": float(len(moving) / len(f)),
            "distance_covered_body_heights": float(np.sum(f) / eff_fps),
        })
        if body_m:
            movement["avg_speed_m_s_approx"] = float(np.mean(f) * body_m)
            movement["top_speed_m_s_approx"] = float(np.percentile(f, 95) * body_m)

    # cadence from ankle crossings: every time one foot passes the other is one step
    da = np.stack([(P(L_AN) - P(R_AN))[:, 0], (P(L_AN) - P(R_AN))[:, 1]], axis=1) / body_h
    axis = da[:, 0] if np.nanvar(da[:, 0]) >= np.nanvar(da[:, 1]) else da[:, 1]
    sig = axis - np.nanmedian(axis)
    strong = np.where(np.abs(sig) > 0.04, np.sign(sig), np.nan)
    sgn = strong[~np.isnan(strong)]
    valid_s = float((~np.isnan(sig)).sum() / eff_fps)
    if len(sgn) >= 4 and valid_s >= 2:
        steps = int(np.sum(sgn[1:] != sgn[:-1]))
        movement["step_count"] = steps
        movement["cadence_steps_per_min"] = steps / valid_s * 60
        mv = movement.get("moving_avg_speed_body_heights_per_s")
        if fixed_camera and body_m and mv and steps >= 4:
            movement["approx_step_length_m"] = (mv * body_m) / max(1e-6, steps / valid_s)

    # ---- events
    events: dict = {}
    reach = np.maximum(
        (P(L_SH)[:, 1] - P(L_WR)[:, 1]) / torso_len, (P(R_SH)[:, 1] - P(R_WR)[:, 1]) / torso_len
    )  # >0.1 means a wrist is clearly above its shoulder (image y grows downward)
    over = _runs(np.nan_to_num(reach, nan=-1) > 0.1)
    events["overhead_reach"] = {"count": len(over), "max_reach_in_torso_lengths": float(np.nanmax(reach)) if np.isfinite(reach).any() else None}

    rel_l, rel_r = P(L_AN) - hip_mid, P(R_AN) - hip_mid
    sw = np.fmax(
        np.r_[np.nan, np.linalg.norm(np.diff(rel_l, axis=0), axis=1)],
        np.r_[np.nan, np.linalg.norm(np.diff(rel_r, axis=0), axis=1)],
    ) * eff_fps / body_h
    swf = np.nan_to_num(sw, nan=0.0)
    pk, _ = find_peaks(swf, height=5.0, distance=max(1, int(0.4 * eff_fps)))
    events["fast_leg_swings"] = {
        "count": int(len(pk)),
        "peak_speed_body_heights_per_s": float(swf[pk].max()) if len(pk) else None,
    }

    knee_min = np.fmin(ang["knee_left"], ang["knee_right"])
    deep = _runs(np.nan_to_num(knee_min, nan=180) < 110)
    events["deep_knee_flexion"] = {"count": len(deep), "min_knee_angle_deg": float(np.nanmin(knee_min)) if np.isfinite(knee_min).any() else None}

    hy = _fill_short_gaps(hip_mid[:, 1], max_gap=6)
    jumps = {"count": 0, "peak_rise_body_heights": None}
    if np.isfinite(hy).sum() > 10:
        hy_f = np.where(np.isnan(hy), np.nanmedian(hy), hy)
        w = max(5, int(1.5 * eff_fps) | 1)
        pad = np.pad(hy_f, (w // 2, w // 2), mode="edge")
        base = np.array([np.median(pad[i : i + w]) for i in range(len(hy_f))])
        rise = (base - hy_f) / body_h
        jp, _ = find_peaks(rise, height=0.10, distance=max(1, int(0.5 * eff_fps)))
        jumps = {"count": int(len(jp)), "peak_rise_body_heights": float(rise[jp].max()) if len(jp) else None}
        if len(jp) and body_m:
            jumps["peak_rise_cm_approx"] = float(rise[jp].max() * body_m * 100)
    events["jumps"] = jumps

    # ---- heuristic score
    lean_std = joint_stats["trunk_lean_from_vertical"].get("std") if joint_stats["trunk_lean_from_vertical"]["available"] else None
    offs = (hip_mid[:, 0] - _mid(P(L_AN), P(R_AN))[:, 0]) / body_h
    offs_std = float(np.nanstd(offs)) if np.isfinite(offs).sum() > 5 else None
    scores: dict = {}
    if lean_std is not None and offs_std is not None:
        scores["posture_stability"] = {
            "value": float(np.clip(100 - 3.0 * lean_std - 120 * offs_std, 0, 100)),
            "heuristic": True,
            "basis": "variability of trunk lean and of hip-over-feet offset; indicative only, not a clinical measure",
        }

    notes = []
    if not fixed_camera:
        notes.append("Footage is not from a fixed camera: speed and distance are relative and may be distorted by camera movement.")
    if not height_cm:
        notes.append("No registered height, so real-world units are unavailable.")
    if avg_conf < 0.6:
        notes.append("Average pose-detection confidence is low (distance, motion blur, lighting or occlusion); treat fine details cautiously.")

    return _clean({
        "meta": meta, "joint_angles": joint_stats, "symmetry": symmetry, "movement": movement,
        "events": events, "scores": scores, "notes": notes,
    })


def history_summary(metrics: dict) -> dict:
    """The handful of numbers worth comparing session to session."""
    m = metrics or {}
    ja, mv, ev = m.get("joint_angles", {}), m.get("movement", {}), m.get("events", {})
    get = lambda d, *ks: (d.get(ks[0], {}) or {}).get(ks[1]) if len(ks) == 2 else d.get(ks[0])  # noqa: E731
    return {
        "knee_left_rom": get(ja, "knee_left", "range"),
        "knee_right_rom": get(ja, "knee_right", "range"),
        "trunk_lean_mean": get(ja, "trunk_lean_from_vertical", "mean"),
        "avg_speed_bh_s": mv.get("avg_speed_body_heights_per_s"),
        "cadence_spm": mv.get("cadence_steps_per_min"),
        "posture_stability": ((m.get("scores") or {}).get("posture_stability") or {}).get("value"),
        "jumps": (ev.get("jumps") or {}).get("count"),
        "overhead_reach": (ev.get("overhead_reach") or {}).get("count"),
    }
