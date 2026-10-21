import numpy as np

from app.features.pipeline.metrics import compute_metrics, history_summary
from tests.synthetic import stick_track


def test_speed_and_cadence_match_ground_truth():
    tr = stick_track(seconds=10, speed_px_s=200, step_hz=1.0)
    m = compute_metrics(tr, stride=2, fps=30, height_cm=170, fixed_camera=True)
    assert m["meta"]["usable"]
    mv = m["movement"]
    # body height ~390px (P75 of box heights), 200px/s -> ~0.51 body heights/s
    assert 0.45 < mv["avg_speed_body_heights_per_s"] < 0.60
    assert 0.75 < mv["avg_speed_m_s_approx"] < 1.05
    # foot passes the other twice per second -> ~120 steps/min
    assert 100 <= mv["cadence_steps_per_min"] <= 135
    assert mv["speed_reliable"] is True


def test_joint_angles_and_asymmetry_detected():
    sym = compute_metrics(stick_track(asym=False), 2, 30, 170)
    asy = compute_metrics(stick_track(asym=True), 2, 30, 170)
    assert sym["joint_angles"]["knee_left"]["available"]
    assert asy["symmetry"]["knee_rom_asymmetry_pct"] > sym["symmetry"]["knee_rom_asymmetry_pct"] + 15


def test_events_jumps_and_overhead():
    m = compute_metrics(stick_track(seconds=10, jumps=1, arm_up=True), 2, 30, 170)
    assert m["events"]["jumps"]["count"] >= 3
    assert m["events"]["overhead_reach"]["count"] >= 3
    none = compute_metrics(stick_track(seconds=10), 2, 30, 170)
    assert none["events"]["jumps"]["count"] == 0
    assert none["events"]["overhead_reach"]["count"] == 0


def test_handheld_footage_flags_speed_unreliable():
    m = compute_metrics(stick_track(), 2, 30, None, fixed_camera=False)
    assert m["movement"]["speed_reliable"] is False
    assert "avg_speed_m_s_approx" not in m["movement"]
    assert any("fixed camera" in n for n in m["notes"])


def test_too_little_data_is_flagged_unusable_not_invented():
    m = compute_metrics(stick_track(seconds=1.0), 2, 30, 170)
    assert m["meta"]["usable"] is False
    assert m["joint_angles"] == {} and m["movement"] == {}


def test_low_confidence_keypoints_are_ignored():
    m = compute_metrics(stick_track(conf=0.2), 2, 30, 170)
    assert m["meta"]["usable"] is False


def test_output_is_json_serialisable_and_history_summary_works():
    import json
    m = compute_metrics(stick_track(), 2, 30, 170)
    json.dumps(m)
    h = history_summary(m)
    assert "cadence_spm" in h and h["cadence_spm"] is not None
