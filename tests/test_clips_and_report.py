import cv2
import numpy as np

from app.features.pipeline.clips import build_segments, write_student_clip
from app.features.pipeline.metrics import compute_metrics
from app.features.pipeline.report_llm import ReportContent, _extract_json, flatten_for_columns, rule_based_report
from app.features.pipeline.video_utils import probe_video, sample_frames
from tests.synthetic import stick_track
from tests.test_identity import make_track


def make_video(path, w=640, h=360, fps=30, seconds=4):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(fps * seconds)):
        f = np.full((h, w, 3), 40, dtype=np.uint8)
        x = 50 + i * 4
        cv2.rectangle(f, (x, 100), (x + 60, 300), (0, 200, 0), -1)
        vw.write(f)
    vw.release()


def test_segments_bridge_small_gaps_and_drop_blips():
    frames = list(range(0, 100, 2)) + list(range(110, 200, 2)) + [400, 402] + list(range(600, 700, 2))
    segs = build_segments(frames, stride=2, fps=30)
    assert segs[0][0] == 0 and segs[0][1] >= 196  # 10-frame gap bridged
    assert all(b - a > 40 for a, b in segs)       # the 2-frame blip is gone
    assert len(segs) == 2


def test_probe_and_sample(tmp_path):
    p = tmp_path / "v.mp4"; make_video(p)
    info = probe_video(str(p))
    assert (info.width, info.height) == (640, 360) and 3.5 < info.duration_s < 4.5
    assert len(sample_frames(str(p), n=5)) == 5


def test_clip_is_cropped_zoomed_square_and_follows_subject(tmp_path):
    p = tmp_path / "v.mp4"; make_video(p)
    info = probe_video(str(p))
    tr = make_track(1, list(range(0, 118, 2)))
    tr.boxes = [np.array([50 + f * 4, 100, 110 + f * 4, 300], dtype=np.float32) for f in tr.frames]
    segs = build_segments(tr.frames, 2, info.fps)
    out = tmp_path / "clip.mp4"
    res = write_student_clip(str(p), str(out), tr, segs, info, out_size=256)
    assert res.n_frames > 100 and res.seconds > 3
    cap = cv2.VideoCapture(str(out)); ok, fr = cap.read()
    assert ok and fr.shape[:2] == (256, 256)
    green = ((fr[:, :, 1] > 150) & (fr[:, :, 0] < 80)).mean()
    assert green > 0.10  # subject fills a meaningful part of the crop (was ~9% of the full frame)


def test_rule_based_report_uses_only_measured_numbers():
    m = compute_metrics(stick_track(seconds=10, asym=True), 2, 30, 170)
    rep = rule_based_report("Soccer", m)
    assert rep.generated_by == "rule_based" and rep.key_metrics and rep.scope_of_improvement
    assert any("Left/right" in i.area for i in rep.scope_of_improvement)  # asymmetry finding triggered
    motion, tech, scope = flatten_for_columns(rep)
    assert motion and tech and scope


def test_unusable_data_gives_honest_report():
    rep = rule_based_report("Soccer", compute_metrics(stick_track(seconds=1), 2, 30, 170))
    assert "Not enough" in rep.headline and not rep.key_metrics


def test_llm_json_extraction_tolerates_fences():
    txt = '```json\n{"headline":"h","motion_tracking_analysis":"m","technique_analysis":{"summary":"s","observations":[]},"scope_of_improvement":[]}\n```'
    rep = ReportContent(**_extract_json(txt))
    assert rep.headline == "h"
