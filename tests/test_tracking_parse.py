"""Feeds fake ultralytics-style results through track_people() to verify the parsing/aggregation code."""
import numpy as np

from app.features.pipeline import tracking
from app.features.pipeline.video_utils import VideoInfo


class T:  # tiny stand-in for a torch tensor
    def __init__(self, a): self.a = np.asarray(a)
    def int(self): return self
    def cpu(self): return self
    def numpy(self): return self.a
    def tolist(self): return self.a.tolist()


class Boxes:
    def __init__(self, ids, xyxy): self.id = None if ids is None else T(ids); self.xyxy = T(xyxy)


class Kpts:
    def __init__(self, d): self.data = T(d)


class Result:
    def __init__(self, ids, xyxy, kps, frame):
        self.boxes = Boxes(ids, xyxy); self.keypoints = Kpts(kps); self.orig_img = frame


class FakeModel:
    def __init__(self, results): self._r = results
    def track(self, **kw):
        self.kw = kw
        return iter(self._r)


def kp_with_face(cx, cy, eye=20):
    k = np.zeros((17, 3), dtype=np.float32)
    k[0] = (cx, cy + 5, 0.9); k[1] = (cx - eye / 2, cy, 0.9); k[2] = (cx + eye / 2, cy, 0.9)
    k[3] = (cx - eye, cy + 2, 0.8); k[4] = (cx + eye, cy + 2, 0.8)
    return k


def test_track_people_groups_by_id_and_collects_faces(monkeypatch):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    faceless = np.zeros((17, 3), dtype=np.float32)
    results = []
    for i in range(30):
        ids = [1, 2] if i % 5 else [1]  # person 2 flickers out every 5th frame
        xyxy = [[100, 50, 200, 350]] + ([[400, 60, 500, 360]] if len(ids) == 2 else [])
        kps = [kp_with_face(150, 90)] + ([faceless] if len(ids) == 2 else [])
        results.append(Result(ids, xyxy, np.stack(kps), frame))
    results.append(Result(None, np.zeros((0, 4)), np.zeros((0, 17, 3)), frame))  # frame with no tracked people
    fm = FakeModel(results)
    monkeypatch.setattr(tracking, "get_model", lambda: fm)

    info = VideoInfo(640, 480, 30.0, 62)
    tracks = tracking.track_people("dummy.mp4", info, stride=2)

    assert set(tracks) == {1, 2}
    assert len(tracks[1].frames) == 30
    assert tracks[1].frames[:3] == [0, 2, 4]            # stride respected in frame indices
    assert len(tracks[2].frames) == 24
    assert len(tracks[1].face_candidates) == tracking.MAX_FACE_CANDIDATES  # heap capped
    assert len(tracks[2].face_candidates) == 0         # no visible face -> no crops
    assert min(tracks[1].face_candidates[0].crop.shape[:2]) >= tracking.MIN_FACE_CROP_SIDE
    assert fm.kw["classes"] == [0] and fm.kw["vid_stride"] == 2 and fm.kw["stream"] is True


def test_merge_tracks_orders_and_dedupes():
    from tests.test_identity import make_track
    a, b = make_track(1, [0, 2, 4]), make_track(2, [4, 6, 8])
    merged = tracking.merge_tracks([b, a])
    assert merged.frames == [0, 2, 4, 6, 8]
