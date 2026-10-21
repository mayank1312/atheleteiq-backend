import numpy as np

from app.features.pipeline.identity import RosterEntry, dominant_track, match_tracks
from app.features.pipeline.tracking import FaceCandidate, Track


def unit(seed, dim=64):
    v = np.random.default_rng(seed).normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def noisy(base, seed, amt=0.15):
    v = base + np.random.default_rng(seed).normal(scale=amt / np.sqrt(len(base)), size=len(base))
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEmbedder:
    """Crop pixel value encodes which embedding the 'face' has."""
    def __init__(self):
        self.table = {}

    def register(self, key, vec):
        self.table[key] = vec

    def __call__(self, crop):
        key = int(crop.flat[0])
        return (self.table[key], 0.99) if key in self.table else None


def make_track(tid, frames, faces=(), box=(0, 0, 100, 300)):
    t = Track(track_id=tid)
    t.frames = list(frames)
    t.boxes = [np.array(box, dtype=np.float32) for _ in frames]
    t.kpts = [np.zeros((17, 3), dtype=np.float32) for _ in frames]
    for i, key in enumerate(faces):
        t.add_face(FaceCandidate(score=0.9 - i * 0.01, frame_idx=i, crop=np.full((4, 4, 3), key, dtype=np.uint8)))
    return t


def setup():
    a, b = unit(1), unit(2)
    emb = FakeEmbedder()
    emb.register(10, noisy(a, 11)); emb.register(11, noisy(a, 12))
    emb.register(20, noisy(b, 21)); emb.register(21, noisy(b, 22))
    roster = [RosterEntry(101, a), RosterEntry(202, b)]
    return roster, emb


def test_tracks_matched_to_correct_students():
    roster, emb = setup()
    tracks = {1: make_track(1, range(0, 200, 2), faces=[10, 11]), 2: make_track(2, range(0, 200, 2), faces=[20, 21])}
    m = match_tracks(tracks, roster, embed=emb)
    assert m[1].student_id == 101 and m[2].student_id == 202


def test_track_with_no_visible_face_stays_unlabelled():
    roster, emb = setup()
    m = match_tracks({1: make_track(1, range(0, 100, 2), faces=[])}, roster, embed=emb)
    assert m[1].student_id is None and m[1].n_faces == 0


def test_stranger_below_threshold_is_not_assigned():
    roster, emb = setup()
    emb.register(30, unit(999))  # unrelated face
    m = match_tracks({1: make_track(1, range(0, 100, 2), faces=[30])}, roster, embed=emb)
    assert m[1].student_id is None


def test_fragments_of_same_student_both_kept_when_not_overlapping():
    roster, emb = setup()
    tracks = {1: make_track(1, range(0, 100, 2), faces=[10]), 3: make_track(3, range(200, 300, 2), faces=[11])}
    m = match_tracks(tracks, roster, embed=emb)
    assert m[1].student_id == 101 and m[3].student_id == 101


def test_face_match_recovers_nearby_faceless_fragment_by_appearance():
    roster, emb = setup()
    anchor = make_track(1, range(0, 30), faces=[10])
    fragment = make_track(3, range(45, 75))
    descriptor = np.array([1.0, 0.0], dtype=np.float32)
    anchor.add_appearance(descriptor)
    fragment.add_appearance(descriptor)

    m = match_tracks({1: anchor, 3: fragment}, roster, embed=emb)

    assert m[1].student_id == 101
    assert m[3].student_id == 101 and m[3].source == "appearance_linked"


def test_jersey_colour_alone_does_not_bridge_a_long_gap():
    roster, emb = setup()
    anchor = make_track(1, range(0, 30), faces=[10])
    fragment = make_track(3, range(200, 230))
    descriptor = np.array([1.0, 0.0], dtype=np.float32)
    anchor.add_appearance(descriptor)
    fragment.add_appearance(descriptor)

    m = match_tracks({1: anchor, 3: fragment}, roster, embed=emb)

    assert m[3].student_id is None


def test_two_simultaneous_tracks_cannot_both_be_one_student():
    roster, emb = setup()
    tracks = {1: make_track(1, range(0, 200, 2), faces=[10]), 3: make_track(3, range(0, 200, 2), faces=[11])}
    m = match_tracks(tracks, roster, embed=emb)
    assert sorted([m[1].student_id is None, m[3].student_id is None]) == [False, True]


def test_lookalike_rejected_by_margin():
    a = unit(1)
    twin = noisy(a, 5, amt=0.3)
    roster = [RosterEntry(1, a), RosterEntry(2, twin)]
    emb = FakeEmbedder(); emb.register(10, noisy(a, 7, amt=0.05))
    m = match_tracks({1: make_track(1, range(0, 100, 2), faces=[10])}, roster, threshold=0.3, margin=0.5, embed=emb)
    assert m[1].student_id is None  # too close to call -> refuse rather than mislabel


def test_dominant_track_picks_longest_largest():
    tracks = {1: make_track(1, range(0, 40, 2), box=(0, 0, 50, 100)), 2: make_track(2, range(0, 200, 2), box=(0, 0, 100, 300))}
    assert dominant_track(tracks) == 2
