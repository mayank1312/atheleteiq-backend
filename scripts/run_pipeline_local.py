"""
Try the whole vision pipeline on a video file - no database, no Celery, no Supabase needed.

  # one person (like a self-upload): finds you by face, falls back to the most prominent player
  python -m scripts.run_pipeline_local --video ~/Desktop/game.mp4 --face ~/Desktop/face.jpg --sport Soccer --height 170

  # group footage: pass every student's face, get a clip + report for each one it recognises
  python -m scripts.run_pipeline_local --video group.mp4 --group --face asha=asha.jpg --face ravi=ravi.jpg --sport Badminton

Writes to ./out/ : a cropped clip per student, metrics.json and report.json. Add ANTHROPIC_API_KEY to .env
for the AI-written report; without it you get the rule-based fallback.
"""
import argparse
import json
import os
import time

from app.features.pipeline.faces import embed_from_bytes
from app.features.pipeline.identity import RosterEntry
from app.features.pipeline.orchestrator import analyze_video
from app.features.pipeline.report_llm import generate_report

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--face", action="append", default=[], help="path, or name=path (repeatable)")
    ap.add_argument("--group", action="store_true", help="academy mode: every recognised student, no fallback")
    ap.add_argument("--sport", default="Soccer")
    ap.add_argument("--height", type=float, default=None, help="athlete height in cm (enables approximate m/s)")
    ap.add_argument("--fixed-camera", action="store_true", help="footage comes from a fixed camera (speed becomes reliable)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    names, roster = {}, []
    for i, spec in enumerate(a.face, start=1):
        name, path = spec.split("=", 1) if "=" in spec else (f"student{i}", spec)
        emb, n = embed_from_bytes(open(os.path.expanduser(path), "rb").read())
        v = np.asarray(emb, dtype=np.float32)
        roster.append(RosterEntry(i, v / np.linalg.norm(v)))
        names[i] = name
        print(f"enrolled {name} ({n} face in photo)")
    only = None if a.group else 1
    if not roster and not a.group:
        names[1] = "athlete"

    t0 = time.time()
    res = analyze_video(
        os.path.expanduser(a.video), roster, a.out, {i: a.height for i in names}, fixed_camera=a.fixed_camera,
        only_student=only, on_progress=lambda p: print(f"\r  tracking {p:4.0%}", end="", flush=True),
    )
    print(f"\nvideo {res.info.width}x{res.info.height} {res.info.fps:.0f}fps {res.info.duration_s:.0f}s | "
          f"{res.debug.get('tracks_found', 0)} people tracked | {time.time() - t0:.0f}s\n")
    for tid, m in res.debug.get("matches", {}).items():
        who = names.get(m["student"], "-- unmatched --")
        print(f"  track {tid:>3}: {who:<16} similarity={m['sim']:.2f} margin={m['margin']:.2f} faces_used={m['faces']}")
    if not res.students:
        print("\nNo student could be analysed. Check the face photo, or that the person is clearly visible.")
        return
    for sa in res.students:
        name = names.get(sa.student_id, str(sa.student_id))
        print(f"\n== {name}: identity={sa.identity_confidence}, tracks={sa.track_ids}, clip={sa.clip.seconds:.0f}s -> {sa.clip.path}")
        rep = generate_report(a.sport, {"height_cm": a.height}, sa.metrics, "script", sa.identity_confidence, [])
        with open(os.path.join(a.out, f"metrics_{name}.json"), "w") as f:
            json.dump(sa.metrics, f, indent=2)
        with open(os.path.join(a.out, f"report_{name}.json"), "w") as f:
            json.dump(rep.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"   [{rep.generated_by}] {rep.headline}")
        for k in rep.key_metrics:
            print(f"     - {k.label}: {k.value}")
    print(f"\nFiles written to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
