"""Synthetic athletes: stick figures with known motion, so metric code can be checked against ground truth."""
import numpy as np

from app.features.pipeline.tracking import Track


def stick_track(seconds=8.0, fps=30, stride=2, speed_px_s=200.0, step_hz=1.0, arm_up=False, jumps=0,
                asym=False, conf=0.9, track_id=1, x0=300.0, base_y=500.0) -> Track:
    n = int(seconds * fps / stride)
    tr = Track(track_id=track_id)
    for i in range(n):
        t = i * stride / fps
        hx = x0 + speed_px_s * t
        hy = base_y
        if jumps:
            # a jump every 2s: 0.5s smooth bump of 60px
            ph = t % 2.0
            if ph < 0.5:
                hy -= 60 * np.sin(np.pi * ph / 0.5)
        s = np.sin(2 * np.pi * step_hz * t)
        la_x, ra_x = hx + 50 * s, hx - 50 * s
        bend_l = 25 + 20 * abs(s)
        bend_r = (25 + 20 * abs(s)) * (0.4 if asym else 1.0)
        k = np.zeros((17, 3), dtype=np.float32)
        k[:, 2] = conf
        k[0, :2] = (hx, hy - 170); k[1, :2] = (hx - 8, hy - 176); k[2, :2] = (hx + 8, hy - 176)
        k[3, :2] = (hx - 16, hy - 172); k[4, :2] = (hx + 16, hy - 172)
        k[5, :2] = (hx - 30, hy - 130); k[6, :2] = (hx + 30, hy - 130)
        wr_y = hy - 200 if arm_up and (t % 2.0) < 0.6 else hy - 40
        k[7, :2] = (hx - 40, hy - 90); k[8, :2] = (hx + 40, hy - 90)
        k[9, :2] = (hx - 45, wr_y); k[10, :2] = (hx + 45, wr_y)
        k[11, :2] = (hx - 20, hy); k[12, :2] = (hx + 20, hy)
        k[13, :2] = (hx - 20 + (la_x - hx) / 2 + bend_l, hy + 90)
        k[14, :2] = (hx + 20 + (ra_x - hx) / 2 - bend_r, hy + 90)
        k[15, :2] = (la_x - 20, hy + 180); k[16, :2] = (ra_x + 20, hy + 180)
        tr.frames.append(i * stride)
        tr.kpts.append(k)
        tr.boxes.append(np.array([hx - 70, hy - 200, hx + 70, hy + 190], dtype=np.float32))
    return tr
