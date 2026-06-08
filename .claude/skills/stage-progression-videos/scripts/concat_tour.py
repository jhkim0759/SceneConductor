"""Concatenate stage clips into one seamless tour mp4.

    python concat_tour.py <out.mp4> <clip1.mp4> <clip2.mp4> ...

Every input is normalised to 1120x840 (center-crop if larger, e.g. an
un-post-cropped 1160x880 tour clip) and written back-to-back at 24 fps. Inputs
must already share boundary camera poses for the joins to look seamless.
"""
import sys
import cv2

TW, TH, FPS = 1120, 840, 24

out = sys.argv[1]
clips = sys.argv[2:]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (TW, TH))
total = 0
for p in clips:
    cap = cv2.VideoCapture(p)
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if (w, h) != (TW, TH):
            x0 = max((w - TW) // 2, 0)
            y0 = max((h - TH) // 2, 0)
            fr = fr[y0:y0 + TH, x0:x0 + TW]
            if fr.shape[:2] != (TH, TW):
                fr = cv2.resize(fr, (TW, TH))
        vw.write(fr)
        n += 1
    cap.release()
    total += n
    print(f"  + {p.split('/')[-1]}: {n} frames")
vw.release()
print(f"concat done -> {out}  ({total} frames, {total / FPS:.1f}s @ {FPS}fps)")
