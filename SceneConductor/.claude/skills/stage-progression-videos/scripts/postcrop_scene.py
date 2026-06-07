"""Idempotent post-crop of a scene's full-video clip set to 1120x840 (20px/edge
off a 1160x880 base). Skips clips already <=1120 wide. File-based (nohup-safe).
    python postcrop_scene.py <scene_dir>
"""
import cv2, os, sys
SCENE = sys.argv[1].rstrip("/")
SV = f"{SCENE}/report/stage_videos"
PX = 20
CLIPS = ["1_stage1_popup", "2_env_build_interior", "2-1_stage2_external_build",
         "3_stage3_first_refine", "3-1_stage3_to_final", "5_stage3_final_turntable"]
for f in CLIPS:
    p = f"{SV}/{f}.mp4"
    if not os.path.exists(p):
        continue
    c = cv2.VideoCapture(p); fps = c.get(5) or 24
    w = int(c.get(3)); h = int(c.get(4))
    if w <= 1120:
        print(f"{f}: already {w}x{h}, skip"); c.release(); continue
    nw, nh = w - 2 * PX, h - 2 * PX
    t = p + ".t.mp4"
    vw = cv2.VideoWriter(t, cv2.VideoWriter_fourcc(*"mp4v"), fps, (nw, nh)); n = 0
    while True:
        ok, fr = c.read()
        if not ok:
            break
        vw.write(fr[PX:h - PX, PX:w - PX]); n += 1
    c.release(); vw.release()
    if n > 0:
        os.replace(t, p); print(f"{f}: {w}x{h} -> {nw}x{nh} ({n}f)")
    else:
        os.remove(t); print(f"{f}: 0 frames!")
print("postcrop_scene done")
