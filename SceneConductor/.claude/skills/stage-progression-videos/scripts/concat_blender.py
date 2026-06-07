"""Lossless-quality concat via Blender's VSE (no OpenCV mp4v re-encode). Loads the
ordered clips back-to-back on one channel and renders the combined timeline once
with H.264 PERC_LOSSLESS (SPV_CRF). All clips must share resolution + fps.

    blender -b --python concat_blender.py -- <out.mp4> <W> <H> <clip1> <clip2> ...
"""
import bpy, sys, os

tail = sys.argv[sys.argv.index("--") + 1:]
out = tail[0]
W = int(tail[1]); H = int(tail[2])
clips = tail[3:]
FPS = 24

scene = bpy.context.scene
scene.render.fps = FPS
if scene.sequence_editor:
    scene.sequence_editor_clear()
se = scene.sequence_editor_create()

f = 1
for i, clip in enumerate(clips):
    if not os.path.exists(clip):
        print(f"[concat] MISSING {clip}, skipping"); continue
    strip = se.sequences.new_movie(name=f"c{i}", filepath=clip, channel=1, frame_start=f)
    f += strip.frame_final_duration
    print(f"[concat] + {os.path.basename(clip)} ({strip.frame_final_duration}f) -> ends {f-1}")

scene.frame_start = 1
scene.frame_end = max(f - 1, 1)
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = os.environ.get("SPV_CRF", "PERC_LOSSLESS")
scene.render.ffmpeg.ffmpeg_preset = os.environ.get("SPV_FFPRESET", "GOOD")
scene.render.ffmpeg.audio_codec = "NONE"
scene.render.use_sequencer = True
scene.render.use_compositing = False
scene.render.filepath = out
print(f"[concat] {W}x{H} frames 1..{scene.frame_end} -> {out}")
bpy.ops.render.render(animation=True)
print("[concat] done")
