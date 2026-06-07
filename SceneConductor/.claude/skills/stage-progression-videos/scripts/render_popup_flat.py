"""Render the popup phase (frames 1..END) with a flattened interior camera.

    blender -b <demo_animated.blend> --python render_popup_flat.py -- <out_mp4> <start> <end> [samples]
"""
import os
import bpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flatcam

tail = sys.argv[sys.argv.index("--") + 1:]
out_mp4 = tail[0]; f_start = int(tail[1]); f_end = int(tail[2])
samples = int(tail[3]) if len(tail) > 3 else 32

scene = bpy.context.scene
scene.frame_start = f_start
scene.frame_end = f_end
# clip 1 = Stage-1 object pop-in ONLY. Hide the room shell (Floor/Walls/Ceiling) so
# no walls appear here (the demo_animated.blend fades walls in within these frames,
# which leaked into clip 1 as a "walls pre-built" bug). The shell belongs to clip 2.
# With walls hidden, clip 1's last frame (furniture on white) == clip 2's first
# frame (furniture, walls still scale 0) -> seamless cut.
for _nm in ("Floor", "Wall_01", "Wall_02", "Wall_03", "Wall_04", "Ceiling"):
    _o = bpy.data.objects.get(_nm)
    if _o:
        _o.hide_render = True
        _o.hide_viewport = True
        # also kill any fade/scale animation so it can't re-appear mid-clip
        try:
            _o.animation_data_clear()
        except Exception:
            pass
flatcam.white_bg(scene)
# clip 1 is lit by the blend's OWN (original) room lights at full energy (white_bg
# ORIGINAL mode keeps them on). clip 2 uses the SAME full original lighting from
# frame 1 (no lights-after-build reveal), so clip 1's end and clip 2's start match
# (user choice A: keep clip 1's bright original lighting on both, drop the reveal).
# Keep the ORIGINAL camera pose + lens (same reference view as clips 2/3/3-1);
# just clear any camera animation so the pose is static.
if scene.camera:
    scene.camera.animation_data_clear()
    if scene.camera.data.animation_data:
        scene.camera.data.animation_data_clear()
    # Optional wider FOV: scale the original lens (SPV_FOCAL_SCALE < 1 -> wider).
    _fs = float(os.environ.get("SPV_FOCAL_SCALE", "1.0"))
    if _fs != 1.0:
        scene.camera.data.lens *= _fs
flatcam.crop(scene)
flatcam.set_engine(scene, max(samples, 64))
scene.render.fps = 24
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = os.environ.get("SPV_CRF", "HIGH")
scene.render.ffmpeg.ffmpeg_preset = os.environ.get("SPV_FFPRESET", "GOOD")
scene.render.filepath = out_mp4
scene.render.use_compositing = False   # avoid blend compositor (black under Cycles)
if scene.use_nodes and scene.node_tree:
    for n in scene.node_tree.nodes:
        if n.type == "OUTPUT_FILE": n.mute = True
if os.environ.get("SPV_SAVE_BLEND", "1") != "0":
    bpy.ops.wm.save_as_mainfile(filepath=out_mp4.rsplit(".", 1)[0] + ".blend")
print(f"[popup_flat] frames {f_start}..{f_end} -> {out_mp4}")
bpy.ops.render.render(animation=True)
print("[popup_flat] done")
