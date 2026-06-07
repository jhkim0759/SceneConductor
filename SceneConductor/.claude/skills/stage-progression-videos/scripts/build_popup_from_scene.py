"""Synthetic Stage-1 popup straight from a walled scene blend (no demo_animated
.blend needed). Furniture pops in one-by-one (scale 0 -> full, staggered) on the
white background, lit by the blend's own lights, from the ORIGINAL reference
camera (+ optional SPV_FOCAL_SCALE). Shell (Floor/Walls/Ceiling) is hidden so the
objects float on white exactly like the demo-based clip 1 — and its LAST frame
(all furniture present, no walls) matches clip 2's first frame for a seamless cut.

    blender -b <blender_scene.blend> --python build_popup_from_scene.py -- \
        <out_mp4> <f_start> <f_end> [samples]
"""
import os, sys, bpy, math
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flatcam

tail = sys.argv[sys.argv.index("--") + 1:]
out_mp4 = tail[0]; f_start = int(tail[1]); f_end = int(tail[2])
samples = int(tail[3]) if len(tail) > 3 else 48

def _is_shell(nm):
    # match ANY number of walls (Wall_01..Wall_NN), floor, ceiling — by prefix,
    # not a fixed Wall_01..04 set (e.g. hairsalon has Wall_01..Wall_06).
    n = nm.lower()
    return n.startswith("floor") or n.startswith("wall") or n.startswith("ceiling")

scene = bpy.context.scene
scene.frame_start = f_start
scene.frame_end = f_end

pc = bpy.data.objects.get("PointCloud_XZ")
if pc:
    try: bpy.data.objects.remove(pc, do_unlink=True)
    except Exception: pass

# hide the room shell (ALL floor/wall/ceiling meshes) so furniture floats on white
for o in bpy.data.objects:
    if o.type == "MESH" and _is_shell(o.name):
        o.hide_render = True
        o.hide_viewport = True

flatcam.white_bg(scene)

# camera: original reference pose/lens (+ optional wider FOV), static
cam = scene.camera or next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
scene.camera = cam
cam.animation_data_clear()
if cam.data.animation_data:
    cam.data.animation_data_clear()
cam.data.clip_start = 0.01
cam.data.clip_end = max(cam.data.clip_end, 300.0)
_fs = float(os.environ.get("SPV_FOCAL_SCALE", "1.0"))
if _fs != 1.0:
    cam.data.lens *= _fs

# furniture = top-level controllers/meshes that are NOT shell, lights or camera
furn = [o for o in bpy.data.objects
        if o.parent is None and not _is_shell(o.name)
        and o.type in ("EMPTY", "MESH")
        and not o.name.startswith("PointCloud")]
# pop order: obj_ by index, then the rest
def _key(o):
    if o.name.startswith("obj_") and o.name[4:].isdigit():
        return (0, int(o.name[4:]))
    return (1, o.name)
furn.sort(key=_key)

N = max(len(furn), 1)
span = max(f_end - f_start - 10, 1)
RISE = 8
for i, o in enumerate(furn):
    orig = tuple(o.scale)
    if orig == (0.0, 0.0, 0.0):
        orig = (1.0, 1.0, 1.0)
    s0 = f_start + int(span * i / N)
    s1 = s0 + RISE
    o.scale = (0, 0, 0); o.keyframe_insert("scale", frame=f_start)
    o.scale = (0, 0, 0); o.keyframe_insert("scale", frame=s0)
    o.scale = orig;      o.keyframe_insert("scale", frame=s1)
print(f"[popup_scene] {N} furniture objects pop in over frames {f_start}..{f_end}")

for act in bpy.data.actions:
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

flatcam.crop(scene)
flatcam.set_engine(scene, max(samples, 64))
scene.render.fps = 24
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = os.environ.get("SPV_CRF", "HIGH")
scene.render.ffmpeg.ffmpeg_preset = os.environ.get("SPV_FFPRESET", "GOOD")
scene.render.filepath = out_mp4
scene.render.use_compositing = False
if scene.use_nodes and scene.node_tree:
    for nd in scene.node_tree.nodes:
        if nd.type == "OUTPUT_FILE":
            nd.mute = True
print(f"[popup_scene] rendering -> {out_mp4}")
bpy.ops.render.render(animation=True)
print("[popup_scene] done")
