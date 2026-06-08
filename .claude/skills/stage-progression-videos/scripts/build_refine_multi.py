"""Multi-stage refine animation on the scene camera (square render).

    blender -b <base.blend> --python build_refine_multi.py -- \
        <out_mp4> <stage1.json> <stage2.json> [stage3.json ...] [--samples N]

Animates obj_ empties LINEARLY through the given stage transforms (with holds),
using the base blend's existing camera / lights / world. Base must have no baked
mesh offsets so stage empty poses reproduce the intended geometry.
"""
import os
import bpy, sys, json, math
from mathutils import Vector

tail = sys.argv[sys.argv.index("--") + 1:]
samples = 32
if "--samples" in tail:
    i = tail.index("--samples"); samples = int(tail[i + 1]); tail = tail[:i] + tail[i + 2:]
out_mp4 = tail[0]
stage_paths = tail[1:]
stages = [json.load(open(p))["objects"] for p in stage_paths]

FPS = 24
HOLD = 16          # hold at each stage
MOVE = 50          # transition between stages
scene = bpy.context.scene
scene.frame_start = 1

# frame for each stage keyframe
stage_frames = [1]
f = 1 + HOLD
for _ in stages[1:]:
    f += MOVE
    stage_frames.append(f)
    f += HOLD
scene.frame_end = f

_prev = {}
def _unwrap(name, tgt):
    p = _prev.get(name)
    if p is None:
        out = tuple(float(x) for x in tgt)
    else:
        out = tuple(a + ((float(b) - a + math.pi) % (2 * math.pi) - math.pi)
                    for a, b in zip(p, tgt))
    _prev[name] = out
    return out

def key(obj, frame, info):
    obj.location = Vector(info["location"]); obj.keyframe_insert("location", frame=frame)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = _unwrap(obj.name, info["rotation_euler"]); obj.keyframe_insert("rotation_euler", frame=frame)
    obj.scale = Vector(info["scale"]); obj.keyframe_insert("scale", frame=frame)

names = sorted([n for n in stages[0] if n.startswith("obj_")],
               key=lambda x: int(x[4:]) if x[4:].isdigit() else 0)
for name in names:
    obj = bpy.data.objects.get(name)
    if obj is None:
        continue
    prev_info = stages[0][name]
    key(obj, 1, prev_info)
    key(obj, 1 + HOLD, prev_info)
    for k, fr in list(enumerate(stage_frames))[1:]:
        info = stages[k].get(name, prev_info)
        key(obj, fr, info)
        key(obj, fr + HOLD, info)
        prev_info = info
print(f"[refine_multi] {len(names)} objs, {len(stages)} stages, frames 1..{scene.frame_end}")

if bpy.data.objects.get("PointCloud_XZ"):
    bpy.data.objects.remove(bpy.data.objects["PointCloud_XZ"], do_unlink=True)

for act in bpy.data.actions:
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flatcam
flatcam.white_bg(scene)

# Room bbox (for the interior fill light). The camera itself keeps its ORIGINAL
# reference pose (see below) so clips 1/2/3/3-1 all share one viewpoint.
_pts = []
_shell = [bpy.data.objects.get(n) for n in
          ("Floor", "Wall_01", "Wall_02", "Wall_03", "Wall_04", "Ceiling")]
_shell = [o for o in _shell if o and o.type == "MESH"]
_src = _shell if _shell else [o for o in bpy.data.objects
                              if o.type == "MESH" and not o.name.startswith("PointCloud")]
for o in _src:
    for c in o.bound_box:
        _pts.append(o.matrix_world @ Vector(c))
if _pts:
    _mn = Vector((min(p.x for p in _pts), min(p.y for p in _pts), min(p.z for p in _pts)))
    _mx = Vector((max(p.x for p in _pts), max(p.y for p in _pts), max(p.z for p in _pts)))
else:
    _mn, _mx = Vector((-5, 0, -2)), Vector((5, 14, 1))
_ctr = (_mn + _mx) * 0.5
_sz = _mx - _mn
# Interior fill light to keep the enclosed scene lit (Cycles doesn't leak the
# white world into a closed room). Skipped in ORIGINAL-lights mode where the
# blend's own lights illuminate the scene.
if os.environ.get("SPV_ORIGINAL_LIGHTS", "0") != "1":
    _ld = bpy.data.lights.new("InteriorLight", type="AREA")
    _ld.energy = 300.0 * flatcam.light_scale()
    _ld.size = max(_sz.x, _sz.y) * 0.5
    _il = bpy.data.objects.new("InteriorLight", _ld)
    _il.location = Vector((_ctr.x, _ctr.y, _mn.z + _sz.z * 0.78))
    scene.collection.objects.link(_il)
# Use the ORIGINAL scene camera view (same reference pose / lens as clips 1 & 2)
# so all refine stages share one viewpoint. The interior light + use_compositing
# =False below keep this origin camera lit under Cycles. Just clear animation.
cam = scene.camera or next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
scene.camera = cam
if cam.animation_data:
    cam.animation_data_clear()
if cam.data.animation_data:
    cam.data.animation_data_clear()
cam.data.clip_start = 0.01
cam.data.clip_end = max(cam.data.clip_end, 300.0)
# Optional wider FOV: scale the original lens (SPV_FOCAL_SCALE < 1 -> wider), same
# scale as clips 1/2 so the connected refine views share one field of view.
_fs = float(os.environ.get("SPV_FOCAL_SCALE", "1.0"))
if _fs != 1.0:
    cam.data.lens *= _fs

flatcam.crop(scene)
flatcam.set_engine(scene, max(samples, 64))
scene.render.fps = FPS
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = os.environ.get("SPV_CRF", "HIGH")
scene.render.ffmpeg.ffmpeg_preset = os.environ.get("SPV_FFPRESET", "GOOD")
scene.render.filepath = out_mp4
scene.render.use_compositing = False   # bypass blend's compositor (black under Cycles)
if scene.use_nodes and scene.node_tree:
    for nd in scene.node_tree.nodes:
        if nd.type == "OUTPUT_FILE":
            nd.mute = True
if os.environ.get("SPV_SAVE_BLEND", "1") != "0":
    bpy.ops.wm.save_as_mainfile(filepath=out_mp4.rsplit(".", 1)[0] + ".blend")
print(f"[refine_multi] rendering -> {out_mp4}")
bpy.ops.render.render(animation=True)
print("[refine_multi] done")
