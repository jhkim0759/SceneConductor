"""G1 island refinement: linear move from FIRST to LAST iter, render persp or BEV.

    blender -b <iter1/island.blend> --python build_island_2view.py -- \
        <out_mp4> <start.json> <end.json> --view persp|bev [--samples N]

Only the start (iter_1) and end (iter_last) obj_ controller transforms are used;
motion is a single LINEAR interpolation between them (with short holds).
"""
import os
import bpy, sys, json, math
from mathutils import Vector

argv = sys.argv
tail = argv[argv.index("--") + 1:] if "--" in argv else []
view = "persp"; samples = 48
if "--view" in tail:
    i = tail.index("--view"); view = tail[i + 1]; tail = tail[:i] + tail[i + 2:]
if "--samples" in tail:
    i = tail.index("--samples"); samples = int(tail[i + 1]); tail = tail[:i] + tail[i + 2:]
out_mp4, start_json, end_json = tail[0], tail[1], tail[2]

start = json.load(open(start_json))["objects"]
end = json.load(open(end_json))["objects"]
names = sorted([n for n in start if n.startswith("obj_")],
               key=lambda x: int(x[4:]) if x[4:].isdigit() else 0)

FPS = 24
HOLD_A, MOVE, HOLD_B = 12, 60, 20
f1 = 1
f2 = f1 + HOLD_A
f3 = f2 + MOVE
f4 = f3 + HOLD_B
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = f4
scene.render.fps = FPS

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

for name in names:
    obj = bpy.data.objects.get(name)
    if obj is None:
        continue
    key(obj, f1, start[name]); key(obj, f2, start[name])
    key(obj, f3, end.get(name, start[name])); key(obj, f4, end.get(name, start[name]))

for act in bpy.data.actions:
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

# --- island mesh bbox (for BEV framing) ---
pts = []
for o in bpy.data.objects:
    if o.type == "MESH":
        for c in o.bound_box:
            pts.append(o.matrix_world @ Vector(c))
bpy.context.view_layer.update()
if pts:
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
else:
    mn, mx = Vector((-1.5, -1.5, 0)), Vector((1.5, 1.5, 1))
ctr = (mn + mx) * 0.5
size = mx - mn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flatcam
flatcam.white_bg(scene)

if view == "bev":
    cam_data = bpy.data.cameras.new("BevCam"); cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(size.x, size.y) * 1.15
    cam = bpy.data.objects.new("BevCam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = Vector((ctr.x, ctr.y, mx.z + 5.0))
    cam.rotation_euler = (0.0, 0.0, 0.0)   # looks straight down -Z
    cam_data.clip_start = 0.01; cam_data.clip_end = 50.0
    scene.camera = cam
else:
    cam = scene.camera or next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
    scene.camera = cam

flatcam.set_engine(scene, max(samples, 64))
flatcam.crop(scene)
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "HIGH"
scene.render.filepath = out_mp4
if scene.use_nodes and scene.node_tree:
    for nd in scene.node_tree.nodes:
        if nd.type == "OUTPUT_FILE":
            nd.mute = True
if os.environ.get("SPV_SAVE_BLEND", "1") != "0":
    bpy.ops.wm.save_as_mainfile(filepath=out_mp4.rsplit(".", 1)[0] + ".blend")
print(f"[island2v] view={view} objs={len(names)} frames 1..{f4} -> {out_mp4}")
bpy.ops.render.render(animation=True)
print("[island2v] done")
