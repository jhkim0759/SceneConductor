"""Continuous-tour clips 4 (group zoom) & 5 (near-BEV turntable) on the FULL
scene, with boundary-aligned cameras so 3 -> 4 -> 5 concatenate seamlessly.

    blender -b <blender_scene.blend> --python build_tour.py -- <out_mp4> \
        --phase zoom|turntable --group obj_2,obj_4,... \
        [--ginit g_init.json] [--gfinal g_final.json] [--samples N]

phase zoom (clip 4):  camera flies from the ORIGINAL ref cam (P0, == clip 3 end)
  IN to a 3/4 view of the relation group (P_island); the group's members animate
  from their init -> final island poses (others stay final). Ceiling kept.
phase turntable (clip 5):  camera starts at P_island (== zoom end), lifts up and
  back to a near-BEV, then orbits 360. Ceiling + ceiling fixtures hidden.

Both phases compute P_island identically from the group bbox + the ORIGINAL lens,
so clip 4's last frame and clip 5's first frame share the exact camera pose.
"""
import os
import sys
import json
import math
import bpy
from mathutils import Vector

tail = sys.argv[sys.argv.index("--") + 1:]


def opt(flag, d=None):
    return tail[tail.index(flag) + 1] if flag in tail else d


out_mp4 = tail[0]
phase = opt("--phase", "zoom")
group = [g for g in opt("--group", "").split(",") if g]
ginit = opt("--ginit")
gfinal = opt("--gfinal")
samples = int(opt("--samples", "64"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flatcam

scene = bpy.context.scene
FPS = 24
scene.render.fps = FPS

_pc = bpy.data.objects.get("PointCloud_XZ")
if _pc is not None:
    try:
        bpy.data.objects.remove(_pc, do_unlink=True)
    except Exception:
        pass

SHELL = ("Floor", "Wall_01", "Wall_02", "Wall_03", "Wall_04", "Ceiling")


def look_quat(loc, target):
    return (target - loc).normalized().to_track_quat("-Z", "Y")


def keep_sign(q, prev):
    """Flip q so it stays on the same hemisphere as prev (no interp flip)."""
    if prev is not None and q.dot(prev) < 0.0:
        q = q.copy(); q.negate()
    return q


# ---- original reference camera P0 ----
cam = scene.camera or next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
P0_loc = cam.location.copy()
P0_lens = cam.data.lens
P0_q = cam.rotation_euler.to_quaternion()
cam.rotation_mode = "QUATERNION"
cam.animation_data_clear()
if cam.data.animation_data:
    cam.data.animation_data_clear()
cam.data.clip_start = 0.01
cam.data.clip_end = 800.0

# ---- group bbox -> aim point + P_island (shared by both phases) ----
gpts = []
for n in group:
    o = bpy.data.objects.get(n)
    if o:
        for c in o.bound_box:
            gpts.append(o.matrix_world @ Vector(c))
if gpts:
    gmn = Vector((min(p.x for p in gpts), min(p.y for p in gpts), min(p.z for p in gpts)))
    gmx = Vector((max(p.x for p in gpts), max(p.y for p in gpts), max(p.z for p in gpts)))
else:
    gmn, gmx = Vector((-1, 6, -2)), Vector((1, 8, -1))
gctr = (gmn + gmx) * 0.5
gsize = gmx - gmn
aim = Vector((gctr.x, gctr.y, gctr.z - 0.05))               # ~furniture centre
hfov = 2.0 * math.atan((36.0 / 2.0) / P0_lens)

# Room (shell) bbox -> keep the close-up camera INSIDE the room (below the
# ceiling, within the walls) so it never flies above the ceiling into black,
# whatever the group size / position.
_rpts = []
for nm in SHELL:
    o = bpy.data.objects.get(nm)
    if o and o.type == "MESH":
        for c in o.bound_box:
            _rpts.append(o.matrix_world @ Vector(c))
if _rpts:
    rmn = Vector((min(p.x for p in _rpts), min(p.y for p in _rpts), min(p.z for p in _rpts)))
    rmx = Vector((max(p.x for p in _rpts), max(p.y for p in _rpts), max(p.z for p in _rpts)))
else:
    rmn, rmx = Vector((-8, -2, -3)), Vector((8, 18, 3))

# Distance to frame the group footprint, then approach from the ORIGINAL-camera
# side so the fly-in is a natural forward dolly.
extent = max(math.hypot(gsize.x, gsize.y), 1.5)
D = (extent * 0.9) / math.tan(hfov / 2.0)
# Look at the group from the OPEN side of the room (toward the room centre) so a
# wall-adjacent group (e.g. a bed) is framed against its wall instead of off to
# one side with empty floor. Group near the centre -> use the original-cam side.
_rctr = (rmn + rmx) * 0.5
_v = Vector((_rctr.x - aim.x, _rctr.y - aim.y, 0.0))
if _v.length < 1.2:
    _v = Vector((P0_loc.x - aim.x, P0_loc.y - aim.y, 0.0))
if _v.length < 1e-3:
    _v = Vector((0.0, -1.0, 0.0))
_v.normalize()


def _room_t(p, d):
    """How far along d from p before leaving the room (xy, 0.5 m wall margin)."""
    ts = []
    for i, lo, hi in ((0, rmn.x + 0.5, rmx.x - 0.5), (1, rmn.y + 0.5, rmx.y - 0.5)):
        di = d[i]
        if abs(di) > 1e-6:
            ts.append((hi - p[i]) / di if di > 0 else (lo - p[i]) / di)
    ts = [t for t in ts if t > 0]
    return min(ts) if ts else D


Dh = min(D, _room_t(aim, _v))                               # clamp inside walls
rise = max(min(D * 0.45, (rmx.z - 0.4) - aim.z), 0.6)       # clamp below ceiling
Pisl_loc = Vector((aim.x + _v.x * Dh, aim.y + _v.y * Dh, aim.z + rise))
Pisl_q = look_quat(Pisl_loc, aim)

print(f"[tour] phase={phase} group={group}")
print(f"[tour] P_island loc={tuple(round(v,2) for v in Pisl_loc)} aim={tuple(round(v,2) for v in aim)} Dh={Dh:.2f} rise={rise:.2f}")


def apply_pose(info, frame):
    o = bpy.data.objects.get(info[0])
    if o is None:
        return
    d = info[1]
    o.rotation_mode = "XYZ"
    o.location = Vector(d["location"]); o.keyframe_insert("location", frame=frame)
    o.rotation_euler = d["rotation_euler"]; o.keyframe_insert("rotation_euler", frame=frame)
    o.scale = Vector(d["scale"]); o.keyframe_insert("scale", frame=frame)


if phase == "zoom":
    HOLD_A, MOVE, HOLD_B = 10, 64, 14
    f1, f2 = 1, 1 + HOLD_A
    f3 = f2 + MOVE
    f4 = f3 + HOLD_B
    scene.frame_start, scene.frame_end = 1, f4
    cam.data.lens = P0_lens
    # group members: init -> final island poses (others untouched = final)
    if ginit and gfinal:
        gi = json.load(open(ginit))["objects"]
        gf = json.load(open(gfinal))["objects"]
        for n in group:
            if n not in gi:
                continue
            apply_pose((n, gi[n]), f1)
            apply_pose((n, gi[n]), f2)
            apply_pose((n, gf.get(n, gi[n])), f3)
            apply_pose((n, gf.get(n, gi[n])), f4)
    # STATIC camera at the group view (P_island). This clip is made independently
    # (no continuous fly-in from the room camera) and simply concatenated; clip 5
    # then starts from THIS same P_island pose for the turntable.
    cam.location = Pisl_loc
    cam.rotation_quaternion = Pisl_q
    keep_ceiling = True

else:  # turntable
    # group at FINAL (exactly matches zoom's last frame)
    if gfinal:
        gf = json.load(open(gfinal))["objects"]
        for n in group:
            if n in gf:
                o = bpy.data.objects.get(n)
                if o:
                    o.rotation_mode = "XYZ"
                    o.location = Vector(gf[n]["location"])
                    o.rotation_euler = gf[n]["rotation_euler"]
                    o.scale = Vector(gf[n]["scale"])
    # shell bbox
    spts = []
    for nm in SHELL:
        o = bpy.data.objects.get(nm)
        if o and o.type == "MESH":
            for c in o.bound_box:
                spts.append(o.matrix_world @ Vector(c))
    if spts:
        smn = Vector((min(p.x for p in spts), min(p.y for p in spts), min(p.z for p in spts)))
        smx = Vector((max(p.x for p in spts), max(p.y for p in spts), max(p.z for p in spts)))
    else:
        smn, smx = Vector((-6, 0, -2.5)), Vector((6, 16, 2))
    sctr = (smn + smx) * 0.5
    ssize = smx - smn
    # Open the top for the BEV WITHOUT killing the lighting: the original lights
    # ARE the ceiling panels, so make the ceiling + ceiling-mounted fixtures
    # invisible to CAMERA rays only (visible_camera=False) — they still emit /
    # bounce light, so the room stays lit exactly as in clip 4 (seamless 4->5),
    # but the camera looks straight down into the room without them in the way.
    def _cam_invisible(o):
        try:
            o.visible_camera = False
        except Exception:
            try:
                o.cycles_visibility.camera = False
            except Exception:
                pass
    _ceil = bpy.data.objects.get("Ceiling")
    if _ceil:
        _cam_invisible(_ceil)
    _zhi = smn.z + 0.72 * (smx.z - smn.z)
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name in SHELL:
            continue
        _zmin = min((o.matrix_world @ Vector(c)).z for c in o.bound_box)
        if _zmin > _zhi:
            _cam_invisible(o)
    # near-BEV geometry
    bev_lens = 28.0
    rx, ry = 1160, 880
    vfov = 2.0 * math.atan((36.0 * ry / rx / 2.0) / bev_lens)
    foot = math.hypot(ssize.x, ssize.y)
    Dbev = (foot * 1.15 / 2.0) / math.tan(vfov / 2.0)
    th = math.radians(78)
    Hb, Rb = Dbev * math.sin(th), Dbev * math.cos(th)
    blook = Vector((sctr.x, sctr.y, smn.z + ssize.z * 0.1))

    def bev_pose(ang):
        loc = Vector((sctr.x + Rb * math.cos(ang), sctr.y + Rb * math.sin(ang), sctr.z + Hb))
        return loc, look_quat(loc, blook)

    LIFT, ORBIT = 32, 96
    f1 = 1
    f2 = f1 + LIFT
    f3 = f2 + ORBIT
    scene.frame_start, scene.frame_end = 1, f3
    ang0 = math.atan2(Pisl_loc.y - sctr.y, Pisl_loc.x - sctr.x)   # start orbit on the island side
    # start at P_island (P0 lens), lift to bev start (lens 28), then orbit 360
    prevq = None
    cam.location = Pisl_loc; cam.rotation_quaternion = keep_sign(Pisl_q, prevq); prevq = cam.rotation_quaternion.copy()
    cam.data.lens = P0_lens
    cam.keyframe_insert("location", frame=f1)
    cam.keyframe_insert("rotation_quaternion", frame=f1)
    cam.data.keyframe_insert("lens", frame=f1)
    loc0, q0 = bev_pose(ang0); q0 = keep_sign(q0, prevq); prevq = q0.copy()
    cam.location = loc0; cam.rotation_quaternion = q0; cam.data.lens = bev_lens
    cam.keyframe_insert("location", frame=f2)
    cam.keyframe_insert("rotation_quaternion", frame=f2)
    cam.data.keyframe_insert("lens", frame=f2)
    N = 48
    for k in range(1, N + 1):
        t = k / N
        loc, q = bev_pose(ang0 + t * 2.0 * math.pi)
        q = keep_sign(q, prevq); prevq = q.copy()
        fr = f2 + int(round((f3 - f2) * t))
        cam.location = loc; cam.rotation_quaternion = q
        cam.keyframe_insert("location", frame=fr)
        cam.keyframe_insert("rotation_quaternion", frame=fr)
    keep_ceiling = False

# smooth eases on camera; linear on the object refine
for act in bpy.data.actions:
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER" if "rotation_quaternion" in fc.data_path or fc.data_path.endswith("location") or fc.data_path.endswith("lens") else "LINEAR"

flatcam.white_bg(scene)
flatcam.crop(scene)
flatcam.set_engine(scene, max(samples, 64))
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "HIGH"
scene.render.filepath = out_mp4
scene.render.use_compositing = False
if scene.use_nodes and scene.node_tree:
    for nd in scene.node_tree.nodes:
        if nd.type == "OUTPUT_FILE":
            nd.mute = True
if os.environ.get("SPV_SAVE_BLEND", "1") != "0":
    bpy.ops.wm.save_as_mainfile(filepath=out_mp4.rsplit(".", 1)[0] + ".blend")
print(f"[tour] rendering {phase} frames {scene.frame_start}..{scene.frame_end} -> {out_mp4}")
bpy.ops.render.render(animation=True)
print("[tour] done")
