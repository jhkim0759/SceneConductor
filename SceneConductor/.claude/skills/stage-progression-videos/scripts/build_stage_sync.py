"""Stage construction with stage-2 objects present, rendered from interior OR
external camera on an IDENTICAL build schedule (so the two views stay in sync).

    blender -b <walled.blend> --python build_stage_sync.py -- \
        <out_mp4> --view interior|external [--samples N]

Shell (Floor -> Walls -> Ceiling) grows on a fixed frame schedule; furniture is
present throughout. interior = scene Camera + interior area light; external =
elevated corner orbit + sun. Both 1024x1024, 132 frames.
"""
import os
import json
import bpy, sys, math
from mathutils import Vector

tail = sys.argv[sys.argv.index("--") + 1:]
view = "interior"; samples = 32; poses_path = None
if "--view" in tail:
    i = tail.index("--view"); view = tail[i + 1]; tail = tail[:i] + tail[i + 2:]
if "--samples" in tail:
    i = tail.index("--samples"); samples = int(tail[i + 1]); tail = tail[:i] + tail[i + 2:]
if "--poses" in tail:
    i = tail.index("--poses"); poses_path = tail[i + 1]; tail = tail[:i] + tail[i + 2:]
out_mp4 = tail[0]

# ---- FIXED build schedule (identical for both views -> sync) ----
SHELL_ORDER = ["Floor", "Wall_01", "Wall_02", "Wall_03", "Wall_04", "Ceiling"]
FPS = 24
F_START_BUILD = 10
PER = 16
STAGGER = 14
HOLD_END = 36

scene = bpy.context.scene
scene.frame_start = 1

_pc = bpy.data.objects.get("PointCloud_XZ")
if _pc is not None:
    try: bpy.data.objects.remove(_pc, do_unlink=True)
    except Exception: pass

# Optional: pose the obj_ empties to a STAGE json (static, no animation). Used by
# clip 5 so the turntable shows the EXACT planned->final result (= clip 3-1's last
# frame) on the non-baked corrected_planned base, instead of the recorded
# blender_scene (whose island result was mislocated). Walls/Floor/Ceiling untouched.
if poses_path:
    _poses = json.load(open(poses_path))["objects"]
    _n = 0
    for _name, _info in _poses.items():
        _o = bpy.data.objects.get(_name)
        if _o is None:
            continue
        _o.location = Vector(_info["location"])
        _o.rotation_mode = "XYZ"
        _o.rotation_euler = Vector(_info["rotation_euler"])
        _o.scale = Vector(_info["scale"])
        _n += 1
    print(f"[build_stage_sync] applied {_n} poses from {poses_path}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flatcam
flatcam.white_bg(scene)

# shell bbox
pts = []
for nm in SHELL_ORDER:
    o = bpy.data.objects.get(nm)
    if o and o.type == "MESH":
        for c in o.bound_box: pts.append(o.matrix_world @ Vector(c))
if pts:
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
else:
    mn, mx = Vector((-5, 0, -2)), Vector((5, 14, 1))
ctr = (mn + mx) * 0.5; size = mx - mn
radius = max(size.x, size.y) * 1.15

present = [n for n in SHELL_ORDER if bpy.data.objects.get(n)]
if view == "turntable":
    # Completed scene: everything present, no build animation. The camera does a
    # full 360 orbit instead (near-BEV turntable of the finished Stage-3 result).
    # Hide the ceiling AND any ceiling-mounted fixtures (light panels) so the
    # top-down orbit looks straight down into the room and sees the floor layout
    # without floating panels occluding the furniture.
    for nm in present:
        o = bpy.data.objects[nm]; o.hide_render = False; o.hide_viewport = False
    # camera-invisible (not hidden) so original ceiling-panel lights still lit it
    def _cam_invis(o):
        try:
            o.visible_camera = False
        except Exception:
            try:
                o.cycles_visibility.camera = False
            except Exception:
                pass
    # ceiling always camera-invisible (top-down sees INTO the room). Walls: hidden
    # by default (clean floor layout), but kept VISIBLE when SPV_TT_WALLS=1 (user
    # request) -> dollhouse-with-walls. The steep near-BEV (78 deg) means even the
    # near wall doesn't occlude the interior (camera looks down past the wall tops).
    _show_walls = os.environ.get("SPV_TT_WALLS", "0") == "1"
    _invis = ["Ceiling"] if _show_walls else ["Ceiling", "Wall_01", "Wall_02", "Wall_03", "Wall_04"]
    for nm in _invis:
        _w = bpy.data.objects.get(nm)
        if _w:
            _cam_invis(_w)
    _z_hi = mn.z + 0.72 * (mx.z - mn.z)          # "near the ceiling" cutoff
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name in SHELL_ORDER:
            continue
        _zmin = min((o.matrix_world @ Vector(c)).z for c in o.bound_box)
        if _zmin > _z_hi:                        # ceiling-mounted -> camera-invisible
            _cam_invis(o)
    TURN_FRAMES = 120
    f_end = TURN_FRAMES
    scene.frame_end = f_end
else:
    # --- build shell: staggered grow 0 -> 1 (identical schedule both views) ---
    f_last = F_START_BUILD
    for i, nm in enumerate(present):
        o = bpy.data.objects[nm]
        o.hide_render = False; o.hide_viewport = False
        orig = tuple(o.scale)
        s0 = F_START_BUILD + i * STAGGER
        s1 = s0 + PER
        o.scale = (0, 0, 0); o.keyframe_insert("scale", frame=1)
        o.scale = (0, 0, 0); o.keyframe_insert("scale", frame=s0)
        o.scale = orig; o.keyframe_insert("scale", frame=s1)
        f_last = max(f_last, s1)
    f_end = f_last + HOLD_END
    scene.frame_end = f_end

if view in ("external", "turntable"):
    cam_data = bpy.data.cameras.new("ExtCam"); cam_data.lens = 30.0
    cam = bpy.data.objects.new("ExtCam", cam_data); scene.collection.objects.link(cam)
    scene.camera = cam; cam.rotation_mode = "QUATERNION"
    # Same elevated 2-1 vantage (height z + distance radius); look at room centre.
    z = mx.z + size.z * 0.9
    look = Vector((ctr.x, ctr.y, ctr.z + size.z * 0.1))
    if view == "turntable":
        # NEAR-BEV turntable: steep, near-top-down orbit so the camera looks down
        # INTO the room and the final floor layout reads clearly. Auto-fit the
        # rotating room footprint to the (tighter, vertical) frame fov.
        cam_data.lens = 28.0
        _sensor = 36.0
        _rx, _ry = 1160, 880                       # landscape frame -> vertical fov is the limit
        _vfov = 2.0 * math.atan((_sensor * _ry / _rx / 2.0) / cam_data.lens)
        _foot = math.hypot(size.x, size.y)          # room footprint diagonal
        _D = (_foot * 1.05 / 2.0) / math.tan(_vfov / 2.0)   # distance to fit (small margin -> fills frame)
        _theta = math.radians(78)                   # elevation above horizontal (near vertical)
        _H = _D * math.sin(_theta); _R = _D * math.cos(_theta)
        _tlook = Vector((ctr.x, ctr.y, mn.z + size.z * 0.1))   # aim near the floor
        # ONE keyframe per frame (N = f_end-1) so the orbit is perfectly uniform
        # (the old N=48 + round() gave uneven 2/3-frame spacing -> micro-stutter),
        # and keep the look-at quaternions on a SINGLE hemisphere (negate when the
        # dot with the previous flips sign) so the 360 deg sweep never jumps mid-way
        # (quaternion double-cover was the visible "stutter" at the half-turn).
        N = f_end - 1
        _prevq = None
        for k in range(N + 1):
            t = k / N
            ang = t * 2.0 * math.pi
            loc = Vector((ctr.x + _R * math.cos(ang), ctr.y + _R * math.sin(ang), ctr.z + _H))
            cam.location = loc
            q = (_tlook - loc).normalized().to_track_quat("-Z", "Y")
            if _prevq is not None and q.dot(_prevq) < 0.0:
                q.negate()
            _prevq = q
            cam.rotation_quaternion = q
            fr = 1 + k
            cam.keyframe_insert("location", frame=fr)
            cam.keyframe_insert("rotation_quaternion", frame=fr)
    else:  # external: partial elevated orbit synced with the shell build
        N = 24
        for k in range(N + 1):
            t = k / N
            ang = math.radians(35) + t * math.radians(40)
            loc = Vector((ctr.x + radius * math.cos(ang),
                          ctr.y - radius * math.sin(ang) - size.y * 0.2, z))
            cam.location = loc
            cam.rotation_quaternion = (look - loc).normalized().to_track_quat("-Z", "Y")
            fr = 1 + int(round((f_end - 1) * t))
            cam.keyframe_insert("location", frame=fr)
            cam.keyframe_insert("rotation_quaternion", frame=fr)
    cam_data.clip_start = 0.01; cam_data.clip_end = radius * 8
else:  # interior
    # Interior fill light, on from frame 1 (skipped in ORIGINAL-lights mode where
    # the blend's own ceiling lights illuminate the room).
    if os.environ.get("SPV_ORIGINAL_LIGHTS", "0") != "1":
        ld = bpy.data.lights.new("InteriorLight", type="AREA")
        ld.energy = 300.0 * flatcam.light_scale()
        ld.size = max(size.x, size.y) * 0.5
        il = bpy.data.objects.new("InteriorLight", ld)
        il.location = Vector((ctr.x, ctr.y, mn.z + (mx.z - mn.z) * 0.78))
        scene.collection.objects.link(il)
    # Use the ORIGINAL scene camera view (user request): keep its stored
    # location / rotation / lens (the GALP/reference pose matching the input
    # photo); only clear animation so the pose is static. The interior area
    # light + use_compositing=False make this origin camera render fine under
    # Cycles (verified) — no EEVEE fallback needed.
    cam = scene.camera or next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
    scene.camera = cam
    cam.animation_data_clear()
    if cam.data.animation_data:
        cam.data.animation_data_clear()
    cam.data.clip_start = 0.01
    cam.data.clip_end = max(cam.data.clip_end, 300.0)
    # Optional: widen the interior view by scaling the original lens (SPV_FOCAL_SCALE
    # < 1 -> shorter focal -> wider FOV). Same scale used by clips 1/3/3-1 so the
    # connected views share one (wider) field of view.
    _fs = float(os.environ.get("SPV_FOCAL_SCALE", "1.0"))
    if _fs != 1.0:
        cam.data.lens *= _fs

# Lighting: the room's own (original) lights stay ON at full energy for the whole
# build (white_bg ORIGINAL mode keeps them lit from frame 1) -> clip 2/2-1 start
# at the SAME full brightness as clip 1's end, so the 1->2 cut is seamless. No
# "lights-after-build" reveal (user choice A: keep clip 1's bright original
# lighting on both clips, drop the dark-then-on reveal).

for act in bpy.data.actions:
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

flatcam.set_engine(scene, max(samples, 64))
flatcam.crop(scene)
scene.render.fps = FPS
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = os.environ.get("SPV_CRF", "HIGH")
scene.render.ffmpeg.ffmpeg_preset = os.environ.get("SPV_FFPRESET", "GOOD")
scene.render.filepath = out_mp4
scene.render.use_compositing = False   # blend's compositor is black under Cycles w/ origin cam
if scene.use_nodes and scene.node_tree:
    for nd in scene.node_tree.nodes:
        if nd.type == "OUTPUT_FILE": nd.mute = True
if os.environ.get("SPV_SAVE_BLEND", "1") != "0":
    bpy.ops.wm.save_as_mainfile(filepath=out_mp4.rsplit(".", 1)[0] + ".blend")
print(f"[build_stage_sync] view={view} shell={present} frames 1..{f_end} -> {out_mp4}")
bpy.ops.render.render(animation=True)
print("[build_stage_sync] done")
