"""flatten(scene, lens): reduce wide-angle 'convex' look on an interior camera.

Increases focal length to `lens`, dollies the camera straight back along its
view axis to preserve framing, makes walls that end up between the camera and
the room centre camera-invisible (so the pulled-back camera still sees in), and
clears any camera animation so the new static pose sticks.
"""
import math
import os
import bpy
from mathutils import Vector


def set_engine(scene, samples=64):
    """Pick render engine from $SPV_ENGINE (cycles|eevee, default cycles)."""
    eng = os.environ.get("SPV_ENGINE", "cycles").lower()
    if eng == "eevee":
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            pass
        try:
            scene.eevee.taa_render_samples = samples
        except Exception:
            pass
        print(f"[set_engine] EEVEE {samples} spp")
    else:
        set_cycles(scene, samples)


def set_cycles(scene, samples=96):
    """Switch to Cycles with GPU (OPTIX/CUDA) + denoise."""
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        chosen = None
        for be in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = be
                prefs.get_devices()
                if any(d.type == be for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = (d.type == be)
                    chosen = be
                    break
            except Exception:
                continue
        scene.cycles.device = "GPU" if chosen else "CPU"
        print(f"[set_cycles] {samples} spp, device={scene.cycles.device}, backend={chosen}")
    except Exception as e:
        scene.cycles.device = "CPU"
        print(f"[set_cycles] CPU fallback ({e})")


def crop(scene, px=None, base_w=1160, base_h=880):
    """Set the render resolution (default 1160x880). Edge-cropping is done as a
    UNIFORM post step (see post_crop), not in-render: with px=0 (default) the
    full base is rendered, and a 20px/edge post crop -> 1120x840 (exactly 4:3).

    px>0 (from $SPV_CROP_PX) optionally trims a uniform centered render border
    too, but that breaks the exact-4:3 math, so leave it 0 unless you know why."""
    # SPV_RES_SCALE multiplies the render resolution (and crop border stays
    # proportional via SPV_CROP_PX = round(20*scale)) so a 2x HQ pass renders
    # 2320x1760 -> 2240x1680 with IDENTICAL framing/crop ratio (content unchanged).
    res_scale = float(os.environ.get("SPV_RES_SCALE", "1.0"))
    base_w = int(round(base_w * res_scale))
    base_h = int(round(base_h * res_scale))
    if px is None:
        px = int(os.environ.get("SPV_CROP_PX", "0"))
    scene.render.resolution_x = base_w
    scene.render.resolution_y = base_h
    scene.render.resolution_percentage = 100
    if px > 0:
        scene.render.use_border = True
        scene.render.use_crop_to_border = True
        scene.render.border_min_x = px / base_w
        scene.render.border_max_x = 1.0 - px / base_w
        scene.render.border_min_y = px / base_h
        scene.render.border_max_y = 1.0 - px / base_h
    else:
        scene.render.use_border = False
        scene.render.use_crop_to_border = False
    print(f"[crop] base {base_w}x{base_h}, in-render border {px}px/edge "
          f"(edge crop done in post)")


def light_scale():
    """Global lighting multiplier from $SPV_LIGHT_SCALE (default 1.0).

    The driver sets it to <1 for the Cycles clips so they render less bright.
    Interior-light energies in the build/refine scripts multiply by this too."""
    return float(os.environ.get("SPV_LIGHT_SCALE", "1.0"))


def white_bg(scene, sun_energy=3.0):
    """Pure WHITE background: Standard view transform + white world (1.0).
    Drops existing (mis-calibrated) lights and adds one gentle sun so objects
    stay well-exposed (not blown out) with soft shading.

    Honours $SPV_LIGHT_SCALE: the sun and the world's *lighting* contribution
    are multiplied by it, while the camera-visible backdrop stays pure white
    (via a Light-Path mix) so dimming the scene never greys the background."""
    scale = light_scale()
    orig = os.environ.get("SPV_ORIGINAL_LIGHTS", "0") == "1"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    w = scene.world or bpy.data.worlds.new("W")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    if orig:
        # ORIGINAL-lights mode: keep the blend's native lights; white backdrop for
        # CAMERA rays only, world contributes ZERO lighting so the scene is lit
        # purely by its own lights. Optionally scale those lights by SPV_LIGHT_SCALE.
        bg_cam = nt.nodes.new("ShaderNodeBackground")
        bg_cam.inputs["Color"].default_value = (1, 1, 1, 1)
        bg_cam.inputs["Strength"].default_value = 1.0
        bg_lit = nt.nodes.new("ShaderNodeBackground")
        bg_lit.inputs["Color"].default_value = (0, 0, 0, 1)
        bg_lit.inputs["Strength"].default_value = 0.0
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg_lit.outputs["Background"], mix.inputs[1])
        nt.links.new(bg_cam.outputs["Background"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
        if scale != 1.0:
            for o in bpy.data.objects:
                if o.type == "LIGHT":
                    try:
                        o.data.energy *= scale
                    except Exception:
                        pass
        print(f"[white_bg] ORIGINAL lights kept + white backdrop (world lighting=0), scale {scale}")
        return
    if scale == 1.0:
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (1, 1, 1, 1)
        bg.inputs["Strength"].default_value = 1.0
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    else:
        # Camera rays -> white 1.0 (pure white backdrop); lighting rays ->
        # white * scale (dimmer ambient). Mix on Light Path 'Is Camera Ray'.
        bg_cam = nt.nodes.new("ShaderNodeBackground")
        bg_cam.inputs["Color"].default_value = (1, 1, 1, 1)
        bg_cam.inputs["Strength"].default_value = 1.0
        bg_lit = nt.nodes.new("ShaderNodeBackground")
        bg_lit.inputs["Color"].default_value = (1, 1, 1, 1)
        bg_lit.inputs["Strength"].default_value = scale
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg_lit.outputs["Background"], mix.inputs[1])   # Fac=0 rays
        nt.links.new(bg_cam.outputs["Background"], mix.inputs[2])   # Fac=1 camera
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            o.hide_render = True
    sd = bpy.data.lights.new("WB_Sun", type="SUN")
    sd.energy = sun_energy * scale
    so = bpy.data.objects.new("WB_Sun", sd)
    so.rotation_euler = (math.radians(50), math.radians(15), math.radians(40))
    scene.collection.objects.link(so)
    print(f"[white_bg] Standard + white world + sun {sun_energy*scale:.2f} (scale {scale})")


def _mesh_centroid():
    pts = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        if o.name.startswith("PointCloud"):
            continue
        for c in o.bound_box:
            pts.append(o.matrix_world @ Vector(c))
    if not pts:
        return Vector((0, 6, 0))
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (mn + mx) * 0.5


def flatten(scene, lens=60.0):
    cam = scene.camera
    if cam is None:
        return
    cam.animation_data_clear()          # clear object loc/rot anim
    if cam.data.animation_data:          # clear LENS fcurve (lives on cam.data)
        cam.data.animation_data_clear()
    old_lens = cam.data.lens
    ctr = _mesh_centroid()
    fwd = (cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized()
    dist = (ctr - cam.location).dot(fwd)
    if dist <= 0.1:
        dist = 4.0
    new_dist = dist * (lens / old_lens)
    cam.location = cam.location - fwd * (new_dist - dist)
    cam.data.lens = lens
    cam.data.clip_start = 0.01
    cam.data.clip_end = max(cam.data.clip_end, 200.0)

    # hide walls now between the (pulled-back) camera and the room centre
    cam_loc = cam.location
    to_ctr = ctr - cam_loc
    L = to_ctr.length
    if L < 1e-4:
        return
    d = to_ctr / L
    hidden = []
    for o in bpy.data.objects:
        if not o.name.startswith("Wall"):
            continue
        wc = sum((o.matrix_world @ Vector(c) for c in o.bound_box), Vector()) / 8.0
        rel = wc - cam_loc
        proj = rel.dot(d)
        perp = (rel - proj * d).length
        if -0.5 < proj < L and perp < L * 0.9:
            o.visible_camera = False
            hidden.append(o.name)
    print(f"[flatcam] lens {old_lens:.0f}->{lens:.0f}mm, dolly back {new_dist-dist:.2f}m, "
          f"hidden walls={hidden}")
