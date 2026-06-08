"""Fast single-still brightness probe with the PRODUCTION lighting recipe
(white_bg ORIGINAL + original reference camera + optional SPV_FOCAL_SCALE), so the
agent can judge per-scene brightness BEFORE the full render and pick SPV_LIGHT_SCALE.

    blender -b <blender_scene.blend> --python bright_probe.py -- <out_png> [samples]
"""
import os, sys, bpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flatcam

tail = sys.argv[sys.argv.index("--") + 1:]
out_png = tail[0]
samples = int(tail[1]) if len(tail) > 1 else 24

scene = bpy.context.scene
pc = bpy.data.objects.get("PointCloud_XZ")
if pc:
    try: bpy.data.objects.remove(pc, do_unlink=True)
    except Exception: pass
flatcam.white_bg(scene)
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
flatcam.set_engine(scene, samples)
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.use_border = False
scene.render.use_compositing = False
if scene.use_nodes and scene.node_tree:
    for nd in scene.node_tree.nodes:
        if nd.type == "OUTPUT_FILE":
            nd.mute = True
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)
# also print mean luminance for a quick numeric read
try:
    import numpy as np
    img = bpy.data.images.load(out_png)
    px = np.array(img.pixels[:]).reshape(-1, 4)[:, :3]
    print(f"[probe] mean_lum={px.mean():.3f} p50={np.median(px):.3f} -> {out_png}")
except Exception as e:
    print(f"[probe] done -> {out_png} (lum calc skipped: {e})")
