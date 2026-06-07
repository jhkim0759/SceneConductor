"""Reset obj_ empties to a stage JSON's transforms, save to a new blend.

    blender -b <in.blend> --python reset_empties.py -- <stage.json> <out.blend>
"""
import bpy, sys, json
from mathutils import Vector

tail = sys.argv[sys.argv.index("--") + 1:]
stage_json, out_blend = tail[0], tail[1]
objs = json.load(open(stage_json))["objects"]
n = 0
for name, info in objs.items():
    if not name.startswith("obj_"):
        continue
    o = bpy.data.objects.get(name)
    if o is None:
        continue
    o.location = Vector(info["location"])
    o.rotation_mode = "XYZ"
    o.rotation_euler = info["rotation_euler"]
    o.scale = Vector(info["scale"])
    n += 1
bpy.context.view_layer.update()
print(f"[reset] reset {n} obj_ empties from {stage_json}")
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
print(f"[reset] saved {out_blend}")
