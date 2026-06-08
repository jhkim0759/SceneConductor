"""Map a group's FINAL island obj_ poses into scene-world via M_anchor.

    blender -b <final island.blend> --python island_to_scene.py -- <metadata.json> <out.json>

scene_world[member] = M_anchor_4x4 @ (member.matrix_world in island)
Output: {"objects": {obj_N: {location, rotation_euler, scale}}}.
"""
import bpy, sys, json
from mathutils import Matrix

tail = sys.argv[sys.argv.index("--") + 1:]
meta_path, out_path = tail[0], tail[1]
meta = json.load(open(meta_path))
M_anchor = Matrix([meta["M_anchor_4x4"][r] for r in range(4)])
bpy.context.view_layer.update()

out = {"objects": {}}
for o in bpy.data.objects:
    if o.name.startswith("obj_") and o.type == "EMPTY":
        Wscene = M_anchor @ o.matrix_world
        loc, rot, scl = Wscene.decompose()
        out["objects"][o.name] = {"location": list(loc),
                                   "rotation_euler": list(rot.to_euler()),
                                   "scale": list(scl)}
print(f"[island_to_scene] {len(out['objects'])} members from {meta_path}")
json.dump(out, open(out_path, "w"))
