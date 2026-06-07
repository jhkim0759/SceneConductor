# blend_ops — Usage

This folder is vendored inside the `scene-refiner` skill (originally `blend_ops_scripts/` at repo root).

## Run a list of ops

**Python:**
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "core"))
from external_blend_tools import apply_ops

ops = [
    {"action": "update_layout",   "obj_name": "obj_3", "location": [1, 0, 0]},
    {"action": "update_rotation", "obj_name": "obj_5", "rotation_euler": [0, 0, 1.57]},
    {"action": "remove_object",   "obj_name": "obj_7"},
]

apply_ops("input.blend", "output.blend", ops)
```

**CLI:**
```bash
python core/run_ops.py input.blend output.blend ops.json
```

All ops execute in **one Blender session**. On first failure execution stops;
the output `.blend` is saved only if at least one mutating op succeeded.

---

## Operation format

Each op is a JSON object with `"action"` plus action-specific keys.
You may pass either a single op dict or a list of op dicts.

```json
{ "action": "<name>", "...": "..." }
```

---

## Actions

| Action            | Role                                           | Required keys                                  |
|-------------------|------------------------------------------------|------------------------------------------------|
| `list_objects`    | Enumerate scene objects (read-only)            | — (optional `name_prefix`, default `"obj_"`)   |
| `inspect_object`  | Read one object's transform & dimensions       | `obj_name`                                     |
| `update_layout`   | Move object to absolute world location (m)     | `obj_name`, `location: [x, y, z]`              |
| `update_rotation` | Set Euler XYZ rotation (radians)               | `obj_name`, `rotation_euler: [rx, ry, rz]`     |
| `flip_yaw_180`    | Add π to current Z rotation, normalize to (-π, π]. Use when an object is facing the opposite direction from the reference image. | `obj_name`                                     |
| `update_size`     | Set non-uniform scale factors                  | `obj_name`, `scale: [sx, sy, sz]`              |
| `remove_object`   | Delete object **and all children** recursively | `obj_name`                                     |
| `attach`          | Chamfer-based snap of `moving_obj` onto `anchor_obj`. `relation`: `"on"` (Z-stack), `"attached_to"`/`"next_to"` (chamfer pull), `"+x"`/`"-x"`/`"+y"`/`"-y"`/`"+z"`/`"-z"` or `[vx,vy,vz]` (axis-aligned kissing) | `anchor_obj`, `moving_obj` (optional `relation`, `n_samples`) |
| `attach_to_wall`  | **Polygon-aware** wall attach. Reads `polygon_v2.json` to compute the wall's tangent + inward normal. By default aligns `rz` to wall tangent; set `preserve_rotation: true` to keep current rotation. Projects XY onto the wall edge, then offsets inward by `wall_thickness/2 + clearance`. Z, rx, ry always preserved. `t_along_m` explicitly sets along-wall position (meters from wall start); when omitted, current along-wall position is preserved. **Use this instead of `attach` whenever anchor is a `Wall_NN`.** | `wall_obj` (e.g. `"Wall_03"`), `moving_obj`; optional: `polygon_path` (auto-discovered), `clearance` (default 0.02), `t_along_m`, `preserve_rotation` (default false) |
| `render`          | Workbench preview PNG (read-only)              | `output_png` (optional `view`, `resolution`)   |
| `metrics`         | Out-of-bounds + pairwise collision report      | — (optional `room_bbox`, `name_prefix`)        |
