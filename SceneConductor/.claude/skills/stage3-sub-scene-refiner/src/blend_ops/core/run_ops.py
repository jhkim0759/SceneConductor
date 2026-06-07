"""Apply a list of ops to a .blend in a single Blender session.

CLI:
    python run_ops.py <input.blend> <output.blend> <ops.json>

`ops.json` may contain either a single op dict or a JSON list of ops.
Prints the result envelope (JSON) to stdout and exits 0 on full success,
1 otherwise.

Python API (alternative to CLI):
    from external_blend_tools import apply_ops
    apply_ops("input.blend", "output.blend", [
        {"action": "update_layout", "obj_name": "obj_3", "location": [1, 0, 0]},
        {"action": "update_rotation", "obj_name": "obj_5", "rotation_euler": [0, 0, 1.57]},
        {"action": "remove_object", "obj_name": "obj_7"},
    ])
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import external_blend_tools as tools


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <input.blend> <output.blend> <ops.json>", file=sys.stderr)
        sys.exit(2)
    blend_in, blend_out, ops_json = sys.argv[1:4]

    with open(ops_json) as f:
        ops = json.load(f)

    result = tools.apply_ops(blend_in, blend_out, ops)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
