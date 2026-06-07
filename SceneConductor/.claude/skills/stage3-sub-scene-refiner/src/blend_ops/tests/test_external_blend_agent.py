"""End-to-end test: Claude picks SceneWeaver-style modification tools and applies
them to an external aligned_scene.blend.

Usage:
    python test_external_blend_agent.py <input.blend> <object_class.json> <work_dir>
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "app"))

import external_blend_tools as tools
from app.claude_cli_client import ClaudeCLIClient


SYSTEM_PROMPT = """You are a 3D scene-editing agent. You will be given a list of objects
in a Blender scene (each with class label, location, rotation, size) and you must propose
2-4 concrete modifications using ONLY the tools below.

Available tools (mirror SceneWeaver semantics):
- update_layout(obj_name: str, location: [x, y, z])  — move object
- update_rotation(obj_name: str, rotation_euler: [rx, ry, rz])  — radians
- update_size(obj_name: str, scale: [sx, sy, sz])  — multiplicative scale
- remove_object(obj_name: str)  — delete the object and its children

Coordinate frame: Z is up, floor is the XY plane.

When you see suspicious patterns (e.g. duplicate chandeliers, a chandelier on the floor,
a person occluding furniture, vases at impossible positions), choose modifications that
would improve scene plausibility. Be specific and conservative — small targeted changes.

Try to exercise a MIX of tool types in your plan — it is more useful to demonstrate
remove_object, update_layout, update_rotation, AND update_size in a single plan when
the issues you find naturally call for them, rather than only one tool type.

Respond with a SINGLE JSON code block in this EXACT shape:

```json
{
  "reasoning": "<one paragraph explaining what you noticed and your plan>",
  "operations": [
    {"tool": "remove_object", "args": {"obj_name": "obj_5"}},
    {"tool": "update_layout", "args": {"obj_name": "obj_1", "location": [-2.65, 5.60, 0.5]}},
    {"tool": "update_rotation", "args": {"obj_name": "obj_3", "rotation_euler": [0, 0, 1.57]}}
  ]
}
```

Do not write anything outside the fenced code block."""


def build_user_prompt(objects, classes):
    lines = [
        "Scene objects (only obj_N entries shown):",
        "",
        f"{'name':<8} {'class':<22} {'location (x,y,z)':<30} {'rot z (rad)':<14} {'scale':<22}",
    ]
    for o in objects:
        cls = classes.get(o["name"], "?")
        loc = o["location"]
        scl = o["scale"]
        rot_z = o["rotation_euler"][2]
        lines.append(
            f"{o['name']:<8} {cls:<22} ({loc[0]:6.2f},{loc[1]:6.2f},{loc[2]:6.2f})        "
            f"{rot_z:6.2f}        ({scl[0]:.2f},{scl[1]:.2f},{scl[2]:.2f})"
        )
    lines += [
        "",
        "Floor extent: roughly x in [-4.5, +4.5], y in [-5, +10] meters.",
        "",
        "Propose 2-4 modifications that would improve scene plausibility. "
        "Common issues to watch for: duplicate chandeliers, objects below the floor, "
        "objects floating at impossible heights, persons interpenetrating furniture, "
        "obviously redundant items.",
    ]
    return "\n".join(lines)


def parse_plan(claude_output):
    m = re.search(r"```json\s*(.*?)```", claude_output, re.DOTALL)
    raw = m.group(1).strip() if m else claude_output.strip()
    return json.loads(raw)


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <input.blend> <object_class.json> <work_dir>")
        sys.exit(2)
    src_blend = sys.argv[1]
    classes_path = sys.argv[2]
    work_dir = Path(sys.argv[3])
    work_dir.mkdir(parents=True, exist_ok=True)

    blend = work_dir / "scene.blend"
    shutil.copy(src_blend, blend)
    with open(classes_path) as f:
        classes = json.load(f)

    print(f"[1] listing objects in {blend} ...")
    listing = tools.list_objects(str(blend))
    objects_before = listing["objects"]
    print(f"    {len(objects_before)} objects")

    before_png = work_dir / "before.png"
    print(f"    rendering top view -> {before_png}")
    tools.render(str(blend), str(before_png), view="top")

    user_prompt = build_user_prompt(objects_before, classes)

    print("[2] asking Claude to plan modifications ...")
    client = ClaudeCLIClient(model="claude-opus-4-6", max_turns=1, thinking_budget=0)
    response = client.complete(
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        timeout=600,
    )
    print("    Claude output (first 400 chars):")
    print("    " + response[:400].replace("\n", "\n    "))

    plan = parse_plan(response)
    print(f"\n[3] plan parsed: {len(plan['operations'])} operations")
    print(f"    reasoning: {plan['reasoning'][:200]}")

    cur = blend
    results = []
    for i, op in enumerate(plan["operations"]):
        nxt = work_dir / f"scene_step{i + 1}.blend"
        tool_name = op["tool"]
        if tool_name not in tools.TOOL_REGISTRY:
            print(f"    op {i + 1} SKIP unknown tool: {tool_name}")
            continue
        fn = tools.TOOL_REGISTRY[tool_name]
        print(f"    op {i + 1}: {tool_name}({op['args']})")
        r = fn(str(cur), str(nxt), **op["args"])
        ok = "OK" if r.get("success") else "FAIL"
        msg = r.get("message", "")
        print(f"      -> {ok}  {msg[:120]}")
        results.append({"op": op, "result": r})
        if r.get("success"):
            cur = nxt

    print(f"\n[4] re-listing objects in final scene {cur} ...")
    listing_after = tools.list_objects(str(cur))
    objects_after = listing_after["objects"]
    print(f"    {len(objects_after)} objects (was {len(objects_before)})")

    after_png = work_dir / "after.png"
    print(f"    rendering top view -> {after_png}")
    tools.render(str(cur), str(after_png), view="top")

    diff_path = work_dir / "diff_summary.json"
    summary = {
        "input_blend": src_blend,
        "final_blend": str(cur),
        "object_count_before": len(objects_before),
        "object_count_after": len(objects_after),
        "claude_reasoning": plan["reasoning"],
        "operations": results,
    }
    with open(diff_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[5] summary written to {diff_path}")

    before_by_name = {o["name"]: o for o in objects_before}
    after_by_name = {o["name"]: o for o in objects_after}
    removed = sorted(set(before_by_name) - set(after_by_name))
    moved = []
    for name in sorted(set(before_by_name) & set(after_by_name)):
        b = before_by_name[name]
        a = after_by_name[name]
        if b["location"] != a["location"] or b["rotation_euler"] != a["rotation_euler"] or b["scale"] != a["scale"]:
            moved.append(name)
    print(f"\nremoved: {removed}")
    print(f"changed transforms: {moved}")


if __name__ == "__main__":
    main()
