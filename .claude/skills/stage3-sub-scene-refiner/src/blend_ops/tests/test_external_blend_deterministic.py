"""Deterministic 4-tool test (no LLM) — proves each modification primitive works
on aligned_scene.blend independent of Claude's reasoning.

Usage:
    python test_external_blend_deterministic.py <input.blend> <work_dir>
"""
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
import external_blend_tools as tools


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <input.blend> <work_dir>")
        sys.exit(2)
    src = sys.argv[1]
    work_dir = Path(sys.argv[2])
    work_dir.mkdir(parents=True, exist_ok=True)
    blend = work_dir / "scene.blend"
    shutil.copy(src, blend)

    print("[0] before:")
    objs = tools.list_objects(str(blend))["objects"]
    print(f"    {len(objs)} objects")
    tools.render(str(blend), str(work_dir / "before.png"), view="top")

    target = next((o for o in objs if o["name"] == "obj_1"), None)
    assert target, "obj_1 not in scene"

    cur = blend
    log = []

    # 1. update_layout — bump obj_1 by +1m on z
    nxt = work_dir / "step1_layout.blend"
    new_loc = list(target["location"])
    new_loc[2] += 1.0
    r = tools.update_layout(str(cur), str(nxt), "obj_1", new_loc)
    print(f"[1] update_layout obj_1 -> z+=1: success={r['success']} before={r.get('before')} after={r.get('after')}")
    log.append(("update_layout", r))
    cur = nxt

    # 2. update_rotation — rotate obj_1 by +pi/4 around Z
    nxt = work_dir / "step2_rotation.blend"
    target_rot = list(target["rotation_euler"])
    target_rot[2] += math.pi / 4
    r = tools.update_rotation(str(cur), str(nxt), "obj_1", target_rot)
    print(f"[2] update_rotation obj_1 -> rz+=pi/4: success={r['success']} before={r.get('before')} after={r.get('after')}")
    log.append(("update_rotation", r))
    cur = nxt

    # 3. update_size — scale obj_2 down to 0.7x
    nxt = work_dir / "step3_size.blend"
    obj2 = next(o for o in objs if o["name"] == "obj_2")
    new_scale = [s * 0.7 for s in obj2["scale"]]
    r = tools.update_size(str(cur), str(nxt), "obj_2", new_scale)
    print(f"[3] update_size obj_2 -> 0.7x: success={r['success']} before={r.get('before')} after={r.get('after')}")
    log.append(("update_size", r))
    cur = nxt

    # 4. remove_object — drop obj_3
    nxt = work_dir / "step4_remove.blend"
    r = tools.remove_object(str(cur), str(nxt), "obj_3")
    print(f"[4] remove_object obj_3: success={r['success']} removed={r.get('removed')}")
    log.append(("remove_object", r))
    cur = nxt

    print("\n[5] after:")
    objs_after = tools.list_objects(str(cur))["objects"]
    print(f"    {len(objs_after)} objects")
    tools.render(str(cur), str(work_dir / "after.png"), view="top")

    # Verify each change
    after_by = {o["name"]: o for o in objs_after}
    pass_count = 0
    fail = []

    if "obj_1" in after_by:
        a = after_by["obj_1"]
        if abs(a["location"][2] - (target["location"][2] + 1.0)) < 1e-3:
            pass_count += 1
        else:
            fail.append(f"obj_1 z mismatch: {a['location'][2]} vs expected {target['location'][2]+1.0}")
        if abs(a["rotation_euler"][2] - target_rot[2]) < 1e-3:
            pass_count += 1
        else:
            fail.append(f"obj_1 rz mismatch: {a['rotation_euler'][2]} vs expected {target_rot[2]}")
    else:
        fail.append("obj_1 missing")

    if "obj_2" in after_by:
        a = after_by["obj_2"]
        if all(abs(a["scale"][i] - new_scale[i]) < 1e-3 for i in range(3)):
            pass_count += 1
        else:
            fail.append(f"obj_2 scale mismatch: {a['scale']} vs expected {new_scale}")
    else:
        fail.append("obj_2 missing")

    if "obj_3" not in after_by:
        pass_count += 1
    else:
        fail.append("obj_3 was not removed")

    summary = {
        "src": src,
        "tools_passed": pass_count,
        "tools_total": 4,
        "failures": fail,
        "operations": [{"tool": t, "result": r} for t, r in log],
        "before_png": str(work_dir / "before.png"),
        "after_png": str(work_dir / "after.png"),
    }
    with open(work_dir / "deterministic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nverdict: {pass_count}/4 tool primitives verified")
    if fail:
        for line in fail:
            print(f"  FAIL: {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
