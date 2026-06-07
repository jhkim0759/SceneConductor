"""Apply operation_plan.json to a Blender scene via blend_ops.apply_ops.

Usage:
    python apply_plan.py <operation_plan.json> <blend_in> <blend_out>

If blend_in == blend_out, the file is updated in place after the ops run.
Ops are applied best-effort: a failing op (e.g. "object not found") is recorded
but does not abort the remaining ops, and the .blend is saved as long as at least
one mutating op succeeded — see blend_ops/USAGE.md for the runner's contract.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "blend_ops" / "core"))
from external_blend_tools import apply_ops


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <operation_plan.json> <blend_in> <blend_out>", file=sys.stderr)
        sys.exit(2)

    plan_path, blend_in, blend_out = sys.argv[1:4]
    plan = json.loads(Path(plan_path).read_text())
    ops = plan["operation_list"]

    print(f"[apply_plan] {len(ops)} ops from {plan_path}")
    print(f"[apply_plan]   in : {blend_in}")
    print(f"[apply_plan]   out: {blend_out}")

    t0 = time.time()
    result = apply_ops(blend_in, blend_out, ops, timeout=1800)
    dt = time.time() - t0

    print(f"[apply_plan] done in {dt:.1f}s — success={result.get('success')}, "
          f"executed={result.get('n_executed')}/{result.get('n_ops')}")

    failed = [(i, r) for i, r in enumerate(result.get("results", [])) if not r.get("success")]
    if failed:
        print(f"[apply_plan] {len(failed)} failed op(s):")
        for i, r in failed:
            print(f"  op[{i}] action={r.get('action')} message={r.get('message','')[:200]}")

    summary = {
        "success": result.get("success"),
        "n_ops": result.get("n_ops"),
        "n_executed": result.get("n_executed"),
        "output": result.get("output"),
        "elapsed_s": round(dt, 1),
    }
    print("[apply_plan] summary:", json.dumps(summary, indent=2))
    # Best-effort apply: succeed as long as the blend was written (output set),
    # even if some non-critical ops failed (e.g. "object not found" for objects
    # merged away upstream). Failed ops are logged above. Only a save failure or
    # a plan where nothing could be applied (output is None) is fatal.
    sys.exit(0 if result.get("output") else 1)


if __name__ == "__main__":
    main()
