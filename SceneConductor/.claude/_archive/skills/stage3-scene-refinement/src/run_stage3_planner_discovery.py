#!/usr/bin/env python3
"""
run_stage3_planner_discovery.py — invoke the Claude CLI vision agent
(`stage3-scene-discoverer`) to propose ops the heuristic planner may have
missed, WITHOUT ever exposing it to operation_plan.json (Option C, anchoring-free).

Writes `<scene_dir>/json/operation_plan_discoveries.json` with a tamper-evident
`_planner_meta` block. Downstream `run_stage3_planner_review.py` consumes this
file as a candidate list and reconciles it with the heuristic plan.

Usage:
    python run_stage3_planner_discovery.py --scene_dir /path/to/scene \
        [--model opus] [--timeout 600]

Exit codes:
    0 — success; json/operation_plan_discoveries.json written and validated
    1 — any validation/IO error
    2 — claude CLI invocation failed (non-zero return)
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_TOP_LEVEL_KEYS = {"discovered_ops"}
ALLOWED_ACTIONS = {
    "update_layout", "update_rotation", "flip_yaw_180", "update_size",
    "delete_object", "attach", "attach_to_wall",
}
AGENT_SPEC_PATH = Path(__file__).resolve().parents[3] / "agents" / "stage3-scene-discoverer.md"
GENERATED_BY = "run_stage3_planner_discovery.py"
MODE = "discovery"
MAX_DISCOVERIES = 8  # Per spec — cap on discovered_ops per pass

# Regex for valid delete_object target names (must match ^obj_\d+, never Floor/Wall/Ceiling).
_DELETE_OBJ_RE = re.compile(r"^obj_\d+$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_system_prompt(agent_spec: Path) -> str:
    """Read the agent spec and strip the YAML frontmatter (--- ... ---)."""
    text = agent_spec.read_text(encoding="utf-8")
    stripped = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
    return stripped.strip()


def _classify_wall_normal(nx: float, ny: float) -> str:
    if abs(nx) >= abs(ny):
        return "LEFT wall (normal -X)" if nx < 0 else "RIGHT wall (normal +X)"
    return "FRONT-NEAR wall (camera side, normal -Y)" if ny < 0 else "BACK wall (far end of room, normal +Y)"


def _build_wall_geometry_hint(scene_dir: Path) -> str:
    """Reuses the same wall-hint format as planner_review (lets the LLM pick wall_obj names)."""
    bs_path = scene_dir / "json" / "blender_scene.json"
    if not bs_path.exists():
        return ""
    try:
        bs = json.loads(bs_path.read_text(encoding="utf-8"))
        stage = bs.get("stage", {})
        poly_verts = stage.get("polygon_vertices", [])
        wall_edges = stage.get("wall_edges", [])
    except Exception:
        return ""
    if not poly_verts or not wall_edges:
        return ""

    lines = ["Wall geometry (for reference when mapping image observations to wall_obj):"]
    for edge in wall_edges:
        wall_name = edge.get("object", "")
        i_from = edge.get("from")
        i_to = edge.get("to")
        if i_from is None or i_to is None or i_from >= len(poly_verts) or i_to >= len(poly_verts):
            continue
        try:
            x1, y1 = poly_verts[i_from]
            x2, y2 = poly_verts[i_to]
        except Exception:
            continue
        tx, ty = (x2 - x1), (y2 - y1)
        # Outward normal = rotate tangent by -90° (right-hand rule, CCW polygon)
        nx, ny = ty, -tx
        label = _classify_wall_normal(nx, ny)
        lines.append(f"  - {wall_name}: {label}")
    return "\n".join(lines)


def run_claude(scene_dir: Path, system_prompt: str, model: str, timeout: int) -> None:
    """Call the claude CLI with the discoverer agent spec as system prompt."""
    wall_geometry_hint = _build_wall_geometry_hint(scene_dir)
    user_prompt = (
        f"You are running as `stage3-scene-discoverer` — the anchoring-free vision-first discovery pass.\n"
        f"\n"
        f"**HARD CONSTRAINT**: Do NOT Read `operation_plan.json` or `operation_plan_revised.json` "
        f"under any circumstances. Your value comes from independent judgment uncorrupted by the "
        f"heuristic planner's existing list.\n"
        f"\n"
        f"Goal: inspect the reference photo and current scene state, then propose ops the heuristic "
        f"planner is likely to miss. Output a discovery list — these are CANDIDATES that a downstream "
        f"reconciliation step will weigh against the heuristic plan.\n"
        f"\n"
        f"Read ONLY these files (use the Read tool):\n"
        f"  1. {scene_dir}/image.png\n"
        f"  2. {scene_dir}/inputs/object_state_annotated_mask.png\n"
        f"  3. {scene_dir}/inputs/object_class.json\n"
        f"  4. {scene_dir}/inputs/relation_graph.json\n"
        f"  5. {scene_dir}/json/blend_info.json\n"
        f"  6. {scene_dir}/json/object_state.json\n"
        f"\n"
        + (f"{wall_geometry_hint}\n\n" if wall_geometry_hint else "")
        + f"Allowed actions and their required fields (same 7 as planner_review):\n"
        f"  - update_layout:   obj_name, location ([x,y,z] absolute world coords), reason\n"
        f"  - update_rotation: obj_name, rotation_euler ([rx,ry,rz] in radians), reason\n"
        f"  - flip_yaw_180:    obj_name, reason\n"
        f"  - update_size:     obj_name, scale ([sx,sy,sz]), reason. WARNING: use sparingly.\n"
        f"  - delete_object:   obj_name (matching ^obj_\\d+; NEVER Floor/Wall/Ceiling), reason\n"
        f"  - attach:          anchor_obj, moving_obj, relation (e.g. \"on\"), reason, priority, source\n"
        f"  - attach_to_wall:  wall_obj, moving_obj, wall_ambiguous (bool), reason, priority, source, t_along_m (float). "
        f"Do NOT include preserve_rotation — wall-tangent alignment is the Stage-3 standard.\n"
        f"\n"
        f"**Discovery rules**:\n"
        f"  - Focus on what the heuristic planner CANNOT derive: wall-mounted single decor items, "
        f"ghost duplicates (high collision overlap, same class), yaw flips (chair facing wrong way), "
        f"stacking gaps (object should rest on something but is floating), out-of-room outliers.\n"
        f"  - Cap your output at {MAX_DISCOVERIES} discovered_ops max per pass. Pick highest-confidence.\n"
        f"  - When in doubt, do NOT emit. False positives hurt more than false negatives — "
        f"the reconciliation step trusts your discoveries.\n"
        f"  - It is OK to emit an empty list if the scene already matches the photo.\n"
        f"\n"
        f"Write the result to {scene_dir}/json/operation_plan_discoveries.json with this schema:\n"
        f"{{\n"
        f"  \"discovered_ops\": [\n"
        f"    {{\"action\": \"attach_to_wall\", \"moving_obj\": \"obj_5\", \"wall_obj\": \"Wall_03\", "
        f"\"wall_ambiguous\": false, \"reason\": \"...\", \"priority\": 4, \"source\": \"discovery\", \"t_along_m\": 0.5}},\n"
        f"    {{\"action\": \"delete_object\", \"obj_name\": \"obj_12\", \"reason\": \"...\"}},\n"
        f"    ...\n"
        f"  ],\n"
        f"  \"review_notes\": \"<natural language summary: count, categories, what you intentionally omitted>\"\n"
        f"}}\n"
        f"\n"
        f"The review_notes must be a genuine summary — never a template placeholder. Cite which "
        f"image observation + which JSON field motivated each discovery in its `reason`."
    )
    cmd = [
        "claude", "-p", user_prompt,
        "--system-prompt", system_prompt,
        "--tools", "Read,Write",
        "--permission-mode", "bypassPermissions",
        "--model", model,
        "--output-format", "text",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        print(
            f"[run_stage3_planner_discovery] ERROR: claude CLI timed out after {timeout}s. "
            "Re-run or increase --timeout.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        print(
            "[run_stage3_planner_discovery] ERROR: 'claude' CLI not found in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"[run_stage3_planner_discovery] ERROR: claude exited {result.returncode}.\n"
            f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}",
            file=sys.stderr,
        )
        sys.exit(2)


def validate_and_inject(scene_dir: Path, cache_anchor_path: Path, model: str) -> None:
    """Parse, validate, inject _planner_meta, atomic-write operation_plan_discoveries.json."""
    plan_path = scene_dir / "json" / "operation_plan_discoveries.json"
    if not plan_path.exists():
        print(
            "[run_stage3_planner_discovery] ERROR: json/operation_plan_discoveries.json was not "
            "written by the claude agent.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        # Strip stale preserve_rotation defensively (mirrors planner_review behavior).
        for _op in plan.get("discovered_ops", []):
            if isinstance(_op, dict) and _op.get("action") == "attach_to_wall" and "preserve_rotation" in _op:
                del _op["preserve_rotation"]
    except json.JSONDecodeError as exc:
        raw = plan_path.read_text(encoding="utf-8")
        print(
            f"[run_stage3_planner_discovery] ERROR: operation_plan_discoveries.json is not valid JSON: {exc}\n"
            f"--- raw content (first 2000 chars) ---\n{raw[:2000]}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = REQUIRED_TOP_LEVEL_KEYS - set(plan.keys())
    if missing:
        print(
            f"[run_stage3_planner_discovery] ERROR: missing required top-level keys: {sorted(missing)}. "
            f"Received: {sorted(plan.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    discovered_ops = plan.get("discovered_ops")
    if not isinstance(discovered_ops, list):
        print(
            "[run_stage3_planner_discovery] ERROR: discovered_ops must be a JSON array.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(discovered_ops) > MAX_DISCOVERIES:
        print(
            f"[run_stage3_planner_discovery] ERROR: discovered_ops has {len(discovered_ops)} entries; "
            f"max is {MAX_DISCOVERIES}. Agent must self-cap per spec.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate each op.
    for i, op in enumerate(discovered_ops):
        if not isinstance(op, dict):
            print(
                f"[run_stage3_planner_discovery] ERROR: discovered_ops[{i}] is not a dict: {op!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        action = op.get("action")
        if action not in ALLOWED_ACTIONS:
            print(
                f"[run_stage3_planner_discovery] ERROR: discovered_ops[{i}] has disallowed action: {action!r}. "
                f"Allowed: {sorted(ALLOWED_ACTIONS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if action == "delete_object":
            obj_name = op.get("obj_name", "")
            if not isinstance(obj_name, str) or not _DELETE_OBJ_RE.match(obj_name):
                print(
                    f"[run_stage3_planner_discovery] ERROR: delete_object op[{i}] has invalid obj_name: "
                    f"{obj_name!r}. Must match ^obj_\\d+ (no Floor/Wall/Ceiling).",
                    file=sys.stderr,
                )
                sys.exit(1)

    # review_notes presence + placeholder check (same pattern as planner_review).
    review_notes = plan.get("review_notes")
    if review_notes is None:
        print(
            "[run_stage3_planner_discovery] ERROR: operation_plan_discoveries.json missing 'review_notes'.",
            file=sys.stderr,
        )
        sys.exit(1)
    placeholder_pattern = re.compile(
        r'<(PLACEHOLDER|TODO|FIXME|fill_this_in|template_\w*|agent_response|\w+_here|INSERT_\w*|ADD_\w*)>',
        re.IGNORECASE,
    )
    if not isinstance(review_notes, str) or placeholder_pattern.search(review_notes):
        print(
            f"[run_stage3_planner_discovery] ERROR: review_notes appears to be an unfilled placeholder: "
            f"{review_notes!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    plan["_planner_meta"] = {
        "model": model,
        "image_sha256": sha256_file(cache_anchor_path),
        "generated_by": GENERATED_BY,
        "mode": MODE,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    tmp_path = plan_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, plan_path)
    print(
        f"[run_stage3_planner_discovery] operation_plan_discoveries.json validated "
        f"({len(discovered_ops)} ops) -> {plan_path}",
        file=sys.stderr,
    )


def is_cache_valid(scene_dir: Path, cache_anchor_path: Path) -> bool:
    """Cache anchor is image.png — discovery does NOT depend on operation_plan.json."""
    plan_path = scene_dir / "json" / "operation_plan_discoveries.json"
    if not plan_path.exists():
        return False
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        meta = plan.get("_planner_meta", {})
        if not isinstance(meta, dict):
            return False
        cached_sha = meta.get("image_sha256")
        cached_mode = meta.get("mode")
        if not cached_sha or cached_mode != MODE:
            return False
        return cached_sha == sha256_file(cache_anchor_path)
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invoke Claude CLI stage3-scene-discoverer (anchoring-free) and write json/operation_plan_discoveries.json"
    )
    parser.add_argument("--scene_dir", type=Path, required=True, help="Scene directory containing image.png")
    parser.add_argument(
        "--model", type=str, default="opus", help="Claude model id (default: opus)"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Subprocess timeout in seconds (default: 600)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()

    if not scene_dir.is_dir():
        print(f"[run_stage3_planner_discovery] ERROR: scene_dir not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        print(f"[run_stage3_planner_discovery] ERROR: image.png not found in {scene_dir}", file=sys.stderr)
        sys.exit(1)

    # Cache anchor = image.png (intentional: discovery is independent of operation_plan.json).
    cache_anchor_path = image_path

    if not AGENT_SPEC_PATH.exists():
        print(
            f"[run_stage3_planner_discovery] ERROR: agent spec not found: {AGENT_SPEC_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    if is_cache_valid(scene_dir, cache_anchor_path):
        print(
            "[run_stage3_planner_discovery] discovery cached (image_sha256 match), skipping re-call",
            file=sys.stderr,
        )
        sys.exit(0)

    (scene_dir / "json").mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(AGENT_SPEC_PATH)
    model = args.model

    print(
        f"[run_stage3_planner_discovery] Running claude CLI ({model}) on {scene_dir} ...",
        file=sys.stderr,
    )
    run_claude(scene_dir, system_prompt, model, args.timeout)
    validate_and_inject(scene_dir, cache_anchor_path, model)
    print("[run_stage3_planner_discovery] Done.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[run_stage3_planner_discovery] FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
