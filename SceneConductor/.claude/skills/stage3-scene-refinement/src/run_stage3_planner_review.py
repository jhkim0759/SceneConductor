#!/usr/bin/env python3
"""
run_stage3_planner_review.py — invoke the Claude CLI vision agent to review the
heuristic operation_plan.json and write json/operation_plan_revised.json with a
tamper-evident _planner_meta block.

Usage:
    python run_stage3_planner_review.py --scene_dir /path/to/scene [--model opus] [--timeout 600]

Exit codes:
    0 — success; json/operation_plan_revised.json written and validated
    1 — any error (subprocess failure, timeout, missing file, bad JSON, missing keys,
        invalid action, empty-drop sentinel, template-string review_notes detected)
    2 — claude CLI invocation failed (non-zero return)
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_TOP_LEVEL_KEYS = {"operation_list"}
ALLOWED_ACTIONS = {
    "update_layout", "update_rotation", "flip_yaw_180", "update_size",
    "delete_object", "attach", "attach_to_wall",
}
AGENT_SPEC_PATH = Path(__file__).resolve().parents[3] / "agents" / "stage3-scene-planner.md"
GENERATED_BY = "run_stage3_planner_review.py"
MODE = "planner_review"

# Regex for valid delete_object target names.
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
    # Strip leading YAML frontmatter block (--- ... ---\n)
    stripped = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
    return stripped.strip()


def _classify_wall_normal(nx: float, ny: float) -> str:
    """Return a human-readable spatial label for a wall given its outward XY normal."""
    if abs(nx) >= abs(ny):
        return "LEFT wall (normal -X)" if nx < 0 else "RIGHT wall (normal +X)"
    else:
        return "FRONT-NEAR wall (camera side, normal -Y)" if ny < 0 else "BACK wall (far end of room, normal +Y)"


def _build_wall_geometry_hint(scene_dir: Path) -> str:
    """
    Build a wall geometry hint block from blender_scene.json.

    Uses polygon_vertices + wall_edges to derive each wall's XY bounding box and
    outward normal, then classifies it as LEFT / RIGHT / FRONT-NEAR / BACK.
    Returns an empty string if blender_scene.json is missing or malformed.
    """
    bs_path = scene_dir / "json" / "blender_scene.json"
    if not bs_path.exists():
        return ""
    try:
        bs = json.loads(bs_path.read_text(encoding="utf-8"))
        stage = bs.get("stage", {})
        poly_verts = stage.get("polygon_vertices", [])  # list of [x, y]
        wall_edges = stage.get("wall_edges", [])         # list of {from, to, object, ...}
        wall_ambiguous_path = scene_dir / "json" / "wall_ambiguous.json"
        ambiguous_list: list = []
        if wall_ambiguous_path.exists():
            try:
                wa = json.loads(wall_ambiguous_path.read_text(encoding="utf-8"))
                ambiguous_list = wa.get("ambiguous", [])
            except Exception:
                pass
    except Exception:
        return ""

    if not poly_verts or not wall_edges:
        return ""

    lines = ["Wall geometry (for reference when mapping image observations to wall_id):"]
    for edge in wall_edges:
        wall_name = edge.get("object", "")
        i_from = edge.get("from")
        i_to = edge.get("to")
        if i_from is None or i_to is None or i_from >= len(poly_verts) or i_to >= len(poly_verts):
            continue
        v0 = poly_verts[i_from]
        v1 = poly_verts[i_to]

        x_min = min(v0[0], v1[0])
        x_max = max(v0[0], v1[0])
        y_min = min(v0[1], v1[1])
        y_max = max(v0[1], v1[1])

        # Outward normal: for CCW polygon, outward = (dy, -dx)
        dx = v1[0] - v0[0]
        dy = v1[1] - v0[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            continue
        nx = dy / length
        ny = -dx / length

        label = _classify_wall_normal(nx, ny)
        lines.append(
            f"  - {wall_name}: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]"
            f"  → {label}"
        )

    lines.append("")
    lines.append(
        "When an image shows an object on the RIGHT side of the frame, choose the wall_id "
        "whose label says 'RIGHT'. When on the LEFT, choose the 'LEFT' wall. "
        "FRONT-NEAR is the wall closest to the camera (low y values). "
        "BACK is the far wall at the back of the room (high y values)."
    )

    if not ambiguous_list:
        lines.append("")
        lines.append(
            "wall_ambiguous is EMPTY — the auto-pass wall_id assignments are geometry-confident "
            "(each object was assigned to its nearest wall by bbox distance). "
            "Do NOT change any attach_to_wall.wall_id unless the reference image STRONGLY and "
            "unambiguously contradicts the bbox-based assignment shown above. "
            "If uncertain, keep the auto-pass wall_id as-is."
        )
    else:
        lines.append("")
        lines.append(
            f"wall_ambiguous lists these objects as uncertain: {ambiguous_list}. "
            "For those objects you may reassign wall_id based on image evidence. "
            "For all other attach_to_wall ops, keep the auto-pass wall_id unless the image "
            "STRONGLY contradicts it."
        )

    return "\n".join(lines)


def run_claude(scene_dir: Path, system_prompt: str, model: str, timeout: int) -> None:
    """Call the claude CLI to run mode=planner_review and write json/operation_plan_revised.json."""
    wall_geometry_hint = _build_wall_geometry_hint(scene_dir)

    user_prompt = (
        f"Run in MODE=planner_review.\n"
        f"\n"
        f"Goal: review the heuristic operation_plan.json against the reference image and produce a "
        f"revised operation plan at json/operation_plan_revised.json.\n"
        f"\n"
        f"Read the following files (use the Read tool):\n"
        f"  1. {scene_dir}/image.png\n"
        f"  2. {scene_dir}/inputs/object_state_annotated_mask.png\n"
        f"  3. {scene_dir}/operation_plan.json\n"
        f"  4. {scene_dir}/inputs/relation_graph.json\n"
        f"  5. {scene_dir}/json/object_state.json\n"
        f"  6. {scene_dir}/json/blend_info.json\n"
        f"\n"
        + (f"{wall_geometry_hint}\n\n" if wall_geometry_hint else "")
        + f"Allowed action values and their required fields:\n"
        f"  - update_layout:   obj_name (string), location ([x, y, z] absolute world coords), reason (string)\n"
        f"  - update_rotation: obj_name (string), rotation_euler ([rx, ry, rz] in radians), reason (string)\n"
        f"  - flip_yaw_180:    obj_name (string), reason (string)\n"
        f"  - update_size:     obj_name (string), scale ([sx, sy, sz]), reason (string). "
        f"WARNING: Use sparingly — misuse can corrupt object proportions. Only emit when an object is visibly wrong-sized.\n"
        f"  - delete_object:   obj_name (string matching ^obj_\\d+; never Floor/Wall/Ceiling), reason (string). "
        f"Only delete objects that are clearly spurious or duplicates. When in doubt, keep.\n"
        f"  - attach:          anchor_obj (string), moving_obj (string), relation (e.g. \"on\"), reason (string), "
        f"priority (int), source (string). Used to ground an object onto another (e.g. cup on table).\n"
        f"  - attach_to_wall:  wall_obj (e.g. \"Wall_03\"), moving_obj (string), "
        f"wall_ambiguous (bool), reason (string), priority (int), source (string), t_along_m (float). "
        f"Used to anchor a wall-mounted object (picture, TV, etc.) to a specific wall. "
        f"NOTE: preserve_rotation is NOT a valid field for attach_to_wall — wall-tangent alignment (preserve_rotation=False) is the Stage 3 standard. Do NOT include preserve_rotation in any attach_to_wall op, and strip it from carry-forward ops.\n"
        f"\n"
        f"Full-replacement rule: the output operation_list is a complete replacement for the original. "
        f"Include every operation you want applied — original ops from operation_plan.json that should "
        f"be kept must be carried forward explicitly.\n"
        f"\n"
        f"Conservatism rule for delete_object: only delete objects that are clearly spurious or "
        f"duplicates. When in doubt, keep.\n"
        f"\n"
        f"Write the result to {scene_dir}/json/operation_plan_revised.json with this schema:\n"
        f"{{\n"
        f"  \"operation_list\": [\n"
        f"    {{\"action\": \"update_layout\", \"obj_name\": \"obj_3\", \"location\": [x, y, z], \"reason\": \"...\"}},\n"
        f"    {{\"action\": \"delete_object\", \"obj_name\": \"obj_7\", \"reason\": \"...\"}},\n"
        f"    ...\n"
        f"  ],\n"
        f"  \"review_notes\": \"<natural language summary of what changed and why>\"\n"
        f"}}\n"
        f"\n"
        f"The review_notes field must be a genuine natural-language summary — do not leave it as a "
        f"template placeholder. Report your reasoning concisely."
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
            f"[run_stage3_planner_review] ERROR: claude CLI timed out after {timeout}s. "
            "Re-run or increase --timeout.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        print(
            "[run_stage3_planner_review] ERROR: 'claude' CLI not found in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"[run_stage3_planner_review] ERROR: claude exited {result.returncode}.\n"
            f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}",
            file=sys.stderr,
        )
        sys.exit(2)


def validate_and_inject(scene_dir: Path, cache_anchor_path: Path, model: str) -> None:
    """Parse json/operation_plan_revised.json, validate, inject _planner_meta, rewrite atomically."""
    plan_path = scene_dir / "json" / "operation_plan_revised.json"
    if not plan_path.exists():
        print(
            "[run_stage3_planner_review] ERROR: json/operation_plan_revised.json was not written by the "
            "claude agent. Check that the agent spec instructs the agent to write the file.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        # Stage 3 standard: preserve_rotation=False is enforced by dispatcher fallback.
        # Strip any stale "preserve_rotation" field from carry-forward attach_to_wall ops
        # so the LLM never sees True values that could be re-emitted unchanged.
        for _op in plan.get("operation_list", []):
            if _op.get("action") == "attach_to_wall" and "preserve_rotation" in _op:
                del _op["preserve_rotation"]
    except json.JSONDecodeError as exc:
        raw = plan_path.read_text(encoding="utf-8")
        print(
            f"[run_stage3_planner_review] ERROR: operation_plan_revised.json is not valid JSON: {exc}\n"
            f"--- raw content (first 2000 chars) ---\n{raw[:2000]}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = REQUIRED_TOP_LEVEL_KEYS - set(plan.keys())
    if missing:
        print(
            f"[run_stage3_planner_review] ERROR: operation_plan_revised.json is missing required top-level "
            f"keys: {sorted(missing)}. Received keys: {sorted(plan.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    operation_list = plan.get("operation_list")
    if not isinstance(operation_list, list):
        print(
            "[run_stage3_planner_review] ERROR: operation_list must be a JSON array.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate each operation's action field.
    for i, op in enumerate(operation_list):
        if not isinstance(op, dict):
            print(
                f"[run_stage3_planner_review] ERROR: operation_list[{i}] is not a dict: {op!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        action = op.get("action")
        if action not in ALLOWED_ACTIONS:
            print(
                f"[run_stage3_planner_review] ERROR: operation_list[{i}] has disallowed action: {action!r}. "
                f"Allowed: {sorted(ALLOWED_ACTIONS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if action == "delete_object":
            obj_name = op.get("obj_name", "")
            if not isinstance(obj_name, str) or not _DELETE_OBJ_RE.match(obj_name):
                print(
                    f"[run_stage3_planner_review] ERROR: delete_object op[{i}] has invalid obj_name: {obj_name!r}. "
                    "Must match ^obj_\\d+ (no Floor/Wall/Ceiling).",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Sentinel: LLM dropped all ops when the original plan was non-empty.
    original_plan_path = scene_dir / "operation_plan.json"
    if original_plan_path.exists():
        try:
            original_plan = json.loads(original_plan_path.read_text(encoding="utf-8"))
            original_ops = original_plan.get("operation_list", [])
            if isinstance(original_ops, list) and len(original_ops) > 0 and len(operation_list) == 0:
                print(
                    "[run_stage3_planner_review] ERROR: operation_list is empty but the original "
                    "operation_plan.json had a non-empty list. The LLM appears to have dropped all "
                    "operations. Re-run or inspect the agent output.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except (json.JSONDecodeError, OSError):
            pass  # Cannot read original — skip this sentinel check.

    # Sentinel: review_notes is missing or is an unfilled template placeholder.
    review_notes = plan.get("review_notes")
    if review_notes is None:
        print(
            "[run_stage3_planner_review] ERROR: operation_plan_revised.json is missing 'review_notes'.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Improved check: look for actual template patterns, not just any < and >
    # Template patterns are things like <placeholder>, <TODO>, etc.
    placeholder_pattern = re.compile(r'<(PLACEHOLDER|TODO|FIXME|fill_this_in|template_\w*|agent_response|\w+_here|INSERT_\w*|ADD_\w*)>', re.IGNORECASE)
    if not isinstance(review_notes, str) or placeholder_pattern.search(review_notes):
        print(
            f"[run_stage3_planner_review] ERROR: review_notes appears to be an unfilled template "
            f"placeholder: {review_notes!r}",
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

    # Atomic write: write to .tmp then os.replace.
    tmp_path = plan_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, plan_path)
    print(
        f"[run_stage3_planner_review] operation_plan_revised.json validated and meta injected -> {plan_path}",
        file=sys.stderr,
    )


def is_cache_valid(scene_dir: Path, cache_anchor_path: Path) -> bool:
    """Return True if operation_plan_revised.json exists with _planner_meta matching current anchor sha256 and mode."""
    plan_path = scene_dir / "json" / "operation_plan_revised.json"
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
        current_sha = sha256_file(cache_anchor_path)
        return cached_sha == current_sha
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invoke Claude CLI stage3-scene-planner (mode=planner_review) and inject _planner_meta"
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
        print(f"[run_stage3_planner_review] ERROR: scene_dir not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        print(f"[run_stage3_planner_review] ERROR: image.png not found in {scene_dir}", file=sys.stderr)
        sys.exit(1)

    # Cache anchor is operation_plan.json (the revised plan is derived from it).
    cache_anchor_path = scene_dir / "operation_plan.json"
    if not cache_anchor_path.exists():
        print(
            f"[run_stage3_planner_review] ERROR: operation_plan.json not found in {scene_dir}. "
            "Run the heuristic scene refiner first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not AGENT_SPEC_PATH.exists():
        print(
            f"[run_stage3_planner_review] ERROR: agent spec not found: {AGENT_SPEC_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Idempotent cache check: skip the API call if operation_plan_revised.json already has valid meta
    if is_cache_valid(scene_dir, cache_anchor_path):
        print("[run_stage3_planner_review] planner_review cached, skipping re-call", file=sys.stderr)
        sys.exit(0)

    # Ensure json/ subdirectory exists
    (scene_dir / "json").mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(AGENT_SPEC_PATH)
    model = args.model

    print(f"[run_stage3_planner_review] Running claude CLI ({model}) on {scene_dir} ...", file=sys.stderr)
    run_claude(scene_dir, system_prompt, model, args.timeout)

    validate_and_inject(scene_dir, cache_anchor_path, model)
    print("[run_stage3_planner_review] Done.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[run_stage3_planner_review] FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
