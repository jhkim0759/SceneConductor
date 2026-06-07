#!/usr/bin/env python3
"""
run_stage3_validation.py — invoke the Claude CLI vision agent to run
mode=validation and write json/island_groups.json with a tamper-evident
_planner_meta block.

Usage:
    python run_stage3_validation.py --scene_dir /path/to/scene [--model opus] [--timeout 600]

Exit codes:
    0 — success; json/island_groups.json written and validated
    1 — any error (subprocess failure, timeout, missing file, bad JSON, missing keys,
        template-string rationale sentinel detected)
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from deterministic_gate import run_pregate

REQUIRED_TOP_LEVEL_KEYS = {"groups_needing_island", "rationale", "target_spec"}
AGENT_SPEC_PATH = Path(__file__).parents[3] / "agents" / "stage3-scene-planner.md"
GENERATED_BY = "run_stage3_validation.py"
MODE = "validation"
_PREGATE_VERSION_TAG = "pregate_v1"
_VALIDATION_SCHEMA_VERSION = "v2_synthetic_dedup"

# Regex that matches the deterministic-fallback template-string sentinel rationales.
_TEMPLATE_SENTINEL_RE = re.compile(
    r"^(Needs refinement|Skipped): (\w+) (group|anchor)"
)

_SYNTHETIC_ID_RE = re.compile(r"^S\d+$")
_OBJ_ID_RE       = re.compile(r"^obj_\d+$")
_STAGE_ANCHOR_RE = re.compile(r"^(Floor|Ceiling|Wall_\d+)$")


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


def run_claude(
    scene_dir: Path,
    system_prompt: str,
    model: str,
    timeout: int,
    pregate_result: dict,
) -> None:
    """Call the claude CLI to run mode=validation and write json/island_groups.json."""
    pregate_flagged: list = pregate_result.get("flagged_groups", [])
    pregate_rationale_lines = (
        "\n".join(
            f"  - {g}: {pregate_result['rationale_by_group'].get(g, '(no detail)')}"
            for g in pregate_flagged
        )
        if pregate_flagged
        else "  (none)"
    )

    pregate_block = (
        "DETERMINISTIC PRE-GATE FINDINGS (computed from world AABBs and relation_graph edge types "
        "— these are ground-truth geometric anomalies):\n\n"
        "Groups already flagged by deterministic checks (MUST be included in your "
        "`groups_needing_island` regardless of your visual judgment):\n"
        f"{pregate_rationale_lines}\n\n"
        "Your job is to (a) confirm these flagged groups belong in groups_needing_island and "
        "add a concise rationale, AND (b) flag ANY ADDITIONAL groups whose visual arrangement "
        "disagrees with the reference. The deterministic pre-gate skips wall-anchored groups "
        "and edge types other than on_top_of/seated_around — vision judgment is the only signal "
        "for those."
    )

    user_prompt = (
        f"Read scene at {scene_dir}, including "
        f"render/planned/blender_scene_view_perspective.png and inputs/relation_graph.json. "
        f"Run in MODE=validation: visually inspect the planned render against the reference image.png "
        f"and decide which relation groups need island refinement. "
        f"Write json/island_groups.json with these THREE top-level keys:\n\n"
        f"1. \"groups_needing_island\": [<group_ids>] — list of group ids that need island-level refinement.\n"
        f"2. \"rationale\": {{<group_id>: <natural-language reason citing visual evidence from the reference image>}}. "
        f"The rationale must NOT use template strings — give a sentence per group that mentions "
        f"what you actually see. Report ≤80 words.\n"
        f"3. \"target_spec\": {{<group_id>: <spec object>}} — for EVERY group_id in groups_needing_island, "
        f"emit a target_spec object with these fields:\n"
        f"   - anchor_role: short label (e.g. \"long_table\", \"round_table\", \"tv_stand\", \"bed\", \"sofa\", \"shelf\").\n"
        f"   - member_count: int >= 1, count of non-anchor members in this group (must match the count in relation_graph for that group).\n"
        f"   - pattern: closed enum — \"ring\" | \"row\" | \"2+2+1\" | \"2+2+1+1\" | \"L\" | \"T\" | \"cluster\" | \"free\".\n"
        f"     If \"free\", you MUST also write free_note.\n"
        f"   - facing: closed enum — \"toward_anchor\" | \"away_from_anchor\" | \"parallel_to_anchor_long_axis\" | \"mixed\".\n"
        f"     If \"mixed\", you MUST also write free_note.\n"
        f"   - spacing: closed enum — \"even_along_each_edge\" | \"even_around_anchor\" | \"tight\" | \"loose\".\n"
        f"   - clearance_m: float >= 0.0, minimum perpendicular distance (in metres) from anchor surface to nearest member surface in the intended layout.\n"
        f"   - free_note: short prose; REQUIRED iff pattern=\"free\" or facing=\"mixed\", else OPTIONAL.\n\n"
        f"Derive each target_spec from the REFERENCE IMAGE (not the current render). "
        f"The target_spec is your interpretation of how this group SHOULD look in the final scene — "
        f"this is the SOLE intent signal the island-refiner will use.\n\n"
        f"Examples:\n"
        f"  \"G2_table_middle\": {{\n"
        f"    \"anchor_role\": \"long_table\", \"member_count\": 5,\n"
        f"    \"pattern\": \"2+2+1\", \"facing\": \"toward_anchor\",\n"
        f"    \"spacing\": \"even_along_each_edge\", \"clearance_m\": 0.10,\n"
        f"    \"free_note\": \"classroom-style with 2 chairs per long side at quarter positions + 1 end chair\"\n"
        f"  }}\n"
        f"  \"G_tv_on_shelf\": {{\n"
        f"    \"anchor_role\": \"shelf\", \"member_count\": 1,\n"
        f"    \"pattern\": \"cluster\", \"facing\": \"toward_anchor\",\n"
        f"    \"spacing\": \"tight\", \"clearance_m\": 0.00\n"
        f"  }}\n\n"
        f"{pregate_block}"
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
            f"[run_stage3_validation] ERROR: claude CLI timed out after {timeout}s. "
            "Re-run or increase --timeout.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        print(
            "[run_stage3_validation] ERROR: 'claude' CLI not found in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"[run_stage3_validation] ERROR: claude exited {result.returncode}.\n"
            f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)


_VALID_PATTERNS = {"ring", "row", "2+2+1", "2+2+1+1", "L", "T", "cluster", "free"}
_VALID_FACINGS = {"toward_anchor", "away_from_anchor", "parallel_to_anchor_long_axis", "mixed"}
_VALID_SPACINGS = {"even_along_each_edge", "even_around_anchor", "tight", "loose"}
_REQUIRED_SPEC_KEYS = {"anchor_role", "member_count", "pattern", "facing", "spacing", "clearance_m"}


def _validate_target_spec(gid: str, spec: dict) -> None:
    """Validate a single target_spec dict for group `gid`. Exits on any violation."""
    if not isinstance(spec, dict):
        print(
            f"[run_stage3_validation] ERROR: target_spec['{gid}'] must be a dict "
            f"(got {type(spec).__name__}).",
            file=sys.stderr,
        )
        sys.exit(1)
    missing_spec_keys = _REQUIRED_SPEC_KEYS - set(spec.keys())
    if missing_spec_keys:
        print(
            f"[run_stage3_validation] ERROR: target_spec['{gid}'] is missing required keys: "
            f"{sorted(missing_spec_keys)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    member_count = spec["member_count"]
    if not isinstance(member_count, int) or member_count < 1:
        print(
            f"[run_stage3_validation] ERROR: target_spec['{gid}'].member_count must be int >= 1 "
            f"(got {member_count!r}).",
            file=sys.stderr,
        )
        sys.exit(1)
    clearance_m = spec["clearance_m"]
    if not isinstance(clearance_m, (int, float)) or clearance_m < 0.0:
        print(
            f"[run_stage3_validation] ERROR: target_spec['{gid}'].clearance_m must be float >= 0.0 "
            f"(got {clearance_m!r}).",
            file=sys.stderr,
        )
        sys.exit(1)
    pattern = spec["pattern"]
    if pattern not in _VALID_PATTERNS:
        print(
            f"[run_stage3_validation] ERROR: target_spec['{gid}'].pattern={pattern!r} is not in "
            f"{sorted(_VALID_PATTERNS)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    facing = spec["facing"]
    if facing not in _VALID_FACINGS:
        print(
            f"[run_stage3_validation] ERROR: target_spec['{gid}'].facing={facing!r} is not in "
            f"{sorted(_VALID_FACINGS)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    spacing = spec["spacing"]
    if spacing not in _VALID_SPACINGS:
        print(
            f"[run_stage3_validation] ERROR: target_spec['{gid}'].spacing={spacing!r} is not in "
            f"{sorted(_VALID_SPACINGS)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    free_note_required = pattern == "free" or facing == "mixed"
    if free_note_required:
        free_note = spec.get("free_note", "")
        if not isinstance(free_note, str) or not free_note.strip():
            print(
                f"[run_stage3_validation] ERROR: target_spec['{gid}'].free_note must be a non-empty "
                f"string when pattern='free' or facing='mixed' "
                f"(pattern={pattern!r}, facing={facing!r}).",
                file=sys.stderr,
            )
            sys.exit(1)
    for str_key in ("anchor_role",):
        if not isinstance(spec[str_key], str):
            print(
                f"[run_stage3_validation] ERROR: target_spec['{gid}'].{str_key} must be a string "
                f"(got {type(spec[str_key]).__name__}).",
                file=sys.stderr,
            )
            sys.exit(1)


def validate_and_inject(
    scene_dir: Path, image_path: Path, model: str, pregate_result: dict
) -> None:
    """Parse json/island_groups.json, validate, inject _planner_meta and pre-gate union, rewrite in-place."""
    plan_path = scene_dir / "json" / "island_groups.json"
    if not plan_path.exists():
        print(
            "[run_stage3_validation] ERROR: json/island_groups.json was not written by the "
            "claude agent. Check that the agent spec instructs the agent to write the file.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raw = plan_path.read_text(encoding="utf-8")
        print(
            f"[run_stage3_validation] ERROR: island_groups.json is not valid JSON: {exc}\n"
            f"--- raw content (first 2000 chars) ---\n{raw[:2000]}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = REQUIRED_TOP_LEVEL_KEYS - set(plan.keys())
    if missing:
        print(
            f"[run_stage3_validation] ERROR: island_groups.json is missing required top-level "
            f"keys: {sorted(missing)}. Received keys: {sorted(plan.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Hard-fail if any rationale value matches the deterministic-fallback template sentinel.
    rationale = plan.get("rationale", {})
    if isinstance(rationale, dict):
        for group_id, reason_text in rationale.items():
            if isinstance(reason_text, str) and _TEMPLATE_SENTINEL_RE.match(reason_text):
                print(
                    f"[run_stage3_validation] ERROR: stage3-scene-planner did not produce a real plan. "
                    f"Template-string sentinel detected in rationale[{group_id!r}]: {reason_text!r}",
                    file=sys.stderr,
                )
                sys.exit(1)

    # --- Union: merge pre-gate flagged groups into vision output ---------------
    pregate_flagged: list = pregate_result.get("flagged_groups", [])
    vision_flagged: set = set(plan.get("groups_needing_island", []))
    final_flagged: list = sorted(vision_flagged | set(pregate_flagged))

    plan["groups_needing_island"] = final_flagged

    # For groups caught by pre-gate but not by vision, add pre-gate rationale entry
    if not isinstance(plan.get("rationale"), dict):
        plan["rationale"] = {}
    for g in pregate_flagged:
        if g not in plan["rationale"]:
            pregate_reason = pregate_result.get("rationale_by_group", {}).get(g, "")
            plan["rationale"][g] = f"[pregate] {pregate_reason}"

    # --- Synthetic islands: validate and merge into final_flagged --------------
    raw_synthetics = plan.get("synthetic_groups", [])
    if not isinstance(raw_synthetics, list):
        print(
            "[run_stage3_validation] ERROR: island_groups.json 'synthetic_groups' must be a list "
            f"(got {type(raw_synthetics).__name__}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate schema of every synthetic entry before any dedup logic.
    seen_synthetic_ids: set = set()
    for i, entry in enumerate(raw_synthetics):
        if not isinstance(entry, dict):
            print(
                f"[run_stage3_validation] ERROR: synthetic_groups[{i}] must be a dict "
                f"(got {type(entry).__name__}).",
                file=sys.stderr,
            )
            sys.exit(1)
        required_synthetic_keys = {"group_id", "member", "anchor", "reason", "target_spec"}
        missing_keys = required_synthetic_keys - set(entry.keys())
        if missing_keys:
            print(
                f"[run_stage3_validation] ERROR: synthetic_groups[{i}] is missing required keys: "
                f"{sorted(missing_keys)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        gid = entry["group_id"]
        if not isinstance(gid, str) or not _SYNTHETIC_ID_RE.match(gid):
            print(
                f"[run_stage3_validation] ERROR: synthetic_groups[{i}].group_id={gid!r} does not "
                f"match ^S\\d+$ pattern.",
                file=sys.stderr,
            )
            sys.exit(1)
        if gid in seen_synthetic_ids:
            print(
                f"[run_stage3_validation] ERROR: duplicate synthetic group_id {gid!r} in "
                f"synthetic_groups.",
                file=sys.stderr,
            )
            sys.exit(1)
        seen_synthetic_ids.add(gid)
        member = entry["member"]
        if not isinstance(member, str) or not _OBJ_ID_RE.match(member):
            print(
                f"[run_stage3_validation] ERROR: synthetic_groups[{i}] ({gid}).member={member!r} "
                f"does not match ^obj_\\d+$ pattern.",
                file=sys.stderr,
            )
            sys.exit(1)
        anchor = entry["anchor"]
        if not isinstance(anchor, str) or (
            not _OBJ_ID_RE.match(anchor) and not _STAGE_ANCHOR_RE.match(anchor)
        ):
            print(
                f"[run_stage3_validation] ERROR: synthetic_groups[{i}] ({gid}).anchor={anchor!r} "
                f"must match ^obj_\\d+$ or be one of Floor/Ceiling/Wall_NN.",
                file=sys.stderr,
            )
            sys.exit(1)
        reason = entry["reason"]
        if not isinstance(reason, str) or not reason.strip():
            print(
                f"[run_stage3_validation] ERROR: synthetic_groups[{i}] ({gid}).reason must be a "
                f"non-empty string.",
                file=sys.stderr,
            )
            sys.exit(1)
        if _TEMPLATE_SENTINEL_RE.match(reason):
            print(
                f"[run_stage3_validation] ERROR: template-string sentinel detected in "
                f"synthetic_groups[{i}] ({gid}).reason: {reason!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        _validate_target_spec(gid, entry["target_spec"])

    # Build covered-members set: all members already handled by a flagged graph group.
    relation_graph_path = scene_dir / "inputs" / "relation_graph.json"
    covered_members: set = set()
    if relation_graph_path.exists():
        try:
            rg = json.loads(relation_graph_path.read_text(encoding="utf-8"))
            for grp in rg.get("groups", {}).values() if isinstance(rg.get("groups"), dict) else []:
                gid_key = grp.get("group_id") if isinstance(grp, dict) else None
                if gid_key and gid_key in final_flagged:
                    for m in grp.get("members", []):
                        covered_members.add(m)
            # Also support groups as a list
            if isinstance(rg.get("groups"), list):
                for grp in rg["groups"]:
                    if isinstance(grp, dict):
                        gid_key = grp.get("group_id")
                        if gid_key and gid_key in final_flagged:
                            for m in grp.get("members", []):
                                covered_members.add(m)
        except (json.JSONDecodeError, KeyError):
            pass  # Best-effort; missing/malformed graph means no covered members to deduplicate.

    # Best-effort anchor/member existence check against object_class.json.
    # object_class.json keys are bare numeric strings ("19"), so normalize to "obj_19"
    # before comparing against synthetic member/anchor ids ("obj_19").
    object_class_path = scene_dir / "inputs" / "object_class.json"
    known_obj_ids: set | None = None
    if object_class_path.exists():
        try:
            oc = json.loads(object_class_path.read_text(encoding="utf-8"))
            if isinstance(oc, dict):
                known_obj_ids = {f"obj_{k}" if not str(k).startswith("obj_") else str(k)
                                 for k in oc.keys()}
        except json.JSONDecodeError:
            known_obj_ids = None

    # Dedup and merge surviving synthetic entries into plan.
    kept_synthetics: list = []
    dropped_count = 0
    for entry in raw_synthetics:
        gid = entry["group_id"]
        member = entry["member"]
        if member in covered_members:
            print(
                f"[run_stage3_validation] dropping synthetic {gid}: member {member} already "
                f"covered by an island group",
                file=sys.stderr,
            )
            dropped_count += 1
            continue
        # Validate obj existence if object_class.json is available.
        if known_obj_ids is not None:
            if member not in known_obj_ids:
                print(
                    f"[run_stage3_validation] ERROR: synthetic {gid}.member={member!r} not found "
                    f"in inputs/object_class.json — agent may have hallucinated this object.",
                    file=sys.stderr,
                )
                sys.exit(1)
            anchor = entry["anchor"]
            if _OBJ_ID_RE.match(anchor) and anchor not in known_obj_ids:
                print(
                    f"[run_stage3_validation] ERROR: synthetic {gid}.anchor={anchor!r} not found "
                    f"in inputs/object_class.json — agent may have hallucinated this object.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Stage anchors (Floor/Ceiling/Wall_NN) have no canonical JSON list;
            # skip validation and let the downstream Blender script reject unknown names.
        if gid in final_flagged:
            print(
                f"[run_stage3_validation] ERROR: synthetic group_id {gid!r} collides with an "
                f"existing relation_graph group id in final_flagged — this should not happen with "
                f"the S-prefix convention.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Merge into plan structures.
        # NOTE: plan["groups_needing_island"] is the same list object as final_flagged
        # (bound at line ~302), so appending once is enough — do NOT append to both.
        final_flagged.append(gid)
        plan["rationale"][gid] = entry["reason"]
        if not isinstance(plan.get("target_spec"), dict):
            plan["target_spec"] = {}
        plan["target_spec"][gid] = entry["target_spec"]
        kept_synthetics.append(entry)

    plan["synthetic_groups"] = kept_synthetics

    graph_group_count = len(final_flagged) - len(kept_synthetics)
    print(
        f"[run_stage3_validation] synthetic_groups: kept={len(kept_synthetics)} "
        f"dropped={dropped_count} total_island_groups={len(final_flagged)} "
        f"(graph={graph_group_count} synthetic={len(kept_synthetics)})",
        file=sys.stderr,
    )

    # --- Validate target_spec for every group in groups_needing_island --------
    # This loop now covers both graph groups and surviving synthetic S* groups
    # because synthetic ids were appended to final_flagged above.
    target_spec_map = plan.get("target_spec", {})
    if not isinstance(target_spec_map, dict):
        print(
            "[run_stage3_validation] ERROR: island_groups.json 'target_spec' must be a dict "
            f"(got {type(target_spec_map).__name__}).",
            file=sys.stderr,
        )
        sys.exit(1)

    for gid in final_flagged:
        spec = target_spec_map.get(gid)
        if spec is None:
            print(
                f"[run_stage3_validation] ERROR: target_spec missing for group '{gid}'. "
                f"Every group in groups_needing_island must have a target_spec entry.",
                file=sys.stderr,
            )
            sys.exit(1)
        _validate_target_spec(gid, spec)

    print(
        f"[run_stage3_validation] target_spec validated for {len(final_flagged)} group(s).",
        file=sys.stderr,
    )

    # Inject pre-gate summary block
    plan["_pregate"] = {
        "flagged_groups": pregate_flagged,
        "thresholds": pregate_result.get("thresholds"),
        "generator_version": pregate_result.get("meta", {}).get("version"),
    }

    if pregate_flagged:
        print(
            f"[run_stage3_validation] Pre-gate union: vision={sorted(vision_flagged)} + "
            f"pregate={sorted(pregate_flagged)} → final={final_flagged}",
            file=sys.stderr,
        )

    plan["_planner_meta"] = {
        "model": model,
        "image_sha256": sha256_file(image_path),
        "generated_by": GENERATED_BY,
        "mode": MODE,
        "pregate_version_tag": _PREGATE_VERSION_TAG,
        "validation_schema_version": _VALIDATION_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[run_stage3_validation] island_groups.json validated and meta injected -> {plan_path}",
        file=sys.stderr,
    )


def is_cache_valid(scene_dir: Path, image_path: Path) -> bool:
    """
    Return True if island_groups.json exists with:
      - _planner_meta matching current image sha256 and mode
      - _planner_meta.pregate_version_tag == _PREGATE_VERSION_TAG
      - _pregate.generator_version == "1"
    Any missing or mismatched field invalidates the cache.
    """
    plan_path = scene_dir / "json" / "island_groups.json"
    if not plan_path.exists():
        return False
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        meta = plan.get("_planner_meta", {})
        if not isinstance(meta, dict):
            return False
        cached_sha = meta.get("image_sha256")
        cached_mode = meta.get("mode")
        cached_pregate_tag = meta.get("pregate_version_tag")
        cached_schema_version = meta.get("validation_schema_version")
        if not cached_sha or cached_mode != MODE:
            return False
        if cached_pregate_tag != _PREGATE_VERSION_TAG:
            print(
                f"[run_stage3_validation] Cache miss: pregate_version_tag mismatch "
                f"(cached={cached_pregate_tag!r}, expected={_PREGATE_VERSION_TAG!r})",
                file=sys.stderr,
            )
            return False
        if cached_schema_version != _VALIDATION_SCHEMA_VERSION:
            print(
                f"[run_stage3_validation] Cache miss: validation_schema_version mismatch "
                f"(cached={cached_schema_version!r}, expected={_VALIDATION_SCHEMA_VERSION!r})",
                file=sys.stderr,
            )
            return False
        # Also check the embedded _pregate block
        pregate_block = plan.get("_pregate", {})
        if not isinstance(pregate_block, dict):
            return False
        if pregate_block.get("generator_version") != "1":
            print(
                "[run_stage3_validation] Cache miss: _pregate.generator_version != '1'",
                file=sys.stderr,
            )
            return False
        current_sha = sha256_file(image_path)
        return cached_sha == current_sha
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invoke Claude CLI stage3-scene-planner (mode=validation) and inject _planner_meta"
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
        print(f"[run_stage3_validation] ERROR: scene_dir not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        print(f"[run_stage3_validation] ERROR: image.png not found in {scene_dir}", file=sys.stderr)
        sys.exit(1)

    if not AGENT_SPEC_PATH.exists():
        print(
            f"[run_stage3_validation] ERROR: agent spec not found: {AGENT_SPEC_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Idempotent cache check: skip the API call if island_groups.json already has valid meta
    if is_cache_valid(scene_dir, image_path):
        print("[run_stage3_validation] validation cached, skipping re-call", file=sys.stderr)
        sys.exit(0)

    # Ensure json/ subdirectory exists
    (scene_dir / "json").mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(AGENT_SPEC_PATH)
    model = args.model

    # Run deterministic pre-gate before the vision call
    print(f"[run_stage3_validation] Running deterministic pre-gate on {scene_dir} ...", file=sys.stderr)
    pregate_result = run_pregate(scene_dir)
    pregate_flagged = pregate_result.get("flagged_groups", [])
    if pregate_flagged:
        print(
            f"[run_stage3_validation] Pre-gate flagged {len(pregate_flagged)} group(s): {pregate_flagged}",
            file=sys.stderr,
        )
    else:
        print("[run_stage3_validation] Pre-gate: no groups flagged.", file=sys.stderr)

    print(f"[run_stage3_validation] Running claude CLI ({model}) on {scene_dir} ...", file=sys.stderr)
    run_claude(scene_dir, system_prompt, model, args.timeout, pregate_result)

    validate_and_inject(scene_dir, image_path, model, pregate_result)
    print("[run_stage3_validation] Done.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[run_stage3_validation] FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
