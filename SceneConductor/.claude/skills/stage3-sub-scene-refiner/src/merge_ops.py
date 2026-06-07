"""
merge_ops.py — Merge heuristic_ops.json, graph_ops.json, llm_ops.json, and
polygon_clamp_ops.json into operation_plan.json.

Merge rules:
  1. When two ops target the same (action, primary_object), the op from the
     higher-priority source wins.  Source priority (highest wins):
       polygon_clamp (rank 4)  >  LLM (rank 3)  =  graph_tool (rank 3)
       >  heuristic (rank 1)
  2. No duplicate (action, primary_object) pairs in output.
  3. Final list sorted by op priority ascending, then primary_object name.
  4. island_tasks defaults to [] — determined post-render by mode=validation.

Usage:
  python3 merge_ops.py --scene-dir /path/to/scene_dir
  python3 merge_ops.py --scene-dir /path/to/scene_dir \
      --heuristic-ops /custom/heuristic_ops.json \
      --graph-ops /custom/graph_ops.json \
      --llm-ops /custom/llm_ops.json \
      --polygon-clamp-ops /custom/polygon_clamp_ops.json \
      --output /custom/operation_plan.json
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def primary_object(op: dict) -> str:
    """Return the key object name for conflict-detection purposes."""
    action = op.get("action", "")
    if action in ("attach", "attach_to_wall"):
        return op.get("moving_obj", "")
    return op.get("obj_name", "")


def conflict_key(op: dict) -> tuple:
    """Return a (action, primary_object) tuple used as the dedup key."""
    return (op.get("action", ""), primary_object(op))


def load_json(path: Path, label: str) -> dict | None:
    """
    Load a JSON file.  Returns None if the file does not exist (caller warns).
    Aborts with a non-zero exit code on invalid JSON.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"merge_ops: ERROR — invalid JSON in {label} ({path}): {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge(
    heuristic_ops: list[dict],
    graph_ops: list[dict],
    llm_ops: list[dict],
    island_tasks: list,
    polygon_clamp_ops: list[dict] | None = None,
) -> dict:
    """
    Merge four op lists according to the documented rules and return the
    final operation_plan dict.

    Source priority (higher number = wins on conflict):
      heuristic = 1  |  graph_tool = 3  |  llm = 3  |  polygon_clamp = 4
    graph_tool and llm share rank 3; last-inserted wins when both target the
    same object (llm is inserted after graph_tool, so llm wins ties).
    polygon_clamp is inserted last and wins all conflicts — it is the
    authoritative boundary enforcement pass.
    """
    # Map source label → numeric rank used for conflict resolution.
    # polygon_clamp = 4 (highest): vertex-exact boundary enforcement wins everything.
    # llm = 3, graph_tool = 3: LLM and graph share rank; llm inserted last so wins ties.
    # heuristic = 1: lowest priority, applied first.
    SOURCE_RANK = {"heuristic": 1, "graph_tool": 3, "llm": 3, "polygon_clamp": 4}

    def _source_rank(op: dict) -> int:
        return SOURCE_RANK.get(op.get("source", "heuristic"), 1)

    # Build a map from conflict_key → op, applying source-rank priority.
    merged: dict[tuple, dict] = {}

    def _insert(op: dict) -> bool:
        """Insert op into merged; return True if it displaced an existing op."""
        key = conflict_key(op)
        existing = merged.get(key)
        if existing is None or _source_rank(op) >= _source_rank(existing):
            merged[key] = op
            return existing is not None  # True only if we displaced something
        return False

    for op in heuristic_ops:
        op.setdefault("source", "heuristic")
        _insert(op)

    graph_conflicts = 0
    for op in graph_ops:
        op.setdefault("source", "graph_tool")
        if _insert(op):
            graph_conflicts += 1

    llm_conflicts = 0
    for op in llm_ops:
        op.setdefault("source", "llm")
        if _insert(op):
            llm_conflicts += 1

    poly_clamp_conflicts = 0
    _polygon_clamp_ops = polygon_clamp_ops or []
    for op in _polygon_clamp_ops:
        op.setdefault("source", "polygon_clamp")
        if _insert(op):
            poly_clamp_conflicts += 1

    # Sort: primary key = priority (asc), secondary key = primary_object name.
    sorted_ops = sorted(
        merged.values(),
        key=lambda o: (o.get("priority", 999), primary_object(o)),
    )

    return {
        "operation_list": sorted_ops,
        "island_tasks": island_tasks,
        "_merge_summary": {
            "heuristic_count": len(heuristic_ops),
            "graph_count": len(graph_ops),
            "llm_count": len(llm_ops),
            "polygon_clamp_count": len(_polygon_clamp_ops),
            "graph_conflicts": graph_conflicts,
            "llm_conflicts": llm_conflicts,
            "poly_clamp_conflicts": poly_clamp_conflicts,
            "final_count": len(sorted_ops),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge heuristic_ops.json and llm_ops.json into operation_plan.json."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=Path,
        help="Root scene directory.",
    )
    parser.add_argument(
        "--heuristic-ops",
        type=Path,
        default=None,
        help="Override path to heuristic_ops.json (default: <scene_dir>/json/heuristic_ops.json).",
    )
    parser.add_argument(
        "--graph-ops",
        type=Path,
        default=None,
        help=(
            "Optional path to graph_tool_planner ops JSON (default: <scene_dir>/json/graph_ops.json). "
            "These tool-based ops (attach_to_wall, attach) replace the LLM classification pass."
        ),
    )
    parser.add_argument(
        "--llm-ops",
        type=Path,
        default=None,
        help="Override path to llm_ops.json (default: <scene_dir>/json/llm_ops.json).",
    )
    parser.add_argument(
        "--polygon-clamp-ops",
        type=Path,
        default=None,
        help=(
            "Optional path to polygon_clamp_ops.json produced by polygon_vertex_clamp.py. "
            "When provided, its operation_list is merged AFTER all other sources (rank 4) "
            "and wins all source conflicts. Default: None (no-op if omitted)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: <scene_dir>/operation_plan.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_dir: Path = args.scene_dir.resolve()

    heuristic_path: Path = args.heuristic_ops or scene_dir / "json" / "heuristic_ops.json"
    llm_path: Path = args.llm_ops or scene_dir / "json" / "llm_ops.json"
    graph_path: Path = args.graph_ops or scene_dir / "json" / "graph_ops.json"
    output_path: Path = args.output or scene_dir / "operation_plan.json"
    polygon_clamp_path: Path | None = args.polygon_clamp_ops

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    heuristic_data = load_json(heuristic_path, "heuristic_ops")
    if heuristic_data is None:
        print(
            f"merge_ops: WARNING — heuristic_ops not found at {heuristic_path}; using empty list.",
            file=sys.stderr,
        )
        heuristic_data = {}

    graph_data = load_json(graph_path, "graph_ops")
    if graph_data is None:
        print(
            f"merge_ops: WARNING — graph_ops not found at {graph_path}; using empty list.",
            file=sys.stderr,
        )
        graph_data = {}

    llm_data = load_json(llm_path, "llm_ops")
    if llm_data is None:
        print(
            f"merge_ops: WARNING — llm_ops not found at {llm_path}; using empty list.",
            file=sys.stderr,
        )
        llm_data = {}

    polygon_clamp_data: dict = {}
    if polygon_clamp_path is not None:
        if polygon_clamp_path.exists():
            polygon_clamp_data = load_json(polygon_clamp_path, "polygon_clamp_ops") or {}
            if not polygon_clamp_data:
                print(
                    f"merge_ops: WARNING — polygon_clamp_ops not found or empty at {polygon_clamp_path}; using empty list.",
                    file=sys.stderr,
                )
        else:
            print(
                f"merge_ops: WARNING — polygon_clamp_ops file not found at {polygon_clamp_path}; using empty list.",
                file=sys.stderr,
            )

    if not heuristic_data and not graph_data and not llm_data and not polygon_clamp_data:
        print(
            "merge_ops: WARNING — heuristic_ops, graph_ops, llm_ops, and polygon_clamp_ops are all "
            "missing or empty; writing empty operation_plan.json.",
            file=sys.stderr,
        )

    heuristic_ops: list[dict] = heuristic_data.get("operation_list", [])
    graph_ops: list[dict] = graph_data.get("operation_list", [])
    llm_ops: list[dict] = llm_data.get("operation_list", [])
    polygon_clamp_ops: list[dict] = polygon_clamp_data.get("operation_list", [])
    island_tasks: list = []  # determined post-render by mode=validation

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    plan = merge(heuristic_ops, graph_ops, llm_ops, island_tasks, polygon_clamp_ops)
    summary = plan.pop("_merge_summary")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Print summary line
    # ------------------------------------------------------------------
    print(
        f"merge_ops: heuristic={summary['heuristic_count']} "
        f"graph_tool={summary['graph_count']} "
        f"llm={summary['llm_count']} "
        f"polygon_clamp={summary['polygon_clamp_count']} "
        f"final={summary['final_count']}"
    )
    print(f"merge_ops: output written to {output_path}")


if __name__ == "__main__":
    main()
