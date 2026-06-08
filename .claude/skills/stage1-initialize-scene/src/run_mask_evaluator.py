#!/usr/bin/env python3
"""
run_mask_evaluator.py — invoke the Claude CLI vision agent to evaluate masks
and write merge_plan.json with a tamper-evident _evaluator_meta block.

Usage:
    python run_mask_evaluator.py --scene_dir /path/to/scene [--model opus] [--timeout 600]

Exit codes:
    0 — success; merge_plan.json written and validated
    1 — any error (subprocess failure, timeout, missing file, bad JSON, missing keys)
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_TOP_LEVEL_KEYS = {"merge_groups", "mesh_groups"}
SCRIPT_PATH = Path(__file__).resolve()
AGENT_SPEC_PATH = SCRIPT_PATH.parents[3] / "agents" / "stage1-mask-evaluator.md"
GENERATED_BY = "run_mask_evaluator.py"
SCHEMA_VERSION = "stage1-mask-evaluator-v2-lowest-keep-id"


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


def run_claude(scene_dir: Path, system_prompt: str, model: str, timeout: int) -> float:
    """Call the claude CLI to evaluate masks and write merge_plan.json.

    Returns wall-clock elapsed seconds of the subprocess call.
    """
    user_prompt = (
        f"Evaluate masks at {scene_dir}. "
        f"Write merge_plan.json to {scene_dir}."
    )
    cmd = [
        "claude", "-p", user_prompt,
        "--system-prompt", system_prompt,
        "--tools", "Read,Write",
        "--permission-mode", "bypassPermissions",
        "--model", model,
        "--output-format", "text",
    ]
    t0 = time.monotonic()
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
            f"[run_mask_evaluator] ERROR: claude CLI timed out after {timeout}s. "
            "Re-run --phase eval or increase --timeout.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        print(
            "[run_mask_evaluator] ERROR: 'claude' CLI not found in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(
            f"[run_mask_evaluator] ERROR: claude exited {result.returncode}.\n"
            f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)
    return elapsed


def validate_and_inject(
    scene_dir: Path,
    image_path: Path,
    model: str,
    wall_clock_sec: float,
) -> None:
    """Parse merge_plan.json, inject _evaluator_meta, rewrite in-place.

    Also saves the raw (pre-injection) plan to logs/stage1_eval_raw_response.json.
    """
    plan_path = scene_dir / "merge_plan.json"
    if not plan_path.exists():
        print(
            "[run_mask_evaluator] ERROR: merge_plan.json was not written by the claude agent. "
            "Check that the agent spec instructs the agent to write the file.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_size = plan_path.stat().st_size
    if raw_size == 0:
        print(
            "[run_mask_evaluator] ERROR: merge_plan.json is empty (0 bytes).",
            file=sys.stderr,
        )
        sys.exit(1)

    logs_dir = scene_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = logs_dir / "stage1_eval_raw_response.json"
    shutil.copyfile(plan_path, raw_copy)

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raw = plan_path.read_text(encoding="utf-8")
        print(
            f"[run_mask_evaluator] ERROR: merge_plan.json is not valid JSON: {exc}\n"
            f"--- raw content (first 2000 chars) ---\n{raw[:2000]}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = REQUIRED_TOP_LEVEL_KEYS - set(plan.keys())
    if missing:
        print(
            f"[run_mask_evaluator] ERROR: merge_plan.json is missing required top-level "
            f"keys: {sorted(missing)}. Received keys: {sorted(plan.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    if plan.get("schema_version") != SCHEMA_VERSION:
        print(
            f"[run_mask_evaluator] ERROR: evaluator returned plan without "
            f"schema_version='{SCHEMA_VERSION}'; the prompt may be stale "
            f"(got schema_version={plan.get('schema_version')!r}).",
            file=sys.stderr,
        )
        sys.exit(1)

    plan["_evaluator_meta"] = {
        "model": model,
        "image_sha256": sha256_file(image_path),
        "generated_by": GENERATED_BY,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": SCHEMA_VERSION,
        "evaluator_prompt_sha256": sha256_file(AGENT_SPEC_PATH),
        "evaluator_script_sha256": sha256_file(SCRIPT_PATH),
        "cache_hit": False,
        "wall_clock_sec": round(float(wall_clock_sec), 3),
        "response_byte_size": int(raw_size),
    }

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[run_mask_evaluator] merge_plan.json validated and meta injected -> {plan_path} "
        f"(raw_bytes={raw_size}, wall_sec={wall_clock_sec:.2f}, raw_copy={raw_copy})",
        file=sys.stderr,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invoke Claude CLI mask evaluator and inject _evaluator_meta into merge_plan.json"
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
        print(f"[run_mask_evaluator] ERROR: scene_dir not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        print(f"[run_mask_evaluator] ERROR: image.png not found in {scene_dir}", file=sys.stderr)
        sys.exit(1)

    if not AGENT_SPEC_PATH.exists():
        print(
            f"[run_mask_evaluator] ERROR: agent spec not found: {AGENT_SPEC_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    system_prompt = load_system_prompt(AGENT_SPEC_PATH)
    model = args.model

    print(f"[run_mask_evaluator] Running claude CLI ({model}) on {scene_dir} ...", file=sys.stderr)
    elapsed = run_claude(scene_dir, system_prompt, model, args.timeout)

    validate_and_inject(scene_dir, image_path, model, elapsed)
    print("[run_mask_evaluator] Done.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[run_mask_evaluator] FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
