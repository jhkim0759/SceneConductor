#!/usr/bin/env python3
"""
run_stage2_director.py — invoke the Claude CLI vision agent to produce
stage2_plan.json and inject a tamper-evident _director_meta block.

Usage:
    python run_stage2_director.py --scene_dir /path/to/scene [--model opus] [--timeout 600]

Exit codes:
    0 — success; json/stage2_plan.json written and validated
    1 — any error (subprocess failure, timeout, missing file, bad JSON, missing keys)
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_TOP_LEVEL_KEYS = {"scene_summary", "polygon_brief", "materials_hint", "lighting_hint"}
AGENT_SPEC_PATH = Path(__file__).parents[3] / "agents" / "stage2-environment-planner.md"
GENERATED_BY = "run_stage2_director.py"


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


def run_claude(scene_dir: Path, system_prompt: str, model: str, timeout: int) -> None:
    """Call the claude CLI to produce json/stage2_plan.json."""
    user_prompt = (
        f"Read image and inputs at {scene_dir} and write json/stage2_plan.json. "
        f"Report ≤80 words."
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
            f"[run_stage2_director] ERROR: claude CLI timed out after {timeout}s. "
            "Re-run or increase --timeout.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        print(
            "[run_stage2_director] ERROR: 'claude' CLI not found in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"[run_stage2_director] ERROR: claude exited {result.returncode}.\n"
            f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_and_inject(scene_dir: Path, image_path: Path, model: str) -> None:
    """Parse json/stage2_plan.json, inject _director_meta, rewrite in-place."""
    plan_path = scene_dir / "json" / "stage2_plan.json"
    if not plan_path.exists():
        print(
            "[run_stage2_director] ERROR: json/stage2_plan.json was not written by the claude agent. "
            "Check that the agent spec instructs the agent to write the file.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raw = plan_path.read_text(encoding="utf-8")
        print(
            f"[run_stage2_director] ERROR: stage2_plan.json is not valid JSON: {exc}\n"
            f"--- raw content (first 2000 chars) ---\n{raw[:2000]}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = REQUIRED_TOP_LEVEL_KEYS - set(plan.keys())
    if missing:
        print(
            f"[run_stage2_director] ERROR: stage2_plan.json is missing required top-level "
            f"keys: {sorted(missing)}. Received keys: {sorted(plan.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    plan["_director_meta"] = {
        "model": model,
        "image_sha256": sha256_file(image_path),
        "generated_by": GENERATED_BY,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[run_stage2_director] stage2_plan.json validated and meta injected -> {plan_path}",
        file=sys.stderr,
    )


def is_cache_valid(scene_dir: Path, image_path: Path) -> bool:
    """Return True if stage2_plan.json exists with a _director_meta matching current image.png sha256."""
    plan_path = scene_dir / "json" / "stage2_plan.json"
    if not plan_path.exists():
        return False
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        meta = plan.get("_director_meta", {})
        if not isinstance(meta, dict):
            return False
        cached_sha = meta.get("image_sha256")
        if not cached_sha:
            return False
        current_sha = sha256_file(image_path)
        return cached_sha == current_sha
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invoke Claude CLI Stage 2 director and inject _director_meta into stage2_plan.json"
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
        print(f"[run_stage2_director] ERROR: scene_dir not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        print(f"[run_stage2_director] ERROR: image.png not found in {scene_dir}", file=sys.stderr)
        sys.exit(1)

    if not AGENT_SPEC_PATH.exists():
        print(
            f"[run_stage2_director] ERROR: agent spec not found: {AGENT_SPEC_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Idempotent cache check: skip the API call if stage2_plan.json already has valid meta
    if is_cache_valid(scene_dir, image_path):
        print("[run_stage2_director] director cached, skipping re-call", file=sys.stderr)
        sys.exit(0)

    # Ensure json/ subdirectory exists
    (scene_dir / "json").mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(AGENT_SPEC_PATH)
    model = args.model

    print(f"[run_stage2_director] Running claude CLI ({model}) on {scene_dir} ...", file=sys.stderr)
    run_claude(scene_dir, system_prompt, model, args.timeout)

    validate_and_inject(scene_dir, image_path, model)
    print("[run_stage2_director] Done.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[run_stage2_director] FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
