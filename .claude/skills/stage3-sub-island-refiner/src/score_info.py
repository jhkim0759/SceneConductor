"""score_info.py — INFO ONLY stub. Scoring has been removed from the island-refiner.

Writes a minimal info.json containing only `iter` and `notes`.
The --render, --masked, --iter-dir, --out, --iter0-info flags are accepted
for backwards compatibility but ignored — no images are loaded or processed.

Usage:
    python score_info.py \\
        --iter-dir   PATH \\
        --masked     PATH \\
        [--iter0-info PATH]

    # Legacy flat-flag form (also accepted, ignored):
    python score_info.py --render PATH --masked PATH --out PATH

Output: <iter-dir>/info.json  (or --out if supplied)

info.json structure:
    {
      "iter": N,
      "notes": "INFO ONLY — score-based gating has been removed from the island-refiner."
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="score_info",
        description=(
            "INFO ONLY stub. Writes minimal info.json (iter + notes). "
            "Score-based gating has been removed from the island-refiner."
        ),
    )
    # Primary interface (iter_step.py uses these)
    ap.add_argument(
        "--iter-dir",
        default=None,
        metavar="PATH",
        dest="iter_dir",
        help="Directory containing render_persp.png. iter number is inferred from dir name.",
    )
    ap.add_argument(
        "--masked",
        default=None,
        metavar="PATH",
        help="Accepted but ignored.",
    )
    ap.add_argument(
        "--iter0-info",
        default=None,
        metavar="PATH",
        dest="iter0_info",
        help="Accepted but ignored.",
    )
    # Legacy flat-flag form kept so any caller using --render/--out doesn't break
    ap.add_argument("--render", default=None, metavar="PATH", help="Accepted but ignored.")
    ap.add_argument("--out", default=None, metavar="PATH", help="Override output path.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Determine output path and iter-dir for iter-number inference.
    if args.iter_dir:
        iter_dir = Path(args.iter_dir).resolve()
        out_path = Path(args.out).resolve() if args.out else iter_dir / "info.json"
    elif args.out:
        out_path = Path(args.out).resolve()
        iter_dir = out_path.parent
    else:
        print("[score_info] ERROR: supply --iter-dir or --out", file=sys.stderr)
        sys.exit(1)

    # Infer iter number from iter_dir directory name (iter_N).
    dir_name = iter_dir.name
    try:
        iter_n = int(dir_name.split("_", 1)[1]) if "_" in dir_name else -1
    except (IndexError, ValueError):
        iter_n = -1

    result = {
        "iter": iter_n,
        "notes": "INFO ONLY — score-based gating has been removed from the island-refiner.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[score_info] iter={iter_n}  Written: {out_path}")


if __name__ == "__main__":
    main()
