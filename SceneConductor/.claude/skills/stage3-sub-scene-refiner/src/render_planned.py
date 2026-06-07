"""Render the 5 multi-view PNGs of a planned .blend.

Usage:
    python render_planned.py <scene_dir> <blend_in> [output_subdir]

Defaults `output_subdir` to "planned" → outputs to <scene_dir>/render/<subdir>/.
Wraps blend_ops.session_runner.render_multi_view (Cycles, 5 views).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "blend_ops" / "session_runner"))
from external_blend_tools import render_multi_view


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(f"usage: {sys.argv[0]} <scene_dir> <blend_in> [output_subdir=planned]", file=sys.stderr)
        sys.exit(2)
    scene_dir = Path(sys.argv[1]).resolve()
    blend_in = Path(sys.argv[2]).resolve()
    subdir = sys.argv[3] if len(sys.argv) == 4 else "planned"
    out_dir = scene_dir / "render" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render_planned] blend : {blend_in}")
    print(f"[render_planned] out   : {out_dir}")

    r = render_multi_view(
        blend_in=str(blend_in),
        scene_dir=str(scene_dir),
        output_dir=str(out_dir),
        samples=32,
        resolution=(1024, 768),
        timeout=900,
    )
    print(f"[render_planned] success={r.get('success')} exit={r.get('exit_code')} produced={len(r.get('produced', []))}/{len(r.get('expected', []))}")
    for p in r.get("produced", []):
        print(f"  ✓ {p}")
    missing = [p for p in r.get("expected", []) if p not in r.get("produced", [])]
    for p in missing:
        print(f"  ✗ {p}")
    sys.exit(0 if r.get("success") else 1)


if __name__ == "__main__":
    main()
