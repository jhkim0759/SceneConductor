"""Orchestrate the SceneConductor stage-progression report videos.

Given a processed scene_dir, produce report/stage_videos/*.mp4 (+ matching
.blend) for the pipeline stages, using the conventions developed for the TEASER
report: white background (Standard view transform + white world + gentle sun),
a moderate interior focal length, a small render-border crop plus a post-render
edge crop, Cycles by default (EEVEE fallback for the refine views whose GALP
origin camera renders black under Cycles), and a per-video .blend saved next to
each mp4.

Run inside the `sceneconductor` conda env (needs cv2 for the post-crop):
    python make_stage_videos.py <scene_dir> [options]

This driver only shells out to Blender (per render script) and OpenCV (post
crop); it makes no rendering decisions of its own beyond wiring paths + args.
See SKILL.md for prerequisites and the full data-flow explanation.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# External pipeline scripts this workflow reuses (override via env if moved).
APPLY_PLAN = os.environ.get(
    "APPLY_PLAN",
    str(Path.home() / "3dscene/SceneConductor/.claude/skills/stage3-sub-scene-refiner/src/apply_plan.py"),
)
EXTRACT_STAGE = os.environ.get(
    "EXTRACT_STAGE",
    str(Path.home() / ".claude/skills/scene-conductor-demo-video/scripts/extract_stage.py"),
)
DEFAULT_BLENDER = os.environ.get(
    "BLENDER", str(Path.home() / "blender/blender-4.2.1-linux-x64/blender"))

ALL_VIDEOS = ["1_popup", "2_env_interior", "2-1_env_external",
              "3_first_refine", "3-1_to_final", "4_island_persp", "4_island_bev",
              "5_turntable"]


def run(cmd, log):
    print("$", " ".join(str(c) for c in cmd))
    with open(log, "w") as fh:
        p = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"  -> exit {p.returncode} (see {log})")
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_dir")
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--videos", default="all",
                    help="comma list from: " + ",".join(ALL_VIDEOS) + " (or 'all')")
    ap.add_argument("--focal-interior", type=float, default=24.0,
                    help="lens (mm) for interior perspective views (1,3,3-1)")
    ap.add_argument("--crop-render-px", type=int, default=0,
                    help="optional in-render border crop per edge (default 0; "
                         "edge cropping is done uniformly in post instead)")
    ap.add_argument("--crop-post-px", type=int, default=20,
                    help="uniform post crop per edge on every finished mp4 "
                         "(1160x880 base - 20/edge -> 1120x840 = 4:3)")
    ap.add_argument("--engine", default="cycles", choices=["cycles", "eevee"],
                    help="engine for popup / env / island views")
    ap.add_argument("--refine-engine", default="eevee", choices=["cycles", "eevee"],
                    help="engine for 3 / 3-1 (their GALP origin cam is black under Cycles)")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--cycles-light-scale", type=float, default=1.0,
                    help="lighting multiplier for the Cycles clips; 1.0 = native "
                         "(default, matches calibrated original lights). Lower to dim.")
    ap.add_argument("--original-lights", dest="original_lights", action="store_true",
                    default=True,
                    help="(DEFAULT) light every clip with the blend's OWN lights "
                         "(white backdrop kept, world lighting=0; no substitute sun/fill)")
    ap.add_argument("--substitute-lights", dest="original_lights", action="store_false",
                    help="opposite of --original-lights: use the white-bg substitute "
                         "sun + interior fill instead of the blend's own lights")
    ap.add_argument("--save-blend", action="store_true",
                    help="also save a full .blend next to each mp4 (reopen/tweak a "
                         "stage). OFF by default: each .blend is a full scene copy "
                         "and 7/scene fills the disk fast.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--operation-plan", default=None,
                    help="plan to apply (default <scene>/operation_plan.json)")
    ap.add_argument("--size-from", default=None,
                    help="extra plan whose update_size ops are merged in "
                         "(fixes plans that dropped mesh-group size ops)")
    ap.add_argument("--island-group", default=None,
                    help="relation group for video 4 (default: first table group found)")
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse existing work/corrected_planned.* and stage jsons")
    args = ap.parse_args()

    scene = Path(args.scene_dir).resolve()
    work = scene / "report" / "stage_videos"
    work.mkdir(parents=True, exist_ok=True)
    tmp = scene / "report" / "_stage_videos_work"
    tmp.mkdir(parents=True, exist_ok=True)
    logs = tmp / "logs"
    logs.mkdir(exist_ok=True)
    B = args.blender
    videos = ALL_VIDEOS if args.videos == "all" else args.videos.split(",")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), BLENDER=B,
               SPV_SAVE_BLEND="1" if args.save_blend else "0",
               SPV_ORIGINAL_LIGHTS="1" if args.original_lights else "0")

    # --- key inputs ---
    blend = scene / "blend"
    raw_json = scene / "report" / "demo" / "stage_data" / "05_s2_scene.json"
    demo_blend = scene / "report" / "demo" / "demo_animated.blend"
    planned_src = blend / "stage3-sub-planned.blend"   # walled base to reset
    corrected = tmp / "corrected_planned.blend"
    corrected_json = tmp / "corrected_planned.json"
    final_stage = tmp / "planned_final_stage.json"

    eng = lambda v: args.refine_engine if v in ("3_first_refine", "3-1_to_final") else args.engine

    # ---------------- data prep ----------------
    if not args.skip_prep:
        # 1) plan: optionally merge update_size ops from a fuller plan
        plan_path = Path(args.operation_plan) if args.operation_plan else scene / "operation_plan.json"
        plan = json.loads(plan_path.read_text())
        ops = plan["operation_list"]
        if args.size_from:
            extra = json.loads(Path(args.size_from).read_text())["operation_list"]
            have = {o.get("obj_name") for o in ops if o["action"] == "update_size"}
            add = [o for o in extra if o["action"] == "update_size" and o.get("obj_name") not in have]
            ops = ([o for o in ops if o["action"] == "update_size"] + add
                   + [o for o in ops if o["action"] != "update_size"])
            plan["operation_list"] = ops
            print(f"[prep] merged size ops: +{len(add)} -> {sum(1 for o in ops if o['action']=='update_size')} total")
        full_plan = tmp / "operation_plan_full.json"
        full_plan.write_text(json.dumps(plan))

        # 2) reset walled base to raw, apply full plan -> corrected_planned
        base_raw = blend / "_spv_base_raw.blend"
        run([B, "-b", planned_src, "--python", HERE / "reset_empties.py", "--",
             raw_json, base_raw], logs / "reset.log")
        rc = subprocess.run([sys.executable, APPLY_PLAN, str(full_plan), str(base_raw), str(corrected)],
                            env=env)
        if rc.returncode != 0:
            print("[prep] apply_plan failed");
        base_raw.unlink(missing_ok=True)

        # 3) extract corrected_planned obj transforms
        run([B, "-b", corrected, "--python", EXTRACT_STAGE, "--", corrected_json],
            logs / "extract.log")

        # 4) planned -> final stage = corrected_planned + island member loc/rot.
        # WALL-mounted decor (board/poster/tv/picture/shelf/...) is EXCLUDED from
        # the island override: the table+chair island align (M_anchor) wrongly
        # 180-flips/relocates wall items, so keep them at their planned (clip-3
        # end) pose. clip 3-1 then moves only the floor furniture (user request).
        cp = json.loads(corrected_json.read_text())
        out = {"objects": dict(cp["objects"])}
        _wall_kw = ("poster", "board", "bulletin", "chalkboard", "whiteboard",
                    "television", " tv", "picture", "painting", "mirror", "window",
                    "shelf", "clock", "sign", "banner", "frame")
        _cls_path = scene / "inputs" / "object_class.json"
        _cls = json.loads(_cls_path.read_text()) if _cls_path.exists() else {}
        def _is_wall(name):
            idx = name.split("_")[1] if "_" in name else name
            c = (_cls.get(idx) or "").lower()
            return any(k.strip() in c for k in _wall_kw)
        groups = sorted(
            Path(p).parent for p in glob.glob(str(scene / "relation_groups" / "*" / "metadata.json")))
        _frozen = []
        for g in groups:
            island = g / "island.blend"
            meta = g / "metadata.json"
            if not island.exists() or not meta.exists():
                continue
            gjson = tmp / f"{g.name}_scene.json"
            run([B, "-b", island, "--python", HERE / "island_to_scene.py", "--", meta, gjson],
                logs / f"island_{g.name}.log")
            if not gjson.exists():
                continue
            members = json.loads(gjson.read_text())["objects"]
            for n, info in members.items():
                if n in out["objects"]:
                    if _is_wall(n):                      # keep planned pose for wall decor
                        _frozen.append(n); continue
                    out["objects"][n] = {"location": info["location"],
                                         "rotation_euler": info["rotation_euler"],
                                         "scale": out["objects"][n]["scale"]}  # keep planned scale
        final_stage.write_text(json.dumps(out))
        print(f"[prep] planned_final_stage written ({len(groups)} groups, "
              f"froze {len(_frozen)} wall objs: {_frozen})")

    # pick island group for video 4 (handles both layouts: an explicit
    # island_init.blend, or per-iter simple_refiner/refiner iter_*/island.blend).
    import re

    def _iter_blends(gdir):
        cands = (glob.glob(str(gdir / "simple_refiner" / "iter_*" / "island.blend"))
                 + glob.glob(str(gdir / "refiner" / "iter_*" / "island.blend")))
        cands = [c for c in cands if ".bak" not in c and "before_" not in c]
        return sorted(cands, key=lambda c: (int(re.search(r"iter_(\d+)", c).group(1))
                                            if re.search(r"iter_(\d+)", c) else 9999))

    cand_groups = ([scene / "relation_groups" / args.island_group] if args.island_group
                   else sorted(Path(p).parent for p in
                               glob.glob(str(scene / "relation_groups" / "*" / "island.blend"))))
    grp = g_init = g_final = g_initjson = g_finaljson = None
    best = -1
    for g in cand_groups:
        if not (g / "island.blend").exists() or not (g / "metadata.json").exists():
            continue
        its = _iter_blends(g)
        init = (g / "island_init.blend") if (g / "island_init.blend").exists() else (
            Path(its[0]) if its else None)
        if init is None:
            continue
        score = len(its) + (5 if "table" in g.name.lower() else 0)  # prefer most-refined / table groups
        if score > best:
            best, grp, g_init, g_final = score, g, init, g / "island.blend"
    if grp:
        g_initjson = tmp / f"{grp.name}_init.json"
        g_finaljson = tmp / f"{grp.name}_final.json"
        run([B, "-b", g_init, "--python", EXTRACT_STAGE, "--", g_initjson], logs / "g_init.log")
        run([B, "-b", g_final, "--python", EXTRACT_STAGE, "--", g_finaljson], logs / "g_final.log")
    if not (g_init and Path(g_init).exists() and g_initjson and Path(g_initjson).exists()):
        skip = [v for v in videos if v.startswith("4_island")]
        if skip:
            print(f"[render] no usable island data -> skipping {skip}")
        videos = [v for v in videos if not v.startswith("4_island")]
    else:
        print(f"[render] island group = {grp.name}")

    # ---------------- render each video ----------------
    # All tunables flow to the render scripts via env (SPV_LENS / SPV_CROP_PX /
    # SPV_ENGINE) so the per-script positional arg parsing stays simple.
    S = HERE
    interior = {"1_popup", "3_first_refine", "3-1_to_final"}   # views that use --focal-interior
    jobs = {
        "1_popup":         [B, "-b", demo_blend, "--python", S/"render_popup_flat.py", "--",
                            work/"1_stage1_popup.mp4", 1, 154, args.samples],
        "2_env_interior":  [B, "-b", blend/"blender_scene.blend", "--python", S/"build_stage_sync.py", "--",
                            work/"2_env_build_interior.mp4", "--view", "interior", "--samples", args.samples],
        "2-1_env_external":[B, "-b", blend/"blender_scene.blend", "--python", S/"build_stage_sync.py", "--",
                            work/"2-1_stage2_external_build.mp4", "--view", "external", "--samples", args.samples],
        "3_first_refine":  [B, "-b", corrected, "--python", S/"build_refine_multi.py", "--",
                            work/"3_stage3_first_refine.mp4", raw_json, corrected_json, "--samples", args.samples],
        "3-1_to_final":    [B, "-b", corrected, "--python", S/"build_refine_multi.py", "--",
                            work/"3-1_stage3_to_final.mp4", corrected_json, final_stage, "--samples", args.samples],
        "4_island_persp":  [B, "-b", g_init, "--python", S/"build_island_2view.py", "--",
                            work/"4_island_G1_perspective.mp4", g_initjson, g_finaljson,
                            "--view", "persp", "--samples", args.samples],
        "4_island_bev":    [B, "-b", g_init, "--python", S/"build_island_2view.py", "--",
                            work/"4_island_G1_bev.mp4", g_initjson, g_finaljson,
                            "--view", "bev", "--samples", args.samples],
        "5_turntable":     [B, "-b", corrected, "--python", S/"build_stage_sync.py", "--",
                            work/"5_stage3_final_turntable.mp4", "--view", "turntable",
                            "--poses", final_stage, "--samples", args.samples],
    }
    rendered = []
    for v in videos:
        cmd = jobs.get(v)
        if cmd is None:
            print(f"[render] unknown video {v}"); continue
        envv = dict(env, SPV_ENGINE=eng(v), SPV_CROP_PX=str(args.crop_render_px))
        if eng(v) == "cycles":   # dim only the Cycles clips; leave EEVEE refine as-is
            envv["SPV_LIGHT_SCALE"] = str(args.cycles_light_scale)
        if v in interior:
            envv["SPV_LENS"] = str(args.focal_interior)
        print(f"[render] {v} ({eng(v)})")
        with open(logs / f"{v}.log", "w") as fh:
            subprocess.run([str(c) for c in cmd], env=envv, stdout=fh, stderr=subprocess.STDOUT)
        rendered.append(str(cmd[6]))   # the output .mp4 path

    # ---------------- post crop (only the mp4s rendered this run) ----------------
    if args.crop_post_px > 0:
        post_crop(rendered, args.crop_post_px)
    print(f"\nDone. Videos in {work}")


def post_crop(paths, px):
    """Crop `px` px off ALL FOUR edges of every finished mp4 (uniform). With the
    1160x880 render base, px=20 -> 1120x840 (exactly 4:3)."""
    import cv2
    for p in [pp for pp in paths if os.path.exists(pp)]:
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap.get(3)); h = int(cap.get(4))
        if w <= 2 * px or h <= 2 * px:
            print(f"[postcrop] SKIP {os.path.basename(p)} ({w}x{h} too small / unreadable)")
            cap.release(); continue
        nw, nh = w - 2 * px, h - 2 * px
        tmpf = p + ".t.mp4"
        vw = cv2.VideoWriter(tmpf, cv2.VideoWriter_fourcc(*"mp4v"), fps, (nw, nh))
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            vw.write(fr[px:h - px, px:w - px]); n += 1
        cap.release(); vw.release()
        if n > 0:
            os.replace(tmpf, p)
            print(f"[postcrop] {os.path.basename(p)} {w}x{h} -> {nw}x{nh} ({round(nw/nh,3)}, {n}f)")
        else:
            os.remove(tmpf)
            print(f"[postcrop] SKIP {os.path.basename(p)} (0 frames)")


if __name__ == "__main__":
    main()
