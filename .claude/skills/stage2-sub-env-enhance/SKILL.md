---
name: stage2-sub-env-enhance
description: Interior look-dev pass that enhances ONLY the lights and stage wall/floor/ceiling colors of an existing Blender `.blend` to match a reference image, leaving object materials and geometry untouched. Trigger on "/stage2-sub-env-enhance".
---

## What this skill does

Given **(a) a reference interior image** and **(b) an existing Blender `.blend`** that already contains a fully populated scene (stage walls/floor/ceiling + layout objects + camera), this skill applies a **look-development pass on the interior environment only**:

1. **Interior lights** — builds the interior rig: `Area_Fill` (overhead cool bounce) + `Practical_L/R` (warm POINT lights) + class-driven lights at every detected lamp/pendant/fluorescent mesh. If the rig ends up with zero LIGHT objects, force-creates a central `Fallback_Light` at 200 W.
2. **Stage material colors** — updates Base Color + Roughness of `Mat_Walls_Stage`, `Mat_Floor_Stage`, `Mat_Ceiling_Stage` (and stage-prefixed siblings like `Mat_Wainscot*`) to match the reference image. Applies the PBR albedo clamp so dark photo pixels don't become near-black albedos.
3. **World** — overwrites to a flat black Background at strength 0. The room is sealed; exterior light cannot reach the interior, so the world contributes zero and any positive strength wastes Cycles samples.
4. **Render + compositor** — Cycles 512 samples, AgX view transform, OIDN denoise, optional Glare and Lens Distortion.

It never touches anything else.

**Exterior light (Sun, Sky, window portals) is intentionally NOT modeled.** Sealed walls block 100% of world emission. See `references/lighting_recipes.md` "What is NOT modeled by this skill" for the conditions under which daylight would need to be reintroduced (requires Stage 3 to author OPEN edges or a future Stage 4.5 boolean window-cut pass).

## Hard rules — what the skill must NOT modify

These rules exist because object materials come from an upstream layout-prediction pipeline; changing them breaks downstream evaluation and consistency.

- **Never touch** any material whose name matches `Material_0*` (e.g. `Material_0.001`, `Material_0.020`).
- **Never touch** any material attached to a mesh whose name starts with `geometry_` (e.g. `geometry_0.005`).
- **Never touch** mesh geometry, camera transform/lens, or object transforms (the skill may only *add* new light objects).
- When uncertain whether a material is "stage" or "object", prefer *not modifying*.
- **Any newly added light object must use fixed energy `200W`** at creation. Built-in named lights (Area_Fill, Practical_L/R) can then have their energy adjusted by the mood presets.
- **If the scene ends up with zero LIGHT objects after the rig builds, force-create `Fallback_Light` at the room centroid.** This guarantee always runs.
- **Strip any pre-existing `Sun` / `Area_Window` / `Ambient_Fill` / `Portal_Window_*` lights** at the start of every run — they're exterior-light leftovers from older runs and never re-created.

See `references/env_rules.md` for the full touchable-vs-protected table.

## Inputs

| Input | Purpose |
|---|---|
| Reference image path | Visual target for wall/floor/ceiling colors and lighting mood |
| Existing `.blend` path | Scene to modify in place |
| Output `.blend` path (optional) | If omitted, overwrites the input |
| Preview PNG path (optional) | If omitted, writes next to the output blend as `<stem>_env_preview.png` |

## Outputs

| Artifact | Notes |
|---|---|
| `<scene_dir>/blend/blender_scene.blend` | Modified in place with environment changes |
| `<scene_dir>/render/blender_scene_env_preview*.png` | Cycles 1024×682 render from the scene's fixed camera, AgX + compositor applied |

## The 3-phase pipeline

### Phase 1 — Read the reference image and derive environment parameters

Open the reference image. Decide on:

- **Wall base color** (sRGB hex) — the dominant non-furniture wall tone, averaged over a clean wall region.
- **Wall lower band color** (optional, sRGB hex) — if the reference shows wainscoting or a two-tone wall, pick the darker lower tone.
- **Floor color** (sRGB hex) — the mid-tone of the floor material.
  - **PBR albedo clamp (HARD)**: floor base color is a *diffuse albedo*, not a sampled photo pixel. After picking the hex, compute Rec.709 luma in **linear** space (`L = 0.2126*R + 0.7152*G + 0.0722*B` with each channel sRGB-decoded to linear). If `L < 0.08`, brighten the hex until `L ≈ 0.10–0.20`. Photo pixels in shadow can be near-black, but a near-black albedo absorbs ~94% of first-bounce light and kills the whole indirect lighting budget — the room will render visibly darker than the reference even with identical lights. Walls have the same risk; clamp similarly to `L ≥ 0.20`.
- **Ceiling color** (sRGB hex) — warm off-white for typical interiors unless the reference says otherwise.
- **Mood** — `neutral_balanced` (default), `cool_fresh`, `soft_diffuse`, or `warm_amber`. Mood shifts only the **color temperature and energy** of the interior light objects — it does not model exterior daylight, since the room is sealed. Legacy time-of-day names (`afternoon` / `morning` / `overcast` / `golden_hour`) are still accepted as aliases for back-compat.
- **Scene scale** — check `scene.unit_settings.scale_length` and the scene bounding box. If the room footprint is under ~1 m, treat as **1/10 scale** and divide all light energies by 10. Full-scale rooms use the default energies.

> If `<scene_dir>/json/stage2_plan.json` exists (written by the Stage 0 director agent), `enhance_env.py` reads it and uses the `materials_hint` hex values + `lighting_hint.mood` as defaults — CLI flags still win when provided.

### Phase 2 — Run the enhancement script

```bash
blender --background <scene_dir>/blend/blender_scene.blend \
        --python src/enhance_env.py -- \
    --output <scene_dir>/blend/blender_scene.blend \
    --preview <scene_dir>/render/blender_scene_env_preview.png \
    --mood neutral_balanced \
    --wall-hex C2A8A6 \
    --wall-lower-hex 9A807C \
    --floor-hex 6F4A2C \
    --ceiling-hex E8DECF
```

All color flags are optional. Omit `--wall-lower-hex` if there is no wainscoting. The mood flag (`neutral_balanced` / `cool_fresh` / `soft_diffuse` / `warm_amber`) selects default color temperatures and energies for the interior light rig; explicit overrides are available (see `--help`).

Scale handling is automatic: the script reads the scene bbox and applies the ×0.1 light-energy factor if the room spans under 1 m. You can force it with `--scale 1_10` or `--scale full`.

The script always:
- Operates in one Blender session (no open/save/reopen) to avoid view_transform and world-node-tree persistence bugs.
- Prints a **protection report** listing every skipped object material by name before it begins modifying anything.
- Prints a **post-config verify** block with world nodes, view transform, samples, and final light energies.

### Phase 3 — Review the preview

Open the preview PNG. Check against the reference for: wall-color hue parity, overall exposure, and the balance between cool `Area_Fill` and warm `Practical_L/R`. If off, rerun Phase 2 with revised color hexes or a different mood; do not modify the skill's protected materials to try to fix it.

See `references/lighting_recipes.md` for mood presets, energy scaling, and common fixes (too yellow → drop practical-energy; too flat → raise fill-energy; too noisy → more samples + clamp indirect).

## Common pitfalls

- **Warm-monoculture amber cast.** If every light source is warm, interior walls always look orange under AgX. The default moods pair a **slightly cool `Area_Fill`** with **warm `Practical_L/R`** — keep that contrast.
- **Light energy wrong for scale.** Scenes authored at ~1/10 scale need ~1/10 light energy. The script auto-detects, but check the log if the first render is blown out.
- **Persistence bugs.** Historical failure: setting `view_transform='AgX'`, saving, reopening — the settings silently revert. The script sidesteps this by doing everything in one Blender session.
- **Touching object materials.** If a prior run modified `Material_0.*` materials, that is out of scope. This skill's job is to leave them alone.
- **Expecting daylight.** The room is sealed. If the reference image shows a bright window, the window itself is a *texture* you'll see on the wall — env-enhance does not model it as a light source. Future Stage 4.5 (wall-opening cuts) will change this; until then, daylight contribution is zero.

## Multi-view render — delegated

This skill **no longer owns the multi-view render step**. The 5-view render (perspective + bev + wide + topcorner + topcorner_opposite) is produced by the project-wide canonical renderer:

> **`.claude/skills/general-multi-view-render/src/render_multi_view.py`**

It consumes the env-enhanced `.blend` directly (post-alignment lights, materials, camera, world) and writes all 5 PNGs to `<scene_dir>/render/`. Brightness alignment from this skill is preserved on disk via `<scene_dir>/render/brightness_align_log.json`; the multi-view renderer reads it via `--brightness-log` to undo the per-primary-vantage calibration for the BEV view.

See `.claude/skills/general-multi-view-render/SKILL.md` for the renderer's full contract (perspective vantage rules, BEV albedo strategy, blocking-wall handling, scene-bbox definition, idempotency).
