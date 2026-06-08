# `stage2_plan.json` — Schema v1.1

> Single source of vision-derived hints consumed by every Stage 2 sub-skill.

## Purpose

`stage2_plan.json` is produced by the `stage2-environment-planner` agent at the very start of Stage 2 (before Stage 1 even runs). It reads the reference image once and emits a structured plan of hints that downstream stages consume as defaults.

**Every field is advisory.** Algorithmic validators downstream (the polygon fitter, the floor luma clamp, the build-stage manifold check) still have final say. If the file is absent or malformed, sub-skills fall back to their current default behaviors.

## File location

`<scene_dir>/json/stage2_plan.json`

## Top-level structure

```json
{
  "schema_version":  "1.1",
  "scene_summary":   "Child's bedroom, ~3m wide, single window on the camera-left wall, vivid yellow-green walls.",
  "polygon_brief":   "rectangular, single window on the camera-left wall, no visible doorway",
  "materials_hint":  { ... },
  "lighting_hint":   { ... },
  "openings_hint":   [ ... ],
  "scale_prior":     { ... },
  "confidence":      { "polygon": 0.9, "materials": 0.85, "openings": 0.6 }
}
```

| Field | Type | Required | Consumed by | Description |
|---|---|---|---|---|
| `schema_version` | string | yes | inspector | `"1.1"` for this revision |
| `scene_summary` | string | yes | stage2-floor-plan-designer | One-paragraph human-readable summary: room type, approximate dimensions, dominant colors, visible openings, notable wall-mount objects |
| `polygon_brief` | string | yes | stage2-floor-plan-designer | Short shape hint: "rectangular" / "L-shaped, alcove on right" / "corner shot, two visible walls meet ~45°" plus opening side hints |
| `materials_hint` | object | yes | enhance_env.py | Wall/floor/ceiling base colors (see §materials_hint) |
| `lighting_hint` | object | yes | enhance_env.py | Mood + interior-light strategy (see §lighting_hint) |
| `openings_hint` | array | no | Stage 4.5 (future) | Detected doorways / windows / archways with rough placement |
| `scale_prior` | object | no | convert.py | Rough monocular estimate of room scale, used to sanity-gate the computed world-scale factor (see §scale_prior) |
| `confidence` | object | no | orchestrator | Per-section self-rated confidence in [0, 1]; sections with confidence < 0.5 are treated as "skip the hint, use algorithmic default" by sub-skills |

---

## `materials_hint`

```json
{
  "materials_hint": {
    "wall_hex":            "A8A93F",
    "wall_lower_hex":      null,
    "floor_hex":           "C0B7A0",
    "ceiling_hex":         "E8DECF"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `wall_hex` | string (6 hex chars, no `#`) | yes | Dominant wall base color in sRGB |
| `wall_lower_hex` | string \| null | no | Lower wall (wainscoting) color if the reference shows a two-tone wall, else `null` |
| `floor_hex` | string | yes | Mid-tone of the floor material |
| `ceiling_hex` | string | yes | Default `"E8DECF"` (warm off-white) for typical interiors |

**PBR albedo clamp:** every hex value must, after sRGB→linear conversion, have Rec.709 luma ≥ 0.20 (walls) / ≥ 0.10 (floor). `enhance_env.py` clamps lower-luma values up to the threshold and logs a warning. The director should already respect this when picking colors.

---

## `lighting_hint`

```json
{
  "lighting_hint": {
    "mood": "neutral_balanced"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `mood` | string | yes | One of `"neutral_balanced"`, `"cool_fresh"`, `"soft_diffuse"`, `"warm_amber"`. Selects interior-light color temperature and energy. External lighting (Sun/Sky/HDRI/window portals) is always disabled — `mood` is the only knob currently wired. Legacy time-of-day names (`afternoon` / `morning` / `overcast` / `golden_hour`) are accepted as aliases for back-compat |

---

## `openings_hint`

> **STATUS: not yet consumed by any stage.** The director writes this block, but Stage 4.5 (boolean wall cuts) does not exist yet, and Stage 3's polygon fitter does not read it either. Keep emitting it so the data is captured at vision time — wiring will happen when Stage 4.5 lands.

Each entry describes one visible doorway / window / archway. Future Stage 4.5 will consume `kind`, `camera_side`, and size estimates for boolean wall cuts. Future Stage 3 may consume `camera_side` + `kind` to bias edge type labels (doorway → `OPEN`, window → `WALL` with hint for cut location).

```json
{
  "openings_hint": [
    {
      "kind":            "window",
      "camera_side":     "left",
      "approx_height":   1.2,
      "approx_width":    0.9,
      "z_from_floor":    0.9,
      "notes":           "single window visible at upper-left of frame, curtain partially closed"
    },
    {
      "kind":            "doorway",
      "camera_side":     "behind",
      "approx_height":   2.0,
      "approx_width":    0.8,
      "z_from_floor":    0.0,
      "notes":           "doorway behind camera — not visible in image; inferred from object arrangement"
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | string | yes | `"window"` \| `"doorway"` \| `"archway"` |
| `camera_side` | string | yes | `"left"` \| `"right"` \| `"front"` \| `"behind"` — which wall (in camera-relative terms) the opening sits on |
| `approx_height` | float | no | Approximate opening height in meters |
| `approx_width` | float | no | Approximate opening width in meters |
| `z_from_floor` | float | no | Approximate bottom-of-opening height above floor in meters. `0` for doorways, `~0.9` for windows |
| `notes` | string | no | Free-form rationale |

`kind == "doorway"` and `kind == "archway"` cause the matching wall edge to be set to `OPEN`. `kind == "window"` keeps the wall edge as `WALL` (the opening is a boolean cut, not a missing wall).

---

## `scale_prior`

> **Consumed by `convert.py`** to sanity-gate the world-scale factor it computes algorithmically. This is a *prior*, not ground truth: `convert.py` keeps its own computed scale and only uses this block to flag/clamp wildly off results.

```json
{
  "scale_prior": {
    "scene_scale_class":        "child",
    "expected_room_footprint_m": [2.5, 5.0],
    "expected_room_height_m":    [2.2, 3.0],
    "reasoning":                 "Kindergarten-height tables and small chairs imply a child-scaled room.",
    "confidence":                0.55
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `scene_scale_class` | string | yes | `"child"` \| `"standard"` \| `"oversized"` \| `"unknown"`. `"child"` = child-sized furniture (kindergarten / nursery); `"standard"` = normal adult interior; `"oversized"` = unusually large furniture/room; `"unknown"` = no scale signal in the image |
| `expected_room_footprint_m` | `[lo, hi]` float pair | yes | Estimated **largest horizontal room dimension** (width or depth, whichever is greater), in meters, as a `[low, high]` range. Convention: `lo ≤ hi`; the true value is expected to fall inside the range |
| `expected_room_height_m` | `[lo, hi]` float pair | yes | Estimated floor-to-ceiling height range in meters, `[low, high]`. **Informational only** for now — `convert.py` does not gate on this yet |
| `reasoning` | string | yes | One short sentence naming the scale cue(s) used (e.g. door height, human reference, chair/table proportions) |
| `confidence` | float | yes | `0.0`–`1.0`. **Consumers IGNORE this entire block when `confidence < 0.5`**, falling back to a hardcoded extent guard |

**CRITICAL — monocular scale is ambiguous.** Absolute scale cannot be recovered from a single image without a known reference. The agent **must give GENEROUS `[lo, hi]` ranges** and set `confidence` near `0` when the image lacks scale cues (no people, doors, or recognizable standard-sized objects). **Never emit a narrow range.** A tight range with high confidence on a cueless image is a contract violation — when in doubt, widen the range and lower confidence so the downstream guard falls back to its hardcoded default.

---

## Versioning and back-compat

- `schema_version = "1.1"`: this version. Required field. **v1.1 adds the optional `scale_prior` block** (consumed by `convert.py`); v1.0 producers/consumers remain compatible since the field is optional.
- If `schema_version` is missing or unknown, sub-skills must log a warning and fall back to defaults.
- Adding new optional fields is non-breaking. Adding required fields or removing fields requires bumping `schema_version`.

---

## Consumer fall-back behavior

| Field absent | Consumer behavior |
|---|---|
| Entire `stage2_plan.json` missing | Each sub-skill uses its current default (no behavior change vs pre-director pipeline) |
| `materials_hint.wall_hex` missing | `enhance_env.py` uses CLI value; if CLI value also absent, uses preset wall color from the chosen mood |
| `polygon_brief` missing | `stage2-floor-plan-designer` runs with no extra context (same as today) |
| `openings_hint` missing or empty | No opening biasing; all edges default to `WALL` |
| `lighting_hint.mood` missing | `enhance_env.py --mood` default (`neutral_balanced`) |
| `scale_prior` missing or `scale_prior.confidence < 0.5` | `convert.py` ignores the prior and uses its hardcoded extent guard (no scale gating from vision) |
| `confidence.<section> < 0.5` | Consumer ignores that hint section and falls back as if absent |
