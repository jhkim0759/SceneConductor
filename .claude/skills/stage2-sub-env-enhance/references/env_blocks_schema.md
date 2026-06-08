# `blender_scene.json` — env-enhance owned blocks

> **Owned by `stage2-sub-env-enhance`.** All other top-level blocks (`meta`, `camera`, `scene`, `objects`, `point_cloud`, `stage`) are owned by `stage2-sub-pointmap-to-separable-stage`. See its `references/scene_json_schema.md` for the canonical schema overview and round-trip invariant.

This file documents only what env-enhance writes:

- [`lighting`](#lighting-array) — interior light rig
- [`world`](#world-block) — sealed-interior world (flat black)
- [`stage_materials`](#stage_materials-block) — wall / floor / ceiling colors
- [`render`](#render-block) — Cycles render + color-management settings
- [`compositor`](#compositor-block) — optional post-render node tree

All five blocks are optional in v2.0. When absent, the builder uses Blender defaults (see `scene_json_schema.md` §2 "Backward compatibility").

---

## `lighting` array

Serialized light rig. Each entry describes one Blender light object. **Interior-only since v2.1**: no Sun, no Sky portals — all entries are POINT / AREA lights placed inside the room. The room is a sealed box (Stage 3 builds Floor + Walls + Ceiling), so exterior light contributes zero and is omitted.

```json
{
  "lighting": [
    {
      "name":              "Area_Fill",
      "type":              "AREA",
      "collection":        "Lighting_Env",
      "location":          [0.0, 1.5, 2.7],
      "rotation_euler":    [3.1416, 0.0, 0.0],
      "energy":            150.0,
      "color":             [0.95, 0.97, 1.0],
      "size":              3.0,
      "cycles":            { "cast_shadow": true }
    },
    {
      "name":              "Practical_L",
      "type":              "POINT",
      "collection":        "Lighting_Env",
      "location":          [-0.9, 0.3, 2.0],
      "rotation_euler":    [0.0, 0.0, 0.0],
      "energy":            15.0,
      "color":             [1.0, 0.88, 0.70],
      "cycles":            { "cast_shadow": true }
    },
    {
      "name":              "Fallback_Light",
      "type":              "POINT",
      "collection":        "Lighting_Env",
      "location":          [0.0, 1.5, 1.3],
      "rotation_euler":    [0.0, 0.0, 0.0],
      "energy":            200.0,
      "color":             [1.0, 1.0, 1.0],
      "cycles":            { "cast_shadow": true }
    }
  ]
}
```

### Light object schema

| Field | Type | Required | Unit | Default | Description |
|-------|------|----------|------|---------|-------------|
| `name` | string | **yes** | — | — | Blender object name; must be unique in the scene |
| `type` | string | **yes** | — | — | `"AREA"` \| `"POINT"` \| `"SPOT"`. `"SUN"` is no longer produced by env-enhance |
| `collection` | string | no | — | `"Lighting_Env"` | Blender collection to link this light into |
| `location` | float[3] | **yes** | meters | — | World position; must be inside the room polygon |
| `rotation_euler` | float[3] | **yes** | radians, XYZ | — | World rotation |
| `energy` | float | **yes** | Watts | — | > 0. New env-enhance-created lights default to 200 W (per skill hard rule); built-in named lights use preset values |
| `color` | float[3] | **yes** | linear RGB [0, 1] | `[1.0, 1.0, 1.0]` | |
| `size` | float\|null | no | meters | `null` | AREA / SPOT only |
| `spread_deg` | float\|null | no | degrees | `null` | SPOT cone half-angle |
| `shadow_soft_size` | float\|null | no | meters | `null` | POINT soft-shadow radius |
| `cycles.cast_shadow` | bool | no | — | `true` | Disable per-light shadow casting |

`SUN`-specific fields (`sun_elevation_deg`, `sun_rotation_deg`) and `cycles.is_portal` were removed in v2.1. They were only used by the Sun + Window-portal rig, which no longer exists.

### Named lights produced by `enhance_env.py`

| Name | Role | Type | Default energy |
|---|---|---|---|
| `Area_Fill` | Overhead cool bounce (ceiling diffuser) | AREA | 150 W |
| `Practical_L` | Warm interior point light (left) | POINT | 15 W |
| `Practical_R` | Warm interior point light (right) | POINT | 15 W |
| `Fallback_Light` | Auto-created central POINT light when the rig would otherwise have zero lights (Rule 2 guarantee) | POINT | 200 W |
| `<class-driven>` | One light per detected lamp / fluorescent / pendant mesh, name derived from object class | AREA / POINT | preset per class |

`Sun`, `Area_Window`, `Ambient_Fill`, `Portal_Window_*` are **deprecated** — env-enhance strips them from any prior run and never re-creates them. A JSON containing those names is tolerated but the lights themselves are deleted on the next env-enhance pass.

---

## `world` block

Sealed-interior world: a flat Background at strength 0. The room is fully enclosed so any positive world strength would be wasted Cycles samples.

```json
{
  "world": {
    "mode":     "flat",
    "color":    [0.0, 0.0, 0.0],
    "strength": 0.0
  }
}
```

| Field | Type | Required | Unit | Default | Description |
|---|---|---|---|---|---|
| `mode` | string | **yes** | — | `"flat"` | Always `"flat"`. Kept as a single-valued constant so `build.py`'s world-dispatch loader still has a key to switch on; no other values are produced by env-enhance. |
| `color` | float[3] | **yes** | linear RGB [0, 1] | `[0.0, 0.0, 0.0]` | Background color |
| `strength` | float | no | dimensionless | `0.0` | Background Strength input |

`"nishita_sky"` and `"hdri"` modes were removed in v2.1. `enhance_env.py` always overwrites the world to flat black at strength 0 on every run. The temporary BEV render in the canonical multi-view renderer (`general-multi-view-render/src/render_multi_view.py`) swaps in its own albedo world for that single view and restores the flat world in a `try/finally`.

---

## `stage_materials` block

PBR material parameters for the stage surfaces (walls / floor / ceiling). Covers only `Mat_Walls_Stage`, `Mat_Floor_Stage`, `Mat_Ceiling_Stage`, and per-wall overrides. Never covers `Material_0.*` materials — those belong to the upstream layout-prediction pipeline (see `scene_json_schema.md` §4).

```json
{
  "stage_materials": {
    "floor": {
      "base_color_linear":   [0.541, 0.479, 0.380, 1.0],
      "base_color_srgb_hex": "C0B7A0",
      "roughness":           0.55,
      "metallic":            0.0,
      "specular_ior_level":  0.5,
      "normal_strength":     1.0,
      "texture_image_path":  null
    },
    "ceiling": {
      "base_color_linear":   [0.810, 0.706, 0.604, 1.0],
      "base_color_srgb_hex": "E8DECF",
      "roughness":           0.9,
      "metallic":            0.0,
      "normal_strength":     1.0,
      "texture_image_path":  null
    },
    "walls": {
      "__default__": {
        "base_color_linear":   [0.392, 0.398, 0.046, 1.0],
        "base_color_srgb_hex": "A8A93F",
        "roughness":           0.85,
        "metallic":            0.0,
        "normal_strength":     1.0,
        "texture_image_path":  null
      },
      "Wall_01": {
        "base_color_linear":   [0.330, 0.233, 0.222, 1.0],
        "base_color_srgb_hex": "9A807C",
        "roughness":           0.75,
        "metallic":            0.0,
        "normal_strength":     1.0,
        "texture_image_path":  null
      }
    }
  }
}
```

### PBR material schema

| Field | Type | Required | Unit | Default | Description |
|---|---|---|---|---|---|
| `base_color_linear` | float[4] | **yes** | linear RGBA [0, 1] | — | Principled BSDF Base Color; alpha = 1.0 |
| `base_color_srgb_hex` | string | no | — | — | Human-readable sRGB hex (6 digits, no `#`) |
| `roughness` | float | **yes** | dimensionless | — | Principled BSDF Roughness ∈ [0, 1] |
| `metallic` | float | no | dimensionless | `0.0` | Principled BSDF Metallic ∈ [0, 1] |
| `specular_ior_level` | float\|null | no | dimensionless | `null` | `null` means leave at Blender default (0.5) |
| `normal_strength` | float | no | dimensionless | `1.0` | Normal Map node Strength; only applied when a normal texture is present |
| `texture_image_path` | string\|null | no | relative path | `null` | Diffuse texture image, relative to scene directory |

### `walls` sub-block

- `__default__` — applied to all wall objects not explicitly listed.
- `"Wall_NN"` — per-wall override; must match an entry in `stage.wall_objects[]`.

### PBR albedo clamp (hard)

Every base color is a *diffuse albedo*, not a sampled photo pixel. After picking the hex, compute Rec.709 luma in linear space:

```
L = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin
```

with each channel sRGB-decoded. If `L < 0.20` for walls/ceiling or `L < 0.10` for floor, brighten until the threshold is met. A near-black albedo absorbs > 90 % of indirect light and renders the room visibly darker than the reference. `enhance_env.py` enforces this floor and logs a warning when it has to brighten an incoming hex.

---

## `render` block

Cycles render and color-management settings.

```json
{
  "render": {
    "engine":          "CYCLES",
    "samples":         512,
    "resolution":      [1024, 682],
    "view_transform":  "AgX",
    "look":            "AgX - Medium High Contrast",
    "exposure":        0.0,
    "clamp_indirect":  5.0,
    "use_denoising":   true,
    "denoiser":        "OPENIMAGEDENOISE",
    "max_bounces":     12,
    "diffuse_bounces": 6,
    "glossy_bounces":  6,
    "file_format":     "PNG",
    "color_depth":     "16"
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `engine` | string | `"CYCLES"` | `"CYCLES"` \| `"EEVEE"` |
| `samples` | int | `512` | Path-tracing sample count |
| `resolution` | int[2] | from `camera.resolution` | Overrides camera resolution for the final render |
| `view_transform` | string | `"AgX"` | Color-management view transform |
| `look` | string | `"AgX - Medium High Contrast"` | AgX look preset |
| `exposure` | float | `0.0` | Color management exposure offset (stops) |
| `clamp_indirect` | float | `5.0` | `cycles.sample_clamp_indirect` |
| `use_denoising` | bool | `true` | Enable OIDN / OptiX denoiser |
| `denoiser` | string | `"OPENIMAGEDENOISE"` | `"OPENIMAGEDENOISE"` \| `"OPTIX"` |
| `max_bounces` | int | `12` | Cycles max bounces |
| `diffuse_bounces` | int | `6` | Diffuse bounces |
| `glossy_bounces` | int | `6` | Glossy bounces |
| `file_format` | string | `"PNG"` | Output image format |
| `color_depth` | string | `"16"` | Output bit depth: `"8"` \| `"16"` |

---

## `compositor` block

Optional post-render node tree.

```json
{
  "compositor": {
    "glare": {
      "enabled":    true,
      "glare_type": "FOG_GLOW",
      "mix":        -0.92,
      "threshold":  1.0,
      "size":       6
    },
    "lens_distortion": {
      "enabled":    true,
      "distortion": 0.006,
      "dispersion": 0.002
    }
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `glare.enabled` | bool | `true` | Insert a Glare node |
| `glare.glare_type` | string | `"FOG_GLOW"` | Blender Glare node type |
| `glare.mix` | float | `-0.92` | Glare mix factor ∈ [-1, 1]; negative reduces glare |
| `glare.threshold` | float | `1.0` | Pixel brightness above which glare applies |
| `glare.size` | int | `6` | Glare quality/size exponent |
| `lens_distortion.enabled` | bool | `true` | Insert a Lens Distortion node |
| `lens_distortion.distortion` | float | `0.006` | Barrel/pincushion amount |
| `lens_distortion.dispersion` | float | `0.002` | Chromatic aberration amount |

Skip the entire compositor by passing `--no-compositor` to `enhance_env.py`; the block is then omitted on output.
