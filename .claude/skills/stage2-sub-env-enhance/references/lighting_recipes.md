# Lighting recipes — interior-only

The room is a sealed box (Stage 3 builds Floor + Walls + Ceiling), so there is no Sun, no Sky, and no window portals. The mood presets shift only the **color temperature and energy** of the three interior light objects (`Area_Fill`, `Practical_L`, `Practical_R`) plus any class-driven lamps the scene already contains.

Preset names are **color-semantic** (not time-of-day) to avoid confusion with exterior lighting models that this skill does not implement.

## Mood presets

Pick one based on the reference image's interior color cast.

### `neutral_balanced` (default — cool fill + warm practicals)

| Parameter | Full scale | 1/10 scale | Notes |
|---|---|---|---|
| Area_Fill energy | 150 W | 15 W | Overhead cool bounce |
| Area_Fill color | (0.95, 0.97, 1.0) | — | Slightly cool — pairs with warm practicals |
| Practical energy | 15 W each | 1.5 W each | Two POINT lights L/R |
| Practical color | (1.0, 0.88, 0.70) | — | Tungsten warm |
| Exposure | 0.0 | — | |

Good for living rooms, bedrooms, anywhere with a balanced interior look.

### `cool_fresh` — bluish, airy

| Parameter | Full scale | Notes |
|---|---|---|
| Area_Fill color | (0.92, 0.97, 1.0) | More blue |
| Practical color | (1.0, 0.92, 0.80) | Less amber than neutral_balanced |
| Exposure | 0.0 | |

Good for clean, airy interiors.

### `soft_diffuse` — low-contrast, near-neutral

| Parameter | Full scale | Notes |
|---|---|---|
| Area_Fill energy | 200 W | Higher overhead fill to compensate for muted practicals |
| Practical energy | 8 W each | Dimmed |
| Practical color | (1.0, 0.95, 0.90) | Near-neutral |
| Exposure | +0.2 | Brighten the low-key result |

Good for soft, shadowless scenes.

### `warm_amber` — strong warm cast

| Parameter | Full scale | Notes |
|---|---|---|
| Area_Fill color | (1.0, 0.92, 0.78) | Warm overhead |
| Practical color | (1.0, 0.78, 0.55) | Heavy amber |
| Practical energy | 20 W each | Brighter |
| Exposure | -0.3 | |

Good for dramatic interiors where the warm cast dominates.

## Legacy alias names

For back-compat, the old time-of-day names are still accepted on `--mood` and in `stage2_plan.json` `lighting_hint.mood`, mapped to the canonical key with a console warning:

| Legacy name | Canonical name |
|---|---|
| `afternoon` | `neutral_balanced` |
| `morning` | `cool_fresh` |
| `overcast` | `soft_diffuse` |
| `golden_hour` | `warm_amber` |

New callers should use the canonical names directly.

## Scale handling

The script reads the scene bounding box.

- Room footprint diagonal ≥ 2.0 m → **full scale** — use the energies above.
- Room footprint diagonal < 2.0 m → **1/10 scale** — divide every energy by 10 (values shown in the 1/10 column for `neutral_balanced`; other presets scale identically).

Override with `--scale full` or `--scale 1_10`.

## Light rig structure

After the rig builds, the scene always contains at minimum:

| Light | Type | Location |
|---|---|---|
| `Area_Fill` | AREA | Just below the ceiling, centered above the room |
| `Practical_L` | POINT | Left side of the room, ~70% of room height |
| `Practical_R` | POINT | Right side of the room, ~70% of room height |
| 0..N class-driven | varies | At the location of each detected lamp / pendant / fluorescent mesh |
| `Fallback_Light` | POINT | Room centroid; **only created when the entire rig would otherwise have zero lights** (Rule 2 guarantee) |

## Common fixes

| Symptom | Fix |
|---|---|
| Render is blown out | Pass `--scale 1_10` (script should auto-detect; if not, force it). |
| Whole image is yellow / amber | Drop `--practical-energy` by half. Warm practicals balanced by the slightly-cool `Area_Fill` is the target ratio. |
| Too dark in corners | Raise `--fill-energy` 1.5×. The room is sealed, so corners only see what `Area_Fill` and `Practical_*` give them. |
| Too cold / clinical | Drop `Area_Fill` energy by half, or shift its color toward warm. Or pick `warm_amber` mood. |
| Fireflies on reflective surfaces | Default `clamp_indirect = 5.0` is usually enough; if not, lower to 3.0. |
| Wall geometry missing / wrong | Out of scope — stage geometry is built by `stage2-sub-pointmap-to-separable-stage`. |

## Compositor

Default chain: `RenderLayers → Glare (FOG_GLOW, mix −0.92, threshold 1.0, size 6) → Lens Distortion (0.006 distortion, 0.002 dispersion) → Composite`.

Skip the compositor with `--no-compositor` if you want raw AgX output.

## What is NOT modeled by this skill

- **Sun / Sky / daylight.** Sealed rooms don't receive it. If you need a daylit scene, the upstream Stage 3 must mark at least one edge OPEN (doorway / archway) so the world can actually contribute, and a future Stage 4.5 must boolean-cut windows into walls; only then does env-enhance need a daylight code path.
- **Window glass refraction / colored daylight.** Not relevant without windows.
- **HDRI environment maps.** Same reason — sealed walls absorb everything.

Bringing these back requires both schema changes (`world.mode` would re-add `"nishita_sky"` / `"hdri"` values) and a new env-enhance branch keyed on the presence of `stage.openings[]` / `stage.open_edges[]`. Until that happens, exterior light is a no-op.
