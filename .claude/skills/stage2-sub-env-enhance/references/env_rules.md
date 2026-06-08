# Environment scope rules

This skill only modifies data that belongs to the **room shell + interior illumination**. Object-owned data must remain untouched.

The room is treated as a **sealed box** — Stage 3 builds Floor + Walls + Ceiling, so exterior light cannot reach the interior. All lighting decisions are interior-only.

## Touchable (in scope)

| Data-block class | Name pattern | Allowed modifications |
|---|---|---|
| World | `bpy.data.worlds['World']` | Rebuild node tree as flat black Background at strength 0 (sealed-interior; no Sky / HDRI). |
| Stage wall material | `Mat_Walls_Stage`, `Mat_Wall*`, `Mat_Wainscot*` | Principled BSDF Base Color, Roughness, Metallic. Never add Image Texture nodes. |
| Stage floor material | `Mat_Floor_Stage`, `Mat_Floor*` | Same as walls + Specular IOR Level. |
| Stage ceiling material | `Mat_Ceiling_Stage`, `Mat_Ceiling*` | Same as walls. |
| Interior lights | `Area_Fill`, `Practical_L`, `Practical_R`, `Fallback_Light` | Create in collection `Lighting_Env`. Energy, color, size, location, rotation. |
| Class-driven lights | `<class>_<obj_id>` (e.g. `lamp_obj_7`) | One Blender LIGHT spawned per detected lamp / fluorescent / pendant mesh. |
| Scene render settings | `scene.render`, `scene.cycles`, `scene.view_settings` | Samples, denoiser, bounces, AgX view transform, exposure, resolution. |
| Compositor | `scene.node_tree` | Glare, Lens Distortion, File Output. |

## Protected (out of scope — do NOT modify)

| Data-block | Reason |
|---|---|
| Any material named `Material_0*` | Owned by layout-prediction pipeline. |
| Any material assigned to a mesh whose name starts with `geometry_` | Per-object GLB imports; materials owned upstream. |
| Mesh data on any object | Geometry is authoritative from upstream. |
| Camera object transform, `data.lens`, `data.shift_x/y`, `data.dof` | Camera is fixed by the predictor. |
| Object world-space positions / rotations | Layout is authoritative. |
| Collection membership of existing objects | Leave as is; only add a new `Lighting_Env` collection. |
| Stage geometry (`Floor`, `Wall_NN`, `Ceiling`) | Built by `stage2-sub-pointmap-to-separable-stage` (Stage 3). env-enhance only touches their *materials*, never their mesh data. |

## Deprecated / removed (stripped on every run)

| Name | Reason |
|---|---|
| `Sun` | External light; sealed walls block its contribution entirely. |
| `Area_Window` | Window-portal helper from the daylight era. |
| `Ambient_Fill` | Old global fill; replaced by `Area_Fill`. |
| `Portal_Window_1`, `Portal_Window_2` | Cycles portal lights; only useful with Sky + windows, neither of which we have. |
| World node `ShaderNodeTexSky` | Nishita sky; replaced by flat black Background. |

If any of these names exist in the incoming `.blend` (e.g. left over from a prior run), `enhance_env.py` deletes them before building the interior rig. They are never re-created.

## Hard rules

- **Any new light object must use fixed energy 200 W** at creation. Built-in named lights (Area_Fill, Practical_L/R) can then have their energy adjusted by mood presets.
- **If the scene ends up with zero LIGHT objects after the rig builds, force-create `Fallback_Light` at the room centroid at fixed 200 W.** This guarantee always runs (no early-return / no exception path).
- **Never touch a material when uncertain whether it is "stage" or "object".** Err on the side of skipping.

## Decision procedure when inspecting materials

```
for mat in bpy.data.materials:
    if mat.name.startswith('Material_0'):
        SKIP (protected)
    elif mat.name.startswith('Mat_') and any(s in mat.name for s in ['Wall', 'Floor', 'Ceiling', 'Wainscot']):
        ALLOW (stage — tune Base Color / Roughness only)
    else:
        SKIP (unknown — err on the side of not modifying)
```

When reporting, always log both the modified and skipped lists so the user can audit.
