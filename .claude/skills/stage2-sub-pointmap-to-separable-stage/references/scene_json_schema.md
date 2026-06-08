# `blender_scene.json` — Schema v2.0

> **Canonical reference for every skill that reads or writes `blender_scene.json`.**
> Version: **2.0** | Coordinate system: Blender (X right, Y forward, Z up) | Units: meters / radians

---

## Table of Contents

1. [Invariant: JSON ↔ .blend round-trip](#1-invariant-json--blend-round-trip)
2. [Backward compatibility](#2-backward-compatibility)
3. [Per-stage write responsibilities](#3-per-stage-write-responsibilities)
4. [What is NOT serialized](#4-what-is-not-serialized)
5. [Top-level structure](#5-top-level-structure)
6. [`meta` block](#6-meta-block-required)
7. [`camera` block](#7-camera-block-required)
8. [`scene` block](#8-scene-block-required)
9. [`objects` array](#9-objects-array-required)
10. [`point_cloud` block](#10-point_cloud-block-required)
11. [`stage` block](#11-stage-block-optional)
12. [env-enhance owned blocks](#12-env-enhance-owned-blocks-lighting--world--stage_materials--render--compositor)

---

## 1. Invariant: JSON ↔ .blend round-trip

`blender_scene.json` and its paired `.blend` are a **single logical artifact**. Every field in the JSON has a direct correspondent in Blender scene state, and every element placed in the `.blend` must be described by a field in the JSON. Any direct manipulation inside Blender must be exported back to `blender_scene.json` in the same pass before the iteration is considered complete. Any write to `blender_scene.json` must be followed by a rebuild of the `.blend` via the `stage2-sub-pointmap-to-separable-stage` skill before downstream skills consume it. Treat `(blender_scene.json, *.blend)` as one atomic unit.

---

## 2. Backward compatibility

Legacy JSONs (Stage 1 output, pre-v2.0) may omit the `stage`, `lighting`, `world`, `stage_materials`, `render`, and `compositor` blocks. The `stage2-sub-pointmap-to-separable-stage` builder must treat all of these as optional and apply the following defaults when absent:

| Missing block       | Default behavior |
|---------------------|-----------------|
| `stage`             | No stage geometry built; skip Stage collection |
| `lighting`          | No lights added by the builder; Blender scene has no light rig |
| `world`             | Blender default world (solid gray, `world_strength = 1.0`) |
| `stage_materials`   | Stage mesh uses Blender default material (Principled BSDF, base color `(0.8, 0.8, 0.8, 1.0)`) |
| `render`            | Blender default render settings (EEVEE, 64 samples, 1920×1080) |
| `compositor`        | No compositor node tree added |

`point_cloud` is **required** in v2.0 — Stage 1 emits it so that Stage 2's build.py imports `PointCloud_XZ` for Stage 3 to consume. A pre-v2.0 JSON missing this block is downgraded: build.py logs a warning and skips PLY import (Stage 3 will then fail until a `point_cloud` block is added or `pointmap_xz.ply` is imported manually).

**Removed in v2.0:** the legacy top-level `floor` key. Stage 3 (`stage2-sub-pointmap-to-separable-stage`) builds the authoritative `Stage` collection's `Floor` mesh from the polygon, so emitting a separate `floor.obj` import creates a duplicate / Z-fighting floor. Stage 1 must NOT emit this block. Older JSONs that still contain `floor` are tolerated by build.py for back-compat, but the resulting `.blend` will have two floors — fix the upstream emitter instead.

The `meta.schema_version` field distinguishes v2.0 from earlier builds; its absence implies pre-v2.0.

---

## 3. Per-stage write responsibilities

| Block             | Written by                        | Pipeline stage         | Notes |
|-------------------|-----------------------------------|------------------------|-------|
| `meta`            | `stage2-sub-pointmap-to-separable-stage`          | Stage 1                | `schema_version` added in v2.0 |
| `camera`          | `stage2-sub-pointmap-to-separable-stage`          | Stage 1                | |
| `scene`           | `stage2-sub-pointmap-to-separable-stage`          | Stage 1                | |
| `objects`         | `stage2-sub-pointmap-to-separable-stage`          | Stage 1                | Edited by Stage 2 correction ops |
| `point_cloud`     | `stage2-sub-pointmap-to-separable-stage` (convert.py) | Stage 1            | PLY import metadata — required so Stage 2 build.py imports PointCloud_XZ |
| `stage`           | `stage2-sub-pointmap-to-separable-stage`     | Stage 3                | Polygon, floor/ceiling z, wall edges, openings |
| `lighting`        | `stage2-sub-env-enhance` (exporter)    | Stage 4                | Serialized from light rig |
| `world`           | `stage2-sub-env-enhance` (exporter)    | Stage 4                | |
| `stage_materials` | `stage2-sub-env-enhance` (exporter)    | Stage 4                | |
| `render`          | `stage2-sub-env-enhance` (exporter)    | Stage 4                | Only written when non-default |
| `compositor`      | `stage2-sub-env-enhance` (exporter)    | Stage 4                | Only written when non-default |

---

## 4. What is NOT serialized

The following are deliberately excluded from `blender_scene.json`:

- **Per-object materials** (`Material_0`, `Material_0.001`, … attached to `geometry_*` meshes). Owned by the upstream layout-prediction pipeline and protected by the env-only rule (`feedback_blender_scene_env_only.md`): only `Mat_Walls_Stage`, `Mat_Floor_Stage`, `Mat_Ceiling_Stage`, and wainscot materials are in scope for Stage 5; `Material_0.*` materials are always protected.
- **Geometry mesh content** (`geometry_0`, `geometry_1`, … vertex/face data). The JSON holds only a `mesh_path` reference to each asset file (GLB or OBJ).
- **Point cloud vertex data** (`pointmap_xz.ply`). The `point_cloud` block stores only the relative path, axis remap, and import options.

---

## 5. Top-level structure

```json
{
  "meta":            { },
  "camera":          { },
  "scene":           { },
  "objects":         [ ],
  "point_cloud":     { },
  "stage":           { },
  "lighting":        [ ],
  "world":           { },
  "stage_materials": { },
  "render":          { },
  "compositor":      { }
}
```

**Required top-level keys:** `meta`, `camera`, `scene`, `objects`, `point_cloud`.
All other keys are optional. Absence means "use builder defaults" (see §2).

---

## 6. `meta` block (required)

Describes the provenance and coordinate conventions of this file. The builder asserts `coordinate_system == "blender"` and `rotation_unit == "radians"` before processing any transforms.

```json
{
  "meta": {
    "schema_version":       "2.0",
    "source":               "path/to/layout_prediction.json",
    "coordinate_system":    "blender",
    "conversion":           "trimesh(x,y,z) -> blender(-x,z,y)",
    "units":                "meters",
    "rotation_order":       "XYZ",
    "rotation_unit":        "radians",
    "camera_rotation_fix":  "rz -> rz + pi (normalized to (-pi, pi])",
    "object_rotation_fix":  "objects[].rotation_euler[z] -> rz + pi (normalized to (-pi, pi]); floor exempt"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | recommended | `"2.0"` for this revision; absent in legacy files |
| `source` | string | no | Original input file path; informational only |
| `coordinate_system` | string | **yes** | Must be `"blender"` |
| `conversion` | string | no | Human-readable description of the trimesh → Blender axis swap |
| `units` | string | **yes** | Must be `"meters"` |
| `rotation_order` | string | **yes** | Must be `"XYZ"` |
| `rotation_unit` | string | **yes** | Must be `"radians"` |
| `camera_rotation_fix` | string | no | Informational note on yaw correction applied at conversion time |
| `object_rotation_fix` | string | no | Informational note on per-object Z rotation correction |

---

## 7. `camera` block (required)

Defines the scene camera. The builder creates one `Camera` object and sets it as `scene.camera`. Resolution is also applied to `scene.render`.

```json
{
  "camera": {
    "location":             [0.0, 0.0, 0.0],
    "rotation_euler":       [1.5464, -0.0030, -0.0009],
    "rotation_euler_deg":   [88.6, -0.17, -0.05],
    "lens":                 26.2,
    "sensor_width":         36.0,
    "sensor_fit":           "HORIZONTAL",
    "resolution":           [1920, 1080],
    "clip_start":           0.01,
    "clip_end":             100.0
  }
}
```

| Field | Type | Required | Unit | Default | Constraints |
|-------|------|----------|------|---------|-------------|
| `location` | float[3] | **yes** | meters (Blender XYZ) | — | |
| `rotation_euler` | float[3] | **yes** | radians, XYZ order | — | |
| `rotation_euler_deg` | float[3] | no | degrees | — | Informational copy; not read by builder |
| `lens` | float | **yes** | millimeters | — | > 0 |
| `sensor_width` | float | **yes** | millimeters | — | > 0 |
| `sensor_fit` | string | **yes** | — | `"HORIZONTAL"` | `"HORIZONTAL"` \| `"VERTICAL"` \| `"AUTO"` |
| `resolution` | int[2] | **yes** | pixels [W, H] | — | Both > 0 |
| `clip_start` | float | **yes** | meters | `0.01` | > 0 |
| `clip_end` | float | **yes** | meters | `100.0` | > clip_start |

---

## 8. `scene` block (required)

Stores scene-level metadata produced by the layout converter.

```json
{
  "scene": {
    "world_scale_factor":      1.0,
    "shifted_center_blender":  [0.218, 1.142, 0.072],
    "shifted_scale":           0.895
  }
}
```

| Field | Type | Required | Unit | Default | Description |
|-------|------|----------|------|---------|-------------|
| `world_scale_factor` | float | no | dimensionless | `1.0` | Uniform scale applied to world before import; `1.0` means no additional scale |
| `shifted_center_blender` | float[3] | no | meters | — | AABB center of the layout before the coordinate origin shift |
| `shifted_scale` | float | no | dimensionless | — | The scale factor derived from the layout extent normalization pass |

---

## 9. `objects` array (required)

Ordered list of scene objects. Each entry is imported, unit-cube-normalized, and placed via an `Empty` named `<id>` that carries the JSON transform. Child mesh objects are parented under this Empty.

```json
{
  "objects": [
    {
      "id":               "obj_1",
      "mesh_path":        "/abs/path/to/000000034646.glb",
      "location":         [-0.379, 0.797, -0.017],
      "rotation_euler":   [0.0068, -0.0036, 0.681],
      "rotation_euler_deg": [0.391, -0.206, 38.99],
      "scale":            [0.122, 0.122, 0.122],
      "class":            "chair",
      "wall_attached":    false,
      "wall_id":          null,
      "gap_m":            0.01,
      "visible":          true
    }
  ]
}
```

| Field | Type | Required | Unit | Default | Description |
|-------|------|----------|------|---------|-------------|
| `id` | string | **yes** | — | — | Unique within `objects[]`; becomes the Blender Empty name |
| `mesh_path` | string | **yes** | absolute path | — | `.glb` or `.obj`; by-reference only, content not serialized |
| `location` | float[3] | **yes** | meters | — | Blender XYZ world position of the Empty |
| `rotation_euler` | float[3] | **yes** | radians, XYZ | — | Blender XYZ Euler rotation |
| `rotation_euler_deg` | float[3] | no | degrees | — | Informational; not consumed by builder |
| `scale` | float[3] | **yes** | dimensionless | — | Applied to the Empty after unit-cube normalization |
| `class` | string | no | — | `null` | Semantic category (e.g. `"chair"`, `"sofa"`, `"tv"`); used by Stage 2 correction ops |
| `wall_attached` | bool | no | — | `false` | `true` if this object is physically mounted on a wall; triggers snap-to-wall in Stage 2 |
| `wall_id` | string\|null | no | — | `null` | Name of the host wall (e.g. `"Wall_03"`); set when `wall_attached == true` |
| `gap_m` | float | no | meters | `0.01` | Stand-off gap for wall-attached objects; used by `snap_to_wall` op |
| `visible` | bool | no | — | `true` | If `false`, object is hidden in viewport and render (`hide_render = True`) |

**Constraints:**
- `id` values must be unique across the entire `objects[]` array; `"floor"` is reserved (legacy back-compat only — new files must not include a `"floor"` entry).
- `scale` components must all be > 0.
- If `wall_attached == true`, `wall_id` should name a wall listed in `stage.wall_objects[]`; a null `wall_id` on a wall-attached object is valid only if the wall is inferred at runtime.

---

## 10. `point_cloud` block (required)

Import metadata for the XZ pointmap PLY. Required in v2.0: Stage 1 emits it so that Stage 2's build.py imports `PointCloud_XZ` into the `.blend`, which Stage 3 then reads to fit the floor polygon. The block stores only the import configuration; PLY bytes live at `ply_path`.

```json
{
  "point_cloud": {
    "ply_path":            "inputs/pointmap_xz.ply",
    "axis_remap":          "forward=Z,up=Y",
    "decimate_ratio":      1.0,
    "visible":             false,
    "world_scale_applied": 1.0,
    "name":                "PointCloud_XZ"
  }
}
```

| Field | Type | Required | Unit | Default | Description |
|-------|------|----------|------|---------|-------------|
| `ply_path` | string | **yes** | relative path | — | Path to the PLY file relative to the scene directory; canonical: `"inputs/pointmap_xz.ply"` |
| `axis_remap` | string | no | — | `"forward=Z,up=Y"` | Blender PLY import axis convention string |
| `decimate_ratio` | float | no | dimensionless | `1.0` | Fraction of vertices to keep [0, 1]; `1.0` means no decimation |
| `visible` | bool | no | — | `false` | Whether the point cloud object is visible in viewport and render after import |
| `world_scale_applied` | float | no | dimensionless | `meta.world_scale_factor` | Uniform scale factor baked into the imported PLY coordinates so the cloud aligns with rescaled layout objects |
| `name` | string | no | — | `"PointCloud_XZ"` | Blender object name after import; downstream Stage 3 looks up this exact name |

> **Why required:** Stage 3 (`extract_inputs.py`) does `bpy.data.objects["PointCloud_XZ"]` — if the cloud was never imported during Stage 2's build, Stage 3 crashes. Stage 1 must therefore emit this block.

---

## 11. `stage` block (optional)

Written by `stage2-sub-pointmap-to-separable-stage` (Stage 4). The Stage 4.5 wall-opening pass adds the `openings[]` array. Describes the room polygon and all stage geometry. When present, the builder creates the `Stage` collection containing `Floor`, `Ceiling`, and per-wall `Wall_NN` objects with the specified thicknesses.

```json
{
  "stage": {
    "polygon_vertices":    [[x0,y0], [x1,y1], [x2,y2], [x3,y3]],
    "polygon_centroid_xy": [cx, cy],
    "floor_z":             0.0,
    "ceiling_z":           2.8,
    "wall_thickness":      0.25,
    "floor_thickness":     0.30,
    "ceiling_thickness":   0.30,
    "wall_objects":        ["Wall_01", "Wall_02", "Wall_03"],
    "wall_edges": [
      { "from": 0, "to": 1, "object": "Wall_01", "orientation": "camera_gaze" },
      { "from": 1, "to": 2, "object": "Wall_02", "orientation": "unspecified" },
      { "from": 2, "to": 3, "object": "Wall_03", "orientation": "unspecified" }
    ],
    "open_edges": [
      { "from": 3, "to": 0, "reason": "camera_near" }
    ],
    "openings": [
      {
        "id":        "opening_0001",
        "wall_name": "Wall_03",
        "kind":      "window",
        "xy_range":  [[x0, y0], [x1, y1]],
        "z_range":   [z0, z1],
        "source":    "stage_refine_iter2"
      }
    ],
    "camera_xy":     [0.0, 0.0],
    "camera_source": "blender_camera",
    "source_frame":  "blend_world",
    "buffer_m":      0.05,
    "rect_angle_deg": 12.4,
    "generator":     "stage2-sub-pointmap-to-separable-stage"
  }
}
```

### 11.1 Core fields

| Field | Type | Required | Unit | Default | Constraints |
|-------|------|----------|------|---------|-------------|
| `polygon_vertices` | float[N][2] | **yes** | meters (XY) | — | N >= 3; vertices in polygon order; Y is the Blender forward axis |
| `polygon_centroid_xy` | float[2]\|null | no | meters | `null` | Centroid of the polygon; computed at generation time |
| `floor_z` | float | **yes** | meters | — | Z coordinate of the floor surface (top face) |
| `ceiling_z` | float | **yes** | meters | — | Z coordinate of the ceiling surface (bottom face); must be > `floor_z` |
| `wall_thickness` | float | no | meters | `0.25` | Extrusion depth of each wall slab |
| `floor_thickness` | float | no | meters | `0.30` | Extrusion depth of the floor slab (downward from `floor_z`) |
| `ceiling_thickness` | float | no | meters | `0.30` | Extrusion depth of the ceiling slab (upward from `ceiling_z`) |
| `wall_objects` | string[] | **yes** | — | — | Ordered list of wall Blender object names; index matches wall edge index |
| `source_frame` | string | no | — | `"blend_world"` | Frame in which polygon vertices are expressed; always `"blend_world"` for v2.0 |
| `generator` | string | no | — | — | Skill identifier that created this block |

### 11.2 `wall_edges[]`

Each element describes one WALL polygon edge and its corresponding Blender object.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `from` | int | **yes** | Index into `polygon_vertices[]` of the edge start vertex |
| `to` | int | **yes** | Index into `polygon_vertices[]` of the edge end vertex |
| `object` | string | **yes** | Blender object name (e.g. `"Wall_02"`) |
| `orientation` | string | no | `"camera_gaze"` \| `"camera_near"` \| `"unspecified"` — the wall's spatial relationship to the camera at stage generation time |

### 11.3 `open_edges[]`

Polygon edges that are left open (no wall slab built). Typically used for doorways, camera-side openings, or hallway connections.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `from` | int | **yes** | Start vertex index |
| `to` | int | **yes** | End vertex index |
| `reason` | string | no | `"camera_near"` \| `"camera_gaze"` \| `"doorway"` \| `"unspecified"` |

### 11.4 `openings[]` (Stage 4.5 — wall-opening boolean cuts)

Boolean cuts applied to individual wall faces (windows, doors, arches). Each entry instructs the wall-opening pass to subtract a rectangular volume from the named wall's solid geometry.

```json
{
  "id":        "opening_0001",
  "wall_name": "Wall_03",
  "kind":      "window",
  "xy_range":  [[-0.5, 1.2], [0.5, 1.2]],
  "z_range":   [0.9, 2.1],
  "source":    "stage_refine_iter2"
}
```

| Field | Type | Required | Unit | Constraints | Description |
|-------|------|----------|------|-------------|-------------|
| `id` | string | **yes** | — | Unique within `openings[]` | Stable identifier; preserved across iterations |
| `wall_name` | string | **yes** | — | Must match a name in `stage.wall_objects[]` | Host wall |
| `kind` | string | **yes** | — | `"window"` \| `"door"` \| `"arch"` | Semantic type; `"door"` implies `z_range[0] == floor_z` |
| `xy_range` | float[2][2] | **yes** | meters | `xy_range[0]` < `xy_range[1]` per axis | `[[x0,y0], [x1,y1]]` — the XY footprint along the wall face in world space |
| `z_range` | float[2] | **yes** | meters | `z_range[0]` < `z_range[1]`; must be within `[floor_z, ceiling_z]` | `[z_bottom, z_top]` of the opening |
| `source` | string | no | — | — | Which pass/iteration authored this opening |

### 11.5 Auxiliary fields

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `camera_xy` | float[2]\|null | meters | World XY position of the camera at the time the polygon was computed; used for open-edge decisions |
| `camera_source` | string | — | How `camera_xy` was derived (e.g. `"blender_camera"`) |
| `buffer_m` | float | meters | Inward buffer applied to the pointmap footprint before polygon fitting |
| `rect_angle_deg` | float | degrees | Rotation angle of the minimum-area bounding rectangle; diagnostic |

---

## 12. env-enhance owned blocks (`lighting` / `world` / `stage_materials` / `render` / `compositor`)

These five top-level blocks are written and owned by `stage2-sub-env-enhance` (Stage 4). They appear in `blender_scene.json` after Stage 4 completes. All five are optional — when absent, the builder applies Blender defaults (see §2).

**Full schema for these blocks lives in `stage2-sub-env-enhance/references/env_blocks_schema.md`.** Update that file when env-enhance changes its output. The summary below exists only so a JSON reader can recognise the blocks without context-switching.

| Block | Owner skill | One-line summary |
|---|---|---|
| `lighting` | env-enhance | Array of interior light objects (POINT / AREA). No Sun, no portals — the room is sealed |
| `world` | env-enhance | Always `{ "mode": "flat", "color_linear": [0,0,0], "world_strength": 0.0 }` |
| `stage_materials` | env-enhance | PBR Base Color + Roughness for `Mat_Wall*` / `Mat_Floor*` / `Mat_Ceiling*` plus per-wall overrides |
| `render` | env-enhance | Cycles + color-management settings (samples, AgX view transform, bounces, denoise) |
| `compositor` | env-enhance | Optional Glare + Lens Distortion post-render chain |

The pointmap skill's `build.py` reads these blocks if present (so a round-trip rebuild reproduces the env state) but never writes them.

---

## Appendix A — Pointmap-owned example (annotated)

A `blender_scene.json` containing only the blocks this skill owns. The five env-enhance owned blocks (`lighting`, `world`, `stage_materials`, `render`, `compositor`) are appended later by Stage 4 — see `stage2-sub-env-enhance/references/env_blocks_schema.md` for examples.

```json
{
  "meta": {
    "schema_version":       "2.0",
    "source":               "sample/inputs/layout_prediction.json",
    "coordinate_system":    "blender",
    "conversion":           "trimesh(x,y,z) -> blender(-x,z,y)",
    "units":                "meters",
    "rotation_order":       "XYZ",
    "rotation_unit":        "radians",
    "world_scale_factor":   1.0
  },

  "scene": {
    "shifted_center_blender": [0.218, 1.142, 0.072],
    "shifted_scale":          0.895
  },

  "camera": {
    "location":           [0.0, 0.0, 0.0],
    "rotation_euler":     [1.5464, -0.0030, -0.0009],
    "lens":               26.2,
    "sensor_width":       36.0,
    "sensor_fit":         "HORIZONTAL",
    "resolution":         [1920, 1080],
    "clip_start":         0.01,
    "clip_end":           100.0
  },

  "objects": [
    {
      "id":             "obj_1",
      "mesh_path":      "inputs/object/1.glb",
      "location":       [-0.379, 0.797, -0.017],
      "rotation_euler": [0.007, -0.004, 0.681],
      "scale":          [0.122, 0.122, 0.122],
      "class":          "chair",
      "wall_attached":  false,
      "visible":        true
    }
  ],

  "point_cloud": {
    "ply_path":            "inputs/pointmap_xz.ply",
    "axis_remap":          "forward=Z,up=Y",
    "world_scale_applied": 1.0,
    "visible":             false,
    "name":                "PointCloud_XZ"
  },

  "stage": {
    "polygon_vertices":  [[-2.0, 0.5], [2.0, 0.5], [2.0, 4.0], [-2.0, 4.0]],
    "floor_z":           0.0,
    "ceiling_z":         2.8,
    "wall_thickness":    0.25,
    "floor_thickness":   0.30,
    "ceiling_thickness": 0.30,
    "wall_objects":      ["Wall_01", "Wall_02", "Wall_03"],
    "wall_edges": [
      { "from": 0, "to": 1, "object": "Wall_01", "orientation": "camera_gaze" },
      { "from": 1, "to": 2, "object": "Wall_02", "orientation": "unspecified" },
      { "from": 2, "to": 3, "object": "Wall_03", "orientation": "unspecified" }
    ],
    "open_edges": [
      { "from": 3, "to": 0, "reason": "camera_near" }
    ],
    "openings": [],
    "generator": "stage2-sub-pointmap-to-separable-stage"
  }
}
```

---

## Appendix B — Key invariants summary

| Rule | Detail |
|------|--------|
| Coordinate system | Blender XYZ (X right, Y forward, Z up); 1 unit = 1 meter; never apply trimesh↔Blender swap inside Stage 2 |
| Rotation unit | Always radians in the JSON; `rotation_euler_deg` fields are informational only |
| Normalization contract | The loader bakes unit-cube normalization (AABB center at origin, max half-extent = 1.0) into mesh vertex data before applying JSON transforms |
| JSON ↔ .blend atomicity | Any edit to either side must be immediately propagated to the other; stale pairs are forbidden |
| `floor_z < ceiling_z` | Required; minimum room height is not constrained by the schema but should exceed 2.0 m for interior scenes |
| `polygon_vertices >= 3` | Degenerate polygons (0, 1, 2 vertices) are rejected by the stage builder |
| `opening.z_range` within stage | `z_range[0] >= floor_z` and `z_range[1] <= ceiling_z` must hold |
| Material protection | `Material_0.*` and `geometry_*` mesh materials are never written or modified by env-enhance (env-only rule) |
| Per-stage ownership | Each block has exactly one owning skill; no two stages write the same block (reads are unrestricted). Env-enhance owns `lighting` / `world` / `stage_materials` / `render` / `compositor`; everything else is pointmap-skill owned |
