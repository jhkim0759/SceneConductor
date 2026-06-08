"""
_material_utils.py — Shared PBR material helpers for the stage2-sub-env-enhance skill.

Consumed by:
  - enhance_env.py   (stage material rewrite)
  - export_env_to_json.py  (PBR → dict serialisation)
  - stage2-sub-pointmap-to-separable-stage/build.py  (additive v2.0 world / material loaders)

SAFETY RULE: per-object materials (Material_0.*) and materials bound to
geometry_* meshes are NEVER modified or serialised here.
See: feedback_blender_scene_env_only.md
"""

from __future__ import annotations

import math
import os
from typing import Optional


# ---------------------------------------------------------------------------
# sRGB / linear colour utilities
# ---------------------------------------------------------------------------

def srgb_to_linear(c: float) -> float:
    """Convert a single sRGB channel value [0..1] to linear."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    """Convert a single linear channel value [0..1] to sRGB."""
    if c <= 0.0:
        return 0.0
    if c >= 1.0:
        return 1.0
    if c < 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def linear_rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert linear-light RGB [0..1] to a '#RRGGBB' sRGB hex string.

    The conversion applies standard IEC 61966-2-1 sRGB gamma so the result is
    suitable for display / CSS use only — not for feeding back into Blender as
    a linear colour.
    """
    def clamp_u8(v: float) -> int:
        return max(0, min(255, round(linear_to_srgb(v) * 255.0)))

    return "#{:02X}{:02X}{:02X}".format(clamp_u8(r), clamp_u8(g), clamp_u8(b))


def srgb_hex_to_linear_rgba(hex_str: str) -> tuple:
    """Parse '#RRGGBB' or 'RRGGBB' sRGB hex and return linear (r, g, b, 1.0)."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex colour: {hex_str!r}")
    return (
        srgb_to_linear(int(h[0:2], 16) / 255.0),
        srgb_to_linear(int(h[2:4], 16) / 255.0),
        srgb_to_linear(int(h[4:6], 16) / 255.0),
        1.0,
    )


# ---------------------------------------------------------------------------
# Principled BSDF node access
# ---------------------------------------------------------------------------

def _get_or_create_principled(mat):
    """Return the Principled BSDF node in *mat*, wiring it to the output if
    it had to be created fresh.

    Args:
        mat: A ``bpy.types.Material`` with ``use_nodes`` enabled (this
             function enables it if not already set).

    Returns:
        The ``ShaderNodeBsdfPrincipled`` node.
    """
    # Import here so the module is safe to import outside Blender (e.g. in
    # unit-test stubs that mock bpy).
    import bpy  # noqa: F401 — required at call-site inside Blender

    mat.use_nodes = True
    nt = mat.node_tree
    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    pbsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    pbsdf.location = (0, 0)
    for node in nt.nodes:
        if node.type == "OUTPUT_MATERIAL":
            nt.links.new(pbsdf.outputs["BSDF"], node.inputs["Surface"])
            break
    return pbsdf


def apply_mat_color(mat, linear_rgba: tuple, roughness: float,
                    specular_ior: Optional[float] = None) -> None:
    """Set Base Color, Roughness, and optionally Specular IOR on *mat*.

    This is the canonical way for stage-material writers to touch a material's
    PBR inputs.  It intentionally does NOT touch emission, transmission, or any
    other channel so that hand-authored overrides survive.
    """
    pbsdf = _get_or_create_principled(mat)
    pbsdf.inputs["Base Color"].default_value = linear_rgba
    pbsdf.inputs["Roughness"].default_value = roughness
    if specular_ior is not None:
        for iname in ("Specular IOR Level", "Specular"):
            if iname in pbsdf.inputs:
                pbsdf.inputs[iname].default_value = specular_ior
                break


# ---------------------------------------------------------------------------
# Stage material name predicates
# ---------------------------------------------------------------------------

STAGE_KEYWORDS = ("Wall", "Floor", "Ceiling", "Wainscot")


def is_stage_material(mat_name: str) -> bool:
    """Return True if the name follows the stage-material naming convention."""
    return mat_name.startswith("Mat_") and any(kw in mat_name for kw in STAGE_KEYWORDS)


def classify_stage_material(mat_name: str) -> Optional[str]:
    """Map a stage material name to its logical role key.

    Returns one of:
      ``"floor"``
      ``"ceiling"``
      ``"walls.__default__"``
      ``"walls.Wall_NN"``   (for per-wall overrides like ``Mat_Wall_03``)
      ``None``              (not a recognised stage material)
    """
    if not mat_name.startswith("Mat_"):
        return None

    n = mat_name  # alias

    if "Floor" in n:
        return "floor"
    if "Ceiling" in n:
        return "ceiling"

    # Per-wall override: Mat_Wall_01, Mat_Wall_03, …
    # Distinguish from the global Mat_Walls_Stage.
    if "Wall" in n and not "Walls" in n and not "Wainscot" in n:
        # Extract the wall object name, e.g. "Mat_Wall_03" → "Wall_03"
        # Convention: the suffix after "Mat_" is the object name segment.
        suffix = n[len("Mat_"):]  # e.g. "Wall_03" or "Wall_03_Stage"
        wall_obj = suffix.split("_Stage")[0]  # strip trailing _Stage if present
        return f"walls.{wall_obj}"

    if "Walls" in n or "Wainscot" in n:
        return "walls.__default__"

    return None


# ---------------------------------------------------------------------------
# PBR readback — Principled BSDF → serialisable dict
# ---------------------------------------------------------------------------

def principled_to_pbr_dict(principled_node, scene_dir: str) -> dict:
    """Serialise a Principled BSDF node's key inputs into a plain dict.

    Args:
        principled_node: A ``ShaderNodeBsdfPrincipled`` node.
        scene_dir: Absolute path to the directory that contains
                   ``blender_scene.json``.  Texture paths are stored relative
                   to this directory (forward-slash separators).

    Returns:
        ``{
            "base_color_linear": [r, g, b, a],   # 4-element list, linear
            "base_color_srgb_hex": "#RRGGBB",
            "roughness": float,
            "metallic": float,
            "specular_ior_level": float,
            "normal_strength": float,             # 1.0 if no Normal Map node
            "texture_image_path": str | None,     # relative fwd-slash, or None
        }``
    """
    node = principled_node

    # --- Base Color ---
    bc_socket = node.inputs.get("Base Color")
    if bc_socket is not None:
        raw = bc_socket.default_value
        lin = [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
    else:
        lin = [1.0, 1.0, 1.0, 1.0]

    hex_str = linear_rgb_to_hex(lin[0], lin[1], lin[2])

    # --- Roughness ---
    roughness = _read_float_input(node, "Roughness", 0.5)

    # --- Metallic ---
    metallic = _read_float_input(node, "Metallic", 0.0)

    # --- Specular IOR Level (Blender 4.x) or Specular (legacy) ---
    specular_ior = _read_float_input(node, "Specular IOR Level",
                    _read_float_input(node, "Specular", 0.5))

    # --- Normal strength: look for a ShaderNodeNormalMap linked to Normal ---
    normal_strength = 1.0
    normal_socket = node.inputs.get("Normal")
    if normal_socket is not None and normal_socket.is_linked:
        link = normal_socket.links[0]
        from_node = link.from_node
        if from_node.bl_idname == "ShaderNodeNormalMap":
            strength_socket = from_node.inputs.get("Strength")
            if strength_socket is not None:
                normal_strength = float(strength_socket.default_value)

    # --- Texture image path: ShaderNodeTexImage linked to Base Color ---
    texture_path: Optional[str] = None
    if bc_socket is not None and bc_socket.is_linked:
        link = bc_socket.links[0]
        from_node = link.from_node
        if from_node.bl_idname == "ShaderNodeTexImage" and from_node.image is not None:
            raw_path = from_node.image.filepath
            texture_path = _make_relative_fwdslash(raw_path, scene_dir)

    return {
        "base_color_linear": lin,
        "base_color_srgb_hex": hex_str,
        "roughness": roughness,
        "metallic": metallic,
        "specular_ior_level": specular_ior,
        "normal_strength": normal_strength,
        "texture_image_path": texture_path,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_float_input(node, name: str, default: float) -> float:
    sock = node.inputs.get(name)
    if sock is None:
        return default
    try:
        return float(sock.default_value)
    except (TypeError, ValueError):
        return default


def _make_relative_fwdslash(raw_path: str, scene_dir: str) -> str:
    """Return *raw_path* as a forward-slash path relative to *scene_dir*.

    Blender paths may start with '//' (Blender-relative) or be absolute.
    If the path cannot be made relative (different drive on Windows, etc.),
    it is returned as-is with backslashes replaced.
    """
    # Strip Blender's own '//' relative prefix first.
    if raw_path.startswith("//"):
        raw_path = raw_path[2:]

    abs_path = os.path.abspath(raw_path) if not os.path.isabs(raw_path) else raw_path

    try:
        rel = os.path.relpath(abs_path, scene_dir)
    except ValueError:
        # os.path.relpath raises ValueError on Windows when paths are on
        # different drives.  Fall back to the absolute path.
        rel = abs_path

    return rel.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Shared world builders — consumed by enhance_env.py AND build.py (v2.0)
# ---------------------------------------------------------------------------

def build_nishita_world(
    sun_elev_deg: float,
    sun_rot_deg: float,
    dust_density: float = 1.5,
    ozone_density: float = 1.0,
    altitude: float = 0.0,
    strength: float = 1.0,
    exposure: float = 0.0,
) -> None:
    """Build (or rebuild) the Blender World with a Nishita sky shader.

    Shared implementation used by both ``enhance_env.py`` and the
    ``build.py`` v2.0 ``build_world`` loader so that both paths are always
    in sync.  This is a pure refactor of the ``rebuild_world`` function that
    previously lived only in ``enhance_env.py`` — behaviour is identical.

    Parameters
    ----------
    sun_elev_deg:
        Solar elevation in degrees [-90, 90]. 0 = horizon, 90 = zenith.
    sun_rot_deg:
        Solar azimuth rotation in degrees [0, 360).
    dust_density:
        Nishita dust/aerosol density. Default 1.5.
    ozone_density:
        Nishita ozone density. Default 1.0.
    altitude:
        Camera altitude above sea level in metres. Default 0.0.
    strength:
        Background node Strength input. Default 1.0.
    exposure:
        Color-management exposure offset applied to ``scene.view_settings``.
        Pass ``0.0`` to leave exposure unchanged.
    """
    import math
    import bpy  # noqa: F401

    scene = bpy.context.scene
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type    = "NISHITA"
    sky.sun_elevation  = math.radians(sun_elev_deg)
    sky.sun_rotation   = math.radians(sun_rot_deg)
    sky.dust_density   = dust_density
    sky.ozone_density  = ozone_density
    sky.altitude       = altitude
    sky.location       = (-400, 0)

    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    bg.location = (-100, 0)

    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (200, 0)

    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    if abs(exposure) > 1e-6:
        try:
            scene.view_settings.exposure = exposure
        except Exception as exc:
            print(f"[_material_utils] WARN: could not set exposure={exposure}: {exc}")

    print(
        f"[_material_utils] Nishita sky: elev={sun_elev_deg}°, "
        f"rot={sun_rot_deg}°, dust={dust_density}, "
        f"ozone={ozone_density}, altitude={altitude} m, "
        f"strength={strength}, exposure={exposure}"
    )
