#!/usr/bin/env python3
"""
Generate a GroundedSAM-friendly object-class prompt from a scene image.

Output: <scene_dir>/object_class_prompt.json
  {"prompt": "chair. table. lamp.", "classes": ["chair", "table", "lamp"]}

Usage:
    python generate_object_classes.py --scene_dir /path/to/scene [--model claude-opus-4-6] [--max_tokens 512]

The scene_dir must contain an image named image.png (or image.jpg).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_CLAUDE_MODEL = "claude-opus-4-6"
DEFAULT_FALLBACK_CLASSES = [
    "bed",
    "pillow",
    "window",
    "curtain",
    "nightstand",
    "table lamp",
    "side table",
    "book",
    "stuffed animal",
    "framed picture",
    "tissue box",
    "phone",
]

SYSTEM_PROMPT = (
    "You generate segmentation prompts for Grounded-SAM from a single indoor scene image.\n\n"
    "Return only a dot-separated list of visible object names that should be segmented.\n\n"
    "Rules:\n"
    "1. Output only the final prompt string.\n"
    "2. Use short singular nouns or noun phrases, usually 1 to 3 words.\n"
    "3. Focus on visible items useful for object-level 3D reconstruction and layout.\n"
    "4. Include furniture, decor, containers, electronics, lighting fixtures, windows,\n"
    "   doors, curtains, rugs, and other discrete items a person could identify.\n"
    "5. STRICTLY EXCLUDE ONLY the three room-enclosing surfaces:\n"
    "   NEVER include: wall, walls, accent wall, ceiling, floor, ground.\n"
    "   Windows, doors, curtains, rugs, baseboards, molding, beams, columns ARE allowed —\n"
    "   they are objects, not background.\n"
    "6. LIGHTING FIXTURES — be aggressive about including them whenever any luminaire\n"
    "   is visible in the image. Treat each visible fixture as its own class. Use the\n"
    "   most specific term that fits: pendant light, chandelier, ceiling light,\n"
    "   recessed light, downlight, track light, fluorescent light, fluorescent tube,\n"
    "   linear light, ceiling fan light, table lamp, floor lamp, desk lamp,\n"
    "   wall sconce, picture light. If no lighting fixture is visible, do not invent one.\n"
    "7. Do not add numbering, explanations, markdown, quotes, or categories.\n"
    "8. Keep the prompt concise but reasonably complete for the visible scene."
)

EXTRACTION_INSTRUCTION = (
    "Analyze the image and return a Grounded-SAM prompt.\n\n"
    "Requirements:\n"
    "- Return only a dot-separated list of visible, segmentable objects.\n"
    "- Include furniture, decor, electronics, windows, doors, curtains, rugs, and any other\n"
    "  discrete items visible in the scene.\n"
    "- LIGHTING FIXTURES: scan the ceiling, walls, and surfaces for luminaires and include\n"
    "  every visible fixture as its own class — pendant light, chandelier, ceiling light,\n"
    "  recessed light, downlight, track light, fluorescent light, fluorescent tube,\n"
    "  linear light, table lamp, floor lamp, desk lamp, wall sconce, etc.\n"
    "  Use the most specific term that fits what is shown. Skip lighting only if no\n"
    "  fixture is actually visible in the image — never invent one.\n"
    "- If enough objects exist, aim for around 12 to 24 items.\n"
    "- Use short singular nouns or noun phrases.\n"
    "- STRICTLY EXCLUDE ONLY: wall, ceiling, floor (the three room-enclosing surfaces).\n"
    "  These are NEVER objects. Everything else — including windows, doors, curtains, rugs,\n"
    "  and lighting fixtures — IS an object and should be included if visible.\n\n"
    "Example format:\n"
    "chair. table. lamp. cabinet. pillow. plant. window. curtain. pendant light."
)


def find_image(scene_dir: Path) -> Path:
    for name in ("image.png", "image.jpg", "image.jpeg"):
        candidate = scene_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image.png/jpg found in {scene_dir}")


def run_claude(image_path: Path, model: str) -> str:
    full_prompt = (
        f"Read the following image file and analyze it:\n- {image_path}\n\n{EXTRACTION_INSTRUCTION}"
    )
    cmd = [
        "claude", "-p", full_prompt,
        "--system-prompt", SYSTEM_PROMPT,
        "--tools", "Read",
        "--permission-mode", "bypassPermissions",
        "--model", model,
        "--output-format", "text",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd="/tmp")
    return result.stdout.strip()


def load_existing_prompt(scene_dir: Path) -> dict | None:
    prompt_path = scene_dir / "object_class_prompt.json"
    if not prompt_path.exists():
        return None
    try:
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    classes = payload.get("classes")
    prompt = payload.get("prompt")
    if isinstance(classes, list) and classes and isinstance(prompt, str) and prompt.strip():
        return payload
    return None


def build_fallback_prompt(image_path: Path) -> tuple[str, list[str], str]:
    env_prompt = os.environ.get("OBJECT_CLASS_FALLBACK_PROMPT", "").strip()
    if env_prompt:
        classes = normalize_prompt(env_prompt)
        return env_prompt, classes, "env_fallback"

    prompt = ". ".join(DEFAULT_FALLBACK_CLASSES) + "."
    return prompt, DEFAULT_FALLBACK_CLASSES[:], "default_fallback"


def normalize_prompt(text: str) -> list[str]:
    text = text.strip()
    text = re.sub(r"^\s*(segmentation\s+prompt|prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("\n", ". ").replace(";", ". ").replace(",", ". ")
    text = re.sub(r"[\"`']", "", text)

    classes = []
    seen: set[str] = set()
    for chunk in re.split(r"\.+", text):
        chunk = re.sub(r"^\s*[-*0-9\)\(]+\s*", "", chunk).strip().lower()
        chunk = re.sub(r"\s+", " ", chunk).strip(" .")
        if not chunk or len(chunk.split()) > 4 or chunk in seen:
            continue
        seen.add(chunk)
        classes.append(chunk)

    if not classes:
        raise ValueError("Claude returned an empty or unparseable prompt")
    return classes


def parse_args():
    parser = argparse.ArgumentParser(description="Generate object-class prompt from scene image via Claude CLI")
    parser.add_argument("--scene_dir", type=Path, required=True, help="Scene directory containing image.png")
    parser.add_argument("--model", type=str, default=DEFAULT_CLAUDE_MODEL, help="Claude model id")
    parser.add_argument("--max_tokens", type=int, default=512, help="Placeholder for interface parity")
    return parser.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    if not scene_dir.is_dir():
        raise NotADirectoryError(scene_dir)

    image_path = find_image(scene_dir)
    payload = load_existing_prompt(scene_dir)
    if payload is not None:
        print(json.dumps(payload, indent=2))
        print(
            f"\n[generate_object_classes] Reused existing prompt -> {scene_dir / 'object_class_prompt.json'}",
            file=sys.stderr,
        )
        return

    model_used = args.model
    raw_response = None
    prompt = None
    classes = None
    claude_bin = shutil.which("claude")

    if claude_bin:
        try:
            raw_response = run_claude(image_path, args.model)
            classes = normalize_prompt(raw_response)
            prompt = ". ".join(classes) + "."
        except Exception as exc:
            print(
                f"[generate_object_classes] WARN: Claude invocation failed, falling back: {exc}",
                file=sys.stderr,
            )

    if prompt is None or classes is None or raw_response is None:
        prompt, classes, model_used = build_fallback_prompt(image_path)
        raw_response = prompt

    payload = {
        "prompt": prompt,
        "classes": classes,
        "model": model_used,
        "image": str(image_path),
        "raw_response": raw_response,
    }

    out_path = scene_dir / "object_class_prompt.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\n[generate_object_classes] Saved -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[generate_object_classes] ERROR: {exc}", file=sys.stderr)
        raise
