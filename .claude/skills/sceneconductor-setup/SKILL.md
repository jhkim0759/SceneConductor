---
name: sceneconductor-setup
description: One-shot installer that takes a FRESH clone of SceneConductor to a fully runnable pipeline — initializes submodules, downloads Blender 4.2.1, builds all five conda envs (./setup.sh, including the GroundingDINO torch-compat patch + _C verification), downloads every model checkpoint (GroundedSAM, GALP, SAM3D, Qwen), wires the SAM3D symlink, and verifies everything. Use this skill WHENEVER the user wants to install / set up / provision / bootstrap SceneConductor, is starting from a fresh git clone, hits setup errors such as a missing `groundingdino._C` CUDA extension, a conda env that cannot be found, or absent model checkpoints, or asks "how do I get this running". Trigger on "/sceneconductor-setup", "set up SceneConductor", "install the pipeline", "provision the environment", "bootstrap a fresh clone". Prefer this over hand-running setup.sh — it also fetches checkpoints and validates the install, which setup.sh alone does not.
argument-hint: "[--force] [--gpu N] [--skip-checkpoints] [--skip-qwen] [--skip-envs] [--skip-blender] [--skip-submodules]"
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# sceneconductor-setup

Turns a **fresh `git clone` of SceneConductor into a runnable pipeline with one command.** `setup.sh` on its own only builds the conda envs — it deliberately does **not** download the ~25 GB of model weights, wire the SAM3D symlink, or validate the result. This skill is the full provisioner: it orchestrates everything an end user needs so they can go straight from clone to `/scene-orchestration`.

It exists because verifying the release surfaced two failures a fresh clone hits that a developer's already-set-up machine never sees, both now handled automatically here:

1. **GroundingDINO `_C` won't compile against the pinned `torch 2.9.1`** → `NameError: _C is not defined` in Stage 1. `setup.sh` now patches the CUDA source (`ms_deform_attn_cuda.cu`) for modern libtorch *before* building and **verifies `groundingdino._C` actually imports** instead of silently leaving a broken env.
2. **GALP is shipped as a git submodule** whose `init_ss_generator_v1_4` signature differs from older vendored copies → `TypeError: got multiple values for 'device'`. `run_galp.py` now adapts to whichever signature is present, so the submodule "just works".

## Quick reference

```
/sceneconductor-setup [--force] [--gpu N] [--skip-checkpoints] [--skip-qwen]
                      [--skip-envs] [--skip-blender] [--skip-submodules]
```

The whole flow is one script — run it from the repo root:

```bash
bash .claude/skills/sceneconductor-setup/scripts/provision.sh
```

| Flag | Effect |
|---|---|
| `--force` | rebuild every conda env from scratch (passes `--force` to `setup.sh`) |
| `--gpu N` | GPU index for the optional verify smoke (default 0) |
| `--skip-checkpoints` | don't download weights (envs/blender only) |
| `--skip-qwen` | download everything except the ~10 GB Qwen weights (use the HF cache instead) |
| `--skip-envs` / `--skip-blender` / `--skip-submodules` | skip that stage if already done |

Environment passthrough: `CUDA_HOME` (CUDA toolkit for the source builds), `SC_ENVS_DIR` (place big envs off `/home`), `HF_TOKEN` (needed for the **gated** SAM3D download).

## What it does (5 idempotent stages)

`provision.sh` runs these in order; every stage is skippable and safe to re-run after a partial failure:

1. **Submodules** — `git submodule update --init --recursive` (Grounded-SAM, SAM3D, Qwen3.6, GALP).
2. **Blender 4.2.1** — downloads + extracts the vendored Linux build under the repo root (skipped if present).
3. **Conda envs** — runs repo-root `./setup.sh`, which builds all five envs, applies the GroundingDINO `.cu` torch-compat patch, and verifies `groundingdino._C`.
4. **Checkpoints** — `scripts/download_checkpoints.sh` fetches all four weight sets into `checkpoints/` and wires the SAM3D symlink.
5. **Verify** — `scripts/verify_install.sh` audits Blender, the five envs, `_C`, every checkpoint, and submodule population; prints a PASS/FAIL table.

## How to run it (instructions for the agent)

Run the orchestrator and watch for the two interactive snags below. **Prefer the bundled scripts over re-deriving the steps** — they encode the exact fixes and the correct checkpoint layout.

```bash
cd <repo-root>
bash .claude/skills/sceneconductor-setup/scripts/provision.sh   # add flags as requested
```

This is long-running (envs compile CUDA extensions, ~30–60 min the first time; weights are ~25 GB). For a fresh machine, consider running it in the background and tailing the output. If the user already has the conda envs, pass `--skip-envs` to jump straight to checkpoints + verify.

### Two things that need a human/agent decision

- **CUDA_HOME** — the `scenegen` and `grounded-sam` source builds need a real CUDA toolkit (with `nvcc`). If `/usr/local/cuda` isn't it, export the right one first, e.g. `export CUDA_HOME=/usr/local/cuda-12.8`. The pinned torch is a cu128 build, so a CUDA 12.x toolkit is the safe match.
- **SAM3D is gated** — `facebook/sam-3d-objects` on Hugging Face requires accepting the license and a token. If the SAM3D download fails, tell the user to (1) request access at https://huggingface.co/facebook/sam-3d-objects, (2) make a token at https://huggingface.co/settings/tokens, then (3) re-run with `HF_TOKEN=hf_xxx`. Everything else (GroundedSAM, GALP, Qwen) is public.

### Verifying without a full reinstall

If the user just wants to know whether their checkout is ready (no rebuild), run only the audit — it's read-only and finishes in seconds:

```bash
bash .claude/skills/sceneconductor-setup/scripts/verify_install.sh --gpu 0
```

A non-zero exit means at least one **required** check failed; the table names exactly what to fix.

## Bundled scripts

| Script | Role |
|---|---|
| `scripts/provision.sh` | master orchestrator (the 5 stages above) |
| `scripts/download_checkpoints.sh` | fetches GroundedSAM / GALP / SAM3D / Qwen weights + SAM3D symlink |
| `scripts/verify_install.sh` | non-destructive PASS/FAIL audit of the whole install |

## Troubleshooting

- **`NameError: _C is not defined` (Stage 1 GroundedSAM)** — the GroundingDINO CUDA extension didn't build. Re-run `./setup.sh --grounded-sam --force` with a valid `CUDA_HOME`/`nvcc` and a compatible gcc. `setup.sh` now patches the `.cu` automatically and will report `groundingdino._C did NOT build/import` loudly if it still fails.
- **`conda env not found`** — an env name must match `conda_envs:` in `DIRECTORYS.yaml`. Re-run `provision.sh` (or `./setup.sh`) — it skips envs that already exist.
- **SAM3D weights missing** — gated repo; see the HF_TOKEN steps above. The rest of the pipeline still installs; SAM3D is only needed for Stage 1 textured GLBs.
- **No space on `/home`** — `SC_ENVS_DIR=/big/disk/sc_envs bash provision.sh` puts the large envs elsewhere.
- **Want to re-validate only** — `verify_install.sh` is read-only and safe to run anytime.
