# Project Lessons

## Stage 1 — fresh-checkout setup gotchas (release-check, 2026-06-09)

A clean checkout (submodules freshly cloned) needs two manual fixes before Stage 1
runs end-to-end. Both are environment/build issues, not pipeline-logic bugs.

### 1. GroundingDINO `_C` CUDA extension is not pre-built
- **Symptom:** Step 2 (GroundedSAM) crashes with
  `NameError: name '_C' is not defined` at
  `groundingdino/models/GroundingDINO/ms_deform_attn.py`, preceded by the warning
  "Failed to load custom C++ ops. Running on CPU mode Only!".
- **Cause:** `groundingdino` is pip-installed in env `grounded-sam` but its
  MultiScaleDeformableAttention CUDA extension (`_C*.so`) was never compiled.
- **Fix (build it against the env's torch):**
  ```bash
  cd submodules/Grounded-SAM/GroundingDINO
  scl enable gcc-toolset-9 'conda run -n grounded-sam bash -c \
    "CUDA_HOME=/usr/local/cuda-12.8 pip install -e . --no-build-isolation -v"'
  ```
  - env torch is `2.9.1+cu128` → use the CUDA **12.8** toolkit (`/usr/local/cuda-12.8`,
    has nvcc V12.8.93). Do NOT use `/usr/local/cuda-12.6` (no `bin/`).
  - torch 2.9 needs gcc ≥ 9 → use `scl enable gcc-toolset-9`.
  - torch 2.9 API change required patching
    `groundingdino/.../csrc/MsDeformAttn/ms_deform_attn_cuda.cu`:
    `value.type().is_cuda()` → `value.is_cuda()`,
    `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` → `...(value.scalar_type(), ...)`.
  - Added `etc/conda/activate.d/grounded_sam.sh` to put torch libs on `LD_LIBRARY_PATH`.
- **Verify:** `conda run -n grounded-sam python -c "import groundingdino._C as C; print(C.__file__)"`
  must print a path to `_C.cpython-310-*.so`.

### 2. GALP `init_ss_generator_v1_4` call-site bug (now fixed in repo)
- **Symptom:** Step 6 (GALP) crashes with
  `TypeError: init_ss_generator_v1_4() got multiple values for argument 'device'`
  at `src/run_galp.py:152`.
- **Cause:** the call site was written for the old `init_ss_generator(config, ckpt_path,
  ..., device=...)` signature, but the import now aliases it to
  `init_ss_generator_v1_4(ss_generator_config_path, device="cuda", resolution=32)`.
  The stale `None` (old `ckpt_path`) was passed positionally → landed on `device`,
  colliding with the `device=device` keyword.
- **Fix:** removed the dead `None,` positional arg from the call in `src/run_galp.py`.
  This is committed in the script; re-running `--phase post` (no `--force`, reuses
  SAM3D meshes) completes GALP + finalize.

### General: re-running after a mid-pipeline failure
- `pre`, `eval`, and SAM3D (Step 5) outputs persist on disk. If only GALP fails,
  re-run `--phase post` WITHOUT `--force` to reuse the ~15-min SAM3D meshes rather
  than regenerating them.
