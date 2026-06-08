#!/bin/bash
# Training launcher (3D-FUTURE + ScanNet + COCO, indoor filtered)
# - Floor rotation prediction (xz2f_rot)
# - EMA model enabled
# - Point-map surface loss
# - Select GPUs via CUDA_VISIBLE_DEVICES (default: 2,3,4,5,6,7)

cd "$(dirname "$0")/.."

NUM_MACHINES=1
MACHINE_RANK=0
GPU_LIST="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
IFS=, read -r -a GPU_IDS <<< "$GPU_LIST"
NUM_LOCAL_GPUS="${#GPU_IDS[@]}"
LAUNCHER_PYTHON="${PYTHON:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# NCCL debugging + longer timeout to survive slow rank-0 ops (NFS writes, wandb upload)
export TORCH_NCCL_TRACE_BUFFER_SIZE=8192
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
LOAD_CKPT="${LOAD_CKPT:-checkpoints/galp.pt}"

# Bypass accelerate's CLI entrypoint because it imports a broken torch._dynamo path in trellis.
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$LAUNCHER_PYTHON" -m accelerate.commands.launch \
  --num_machines $NUM_MACHINES \
  --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
  --machine_rank $MACHINE_RANK \
  --main_process_port 29504 \
  -m src.train \
  --config configs/mp8_nt512.yaml \
  --pin_memory \
  --allow_tf32 \
  --gradient_accumulation_steps 8 \
  --output_dir "$OUTPUT_DIR" \
  --num_workers 4 \
  --dataset_mix future+scannet+coco \
  --model_version v1_4 \
  --tag v1_4_coco_final \
  --mesh_aug_prob 1.0 \
  --use_high_pointmap \
  --use_pm_surface_loss \
  --pm_surface_loss_weight 1.0 \
  --ddp_find_unused_parameters \
  --max_val_steps 0 \
  --load_ckpt "$LOAD_CKPT" \
  --post_training \
  --max_grad_norm 0.5 \
  "$@"
  
#   --use_floor_loss --floor_loss_weight 0.1 --floor_loss_threshold 0.01 \
#   --use_penetration_loss --penetration_loss_weight 0.01 --penetration_loss_resolution 64 \
# --use_ema \
# --ema_device cpu \
 
