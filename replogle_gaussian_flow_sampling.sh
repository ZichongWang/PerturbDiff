#!/bin/bash

set -euo pipefail

export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Paste the checkpoint you want to evaluate here, then run this script.
CKPT_PATH="${CKPT_PATH:-/home/zichong/fork/PerturbDiff/checkpoints/from_scratch_replogle_gaussian_flow/6_step_MMD_ckpt/ckpts/epoch=89-step=114828-sampling_validation_mmd_epoch=0.0259.ckpt}"
RUN_NAME="${RUN_NAME:-nommd}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
FLOW_STEPS="${FLOW_STEPS:-5}"
GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-1.5}"
NUM_SAMPLED_BATCHES="${NUM_SAMPLED_BATCHES:-null}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zichong/fork/PerturbDiff/checkpoints/test_samples/${RUN_NAME}}"

if [[ "${CKPT_PATH}" == "/abs/path/to/flow_checkpoint.ckpt" ]]; then
  echo "Set CKPT_PATH in replogle_gaussian_flow_sampling.sh before running." >&2
  exit 1
fi

OVERRIDES=(
  "device=auto"
  "trainer.devices=[0]"
  "trainer.use_distributed_sampler=false"
  "path=trixie_path"
  "data=replogle_finetune"
  "data.normalize_counts=10"
  "data.num_workers=16"
  "data.prefetch_factor=12"
  "data.keep_control_cell=false"
  "data.pad_length=2000"
  "data.embed_key=X_hvg"
  "data.use_cell_set=32"
  "optimization.micro_batch_size=512"
  "cov_encoding=trixie_onehot"
  "cov_encoding.batch_encoding=onehot"
  "cov_encoding.celltype_encoding=llm"
  "cov_encoding.replogle_gene_encoding=genept"
  "run_name=${RUN_NAME}"
  "lightning.logger.project=perturb_flow_gaussian"
  "lightning.logger.name=${RUN_NAME}"
  "data.dataset_path=/home/zichong/fork/PerturbDiff/data/perturb_data/finetune_data/nadig_processed_data/replogle_fake_batch.h5ad"
  "data.perturbseq_batch_col=fake_batch"
  "data.skip_cached_indices=true"
  "model_checkpoint_path='${CKPT_PATH}'"
  "sampling.num_sampled_batches=${NUM_SAMPLED_BATCHES}"
  "sampling.flow_steps=${FLOW_STEPS}"
  "sampling.guidance_strength=${GUIDANCE_STRENGTH}"
  "sampling.progress=true"
  "sampling.output_dir='${OUTPUT_DIR}'"
)

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python ./src/apps/run/rawdata_flow_sampling.py "${OVERRIDES[@]}" "$@"
