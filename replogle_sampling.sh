#!/bin/bash

set -euo pipefail

# Use a full Lightning callback checkpoint (.ckpt), not the manual
# weights-only file at checkpoints/from_scratch_replogle.
CKPT_PATH="${CKPT_PATH:-/home/zichong/fork/PerturbDiff/checkpoints/wandb/diffusion_pertub/ct3oaizj/checkpoints/epoch=155-step=200000-validation_random_Perturbseq_loss=0.00.ckpt}"
RUN_NAME="${RUN_NAME:-replogle_sampling}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
SAMPLED_BATCHES="${SAMPLED_BATCHES:-null}"

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
  "optimization.micro_batch_size=2048"
  "cov_encoding=trixie_onehot"
  "cov_encoding.batch_encoding=onehot"
  "cov_encoding.celltype_encoding=llm"
  "cov_encoding.replogle_gene_encoding=genept"
  "model.p_drop_control=0"
  "lightning.logger.project=diffusion_pertub"
  "lightning.logger.name=${RUN_NAME}"
  "run_name=${RUN_NAME}"
  "data.dataset_path=/home/zichong/fork/PerturbDiff/data/perturb_data/finetune_data/nadig_processed_data/replogle_fake_batch.h5ad"
  "data.perturbseq_batch_col=fake_batch"
  "data.skip_cached_indices=true"
  "model_checkpoint_path=\"${CKPT_PATH}\""
  "sampling.use_ddim=true"
  "sampling.num_sampled_batches=${SAMPLED_BATCHES}"
  "sampling.start_time=100"
  "sampling.eta=0.0"
  "sampling.guidance_strength=1.0"
  "sampling.progress=true"
)

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python ./src/apps/run/rawdata_diffusion_sampling.py "${OVERRIDES[@]}" "$@"
