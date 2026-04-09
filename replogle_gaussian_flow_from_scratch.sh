#!/bin/bash

set -euo pipefail

COMMON_TRAIN_RUNTIME="
trainer.devices=[0]
trainer.use_distributed_sampler=false
data.normalize_counts=10
trainer.max_steps=200000
lightning.callbacks.checkpoint.save_top_k=-1
trainer.limit_val_batches=16
lightning.ema.decay=0.99
lightning.ema.update_steps=10
cov_encoding.batch_encoding=onehot
path=trixie_path
cov_encoding=trixie_onehot
"

COMMON_MODEL_2000="
data.pad_length=2000
model.hidden_num=[2000,512]
model.input_dim=2000
data.embed_key=X_hvg
model.output_activation=relu
model.enable_self_condition=true
optimization.mmd_weight_alpha=0.0 
"

COMMON_REPLOGLE_TRAIN="
optimization.micro_batch_size=512
data.use_cell_set=32
optimization.optimizer.lr=0.0005
optimization.scheduler.min_ratio=0.01
optimization.scheduler.plateau_ratio=0.0
cov_encoding.replogle_gene_encoding=genept
"

COMMON_SCRATCH_DATA="
cov_encoding.celltype_encoding=llm
model.p_drop_cond=0.3
model.p_drop_control=0
data.keep_control_cell=false
"

SCRATCH_REPLOGLE_EXTRA="
data=replogle_finetune
data.num_workers=16
data.prefetch_factor=12
trainer.val_check_interval=0.5
lightning.callbacks.checkpoint.every_n_train_steps=10000
run_name=from_scratch_replogle_gaussian_flow
lightning.logger.project=perturb_flow_gaussian
lightning.logger.name=10_step_woMMD_p0.3
sampling_eval.enabled=true
sampling_eval.flow_steps=10
sampling_eval.guidance_strength=3
"

CKPT_NAMING="
lightning.callbacks.checkpoint.dirpath='\${trainer.default_root_dir}/\${run_name}/\${lightning.logger.name}/ckpts'
lightning.callbacks.checkpoint.monitor=validation_validation_loss_epoch
lightning.callbacks.checkpoint.filename='{epoch}-{step}-{validation_validation_loss_epoch:.4f}'
"

FAKE_BATCH="
data.dataset_path=/home/zichong/fork/PerturbDiff/data/perturb_data/finetune_data/nadig_processed_data/replogle_fake_batch.h5ad
data.perturbseq_batch_col=fake_batch
data.skip_cached_indices=true
"

CUDA_VISIBLE_DEVICES=1 python ./src/apps/run/rawdata_flow_training.py \
$COMMON_TRAIN_RUNTIME \
$COMMON_MODEL_2000 \
$COMMON_SCRATCH_DATA \
$COMMON_REPLOGLE_TRAIN \
$SCRATCH_REPLOGLE_EXTRA \
$FAKE_BATCH \
$CKPT_NAMING \
"$@"
