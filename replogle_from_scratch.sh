#!/bin/bash

COMMON_TRAIN_RUNTIME="
trainer.devices=[0]
trainer.use_distributed_sampler=false
data.normalize_counts=10
trainer.max_steps=200000
lightning.callbacks.checkpoint.save_top_k=-1
trainer.limit_val_batches=32
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
"

COMMON_REPLOGLE_TRAIN="
optimization.micro_batch_size=2048
data.use_cell_set=32
optimization.optimizer.lr=0.0005
cov_encoding.replogle_gene_encoding=genept
"

# -----------------------------
# Shared scratch defaults
# -----------------------------
COMMON_SCRATCH_DATA="
cov_encoding.celltype_encoding=llm
model.p_drop_control=0
data.keep_control_cell=false
"

# -----------------------------
# Shared sampling defaults
# -----------------------------
COMMON_SAMPLING="
trainer.use_distributed_sampler=false
data.normalize_counts=10
data.num_workers=16
data.prefetch_factor=16
lightning.ema.decay=0.99
lightning.ema.update_steps=10
path=trixie_path
cov_encoding=trixie_onehot
cov_encoding.batch_encoding=onehot
model.p_drop_control=0
data.keep_control_cell=false
sampling.use_ddim=true
sampling.num_sampled_batches=null # set to small numbers for fast sampling
"


SCRATCH_REPLOGLE_EXTRA="
data=replogle_finetune
data.num_workers=16
data.prefetch_factor=12
trainer.val_check_interval=1.0
lightning.callbacks.checkpoint.every_n_train_steps=10000
run_name=from_scratch_replogle
lightning.logger.project=diffusion_pertub
lightning.logger.name=replogle_no_batch_bs2048
"

CKPT_NAMING="
lightning.callbacks.checkpoint.dirpath='${trainer.default_root_dir}/${run_name}/ckpts'
lightning.callbacks.checkpoint.monitor=validation_validation_loss_epoch
lightning.callbacks.checkpoint.filename='${run_name}-{epoch}-{step}-{validation_validation_loss_epoch:.4f}'
"


FAKE_BATCH="
data.dataset_path=/home/zichong/fork/PerturbDiff/data/perturb_data/finetune_data/nadig_processed_data/replogle_fake_batch.h5ad
data.perturbseq_batch_col=fake_batch
data.skip_cached_indices=true
"

CUDA_VISIBLE_DEVICES=0 python ./src/apps/run/rawdata_diffusion_training.py \
$COMMON_TRAIN_RUNTIME \
$COMMON_MODEL_2000 \
$COMMON_SCRATCH_DATA \
$COMMON_REPLOGLE_TRAIN \
$SCRATCH_REPLOGLE_EXTRA \
$FAKE_BATCH \
$CKPT_NAMING


