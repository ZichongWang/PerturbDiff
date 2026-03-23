#!/bin/bash

COMMON_TRAIN_RUNTIME="
trainer.devices=[0]
trainer.use_distributed_sampler=false
data.normalize_counts=10
trainer.max_steps=300000
lightning.callbacks.checkpoint.save_top_k=-1
trainer.limit_val_batches=5
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
optimization.micro_batch_size=128
data.use_cell_set=32
optimization.optimizer.lr=0.0001
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
optimization.micro_batch_size=128
lightning.logger.project=diffusion_pertub
lightning.logger.name=from_scratch_replogle_low_lr
"

CUDA_VISIBLE_DEVICES=0 python ./src/apps/run/rawdata_diffusion_training.py \
$COMMON_TRAIN_RUNTIME \
$COMMON_MODEL_2000 \
$COMMON_SCRATCH_DATA \
$COMMON_REPLOGLE_TRAIN \
$SCRATCH_REPLOGLE_EXTRA \


