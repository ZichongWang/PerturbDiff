#!/bin/bash

set -euo pipefail

COMMON_TRAIN_RUNTIME="
trainer.devices=[0]
trainer.use_distributed_sampler=false
data.normalize_counts=10
trainer.max_steps=200000
lightning.callbacks.checkpoint.save_top_k=40
trainer.limit_val_batches=16
lightning.ema.decay=0.99
lightning.ema.update_steps=10
cov_encoding.batch_encoding=onehot
path=trixie_path
cov_encoding=trixie_onehot
"

COMMON_MODEL_500="
data.pad_length=500
model.hidden_num=[500,512]
model.input_dim=500
data.embed_key=X
model.output_activation=relu
model.enable_self_condition=true
optimization.devide_1_t=false
optimization.mmd_weight_alpha=0.01
"

COMMON_K562_TRAIN="
optimization.micro_batch_size=512
data.use_cell_set=32
optimization.optimizer.lr=0.0005
optimization.scheduler.min_ratio=0.01
optimization.scheduler.plateau_ratio=0.0
cov_encoding.replogle_gene_encoding=genept
"

COMMON_SCRATCH_DATA="
cov_encoding.celltype_encoding=onehot
model.p_drop_cond=0.3
model.p_drop_control=0
data.keep_control_cell=false
"

K562_EXTRA="
data=replogle_finetune
data.num_workers=16
data.prefetch_factor=12
trainer.val_check_interval=0.5
run_name=from_scratch_k562_500_gaussian_flow
lightning.logger.project=perturb_flow_gaussian
lightning.logger.name=6_step_MMD_devide_1_t_k562_500
sampling_eval.enabled=true
sampling_eval.flow_steps=6
sampling_eval.guidance_strength=1.5
"

CKPT_NAMING="
lightning.callbacks.checkpoint.every_n_train_steps=null
lightning.callbacks.checkpoint.every_n_epochs=1
+lightning.callbacks.checkpoint.save_on_train_epoch_end=false
lightning.callbacks.checkpoint.mode=min
lightning.callbacks.checkpoint.dirpath='\${trainer.default_root_dir}/\${run_name}/\${lightning.logger.name}/ckpts'
lightning.callbacks.checkpoint.monitor=sampling_validation_mmd_epoch
lightning.callbacks.checkpoint.filename='{epoch}-{step}-{sampling_validation_mmd_epoch:.4f}'
"

K562_DATA="
data.dataset_name=K562_500
data.dataset_path=/home/zichong/fork/PerturbDiff/data/K562_500/K562_500_fake.h5ad
data.selected_gene_file=null
data.pert_col=gene
data.perturbseq_batch_col=fake_batch
data.cell_line_key=cell_type
data.control_pert=non-targeting
data.skip_cached_indices=true
+data.split_gene_files.train=/home/zichong/fork/PerturbDiff/data/K562_500/k562_train_genes.npy
+data.split_gene_files.validation=/home/zichong/fork/PerturbDiff/data/K562_500/k562_val_genes.npy
+data.split_gene_files.test=/home/zichong/fork/PerturbDiff/data/K562_500/k562_test_genes.npy
"

CUDA_VISIBLE_DEVICES=1 python ./src/apps/run/rawdata_flow_training.py \
$COMMON_TRAIN_RUNTIME \
$COMMON_MODEL_500 \
$COMMON_SCRATCH_DATA \
$COMMON_K562_TRAIN \
$K562_EXTRA \
$K562_DATA \
$CKPT_NAMING \
"$@"
