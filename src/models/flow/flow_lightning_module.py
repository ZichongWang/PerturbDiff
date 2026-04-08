"""Lightning module for the rectified-flow pipeline."""

import pickle
import time
from types import SimpleNamespace

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from geomloss import SamplesLoss
from pytorch_lightning.utilities import grad_norm
from transformers.trainer_pt_utils import get_parameter_names

from src.apps.sampling.flow_sampling_generation import sample_flow_dataloader_batches
from src.common.utils import get_short_dsname
from src.models.covariate_encoding import CovEncoder
from src.models.flow.flow_core import create_flow
from src.models.lightning.lightning_factories import get_optimizer, model_init_fn


class FlowPlModel(pl.LightningModule):
    """Lightning module used by the rectified-flow pipeline."""

    def __init__(self, cov_encoding_cfg, model_cfg, py_logger, optimizer_cfg, trainer_cfg, all_split_names, sampling_eval_cfg=None):
        """
        Initialize the class instance.

        :param cov_encoding_cfg: Configuration object for this component.
        :param model_cfg: Model configuration.
        :param py_logger: Python logger instance.
        :param optimizer_cfg: Optimizer configuration.
        :param trainer_cfg: Trainer configuration.
        :param all_split_names: Validation split names.
        :param sampling_eval_cfg: Flow sampling-eval configuration.
        :return: None.
        """
        super().__init__()
        self.cov_encoding_cfg = cov_encoding_cfg
        self.model_cfg = model_cfg
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        self.trainer_cfg = trainer_cfg
        self.all_split_names = all_split_names
        self.sampling_eval_cfg = sampling_eval_cfg

        assert self.optimizer_cfg.mmd_weight_alpha >= 0.0, "optimization.mmd_weight_alpha must be non-negative."
        assert self.optimizer_cfg.mmd_weight_gamma >= 0.0, "optimization.mmd_weight_gamma must be non-negative."

        for split in self.all_split_names:
            setattr(self, f"validation_{split}_step_outputs", [])

        self.cov_encoder = CovEncoder(self.cov_encoding_cfg)
        self.gene_embedding = {}
        if self.model_cfg.use_gene_embedding:
            for gene_emb_file in self.cov_encoding_cfg.gene_embedding_path:
                with open(gene_emb_file, "rb") as f:
                    gene_emb = pickle.load(f)
                    gene_emb = {k: torch.tensor(v, dtype=torch.float32) for k, v in gene_emb.items()}
                    self.gene_embedding.update(gene_emb)
        else:
            self.gene_embedding = None
        self.gene_name_embedding_cache = {}

        self.model = model_init_fn(self.model_cfg, self.cov_encoding_cfg)
        self.flow = create_flow(self.model_cfg)

        self.model_cfg.p_drop_cond = getattr(self.model_cfg, "p_drop_cond", 0.0)

        self._last_logged_batch_start_time = time.monotonic()
        self.validation_step_outputs = []

        blur = self.optimizer_cfg.get("blur", 0.05)
        self.loss_fn = SamplesLoss(loss="energy", blur=blur)

        self.save_hyperparameters()

    def log_data(self, log_dict, train=True):
        """
        Log metrics for training or validation.

        :param log_dict: Metric dictionary.
        :param train: Whether to log in training mode.
        :return: None.
        """
        if train:
            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=self.optimizer_cfg.micro_batch_size,
                logger=True,
                sync_dist=True,
            )
        else:
            self.log_dict(
                log_dict,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.optimizer_cfg.micro_batch_size,
                logger=True,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: Current batch tensor(s).
        :param batch_idx: Batch index.
        :return: Training loss.
        """
        loss_dict = self._compute_loss(batch)
        loss = (loss_dict["loss"]).mean()

        log_dict = {"training_loss_step": loss.detach()}
        # for k, v in loss_dict.items():
        #     if k.startswith("dataset_loss_"):
        #         log_dict[k] = v

        assert self.model.model_name == "Cross_DiT"
        log_dict["training_MSE_loss_step"] = loss_dict["mse"].mean().detach()
        log_dict["training_MMD_loss_step"] = loss_dict["mmd_raw"].mean().detach()
        log_dict["training_weighted_MMD_loss_step"] = loss_dict['mmd_weighted'].mean().detach()

        self.log_data(log_dict, train=True)
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step.

        :param batch: Current batch tensor(s).
        :param batch_idx: Batch index.
        :param dataloader_idx: Validation dataloader index.
        :return: Validation loss mapping.
        """
        split = self.all_split_names[dataloader_idx]

        loss_dict = self._compute_loss(batch)

        loss = (loss_dict["loss"]).mean()

        log_dict = {f"validation_{split}_loss": loss}
        # for k, v in loss_dict.items():
        #     if k.startswith("dataset_loss_"):
        #         log_dict[k] = v

        assert self.model.model_name == "Cross_DiT"
        log_dict[f"validation_{split}_MSE_loss"] = loss_dict["mse"].mean()
        log_dict[f"validation_{split}_MMD_loss"] = loss_dict["mmd_raw"].mean()
        self.log_data(log_dict, train=False)

        self.validation_step_outputs.append({f"validation_{split}_loss": loss})
        getattr(self, f"validation_{split}_step_outputs").append({f"validation_{split}_loss": loss})
        return {f"validation_{split}_loss": loss}

    def on_validation_epoch_end(self):
        """Aggregate validation losses and optionally run flow sampling eval."""
        for split in self.all_split_names:
            arr = getattr(self, f"validation_{split}_step_outputs")
            if len(arr) > 0:
                vals = torch.stack([o[f"validation_{split}_loss"] for o in arr])
                mean_val = vals.mean()
                self.log(
                    f"validation_{split}_loss_epoch",
                    mean_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            getattr(self, f"validation_{split}_step_outputs").clear()

        if getattr(self.sampling_eval_cfg, "enabled", False):
            self._run_sampling_validation_epoch()

    @torch.no_grad()
    def _run_sampling_validation_epoch(self):
        """Run Euler-solver validation metrics on the configured split."""
        if self.trainer.sanity_checking:
            return
        if not self.trainer.is_global_zero:
            if dist.is_initialized():
                dist.barrier()
            return

        split = getattr(self.sampling_eval_cfg, "split", "validation")
        if split not in self.all_split_names:
            raise ValueError(f"sampling_eval.split={split} is not in available splits {self.all_split_names}")

        split_idx = self.all_split_names.index(split)
        dataloaders = self.trainer.val_dataloaders
        if not isinstance(dataloaders, (list, tuple)):
            dataloaders = [dataloaders]
        dataloader = dataloaders[split_idx]
        runtime_cfg = SimpleNamespace(
            data=dataloader.dataset.data_args,
            model=self.model_cfg,
            optimization=self.optimizer_cfg,
        )

        self.py_logger.info("Running flow-based validation on split=%s", split)
        metrics = sample_flow_dataloader_batches(
            self,
            self.flow,
            runtime_cfg,
            self.device,
            self.py_logger,
            dataloader,
            datamodule=self.trainer.datamodule,
            num_sampled_batches=getattr(self.sampling_eval_cfg, "num_batches", 1),
            flow_steps=getattr(self.sampling_eval_cfg, "flow_steps", 8),
            clip_denoised=getattr(self.sampling_eval_cfg, "clip_denoised", True),
            progress=getattr(self.sampling_eval_cfg, "progress", False),
            guidance_strength=getattr(self.sampling_eval_cfg, "guidance_strength", 1.0),
            seed=getattr(self.sampling_eval_cfg, "seed", None),
            collect_covariates=False,
        )
        metric_prefix = "sampling_validation" if split == "validation" else f"sampling_{split}"
        self.log_dict(
            {
                f"{metric_prefix}_fake_r2_epoch": metrics["fake_r2_metric"],
                f"{metric_prefix}_delta_r2_epoch": metrics["delta_r2_metric"],
                f"{metric_prefix}_ctrl_to_pert_r2_epoch": metrics["ctrl_to_pert_r2_metric"],
                f"{metric_prefix}_mmd_epoch": metrics["mmd_metric"],
                f"{metric_prefix}_ctrl_to_pert_mmd_epoch": metrics["ctrl_to_pert_mmd_metric"],
                f"{metric_prefix}_delta_pearson_r": metrics['delta_pearson_r'],
                f"{metric_prefix}_delta_MAE":metrics['delta_mae']
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            rank_zero_only=True,
        )

        if dist.is_initialized():
            dist.barrier()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log elapsed seconds per training step."""
        if batch_idx > 0 and batch_idx % self.trainer_cfg.log_every_n_steps == 0:
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            self._last_logged_batch_start_time = time.monotonic()
            time_per_step = elapsed_time / self.trainer_cfg.log_every_n_steps
            self.log("sec/step", time_per_step, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)

    def _encode_covariates(self, batch):
        """Encode perturbation, cell-type, and batch covariates."""
        return self.cov_encoder(batch["cov_pert"], batch["cov_celltype"], batch["cov_batch"])

    def _compute_loss(self, batch):
        """
        Compute one flow-matching loss value.

        :param batch: Current batch.
        :return: Loss tensor(s) and scalar metrics.
        """
        batch["batch_emb"] = self._encode_covariates(batch)
        pert_emb = batch["pert_emb"]
        device = pert_emb.device

        if self.gene_embedding is None:
            gene_emb = None
        else:
            gene_emb = []
            for i, dataset_name in enumerate(batch["ds_name"]):
                dataset_name = dataset_name[0]
                if dataset_name not in self.gene_name_embedding_cache:
                    self.gene_name_embedding_cache[dataset_name] = torch.stack(
                        [self.gene_embedding.get(idx, torch.zeros(5120)) for idx in batch["col_genes"][i]]
                    )
                gene_emb.append(self.gene_name_embedding_cache[dataset_name])
            gene_emb = torch.stack(gene_emb).to(device)

        cond = {
            "batch_emb": batch["batch_emb"],
            "cont_emb": batch["cont_emb"],
            "gene_emb": gene_emb,
            "cov_celltype": batch["cov_celltype"],
            "cov_pert": batch["cov_pert"],
            "ds_name": batch["ds_name"],
        }

        losses = self.flow.training_losses(
            self.model,
            pert_emb,
            batch["cont_emb"],
            self_condition=cond,
            model_kwargs=None,
            p_drop_cond=self.model_cfg.p_drop_cond,
            MMD_loss_fn=self.loss_fn,
            mmd_weight_alpha=self.optimizer_cfg.mmd_weight_alpha,
            mmd_weight_gamma=self.optimizer_cfg.mmd_weight_gamma,
        )

        mse = losses['mse']
        mmd_weighted = losses["mmd_weighted"]
        mmd_raw = losses["mmd_raw"]
        if self.optimizer_cfg.use_mse_loss:
            loss = mse + mmd_weighted
        else:
            loss = mmd_weighted

        return_dict = {}
        # keys = list(set([get_short_dsname(x) for x in self.model.model_cfg.dataset_dict]))
        # name_arr = np.array([get_short_dsname(x) for x in cond["ds_name"]])
        # for ds_name in keys:
        #     if (name_arr == ds_name).any():
        #         return_dict[f"dataset_loss_mse_{ds_name}"] = mse[name_arr == ds_name].detach().nanmean().item()
        #         return_dict[f"dataset_loss_mmd_{ds_name}"] = mmd_per_sample[name_arr == ds_name].detach().nanmean().item()
        #     else:
        #         return_dict[f"dataset_loss_mse1_{ds_name}"] = 0
        #         return_dict[f"dataset_loss_mmd1_{ds_name}"] = 0

        assert self.model.model_name == "Cross_DiT"
        return_dict = losses
        return_dict['loss'] = loss
        # return_dict["mmd"] = losses["mmd"].detach()
        # return_dict["loss1"] = mse_per_sample.detach().mean()
        # return_dict["loss"] = total_per_sample_loss
        # return_dict["weights"] = losses["weights"]
        return return_dict

    def configure_optimizers(self):
        """Return optimizers and schedulers."""
        decay_names = set(get_parameter_names(self.model, [torch.nn.LayerNorm]))
        decay_names_cov = set(get_parameter_names(self.cov_encoder, [torch.nn.LayerNorm]))
        decay_names.update(decay_names_cov)
        decay_names = {n for n in decay_names if "bias" not in n and "layer_norm" not in n and "layernorm" not in n}

        params_decay, params_nodecay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (params_decay if n in decay_names else params_nodecay).append(p)

        for n, p in self.cov_encoder.named_parameters():
            if not p.requires_grad:
                continue
            (params_decay if n in decay_names else params_nodecay).append(p)

        main_groups = [
            {"params": params_decay, "weight_decay": self.optimizer_cfg.optimizer.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        opt_main = get_optimizer(main_groups, self.optimizer_cfg.optimizer)
        sched_main = hydra.utils.call(self.optimizer_cfg.scheduler, optimizer=opt_main)
        return [opt_main], [{"scheduler": sched_main, "interval": "step", "frequency": 1, "monitor": "validation_loss"}]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before each optimizer step."""
        norms = grad_norm(self.model, norm_type=2)
        device = next(self.parameters()).device
        norms = {k: v.to(device) for k, v in norms.items()}
        self.log_dict(
            norms,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            add_dataloader_idx=False,
        )


__all__ = ["FlowPlModel"]
