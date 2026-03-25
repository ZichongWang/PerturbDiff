"""Training entrypoint for the rectified-flow pipeline."""

import os
import sys
import logging

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import model_summary

torch.set_float32_matmul_precision("high")

exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(exc_dir)
module_logger = logging.getLogger(__name__)
module_logger.info("exc_dir: %s", exc_dir)

from src.apps.training.training_datamodule_builder import build_datamodule, populate_covariate_cfg
from src.apps.training.training_pipeline import run_pipeline
from src.apps.training.training_runtime import setup_trainer
from src.common.utils import setup_loggings


@hydra.main(version_base=None, config_path="../../../configs", config_name="rawdata_flow_training")
def main(cfg):
    """
    Launch rectified-flow training using a Hydra config.
    """
    omegaconf.OmegaConf.resolve(cfg)

    logger = setup_loggings(cfg)
    if cfg.model.ckpt_path:
        logger.info("Resuming flow training from checkpoint %s.", cfg.model.ckpt_path)

    pl.seed_everything(cfg.optimization.seed)
    trainer = setup_trainer(cfg)

    datamodule = build_datamodule(cfg, logger)
    populate_covariate_cfg(cfg, datamodule)

    model = hydra.utils.instantiate(
        cfg.lightning.model_module,
        _recursive_=False,
        cov_encoding_cfg=cfg.cov_encoding,
        model_cfg=cfg.model,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        trainer_cfg=cfg.trainer,
        sampling_eval_cfg=cfg.sampling_eval,
        all_split_names=datamodule.all_split_names,
    )
    summary = model_summary.ModelSummary(model, max_depth=2)
    logger.info(summary)

    model.model.group_mean = None
    model.model.group_mean_ctrl = None

    if cfg.model.reinitial_all_from_scratch:
        model.model.initialize_weights()
        logger.info("Re-initialize all layers")

    logger.info("Model:\n%s", model)
    run_pipeline(trainer, model, datamodule, cfg, logger)


if __name__ == "__main__":
    main()
