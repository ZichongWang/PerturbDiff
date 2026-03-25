"""Sampling entrypoint for the rectified-flow pipeline."""

import os
import sys
import logging

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.utilities import model_summary

exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(exc_dir)
module_logger = logging.getLogger(__name__)
module_logger.info("exc_dir: %s", exc_dir)

from src.apps.sampling.flow_sampling_generation import generate_flow_samples
from src.apps.sampling.flow_sampling_setup import build_sampling_datamodule, load_flow_sampling_model, populate_covariate_cfg
from src.apps.sampling.sampling_utils import setup_device
from src.common.utils import setup_loggings


@hydra.main(version_base=None, config_path="../../../configs", config_name="rawdata_flow_sampling")
def main(cfg):
    """
    Load a pretrained flow model and generate samples.
    """
    omegaconf.OmegaConf.resolve(cfg)

    logger = setup_loggings(cfg)
    logger.info("Starting flow model sampling")
    if getattr(cfg, "log_full_config", False):
        logger.info("Config:\n%s", omegaconf.OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.optimization.seed)

    datamodule = build_sampling_datamodule(cfg, logger)
    populate_covariate_cfg(cfg, datamodule)
    model = load_flow_sampling_model(cfg, logger, datamodule)

    summary = model_summary.ModelSummary(model, max_depth=2)
    logger.info(summary)

    device = setup_device(cfg, logger)

    logger.info("Starting flow sample generation...")
    datamodule.setup_dataset()

    truths, samples, trajectories, decoded = generate_flow_samples(
        model,
        model.flow,
        cfg,
        device,
        logger,
        datamodule,
        pca_for_decode=None,
    )
    _ = truths, trajectories, decoded

    logger.info("Sample statistics:")
    logger.info("  Shape: %s", samples.shape)
    logger.info("  Mean: %.4f", samples.mean())
    logger.info("  Std: %.4f", samples.std())
    logger.info("  Min: %.4f", samples.min())
    logger.info("  Max: %.4f", samples.max())


if __name__ == "__main__":
    main()
