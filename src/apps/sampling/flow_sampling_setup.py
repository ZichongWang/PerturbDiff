"""Setup helpers for the flow sampling entrypoint."""

import torch

from src.apps.sampling.sampling_setup import build_sampling_datamodule, populate_covariate_cfg
from src.apps.training.training_model_checkpoint import allow_omegaconf_checkpoint_unpickling
from src.models.flow.flow_lightning_module import FlowPlModel


def load_flow_sampling_model(cfg, logger, datamodule):
    """
    Load a rectified-flow checkpoint for sampling.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param datamodule: Data module providing datasets and loaders.
    :return: Loaded flow Lightning module.
    """
    allow_omegaconf_checkpoint_unpickling()
    ckpt = torch.load(cfg.model_checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    cov_cfg = hparams.get("cov_encoding_cfg", cfg.cov_encoding)
    model_cfg = hparams.get("model_cfg", cfg.model)
    optimizer_cfg = hparams.get("optimizer_cfg", cfg.optimization)
    sampling_eval_cfg = hparams.get("sampling_eval_cfg", None)
    cov_cfg["celltype_encoding"] = cfg.cov_encoding.celltype_encoding

    if cfg.device == "auto":
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        map_location = cfg.device

    model = FlowPlModel.load_from_checkpoint(
        cfg.model_checkpoint_path,
        cov_encoding_cfg=cov_cfg,
        model_cfg=model_cfg,
        optimizer_cfg=optimizer_cfg,
        py_logger=logger,
        trainer_cfg=cfg.trainer,
        sampling_eval_cfg=sampling_eval_cfg,
        all_split_names=datamodule.all_split_names,
        map_location=map_location,
        weights_only=False,
    )
    return model


__all__ = ["build_sampling_datamodule", "populate_covariate_cfg", "load_flow_sampling_model"]
