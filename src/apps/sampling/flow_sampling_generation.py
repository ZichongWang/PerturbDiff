"""Flow sampling generation helpers and metrics."""

import gc
import json
import random
import time
from pathlib import Path

import anndata
import numpy as np
import torch
from geomloss import SamplesLoss
from pytorch_lightning.utilities import move_data_to_device
from scipy import stats
from sklearn.metrics import r2_score

from src.apps.sampling.sampling_generation_helpers import (
    build_gene_embedding_cache,
    build_obs_data,
    build_self_condition,
    build_x_ctrl,
    collect_batch_covariates,
    load_ctrl_adata,
    load_selected_genes,
)


def save_flow_adata(pred_adata, true_adata, cfg, logger, timestamp=None):
    """
    Save flow predictions and truths with flow-specific filenames.

    :param pred_adata: Predicted AnnData object.
    :param true_adata: Ground-truth AnnData object.
    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param timestamp: Optional timestamp override.
    :return: None.
    """
    if cfg.sampling.output_dir is None:
        output_dir = Path(cfg.save_dir_path) / "samples"
    else:
        output_dir = Path(cfg.sampling.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    pred_path = output_dir / f"flow_predict_{timestamp}.h5ad"
    pred_adata.write_h5ad(pred_path)
    logger.info("Samples saved to: %s", pred_path)

    true_path = output_dir / f"flow_true_{timestamp}.h5ad"
    true_adata.write_h5ad(true_path)
    logger.info("Truths saved to: %s", true_path)


def save_flow_metrics(metrics, cfg, logger, timestamp=None):
    """
    Save aggregate flow sampling metrics as JSON next to the sampled outputs.

    :param metrics: Metric dictionary with JSON-serializable values.
    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param timestamp: Optional timestamp override.
    :return: None.
    """
    if cfg.sampling.output_dir is None:
        output_dir = Path(cfg.save_dir_path) / "samples"
    else:
        output_dir = Path(cfg.sampling.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    metrics_path = output_dir / f"flow_metrics_{timestamp}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    logger.info("Metrics saved to: %s", metrics_path)


def sample_flow_dataloader_batches(
    model,
    flow,
    cfg,
    device,
    logger,
    dataloader,
    *,
    datamodule=None,
    num_sampled_batches,
    flow_steps,
    clip_denoised,
    progress,
    guidance_strength,
    seed=None,
    collect_covariates=False,
):
    """
    Run Euler flow sampling on a dataloader and compute aggregate metrics.
    """
    dataloader_iter = iter(dataloader)
    was_training = model.training
    model = model.to(device)
    model.eval()

    if num_sampled_batches is None:
        num_sampled_batches = len(dataloader)
    else:
        num_sampled_batches = min(int(num_sampled_batches), len(dataloader))
    num_samples = len(dataloader.sampler)

    logger.info("Generating %s flow samples with %s Euler steps", num_samples, flow_steps)

    all_truths = []
    all_samples = []
    all_covariates = []
    batch_fake_r2_metrics = []
    batch_delta_r2_metrics = []
    batch_ctrl_to_pert_r2_metrics = []
    batch_mmd_metrics = []
    batch_ctrl_to_pert_mmd_metrics = []
    batch_pearson_metrics = []
    batch_mae_metrics = []
    reference_col_genes = None

    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    py_state = random.getstate()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        selected_genes = load_selected_genes(cfg)
        assert len(selected_genes) == 2000
        mmd_loss = SamplesLoss(loss="energy", blur=0.05, scaling=0.5).to(device)

        with torch.no_grad():
            for batch_idx in range(num_sampled_batches):
                logger.info("Generating flow batch %s/%s", batch_idx + 1, num_sampled_batches)
                batch_data = next(dataloader_iter)
                batch_data = move_data_to_device(batch_data, device)
                batch_data["batch_emb"] = model._encode_covariates(batch_data)

                pert_emb = batch_data["pert_emb"]
                control_emb = batch_data["cont_emb"]
                if reference_col_genes is None:
                    reference_col_genes = batch_data["col_genes"][0]
                gene_emb = build_gene_embedding_cache(model, batch_data, device)
                self_condition = build_self_condition(cfg, model, batch_data, gene_emb)
                
                mask = ~batch_data["is_padded_list"].bool()
                batch_start_time = time.time()
                x0 = flow.sample_base_state(control_emb)

                sample, _traj = flow.sample_euler_loop(
                    model.model,
                    x_start=x0,
                    control_input_start=control_emb,
                    self_condition=self_condition,
                    flow_steps=flow_steps,
                    guidance_strength=guidance_strength,
                    clip_denoised=clip_denoised,
                    progress=progress,
                    model_kwargs=None,
                )

                pert_emb = pert_emb[mask]
                sample = sample[mask]
                control = control_emb[mask]

                pert_emb_cpu = pert_emb.detach().cpu()
                sample_cpu = sample.detach().cpu()
                control_cpu = control.detach().cpu()

                np_mask = np.isin(batch_data["col_genes"][0], selected_genes)
                pert_np = pert_emb_cpu.numpy()[:, np_mask]
                sample_np = sample_cpu.numpy()[:, np_mask]
                control_np = control_cpu.numpy()[:, np_mask] # [C, G]; C is use_cell_set

                mu_ctrl = control_np.mean(axis=0) # [G]
                mu_pert = pert_np.mean(axis=0)
                mu_sample = sample_np.mean(axis=0)

                delta_real = mu_pert - mu_ctrl
                delta_pred = mu_sample - mu_ctrl

                pearson_metric = stats.pearsonr(delta_real, delta_pred).statistic
                mae = np.mean(np.abs(delta_real - delta_pred))
                fake_r2_metric = r2_score(mu_pert, mu_sample)
                delta_r2_metric = r2_score(delta_real, delta_pred)
                ctrl_to_pert_r2 = r2_score(mu_ctrl, mu_pert)

                torch_mask = torch.as_tensor(np_mask, device=device, dtype=torch.bool)
                sample_eval = sample[:, torch_mask]
                truth_eval = pert_emb[:, torch_mask]
                control_eval = control[:, torch_mask]
                mmd_metric = mmd_loss(sample_eval, truth_eval).item()
                ctrl_to_pert_mmd_metric = mmd_loss(control_eval, truth_eval).item()
                batch_time = time.time() - batch_start_time

                # logger.info(
                #     "Flow batch %s completed in %.2fs, r2_metric: %s, mmd_metric: %s",
                #     batch_idx + 1,
                #     batch_time,
                #     delta_r2_metric,
                #     mmd_metric,
                # )

                batch_mmd_metrics.append(mmd_metric)
                batch_fake_r2_metrics.append(fake_r2_metric)
                batch_delta_r2_metrics.append(delta_r2_metric)
                batch_ctrl_to_pert_r2_metrics.append(ctrl_to_pert_r2)
                batch_ctrl_to_pert_mmd_metrics.append(ctrl_to_pert_mmd_metric)
                batch_pearson_metrics.append(pearson_metric)
                batch_mae_metrics.append(mae)

                all_truths.append(pert_np)
                all_samples.append(sample_np)

                if collect_covariates:
                    all_covariates.extend(collect_batch_covariates(batch_data, dataloader, datamodule, mask))

    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        random.setstate(py_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if was_training:
            model.train()


    all_truths = np.concatenate(all_truths, axis=0) if all_truths else np.empty((0, 0))
    all_samples = np.concatenate(all_samples, axis=0) if all_samples else np.empty((0, 0))
    overall_r2 = float(r2_score(all_truths.mean(0), all_samples.mean(0))) if all_truths.size and all_samples.size else float("nan")
    mean_mmd = float(np.mean(batch_mmd_metrics)) if batch_mmd_metrics else float("nan")
    mean_fake_r2 = float(np.mean(batch_fake_r2_metrics)) if batch_fake_r2_metrics else float("nan")
    mean_delta_r2 = float(np.mean(batch_delta_r2_metrics)) if batch_delta_r2_metrics else float("nan")
    mean_ctrl_to_pert_r2 = float(np.mean(batch_ctrl_to_pert_r2_metrics)) if batch_ctrl_to_pert_r2_metrics else float("nan")
    mean_ctrl_to_pert_mmd = float(np.mean(batch_ctrl_to_pert_mmd_metrics)) if batch_ctrl_to_pert_mmd_metrics else float("nan")
    mean_pearson_metric = float(np.mean(batch_pearson_metrics)) if batch_pearson_metrics else float("nan")
    mean_mae_metric = float(np.mean(batch_mae_metrics)) if batch_mae_metrics else float("nan")

    return {
        "truths": all_truths,
        "samples": all_samples,
        "covariates": all_covariates,
        "reference_col_genes": reference_col_genes,
        "r2_metric": overall_r2,
        "fake_r2_metric": mean_fake_r2,
        "delta_r2_metric": mean_delta_r2,
        "ctrl_to_pert_r2_metric": mean_ctrl_to_pert_r2,
        "mmd_metric": mean_mmd,
        "ctrl_to_pert_mmd_metric": mean_ctrl_to_pert_mmd,
        "delta_pearson_r": mean_pearson_metric,
        "delta_mae": mean_mae_metric,
    }


def generate_flow_samples(
    model,
    flow,
    cfg,
    device,
    logger,
    datamodule,
    pca_for_decode=None,
    store_noised_truth=False,
):
    """
    Generate samples in gene space with the rectified-flow sampler.
    """
    _ = pca_for_decode
    dataloader = datamodule.val_dataloader()[1]

    data_args = getattr(dataloader.dataset, "data_args", None)
    normalize_counts = getattr(data_args, "normalize_counts", None) if data_args is not None else None

    assert store_noised_truth is False

    results = sample_flow_dataloader_batches(
        model,
        flow,
        cfg,
        device,
        logger,
        dataloader,
        datamodule=datamodule,
        num_sampled_batches=cfg.sampling.num_sampled_batches,
        flow_steps=cfg.sampling.flow_steps,
        clip_denoised=cfg.sampling.clip_denoised,
        progress=cfg.sampling.progress,
        guidance_strength=cfg.sampling.guidance_strength,
        seed=getattr(cfg.optimization, "seed", None),
        collect_covariates=True,
    )

    all_truths = results["truths"]
    all_samples = results["samples"]
    all_trajectories = []
    all_covariates = results["covariates"]
    reference_col_genes = results["reference_col_genes"]

    logger.info("Generated %s samples with shape %s", all_samples.shape[0], all_samples.shape)
    logger.info("Overall r2_metric: %s", results["r2_metric"])
    logger.info("fake_r2_metric: %s", results["fake_r2_metric"])
    logger.info("delta_r2_metric: %s", results["delta_r2_metric"])
    logger.info("ctrl_to_pert_r2_metric: %s", results["ctrl_to_pert_r2_metric"])
    logger.info("mmd_metric: %s", results["mmd_metric"])
    logger.info("ctrl_to_pert_mmd_metric: %s", results["ctrl_to_pert_mmd_metric"])
    logger.info("delta_pearson_r: %s", results["delta_pearson_r"])
    logger.info("delta_mae: %s", results["delta_mae"])

    all_pert = np.concatenate([x[0] for x in all_covariates], axis=0)
    all_celltype = np.concatenate([x[1] for x in all_covariates], axis=0)
    all_batch = np.concatenate([x[2] for x in all_covariates], axis=0)

    ctrl_adata, var_index = load_ctrl_adata(cfg)

    selected_genes = load_selected_genes(cfg)
    var_index = selected_genes
    x_ctrl = build_x_ctrl(ctrl_adata, selected_genes, cfg)

    if normalize_counts is not None:
        all_samples *= normalize_counts
        all_truths *= normalize_counts

    obs = build_obs_data(cfg, all_pert, all_celltype, all_batch, ctrl_adata)

    if len(reference_col_genes) > 2000:
        logger.info("Restricting flow outputs to downstream selected genes only...")
        assert (ctrl_adata.var.index[ctrl_adata.var.highly_variable] == selected_genes).all()

        sel_mask = np.isin(reference_col_genes, selected_genes)
        if all_samples.shape[-1] != len(selected_genes):
            assert sel_mask.sum() == 2000
            all_samples = all_samples[:, sel_mask]
            all_truths = all_truths[:, sel_mask]

        cur_genes = np.array(reference_col_genes)[sel_mask].tolist()
        sort_idx = [cur_genes.index(g) for g in selected_genes]
        assert (np.array(cur_genes)[sort_idx] == selected_genes).all()
        all_samples = all_samples[:, sort_idx]
        all_truths = all_truths[:, sort_idx]
        var_index = selected_genes

    pred_adata = anndata.AnnData(X=np.concatenate([all_samples, x_ctrl]), obs=obs)
    true_adata = anndata.AnnData(X=np.concatenate([all_truths, x_ctrl]), obs=obs)
    if var_index is not None:
        pred_adata.var.index = var_index
        true_adata.var.index = var_index

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_flow_adata(pred_adata, true_adata, cfg, logger, timestamp=run_timestamp)
    save_flow_metrics(
        {
            "r2_metric": results["r2_metric"],
            "fake_r2_metric": results["fake_r2_metric"],
            "delta_r2_metric": results["delta_r2_metric"],
            "ctrl_to_pert_r2_metric": results["ctrl_to_pert_r2_metric"],
            "mmd_metric": results["mmd_metric"],
            "ctrl_to_pert_mmd_metric": results["ctrl_to_pert_mmd_metric"],
            "delta_pearson_r": results["delta_pearson_r"],
            "delta_mae": results["delta_mae"],
            "num_truth_cells": int(all_truths.shape[0]),
            "num_sample_cells": int(all_samples.shape[0]),
            "num_genes": int(all_samples.shape[1]) if all_samples.ndim == 2 else 0,
            "flow_steps": int(cfg.sampling.flow_steps),
            "guidance_strength": float(cfg.sampling.guidance_strength),
        },
        cfg,
        logger,
        timestamp=run_timestamp,
    )

    gc.collect()
    return all_truths, all_samples, all_trajectories, None


__all__ = ["sample_flow_dataloader_batches", "generate_flow_samples"]
