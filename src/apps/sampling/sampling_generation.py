"""Sampling generation core split from rawdata_diffusion_sample.py (logic-preserving)."""

import gc
import random
import time

import anndata
import numpy as np
import torch
from geomloss import SamplesLoss
from pytorch_lightning.utilities import move_data_to_device
from sklearn.metrics import r2_score

from src.apps.sampling.sampling_io import save_adata
from src.apps.sampling.sampling_generation_helpers import (
    build_obs_data,
    build_self_condition,
    build_x_ctrl,
    build_gene_embedding_cache,
    collect_batch_covariates,
    load_ctrl_adata,
    load_selected_genes,
    resolve_sampling_runner,
)


def sample_dataloader_batches(
    model,
    diffusion,
    cfg,
    device,
    logger,
    dataloader,
    *,
    datamodule=None,
    num_sampled_batches,
    use_ddim,
    clip_denoised,
    progress,
    guidance_strength,
    start_time,
    nw,
    start_guide_steps,
    eta,
    seed=None,
    collect_covariates=False,
):
    """
    Run reverse-diffusion sampling on a dataloader and compute aggregate metrics.
    """
    dataloader_iter = iter(dataloader)
    was_training = model.training
    model = model.to(device)
    model.eval()

    sampler_batch_size = getattr(dataloader, "batch_size", None) or cfg.optimization.micro_batch_size
    if cfg.data.use_cell_set is not None:
        sampler_batch_size = int(sampler_batch_size // cfg.data.use_cell_set)

    if num_sampled_batches is None:
        num_sampled_batches = len(dataloader)
    else:
        num_sampled_batches = min(int(num_sampled_batches), len(dataloader))
    num_samples = len(dataloader.sampler)

    input_dim = (
        getattr(cfg.model, "input_dim", None)
        or getattr(cfg.model, "output_size", None)
        or getattr(cfg.model, "gene_dim", None)
    )

    logger.info(f"Generating {num_samples} samples with batch size {sampler_batch_size}")
    logger.info(f"Using {'DDIM' if use_ddim else 'DDPM'} sampling")

    cell_set_number = cfg.data.use_cell_set if hasattr(cfg.data, "use_cell_set") and cfg.data.use_cell_set is not None else 1
    sampling_cfg = type(
        "SamplingRuntimeCfg",
        (),
        {
            "sampling": type(
                "SamplingOptions",
                (),
                {
                    "start_time": start_time,
                    "eta": eta,
                    "guidance_strength": guidance_strength,
                    "nw": nw,
                    "start_guide_steps": start_guide_steps,
                },
            )(),
            "model": cfg.model,
        },
    )()

    all_truths = []
    all_samples = []
    all_covariates = []
    batch_mmd_metrics = []
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

        _genes = load_selected_genes(cfg)
        assert len(_genes) == 2000

        with torch.no_grad():
            for batch_idx in range(num_sampled_batches):
                logger.info(f"Generating batch {batch_idx + 1}/{num_sampled_batches}")

                sample_fn, sampling_kwargs = resolve_sampling_runner(sampling_cfg, diffusion, use_ddim)
                batch_data = next(dataloader_iter)
                batch_data = move_data_to_device(batch_data, device)

                batch_data["batch_emb"] = model._encode_covariates(batch_data)
                pert_emb = batch_data["pert_emb"]
                resolved_input_dim = int(input_dim) if input_dim is not None else int(pert_emb.shape[-1])
                device = pert_emb.device
                if reference_col_genes is None:
                    reference_col_genes = batch_data["col_genes"][0]
                gene_emb = build_gene_embedding_cache(model, batch_data, device)
                self_condition = build_self_condition(cfg, model, batch_data, gene_emb)
                mmd_loss = SamplesLoss(loss="energy", blur=0.05, scaling=0.5).to(device)
                mask = ~batch_data["is_padded_list"].bool()
                batch_start_time = time.time()

                sample, _traj = sample_fn(
                    model.model,
                    (len(pert_emb), cell_set_number, resolved_input_dim),
                    self_condition=self_condition,
                    clip_denoised=clip_denoised,
                    device=device,
                    progress=progress,
                    **sampling_kwargs,
                )

                pert_emb = pert_emb[mask]
                if sample is not None:
                    sample = sample[mask]

                pert_emb_cpu = pert_emb.detach().cpu()
                sample_cpu = sample.detach().cpu()

                np_mask = np.isin(batch_data["col_genes"][0], _genes)
                truth_np = pert_emb_cpu.numpy()[:, np_mask]
                sample_np = sample_cpu.numpy()[:, np_mask]

                r2_metric = r2_score(truth_np.mean(0), sample_np.mean(0))
                torch_mask = torch.as_tensor(np_mask, device=device, dtype=torch.bool)
                sample_eval = sample[:, torch_mask]
                truth_eval = pert_emb[:, torch_mask]
                mmd_metric = mmd_loss(sample_eval, truth_eval).item()
                batch_time = time.time() - batch_start_time

                logger.info(
                    "Batch %s completed in %.2fs, r2_metric for this batch: %s, mmd_metric for this batch: %s",
                    batch_idx + 1,
                    batch_time,
                    r2_metric,
                    mmd_metric,
                )

                all_truths.append(truth_np)
                all_samples.append(sample_np)
                batch_mmd_metrics.append(mmd_metric)

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

    all_truths = np.concatenate(all_truths, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)
    overall_r2 = r2_score(all_truths.mean(0), all_samples.mean(0))
    mean_mmd = float(np.mean(batch_mmd_metrics)) if batch_mmd_metrics else float("nan")

    return {
        "truths": all_truths,
        "samples": all_samples,
        "covariates": all_covariates,
        "r2_metric": overall_r2,
        "mmd_metric": mean_mmd,
        "reference_col_genes": reference_col_genes,
    }


def generate_samples(
    model,
    diffusion,
    cfg, 
    device, 
    logger, 
    datamodule, 
    pca_for_decode=None,
    store_noised_truth=False,
):
    """
    Dispatcher that preserves original behavior:
      - If `pca_for_decode` is None -> sample directly in **original gene space**.
      - If `pca_for_decode` is provided -> delegate to `generate_samples_pca` and return its outputs.

    Gene-space mode:
      - pert_emb and samples are [B, S, G] (G ~ 2000)
      - R^2 and MMD are computed in gene space (this is the working space, not a 'decoded' space)
      - Returns (truths, samples, trajectories, decoded=None)
    """
    # Use test split dataloader.
    dataloader = datamodule.val_dataloader()[1] # for test split

    data_args = getattr(dataloader.dataset, "data_args", None)
    normalize_counts = getattr(data_args, "normalize_counts", None) if data_args is not None else None

    assert store_noised_truth == False

    results = sample_dataloader_batches(
        model,
        diffusion,
        cfg,
        device,
        logger,
        dataloader,
        datamodule=datamodule,
        num_sampled_batches=cfg.sampling.num_sampled_batches,
        use_ddim=cfg.sampling.use_ddim,
        clip_denoised=cfg.sampling.clip_denoised,
        progress=cfg.sampling.progress,
        guidance_strength=cfg.sampling.guidance_strength,
        start_time=cfg.sampling.start_time,
        nw=cfg.sampling.nw,
        start_guide_steps=cfg.sampling.start_guide_steps,
        eta=cfg.sampling.eta,
        seed=getattr(cfg.optimization, "seed", None),
        collect_covariates=True,
    )

    all_truths = results["truths"]
    all_samples = results["samples"]
    all_trajectories = []
    all_covariates = results["covariates"]
    reference_col_genes = results["reference_col_genes"]

    logger.info(f"Generated {all_samples.shape[0]} samples with shape {all_samples.shape}")
    logger.info(f"Overall r2_metric: {results['r2_metric']}")


    all_pert = np.concatenate([x[0] for x in all_covariates], axis=0)
    all_celltype = np.concatenate([x[1] for x in all_covariates], axis=0)
    all_batch = np.concatenate([x[2] for x in all_covariates], axis=0)
        
    ctrl_adata, var_index = load_ctrl_adata(cfg)

    # Getting X_ctrl.
    _genes = load_selected_genes(cfg)
    var_index = _genes

    X_ctrl = build_x_ctrl(ctrl_adata, _genes, cfg)

    if normalize_counts is not None:
        all_samples *= normalize_counts
        all_truths *= normalize_counts

    obs = build_obs_data(cfg, all_pert, all_celltype, all_batch, ctrl_adata)
    
    if len(reference_col_genes) > 2000: # i.e., when cfg.data.data_name = "Tahoe100mPBMCPretrain"
        logger.info("Restricting to downstream data selected genes only...")
        assert (ctrl_adata.var.index[ctrl_adata.var.highly_variable] == _genes).all()

        # unmerged data setting, no need of using sel_mask
        sel_mask = np.isin(reference_col_genes, _genes)
        if all_samples.shape[-1] != len(_genes):
        
            assert sel_mask.sum() == 2000
            all_samples = all_samples[:, sel_mask]
            all_truths = all_truths[:, sel_mask]

        # reorder
        cur_genes = np.array(reference_col_genes)[sel_mask].tolist()
        sort_idx = [cur_genes.index(g) for g in _genes]
        assert (np.array(cur_genes)[sort_idx] == _genes).all()
        all_samples = all_samples[:, sort_idx]
        all_truths = all_truths[:, sort_idx]

        var_index = _genes

    pred_adata = anndata.AnnData(X=np.concatenate([all_samples, X_ctrl]), obs=obs)
    true_adata = anndata.AnnData(X=np.concatenate([all_truths, X_ctrl]), obs=obs)
    if var_index is not None:
        pred_adata.var.index = var_index
        true_adata.var.index = var_index

    # Use a shared timestamp so that auxiliary files align with the main outputs.
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    save_adata(pred_adata, true_adata, cfg, logger, timestamp=run_timestamp)

    gc.collect()

    return all_truths, all_samples, all_trajectories, None
