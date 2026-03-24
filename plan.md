## Training-Time Sampling Validation

### Summary
Add an optional sampling-based validation path to training that runs full reverse diffusion from noise on the validation split during each validation event, then logs `R2` and `MMD` without saving `.h5ad` outputs. Implement it as an epoch-end validation routine, not inside `validation_step`, so it runs once per validation event on a fixed small subset and does not multiply cost by every validation batch.

### Key Changes
- Add a new `sampling_eval` section to the training config, separate from the standalone sampling entrypoint config.
  - Fields: `enabled`, `split`, `num_batches`, `use_ddim`, `clip_denoised`, `guidance_strength`, `start_time`, `nw`, `start_guide_steps`, `eta`, `progress`, `seed`.
  - Defaults: disabled, `split=validation`, small fixed `num_batches`, `use_ddim=true`, `progress=false`.
- Extend the Lightning module to run sampling-based evaluation in `on_validation_epoch_end`.
  - Keep the existing denoising-loss validation path unchanged.
  - After logging `validation_<split>_loss_epoch`, run sampling-eval only if `sampling_eval.enabled=true`.
  - Use the selected validation dataloader from `self.trainer.val_dataloaders` rather than rebuilding loaders.
  - Execute only on the configured split, defaulting to the validation loader, not test.
- Factor the reusable sampling/evaluation core out of the standalone sampling pipeline.
  - Extract a helper that accepts `(model, diffusion, cfg-like sampling settings, dataloader, logger)` and returns metrics only.
  - Reuse the existing generation logic: build `batch_emb`, build `self_condition`, call DDIM/DDPM sampler, mask padded duplicates, compute `r2_score` and `SamplesLoss(loss="energy")`.
  - Do not call `save_adata` from training-time validation.
- Define training-time sampling metrics explicitly.
  - `sampling_validation_r2_epoch`: compute exactly like the current sampling pipeline, using concatenated subset predictions/truths and `r2_score(all_truths.mean(0), all_samples.mean(0))`.
  - `sampling_validation_mmd_epoch`: compute as the mean of per-batch energy-distance values across sampled validation batches.
  - Log these as epoch metrics only, with `logger=True`, `prog_bar=True`, and no checkpoint monitor change.
- Make sampling-eval deterministic enough for comparison across epochs.
  - Before iterating the subset, save and restore NumPy/Torch RNG state.
  - Seed the sampling-eval pass from `sampling_eval.seed` so the same subset/control matching path is used each validation event as much as the current dataset code allows.
- Handle distributed training safely.
  - Run sampling-eval only on global rank 0 to avoid duplicated expensive generation.
  - Log with rank-zero semantics and leave existing synchronized denoising-loss logging unchanged.
  - If needed, add a barrier after rank-0 sampling-eval to keep validation flow aligned.

### Interfaces / Behavior
- Public config addition in training: `sampling_eval.*`
- No change to the standalone sampling CLI behavior.
- No automatic sample-file writing during training validation.
- Existing checkpoint monitoring stays on `validation_validation_loss_epoch` by default; sampling metrics are log-only.

### Test Plan
- Smoke test with `sampling_eval.enabled=true`, `sampling_eval.num_batches=1`, `trainer.limit_val_batches=1`, single GPU.
  - Confirm denoising validation still logs.
  - Confirm `sampling_validation_r2_epoch` and `sampling_validation_mmd_epoch` appear once per validation event.
- Compare training-time sampling metrics against the standalone sampling pipeline on the same checkpoint and same small validation subset.
  - Metrics do not need to be bit-identical, but should be directionally consistent and use the same formulas.
- Verify no `.h5ad` files are created during training validation.
- Verify disabled mode leaves current training/validation behavior unchanged.
- Verify the code works with `use_cell_set` and padded groups by checking masked duplicates do not inflate sample counts.

### Assumptions
- Implement sampling-eval in `on_validation_epoch_end`, not literally inside `validation_step`, because the requested behavior is “run inference during validation” and this is the correct Lightning hook for a once-per-validation sampling pass.
- Use validation split only for training-time sampling metrics.
- Use a small fixed subset each validation event for cost control.
- Keep checkpoint selection on the current denoising loss unless changed later.
