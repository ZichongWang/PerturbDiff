# Repository Guidelines

## Project Structure & Module Organization
`src/apps/run/` contains the two CLI entrypoints: `rawdata_diffusion_training.py` and `rawdata_diffusion_sampling.py`. Keep those files thin. Put reusable workflow code in `src/apps/training/` and `src/apps/sampling/`, model code in `src/models/`, dataset and datamodule logic in `src/data/`, and shared helpers in `src/common/`. Hydra configs live under `configs/` and are grouped by concern (`data/`, `model/`, `trainer/`, `path/`, `lightning/`, `optimization/`, `cov_encoding/`). Static figures for the paper/site live in `asset/`; `index.html` backs the project page.

## Different branches
This repo was forked from perturbdiff. The main branch is offical implementation, with slightly modification. It uses diffusion for generation. The branch `perturbflow` is a flow matching one implemented by me. I kept data, dataloader, NN and most things, only modification on loss function to match flow matching object.

## Build, Test, and Development Commands
There is no package build step; development is driven by Python entrypoints plus Hydra overrides.

- use conda environment `torch` for all runs. That include all packages you need.

- `python -c "import hydra,pytorch_lightning,torch,numpy,pandas,anndata,scanpy,h5py,yaml,sklearn,transformers,timm,geomloss; print('env ok')"` checks the runtime environment.
- `python src/apps/run/rawdata_diffusion_training.py run_name=smoke data=pbmc_finetune trainer.accelerator=cpu trainer.devices=1 trainer.max_steps=1 trainer.limit_val_batches=1 optimization.micro_batch_size=32 lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger` runs a one-step training smoke test.
- `python src/apps/run/rawdata_diffusion_sampling.py model_checkpoint_path=/abs/path/model.ckpt data=pbmc_finetune device=cpu sampling.num_sampled_batches=1 lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger` smoke-tests sampling.

Before any run, replace the placeholder roots in `configs/path/trixie_path.yaml`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for modules, functions, and Hydra keys, and `PascalCase` for classes such as `PlModel`. Prefer short docstrings over long inline commentary. Add new configuration as reusable YAML in `configs/` instead of hardcoding paths or hyperparameters in scripts.

## Testing Guidelines
 Contributors should treat smoke runs as required validation. For data-pipeline edits, verify config resolution and datamodule setup on a tiny run. For model or sampling changes, include the exact command you used and the output/checkpoint location you verified. If needed, write test files in `\test` folder.

## Commit & Pull Request Guidelines
Recent history uses short lowercase subjects like `format`, `typo`, and `minor`. Keep commit titles concise and imperative; add a scope when helpful, for example `data: fix pbmc cache path`. Pull requests should summarize the affected dataset or module, list the Hydra overrides used for validation, and mention any required checkpoints or large external files. Include screenshots only for changes to `index.html`, figures in `asset/`, or other user-visible artifacts.
