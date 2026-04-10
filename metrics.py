from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import r2_score


def parse_args() -> argparse.Namespace:
    analyze_dir = Path(__file__).resolve().parent
    repo_root = analyze_dir.parent
    sample_dir = repo_root / "checkpoints/test_samples/nommd"

    parser = argparse.ArgumentParser(
        description=(
            "Compute pseudobulk perturbation metrics from paired true/pred h5ad files. "
            "Metrics follow the perturbation-level definitions in the paper draft."
        )
    )
    parser.add_argument(
        "--true-path",
        type=Path,
        default="/home/zichong/fork/PerturbDiff/checkpoints/replogle_sampling/samples/diffusion_true_20260409_231823.h5ad",
        help="Path to the ground-truth h5ad.",
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default="/home/zichong/fork/PerturbDiff/checkpoints/replogle_sampling/samples/diffusion_predict_20260409_231823.h5ad",
        help="Path to the predicted h5ad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the output CSV/JSON. Defaults to the prediction file directory.",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="gene",
        help="Observation column defining the perturbation label.",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default="non-targeting",
        help="Value in --pert-col identifying control cells.",
    )
    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=None,
        help=(
            "Columns defining one evaluation condition lambda. "
            "Defaults to [pert-col], i.e. one row per perturbation."
        ),
    )
    parser.add_argument(
        "--match-control-cols",
        nargs="+",
        default=None,
        help=(
            "Columns used to match control cells to each perturbation condition. "
            "Defaults to ['cell_line', 'gem_group'] when present."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix. Defaults to the prediction file stem.",
    )
    return parser.parse_args()


def infer_match_control_cols(obs: pd.DataFrame, pert_col: str) -> list[str]:
    preferred = ["cell_line", "gem_group", "donor", "plate", "cell_type"]
    return [col for col in preferred if col in obs.columns and col != pert_col]


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    corr = stats.pearsonr(x, y).statistic
    return float(corr)


def make_tuple_keys(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series([()] * len(df), index=df.index, dtype=object)
    return df[cols].astype(str).agg(tuple, axis=1)


def load_inputs(true_path: Path, pred_path: Path) -> tuple[ad.AnnData, ad.AnnData]:
    true_adata = ad.read_h5ad(true_path)
    pred_adata = ad.read_h5ad(pred_path)

    assert true_adata.n_obs == pred_adata.n_obs, "true/pred number of cells must match"
    assert true_adata.n_vars == pred_adata.n_vars, "true/pred number of genes must match"
    assert true_adata.obs.equals(pred_adata.obs), "true/pred obs must align exactly"
    assert np.array_equal(true_adata.var_names, pred_adata.var_names), "true/pred genes must align exactly"
    return true_adata, pred_adata


def compute_group_metrics(
    obs: pd.DataFrame,
    true_x: np.ndarray,
    pred_x: np.ndarray,
    pert_col: str,
    control_label: str,
    group_cols: list[str],
    match_control_cols: list[str],
) -> pd.DataFrame:
    obs = obs.copy().reset_index(drop=True)
    for col in set(group_cols + match_control_cols + [pert_col]):
        obs[col] = obs[col].astype(str)

    pert_mask = obs[pert_col] != str(control_label)
    pert_obs = obs.loc[pert_mask].copy()
    if pert_obs.empty:
        raise ValueError("No perturbed cells found after excluding control label.")

    pert_obs["_group_key"] = make_tuple_keys(pert_obs, group_cols)
    control_mask_all = obs[pert_col] == str(control_label)
    control_obs_all = obs.loc[control_mask_all].copy()
    control_obs_all["_match_key"] = make_tuple_keys(control_obs_all, match_control_cols)

    rows = []
    for group_key, group_idx in pert_obs.groupby("_group_key").groups.items():
        group_indices = np.asarray(group_idx, dtype=np.int64)
        group_obs = obs.iloc[group_indices]

        allowed_control_keys = set(make_tuple_keys(group_obs, match_control_cols).tolist())
        control_match_mask = control_obs_all["_match_key"].isin(allowed_control_keys)
        control_indices = np.asarray(control_obs_all.index[control_match_mask], dtype=np.int64)
        if control_indices.size == 0:
            raise ValueError(f"No matched control cells found for condition {group_key}.")

        true_group = true_x[group_indices]
        pred_group = pred_x[group_indices]
        ctrl_group = true_x[control_indices]

        mu_true_pert = true_group.mean(axis=0)
        mu_pred_pert = pred_group.mean(axis=0)
        mu_ctrl = ctrl_group.mean(axis=0)

        delta_true = mu_true_pert - mu_ctrl
        delta_pred = mu_pred_pert - mu_ctrl

        row = {
            "num_pert_cells": int(group_indices.size),
            "num_ctrl_cells": int(control_indices.size),
            "overall_r2": float(r2_score(mu_true_pert, mu_pred_pert)),
            "delta_r2": float(r2_score(delta_true, delta_pred)),
            "pdcorr": safe_pearsonr(delta_true, delta_pred),
            "mae": float(np.abs(delta_pred - delta_true).mean()),
        }
        for col, value in zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)):
            row[col] = value
        rows.append(row)

    metric_df = pd.DataFrame(rows)
    ordered_cols = group_cols + [
        "num_pert_cells",
        "num_ctrl_cells",
        "overall_r2",
        "delta_r2",
        "pdcorr",
        "mae",
    ]
    return metric_df.loc[:, ordered_cols].sort_values(group_cols).reset_index(drop=True)


def summarize_metrics(metric_df: pd.DataFrame, group_cols: list[str]) -> dict:
    return {
        "num_conditions": int(len(metric_df)),
        "group_cols": list(group_cols),
        "overall_r2_mean": float(metric_df["overall_r2"].mean()),
        "delta_r2_mean": float(metric_df["delta_r2"].mean()),
        "pdcorr_mean": float(metric_df["pdcorr"].mean()),
        "mae_mean": float(metric_df["mae"].mean()),
    }


def main() -> None:
    args = parse_args()
    true_adata, pred_adata = load_inputs(args.true_path, args.pred_path)

    group_cols = args.group_cols or [args.pert_col]
    match_control_cols = args.match_control_cols
    if match_control_cols is None:
        match_control_cols = infer_match_control_cols(true_adata.obs, args.pert_col)

    output_dir = args.output_dir or args.pred_path.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.pred_path.stem

    metric_df = compute_group_metrics(
        obs=true_adata.obs,
        true_x=to_dense(true_adata.X),
        pred_x=to_dense(pred_adata.X),
        pert_col=args.pert_col,
        control_label=args.control_label,
        group_cols=group_cols,
        match_control_cols=match_control_cols,
    )
    summary = summarize_metrics(metric_df, group_cols)
    summary.update(
        {
            "true_path": str(args.true_path.resolve()),
            "pred_path": str(args.pred_path.resolve()),
            "pert_col": args.pert_col,
            "control_label": args.control_label,
            "match_control_cols": list(match_control_cols),
        }
    )

    csv_path = output_dir / f"{prefix}_perturbation_metrics.csv"
    json_path = output_dir / f"{prefix}_perturbation_metrics_summary.json"
    metric_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"Saved per-condition metrics to: {csv_path}")
    print(f"Saved summary metrics to: {json_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()