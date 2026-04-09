from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


PALETTE = {
    "control": "#9aa0a6",
    "prediction": "#d95f02",
    "ground_truth": "#1b9e77",
}


def parse_args() -> argparse.Namespace:
    analyze_dir = Path(__file__).resolve().parent
    repo_root = analyze_dir.parent
    sample_dir = repo_root / "checkpoints/test_samples/replogle_gaussian_flow_test_sampling"

    parser = argparse.ArgumentParser(
        description="Plot joint UMAPs for control, prediction, and ground truth cells."
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=sample_dir / "flow_predict_20260409_192016.h5ad",
        help="Path to the predicted AnnData file.",
    )
    parser.add_argument(
        "--true-path",
        type=Path,
        default=sample_dir / "flow_true_20260409_192016.h5ad",
        help="Path to the ground-truth AnnData file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=analyze_dir / "umap_plots",
        help="Directory for generated plots and summary csv.",
    )
    parser.add_argument(
        "--top-n-perts",
        type=int,
        default=12,
        help="Use the top-N perturbations by cell count when --selected-perts is omitted.",
    )
    parser.add_argument(
        "--selected-perts",
        nargs="+",
        default=None,
        help="Explicit perturbation names to plot.",
    )
    parser.add_argument(
        "--control-max-cells",
        type=int,
        default=250,
        help="Maximum number of control cells to include in each plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for control subsampling and UMAP.",
    )
    return parser.parse_args()


def subsample_adata(adata: ad.AnnData, max_cells: int | None, seed: int) -> ad.AnnData:
    if max_cells is None or adata.n_obs <= max_cells:
        return adata.copy()
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[picked].copy()


def load_inputs(pred_path: Path, true_path: Path) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, pd.DataFrame]:
    pred_adata = ad.read_h5ad(pred_path)
    true_adata = ad.read_h5ad(true_path)

    assert pred_adata.n_obs == true_adata.n_obs
    assert pred_adata.n_vars == true_adata.n_vars
    assert pred_adata.obs[["gene", "cell_line", "gem_group"]].equals(
        true_adata.obs[["gene", "cell_line", "gem_group"]]
    )

    control_mask = pred_adata.obs["gene"].eq("non-targeting")
    control_adata = pred_adata[control_mask].copy()
    pert_counts = (
        pred_adata.obs.loc[~control_mask, "gene"]
        .value_counts()
        .rename_axis("pert")
        .reset_index(name="n_cells")
    )
    return pred_adata, true_adata, control_adata, pert_counts


def build_plot_adata(
    pert: str,
    pred_adata: ad.AnnData,
    true_adata: ad.AnnData,
    control_adata: ad.AnnData,
    control_max_cells: int,
    seed: int,
) -> ad.AnnData:
    ctrl = subsample_adata(control_adata, control_max_cells, seed)
    pred = pred_adata[pred_adata.obs["gene"].eq(pert)].copy()
    truth = true_adata[true_adata.obs["gene"].eq(pert)].copy()

    if pred.n_obs == 0 or truth.n_obs == 0:
        raise ValueError(f"No cells found for pert={pert}")

    merged = ad.concat(
        {"control": ctrl, "prediction": pred, "ground_truth": truth},
        label="source",
        index_unique="-",
        join="inner",
    )

    merged.X = np.asarray(merged.X, dtype=np.float32)
    sc.pp.scale(merged, max_value=10)

    n_comps = max(2, min(50, merged.n_obs - 1, merged.n_vars - 1))
    n_pcs = max(2, min(30, n_comps))
    n_neighbors = max(5, min(30, merged.n_obs - 1))

    sc.tl.pca(merged, svd_solver="arpack", n_comps=n_comps)
    sc.pp.neighbors(merged, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(merged, random_state=seed)
    return merged


def plot_single_pert(
    pert: str,
    pred_adata: ad.AnnData,
    true_adata: ad.AnnData,
    control_adata: ad.AnnData,
    output_dir: Path,
    control_max_cells: int,
    seed: int,
) -> tuple[ad.AnnData, Path]:
    plot_adata = build_plot_adata(
        pert=pert,
        pred_adata=pred_adata,
        true_adata=true_adata,
        control_adata=control_adata,
        control_max_cells=control_max_cells,
        seed=seed,
    )
    counts = plot_adata.obs["source"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(
        plot_adata,
        color="source",
        palette=PALETTE,
        size=20,
        alpha=0.85,
        frameon=False,
        legend_loc="right margin",
        title=(
            f"{pert} | control={counts.get('control', 0)} | "
            f"prediction={counts.get('prediction', 0)} | "
            f"ground truth={counts.get('ground_truth', 0)}"
        ),
        ax=ax,
        show=False,
    )

    out_path = output_dir / f"{pert}_control_prediction_ground_truth_umap.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_adata, out_path


def main() -> None:
    args = parse_args()

    sc.settings.verbosity = 1
    sc.set_figure_params(dpi=120, facecolor="white")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_adata, true_adata, control_adata, pert_counts = load_inputs(
        pred_path=args.pred_path,
        true_path=args.true_path,
    )
    selected_perts = args.selected_perts or pert_counts.head(args.top_n_perts)["pert"].tolist()

    print(f"pred shape: {pred_adata.shape}")
    print(f"true shape: {true_adata.shape}")
    print(f"control cells: {control_adata.n_obs}")
    print(f"selected perts: {selected_perts}")

    rows = []
    for pert in selected_perts:
        plot_adata, out_path = plot_single_pert(
            pert=pert,
            pred_adata=pred_adata,
            true_adata=true_adata,
            control_adata=control_adata,
            output_dir=args.output_dir,
            control_max_cells=args.control_max_cells,
            seed=args.seed,
        )
        counts = plot_adata.obs["source"].value_counts()
        rows.append(
            {
                "pert": pert,
                "n_control": int(counts.get("control", 0)),
                "n_prediction": int(counts.get("prediction", 0)),
                "n_ground_truth": int(counts.get("ground_truth", 0)),
                "plot_path": str(out_path),
            }
        )
        print(out_path)

    summary_df = pd.DataFrame(rows)
    summary_path = args.output_dir / "selected_perts_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
