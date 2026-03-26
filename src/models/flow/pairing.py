"""Pairing utilities for flow-matching training."""

import ot
import torch


def _sample_derangement(size: int, device) -> torch.Tensor:
    """Sample a permutation with no fixed points."""
    assert size > 1, "Flow pairing requires use_cell_set > 1."
    base = torch.arange(size, device=device)
    perm = base
    while bool((perm == base).any().item()):
        perm = torch.randperm(size, device=device)
    return perm


def _validate_inputs(control_emb: torch.Tensor, pert_emb: torch.Tensor):
    """Validate the shape and device assumptions required by flow pairing."""
    assert control_emb.ndim == 3, "Flow pairing expects control embeddings shaped [B, S, G]."
    assert pert_emb.ndim == 3, "Flow pairing expects perturbed embeddings shaped [B, S, G]."
    assert control_emb.shape == pert_emb.shape, "Flow pairing expects control and perturbed tensors to share shape [B, S, G]."
    _, set_size, _ = control_emb.shape
    assert set_size > 1, "Flow pairing requires use_cell_set > 1."


def _sample_from_row_distribution(coupling: torch.Tensor) -> torch.Tensor:
    """Sample one control index per perturbed row from a Sinkhorn coupling."""
    row_sums = coupling.sum(dim=-1, keepdim=True)
    probs = coupling / row_sums.clamp_min(1e-12)
    zero_rows = row_sums.squeeze(-1) <= 1e-12
    if bool(zero_rows.any().item()):
        probs[zero_rows] = 1.0 / coupling.shape[-1]
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _pair_random(control_emb: torch.Tensor):
    """Randomly pair controls within each set using a derangement."""
    batch_size, _, _ = control_emb.shape
    perms = torch.stack([_sample_derangement(control_emb.shape[1], control_emb.device) for _ in range(batch_size)], dim=0)
    paired = control_emb.gather(1, perms.unsqueeze(-1).expand(-1, -1, control_emb.shape[-1]))
    return paired, {"indices": perms}


def _pair_ot_sinkhorn(
    control_emb: torch.Tensor,
    pert_emb: torch.Tensor,
    *,
    ot_cost: str,
    ot_reg: float,
    ot_num_iters: int,
    ot_sampling: str,
    return_coupling: bool = False,
):
    """Pair controls and perturbations within each set using Sinkhorn OT."""
    if ot_cost != "l2_squared":
        raise NotImplementedError(f"Unsupported OT cost: {ot_cost}")
    if ot_sampling != "row_multinomial":
        raise NotImplementedError(f"Unsupported OT sampling rule: {ot_sampling}")
    assert ot_reg > 0.0, "model.ot_reg must be positive for Sinkhorn OT."
    assert ot_num_iters > 0, "model.ot_num_iters must be a positive integer."

    batch_size, set_size, gene_dim = control_emb.shape
    _ = gene_dim
    uniform_mass = torch.full((set_size,), 1.0 / float(set_size), device=control_emb.device, dtype=control_emb.dtype)

    sampled_indices = []
    paired_controls = []
    couplings = [] if return_coupling else None

    for batch_idx in range(batch_size):
        # Rows index perturbed cells and columns index control cells.
        cost = torch.cdist(pert_emb[batch_idx], control_emb[batch_idx], p=2) ** 2
        coupling = ot.sinkhorn(
            uniform_mass,
            uniform_mass,
            cost,
            reg=ot_reg,
            numItermax=int(ot_num_iters),
            warn=False,
        )
        indices = _sample_from_row_distribution(coupling)
        sampled_indices.append(indices)
        paired_controls.append(control_emb[batch_idx].index_select(0, indices))
        if return_coupling:
            couplings.append(coupling)

    sampled_indices = torch.stack(sampled_indices, dim=0)
    paired_controls = torch.stack(paired_controls, dim=0)
    metadata = {"indices": sampled_indices}
    if return_coupling:
        metadata["coupling"] = torch.stack(couplings, dim=0)
    return paired_controls, metadata


def pair_control_within_set(
    control_emb: torch.Tensor,
    pert_emb: torch.Tensor,
    strategy: str = "within_set_random",
    ot_cost: str = "l2_squared",
    ot_reg: float = 0.05,
    ot_num_iters: int = 200,
    ot_sampling: str = "row_multinomial",
    return_coupling: bool = False,
):
    """
    Pair control cells within each set.

    :param control_emb: Control tensor shaped [B, S, G].
    :param pert_emb: Perturbed tensor shaped [B, S, G].
    :param strategy: Pairing strategy name.
    :param ot_cost: OT cost type for the Sinkhorn strategy.
    :param ot_reg: Entropic regularization strength for Sinkhorn OT.
    :param ot_num_iters: Maximum Sinkhorn iterations.
    :param ot_sampling: Rule used to sample one control partner from the OT coupling.
    :param return_coupling: Whether to return the full OT coupling for debugging.
    :return: Tuple of paired controls and pairing metadata.
    """
    _validate_inputs(control_emb, pert_emb)

    if strategy == "within_set_random":
        return _pair_random(control_emb)
    if strategy == "ot_sinkhorn":
        return _pair_ot_sinkhorn(
            control_emb,
            pert_emb,
            ot_cost=ot_cost,
            ot_reg=ot_reg,
            ot_num_iters=ot_num_iters,
            ot_sampling=ot_sampling,
            return_coupling=return_coupling,
        )
    raise NotImplementedError(f"Unsupported flow pairing strategy: {strategy}")


__all__ = ["pair_control_within_set"]
