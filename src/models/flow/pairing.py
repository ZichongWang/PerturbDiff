"""Pairing utilities for flow-matching training."""

import torch


def _sample_derangement(size: int, device) -> torch.Tensor:
    """Sample a permutation with no fixed points."""
    assert size > 1, "Flow pairing requires use_cell_set > 1."
    base = torch.arange(size, device=device)
    perm = base
    while bool((perm == base).any().item()):
        perm = torch.randperm(size, device=device)
    return perm


def pair_control_within_set(control_emb: torch.Tensor, strategy: str = "within_set_random"):
    """
    Randomly pair control cells within each set.

    :param control_emb: Control tensor shaped [B, S, G].
    :param strategy: Pairing strategy name.
    :return: Tuple of paired controls and permutation indices.
    """
    if strategy != "within_set_random":
        raise NotImplementedError(f"Unsupported flow pairing strategy: {strategy}")
    assert control_emb.ndim == 3, "Flow pairing expects control embeddings shaped [B, S, G]."

    batch_size, set_size, gene_dim = control_emb.shape
    _ = gene_dim
    assert set_size > 1, "Flow pairing requires use_cell_set > 1."

    perms = torch.stack([_sample_derangement(set_size, control_emb.device) for _ in range(batch_size)], dim=0)
    paired = control_emb.gather(1, perms.unsqueeze(-1).expand(-1, -1, control_emb.shape[-1]))
    return paired, perms


__all__ = ["pair_control_within_set"]
