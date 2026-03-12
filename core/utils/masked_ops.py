from __future__ import annotations

import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, considering only
    those elements where mask == 1.

    Args:
        tensor: torch.Tensor
            The tensor to sum and normalize.
        mask: torch.Tensor
            Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float
            The constant to divide by for normalization.
        dim: int | None
            The dimension to sum along before normalization.
            If None, sum over all dimensions.

    Returns:
        torch.Tensor
            The normalized sum, where masked elements (mask == 0) don't
            contribute to the sum.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: The tensor to compute the mean of.
        mask: Same shape as tensor; only elements with mask == 1 are included.
        dim: The dimension to compute the mean along.
            If None, mean over all non-masked elements.

    Returns:
        The mean of the masked elements.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum().clamp(min=1)
    return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)
