from __future__ import annotations

import torch

from core.utils.masked_ops import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch for SFT.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities
            from the SFT policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens,
            0 for prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: The constant by which to divide the sum.

    Returns:
        loss: scalar tensor, the microbatch loss scaled for gradient accumulation.
        metadata: dict with metadata from the underlying loss call.
    """
    batch_size = policy_log_probs.shape[0]

    # Compute negative log-likelihood loss (cross-entropy for SFT)
    # Using masked_normalize to sum only over response tokens
    loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask.float(),
        normalize_constant=normalize_constant * gradient_accumulation_steps * batch_size,
    )

    # Backward pass
    loss.backward()

    # Return loss (detached) for logging
    metadata = {
        "loss": loss.detach(),
    }

    return loss.detach(), metadata
