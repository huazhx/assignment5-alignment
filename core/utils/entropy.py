from __future__ import annotations

import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the
    vocabulary dimension).

    Args:
        logits: torch.Tensor
            Tensor of shape (batch_size, sequence_length, vocab_size) containing
            unnormalized logits.

    Returns:
        torch.Tensor
            Shape (batch_size, sequence_length). The entropy for each
            next-token prediction.
    """
    # log_softmax is numerically stable (uses logsumexp internally)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # H(p) = -sum(p * log(p)), ignoring terms where p == 0
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
