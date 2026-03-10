from __future__ import annotations

import torch
from transformers import PreTrainedModel

from .entropy import compute_entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal language model,
    and optionally the per-token entropy.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring.
        input_ids: torch.Tensor
            Shape (batch_size, sequence_length), concatenated prompt + response tokens.
        labels: torch.Tensor
            Shape (batch_size, sequence_length), shifted input_ids as produced
            by tokenize_prompt_and_output.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": shape (batch_size, sequence_length), log p(x_t | x_{<t}).
            "token_entropy": (only if return_token_entropy=True)
                shape (batch_size, sequence_length), per-token entropy.
    """
    # (batch_size, sequence_length, vocab_size)
    logits = model(input_ids).logits

    # log p(x_t | x_{<t}) for each position — gather the log-prob of the label token
    log_probs_all = torch.log_softmax(logits, dim=-1)
    # labels shape: (batch_size, seq_len) -> gather index: (batch_size, seq_len, 1)
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result
