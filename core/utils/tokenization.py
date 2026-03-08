from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings.
        tokenizer: PreTrainedTokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)

    # Tokenize prompts and outputs separately
    prompt_inputs = tokenizer(
        prompt_strs,
        padding=False,
        return_tensors=None,
    )
    output_inputs = tokenizer(
        output_strs,
        padding=False,
        return_tensors=None,
    )

    prompt_input_ids_list = prompt_inputs["input_ids"]
    output_input_ids_list = output_inputs["input_ids"]

    # Compute lengths
    prompt_lens = [len(ids) for ids in prompt_input_ids_list]
    output_lens = [len(ids) for ids in output_input_ids_list]

    # Concatenate prompt and output for each item in the batch
    combined_input_ids_list = []
    for prompt_ids, output_ids in zip(prompt_input_ids_list, output_input_ids_list):
        combined_input_ids_list.append(prompt_ids + output_ids)

    combined_lens = [prompt_lens[i] + output_lens[i] for i in range(batch_size)]
    max_len = max(combined_lens)

    # Pad to max_len
    padded_input_ids = []
    for combined_ids in combined_input_ids_list:
        pad_len = max_len - len(combined_ids)
        padded = combined_ids + [tokenizer.pad_token_id] * pad_len
        padded_input_ids.append(padded)

    # Convert to tensor
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)

    # Labels: shift input_ids by 1 (remove first token, add pad at end)
    labels = torch.cat([input_ids[:, 1:], torch.full((batch_size, 1), tokenizer.pad_token_id, dtype=torch.long)], dim=1)

    # Create response_mask for labels (before slicing)
    # When we shift input_ids to create labels, we remove the first token.
    # So if prompt_len = 4, the output starts at index 4 in input_ids,
    # but at index 3 in labels (because we dropped input_ids[0]).
    response_masks = []
    for i in range(batch_size):
        prompt_len = prompt_lens[i]
        output_len = output_lens[i]
        combined_len = combined_lens[i]

        # For labels: response starts at prompt_len - 1 (because we removed input_ids[0])
        # and ends at combined_len - 1 (because we removed input_ids[0])
        mask = [False] * (max_len)  # Labels has same shape as input_ids before slicing
        response_start = prompt_len - 1
        response_end = combined_len - 1
        for j in range(response_start, response_end):
            if 0 <= j < max_len:
                mask[j] = True
        response_masks.append(mask)

    response_mask = torch.tensor(response_masks, dtype=torch.bool)

    # Slice off the final token from all tensors
    input_ids = input_ids[:, :-1]
    labels = labels[:, :-1]
    response_mask = response_mask[:, :-1]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
