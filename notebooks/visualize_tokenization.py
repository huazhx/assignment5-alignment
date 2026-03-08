import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Tokenization Utility Visualization

    This notebook walks through `core/utils/tokenization.py`, which implements
    `tokenize_prompt_and_output` — the function that prepares prompt+response
    pairs for language model training with **selective loss masking**.

    ## Why do we need this?

    In RLHF / alignment training, we only want to compute loss on the
    **response tokens**, not the prompt or padding. This function:

    1. Tokenizes prompts and outputs **separately**
    2. Concatenates them and pads to uniform length
    3. Creates shifted `labels` (standard causal LM target)
    4. Builds a `response_mask` that marks only response token positions in `labels`
    """)
    return


@app.cell
def _():
    import pathlib
    import sys
    import os

    project_root = pathlib.Path("/home/xuzhenhua/git/assignment5-alignment")
    sys.path.insert(0, str(project_root))
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: Load a tokenizer

    We use the Qwen2.5-Math-1.5B tokenizer as an example.
    """)
    return


@app.cell
def _():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/home/xuzhenhua/models/Qwen2.5-Math-1.5B")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    return (tokenizer,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: Prepare example inputs

    We create a small batch of 2 prompt-output pairs with **different lengths**
    to demonstrate padding behavior.
    """)
    return


@app.cell
def _():
    prompt_strs = [
        "What is 2+2?",
        "Solve: 3*5",
    ]
    output_strs = [
        "The answer is 4.",
        "15",
    ]
    return output_strs, prompt_strs


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: Run `tokenize_prompt_and_output`

    This is the core function. Let's call it and inspect every output.
    """)
    return


@app.cell
def _(output_strs, prompt_strs, tokenizer):
    from core.utils.tokenization import tokenize_prompt_and_output

    result = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

    input_ids = result["input_ids"]
    labels = result["labels"]
    response_mask = result["response_mask"]

    print(f"input_ids shape:     {input_ids.shape}")
    print(f"labels shape:        {labels.shape}")
    print(f"response_mask shape: {response_mask.shape}")
    return input_ids, labels, response_mask


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: Understand the internal mechanics

    ### 4.1 Separate tokenization

    Prompts and outputs are tokenized independently so we know exactly where
    the prompt ends and the response begins.
    """)
    return


@app.cell
def _(output_strs, prompt_strs, tokenizer):
    prompt_tokens = tokenizer(prompt_strs, padding=False, return_tensors=None)["input_ids"]
    output_tokens = tokenizer(output_strs, padding=False, return_tensors=None)["input_ids"]

    for _i in range(len(prompt_strs)):
        p_ids = prompt_tokens[_i]
        o_ids = output_tokens[_i]
        print(f"--- Example {_i} ---")
        print(f"  Prompt tokens ({len(p_ids)}): {p_ids}")
        print(f"  Prompt decoded:  {tokenizer.decode(p_ids)!r}")
        print(f"  Output tokens ({len(o_ids)}): {o_ids}")
        print(f"  Output decoded:  {tokenizer.decode(o_ids)!r}")
        print()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2 Concatenation & Padding

    Each prompt+output is concatenated, then **right-padded** to the max
    combined length across the batch.

    ```
    Example 0 (longer):  [prompt_ids... | output_ids...]
    Example 1 (shorter): [prompt_ids... | output_ids... | PAD PAD PAD ...]
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 Shifted labels (causal LM)

    For causal language modeling, the label at position `t` is the token at
    position `t+1` in the original sequence. This is done by shifting
    `input_ids` left by 1:

    ```
    input_ids: [A, B, C, D, E]
    labels:    [B, C, D, E, PAD]
    ```

    The final token is sliced off from both, giving us:

    ```
    input_ids: [A, B, C, D]    (predict next token at each position)
    labels:    [B, C, D, E]    (the ground truth next tokens)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.4 Response mask

    The `response_mask` is `True` only for positions in `labels` that correspond
    to response tokens. The key insight:

    - In `input_ids`, the response starts at index `prompt_len`
    - After the left-shift to create `labels`, that becomes index `prompt_len - 1`
    - The mask spans `[prompt_len - 1, combined_len - 1)` in `labels`

    This lets us compute loss **only on response tokens**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: Token-level visualization

    Below we decode each position and show the alignment between `input_ids`,
    `labels`, and `response_mask` for each example in the batch.
    """)
    return


@app.cell
def _(input_ids, labels, response_mask, tokenizer):
    for _i in range(input_ids.shape[0]):
        print(f"{'='*60}")
        print(f"Example {_i}")
        print(f"{'='*60}")
        print(f"{'Pos':>4}  {'input_ids':>10}  {'input_tok':>14}  {'labels':>10}  {'label_tok':>14}  {'mask':>5}")
        print(f"{'-'*4}  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*14}  {'-'*5}")
        for _j in range(input_ids.shape[1]):
            iid = input_ids[_i, _j].item()
            lid = labels[_i, _j].item()
            m = response_mask[_i, _j].item()
            itok = tokenizer.decode([iid]).replace('\n', '\\n')
            ltok = tokenizer.decode([lid]).replace('\n', '\\n')
            marker = " <<" if m else ""
            print(f"{_j:>4}  {iid:>10}  {itok:>14}  {lid:>10}  {ltok:>14}  {str(m):>5}{marker}")
        print()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 6: Visual heatmap of response_mask

    Green = response token (loss computed), Gray = prompt/padding (loss ignored).
    """)
    return


@app.cell
def _(labels, response_mask, tokenizer):
    import matplotlib.pyplot as plt
    import numpy as np

    batch_size, seq_len = response_mask.shape
    mask_np = response_mask.numpy().astype(float)

    fig, axes = plt.subplots(batch_size, 1, figsize=(max(14, seq_len * 0.8), 1.5 * batch_size + 1))
    if batch_size == 1:
        axes = [axes]

    for _i, ax in enumerate(axes):
        ax.imshow(mask_np[_i:_i+1], aspect='auto', cmap='Greens', vmin=0, vmax=1)
        ax.set_yticks([0])
        ax.set_yticklabels([f"Ex {_i}"])
        ax.set_xticks(range(seq_len))

        # Show token text on x-axis
        tok_labels = []
        for _j in range(seq_len):
            t = tokenizer.decode([labels[_i, _j].item()]).strip()
            if labels[_i, _j].item() == tokenizer.pad_token_id:
                t = "[PAD]"
            tok_labels.append(t[:6])
        ax.set_xticklabels(tok_labels, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel("Label tokens")

    fig.suptitle("Response Mask Heatmap (Green = loss computed)", fontsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Output | Shape | Description |
    |--------|-------|-------------|
    | `input_ids` | `(B, L-1)` | Tokenized prompt+output, final token removed |
    | `labels` | `(B, L-1)` | Shifted input_ids (next-token targets) |
    | `response_mask` | `(B, L-1)` | Boolean mask, `True` only on response positions in `labels` |

    Where `L = max(prompt_len + output_len)` across the batch.

    ### Key design choices
    - **Separate tokenization** ensures exact prompt/response boundary detection
    - **Right-padding** keeps prompt tokens left-aligned
    - **Shift-by-1** follows the standard causal LM convention
    - **Response mask** enables selective loss — only backprop through response tokens
    """)
    return


if __name__ == "__main__":
    app.run()
