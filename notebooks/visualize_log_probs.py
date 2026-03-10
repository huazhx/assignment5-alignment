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
    # `get_response_log_probs` — Line-by-Line Walkthrough

    This notebook steps through `core/utils/log_probs.py` one line at a time,
    using a tiny model so every intermediate value is inspectable.

    ## What does this function do?

    Given a causal language model, `input_ids`, and `labels` (shifted input_ids),
    it returns:

    1. **`log_probs`** — the log-probability the model assigns to each label token:
       $\log p_\theta(x_t \mid x_{<t})$
    2. **`token_entropy`** (optional) — how uncertain the model is at each position:
       $H = -\sum_v p_v \log p_v$

    This is the core scoring primitive used in SFT loss, RLHF policy gradients,
    and KL divergence computation.
    """)
    return


@app.cell
def _():
    import pathlib
    import sys
    import os

    _project_root = pathlib.Path("/home/xuzhenhua/git/assignment5-alignment")
    sys.path.insert(0, str(_project_root))
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 0: Load a small model and tokenizer

    We load Qwen2.5-Math-1.5B and create a tiny batch of 2 examples
    to keep the output readable.
    """)
    return


@app.cell
def _():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _model_path = "/home/xuzhenhua/models/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(_model_path, local_files_only=True)
    model.eval()

    print(f"Model: {model.config._name_or_path}")
    print(f"Vocab size: {model.config.vocab_size}")
    return model, tokenizer


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 0b: Prepare `input_ids` and `labels`

    We use `tokenize_prompt_and_output` to get properly formatted inputs.
    Recall that `labels` are `input_ids` shifted left by 1 (the standard
    causal LM target setup).

    ```
    input_ids: [A, B, C, D]    position t   -> model predicts next token
    labels:    [B, C, D, E]    position t   -> ground truth next token
    ```
    """)
    return


@app.cell
def _(tokenizer):
    from core.utils.tokenization import tokenize_prompt_and_output

    prompt_strs = ["What is 2+2?", "Solve: 3*5"]
    output_strs = ["The answer is 4.", "15  whi dsf ds dfg fdg "]

    tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    print(f"input_ids shape: {input_ids.shape}")
    print(f"labels shape:    {labels.shape}")
    print(f"input_ids[0, :8]: {input_ids[0, :].tolist()}")
    print(f"labels[0, :8]:    {labels[0, :8].tolist()}")
    print()
    print("Notice labels[t] == input_ids[t+1] (shifted by 1):")
    print(f"  input_ids[0, 1:6]: {input_ids[0, 1:6].tolist()}")
    print(f"  labels[0, 0:5]:    {labels[0, 0:5].tolist()}")
    return input_ids, labels, response_mask


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: `logits = model(input_ids).logits`

    Forward pass through the model. For each position $t$, the model produces
    a distribution over the entire vocabulary — these are the raw **logits**
    (unnormalized scores).

    Output shape: `(batch_size, sequence_length, vocab_size)`
    """)
    return


@app.cell
def _(input_ids, model):
    import torch

    with torch.no_grad():
        logits = model(input_ids).logits

    print(f"logits shape: {logits.shape}")
    print(f"  batch_size:      {logits.shape[0]}")
    print(f"  sequence_length: {logits.shape[1]}")
    print(f"  vocab_size:      {logits.shape[2]}")
    print()
    print(f"logits[0, 0, :10] (first 10 vocab scores at pos 0):")
    print(f"  {logits[0, 0, :10].tolist()}")
    return logits, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: `log_probs_all = torch.log_softmax(logits, dim=-1)`

    Convert logits to **log-probabilities** over the vocabulary dimension.

    $$\log p(v \mid x_{<t}) = \text{logit}_v - \log\!\left(\sum_{v'} e^{\text{logit}_{v'}}\right)$$

    This uses `logsumexp` internally for numerical stability.

    Output shape: same as logits — `(batch_size, seq_len, vocab_size)`
    """)
    return


@app.cell
def _(logits, torch):
    log_probs_all = torch.log_softmax(logits, dim=-1)

    print(f"log_probs_all shape: {log_probs_all.shape}")
    print()
    print("Sanity check — exp(log_probs) sums to 1 along vocab dim:")
    _sums = log_probs_all[0, 0, :].exp().sum()
    print(f"  sum at pos 0: {_sums.item():.6f}")
    print()
    print("All log-probs are <= 0 (probabilities are <= 1):")
    print(f"  max log_prob: {log_probs_all.max().item():.4f}")
    return (log_probs_all,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: `log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)`

    This is the key line. We want: for each position $t$, what log-probability
    did the model assign to the **actual next token** (i.e., `labels[t]`)?

    ### Breaking it down:

    | Sub-step | Operation | Shape |
    |----------|-----------|-------|
    | `labels` | The target token IDs | `(B, L)` |
    | `.unsqueeze(-1)` | Add vocab dim for gather | `(B, L, 1)` |
    | `log_probs_all.gather(dim=-1, ...)` | Pick one log-prob per position | `(B, L, 1)` |
    | `.squeeze(-1)` | Remove the extra dim | `(B, L)` |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3a: `labels.unsqueeze(-1)`

    `gather` needs the index to have the same number of dims as the source tensor.
    `labels` is `(B, L)`, `log_probs_all` is `(B, L, V)`, so we add a trailing dim.
    """)
    return


@app.cell
def _(labels):
    gather_index = labels.unsqueeze(-1)

    print(f"labels shape:       {labels.shape}")
    print(f"gather_index shape: {gather_index.shape}")
    print()
    print(f"labels[0, :5]:        {labels[0, :5].tolist()}")
    print(f"gather_index[0, :5]:  {gather_index[0, :5].tolist()}")
    return (gather_index,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 3b: `.gather(dim=-1, index=...)` then `.squeeze(-1)`

    `gather` selects one element along the vocab dimension at each position.
    Think of it as: for position $t$, reach into the vocab-size vector and
    pull out the log-prob at index `labels[t]`.

    ```
    log_probs_all[b, t, :]  =  [-8.2, -3.1, -12.0, ..., -0.5, ...]
                                                          ^^^^^
                                             labels[b, t] = this index
    ```
    """)
    return


@app.cell
def _(gather_index, log_probs_all):
    log_probs = log_probs_all.gather(dim=-1, index=gather_index).squeeze(-1)

    print(f"log_probs shape: {log_probs.shape}")
    print()
    print("Per-token log-probs for example 0:")
    for _t in range(min(8, log_probs.shape[1])):
        _lp = log_probs[0, _t].item()
        print(f"  pos {_t}: log p = {_lp:.4f}  (p = {_lp:.4f} -> {2.718281828**_lp:.6f})")
    return (log_probs,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Manual verification

    Let's verify `gather` does what we expect by manually indexing for one position.
    """)
    return


@app.cell
def _(labels, log_probs, log_probs_all):
    # Manually check position 0 of example 0
    _label_token = labels[0, 0].item()
    _manual = log_probs_all[0, 0, _label_token].item()
    _gathered = log_probs[0, 0].item()

    print(f"Label token at [0, 0]: {_label_token}")
    print(f"Manual index:  log_probs_all[0, 0, {_label_token}] = {_manual:.6f}")
    print(f"gather result: log_probs[0, 0]                    = {_gathered:.6f}")
    print(f"Match: {abs(_manual - _gathered) < 1e-6}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: (Optional) `compute_entropy(logits)`

    When `return_token_entropy=True`, we also compute entropy at each position
    using the same logits. This reuses the `compute_entropy` function from
    `core/utils/entropy.py`:

    $$H_t = -\sum_{v} p(v \mid x_{<t}) \cdot \log p(v \mid x_{<t})$$

    High entropy = model is uncertain. Low entropy = model is confident.
    """)
    return


@app.cell
def _(logits):
    from core.utils.entropy import compute_entropy

    token_entropy = compute_entropy(logits)

    print(f"token_entropy shape: {token_entropy.shape}")
    print()
    print("Entropy for example 0:")
    for _t in range(min(8, token_entropy.shape[1])):
        _e = token_entropy[0, _t].item()
        print(f"  pos {_t}: H = {_e:.4f} nats")
    return (token_entropy,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: Verify against the actual function

    Call `get_response_log_probs` and confirm all outputs match.
    """)
    return


@app.cell
def _(input_ids, labels, log_probs, model, token_entropy, torch):
    from core.utils.log_probs import get_response_log_probs

    result = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=True,
    )

    _lp_match = torch.allclose(log_probs, result["log_probs"])
    _ent_match = torch.allclose(token_entropy, result["token_entropy"])
    print(f"log_probs match:     {_lp_match}")
    print(f"token_entropy match: {_ent_match}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 6: Visualize log-probs and entropy

    Two plots side by side:
    - **Left**: per-token log-probabilities (how likely was the actual next token?)
    - **Right**: per-token entropy (how uncertain was the model?)

    The `response_mask` region is highlighted — that's where loss is computed during training.
    """)
    return


@app.cell
def _(log_probs, response_mask, token_entropy):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    _seq_len = log_probs.shape[1]
    _positions = np.arange(_seq_len)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 4))

    # --- Left: log-probs ---
    _lp = log_probs[0].detach().numpy()
    _mask = response_mask[0].numpy()
    _colors_lp = ['#2ecc71' if _mask[_j] else '#bdc3c7' for _j in range(_seq_len)]
    _ax1.bar(_positions, _lp, color=_colors_lp, edgecolor='black', linewidth=0.3)
    _ax1.set_ylabel('log p(label | context)')
    _ax1.set_title('Per-token Log-Probabilities')
    _ax1.set_xlabel('Position')

    # --- Right: entropy ---
    _ent = token_entropy[0].detach().numpy()
    _colors_ent = ['#e74c3c' if _mask[_j] else '#bdc3c7' for _j in range(_seq_len)]
    _ax2.bar(_positions, _ent, color=_colors_ent, edgecolor='black', linewidth=0.3)
    _ax2.set_ylabel('Entropy (nats)')
    _ax2.set_title('Per-token Entropy')
    _ax2.set_xlabel('Position')

    # Shared legend
    _green = mpatches.Patch(color='#2ecc71', label='Response token')
    _gray = mpatches.Patch(color='#bdc3c7', label='Prompt / padding')
    _ax1.legend(handles=[_green, _gray], fontsize=8)

    _red = mpatches.Patch(color='#e74c3c', label='Response token')
    _ax2.legend(handles=[_red, _gray], fontsize=8)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 7: Token-level detail table

    Decode each position and show its log-prob, probability, and entropy.
    """)
    return


@app.cell
def _(labels, log_probs, response_mask, token_entropy, tokenizer):
    import math

    print(f"{'Pos':>4}  {'Label':>12}  {'log_prob':>10}  {'prob':>10}  {'entropy':>10}  {'mask':>5}")
    print(f"{'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*5}")
    for _t in range(log_probs.shape[1]):
        _tok = tokenizer.decode([labels[0, _t].item()]).replace('\n', '\\n').strip()
        _lp = log_probs[0, _t].item()
        _p = math.exp(_lp)
        _e = token_entropy[0, _t].item()
        _m = response_mask[0, _t].item()
        _marker = " <<" if _m else ""
        print(f"{_t:>4}  {_tok:>12}  {_lp:>10.4f}  {_p:>10.6f}  {_e:>10.4f}  {str(_m):>5}{_marker}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    ```python
    def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
        logits = model(input_ids).logits                                          # 1
        log_probs_all = torch.log_softmax(logits, dim=-1)                         # 2
        log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1))      # 3
                                 .squeeze(-1)
        result = {"log_probs": log_probs}
        if return_token_entropy:
            result["token_entropy"] = compute_entropy(logits)                     # 4
        return result
    ```

    | Line | What it does | Shape change |
    |------|-------------|-------------|
    | 1. `model(input_ids).logits` | Forward pass, get raw scores | `(B, L)` -> `(B, L, V)` |
    | 2. `log_softmax` | Normalize to log-probabilities | `(B, L, V)` -> `(B, L, V)` |
    | 3. `gather` + `squeeze` | Pick log-prob of each label token | `(B, L, V)` -> `(B, L)` |
    | 4. `compute_entropy` | Uncertainty at each position | `(B, L, V)` -> `(B, L)` |
    """)
    return


if __name__ == "__main__":
    app.run()
