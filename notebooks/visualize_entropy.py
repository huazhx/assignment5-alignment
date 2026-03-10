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
    # `compute_entropy` — Line-by-Line Walkthrough

    This notebook steps through `core/utils/entropy.py` one line at a time,
    using a tiny vocabulary so every number is easy to inspect.

    ## What is entropy?

    Entropy measures the **uncertainty** of a probability distribution:

    $$H(p) = -\sum_{i} p_i \log p_i$$

    - **High entropy** = model is unsure (uniform-ish distribution)
    - **Low entropy** = model is confident (one token dominates)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 0: Create toy logits

    We use `batch_size=1`, `seq_len=3`, `vocab_size=4` so every value fits on screen.
    The three positions simulate:

    | Position | Pattern | Expected entropy |
    |----------|---------|-----------------|
    | 0 | One token dominates | Low |
    | 1 | Roughly uniform | High |
    | 2 | Two tokens compete | Medium |
    """)
    return


@app.cell
def _():
    import torch

    logits = torch.tensor([
        # pos 0: confident — token 0 dominates
        [10.0, 0.0, 0.0, 0.0],
        # pos 1: uncertain — roughly uniform
        [1.0, 1.0, 1.0, 1.0],
        # pos 2: two tokens compete
        [5.0, 5.0, 0.0, 0.0],
    ]).unsqueeze(0)  # shape: (1, 3, 4)

    print(f"logits shape: {logits.shape}")
    print(f"logits:\n{logits}")
    return logits, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: `log_probs = torch.log_softmax(logits, dim=-1)`

    This converts raw logits to **log-probabilities** along the vocabulary dimension.

    Under the hood it computes:

    $$\log p_i = x_i - \log\!\left(\sum_j e^{x_j}\right)$$

    which is equivalent to `logits - logsumexp(logits)`. This avoids computing
    `softmax` first (which can overflow for large logits) and then taking `log`
    (which can produce `-inf` for tiny probabilities).
    """)
    return


@app.cell
def _(logits, torch):
    log_probs = torch.log_softmax(logits, dim=-1)

    print(f"log_probs shape: {log_probs.shape}")
    print(f"log_probs:\n{log_probs}")
    print()
    print("Sanity check — exp(log_probs) should sum to 1 along vocab dim:")
    print(log_probs.exp().sum(dim=-1))
    return (log_probs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: `probs = log_probs.exp()`

    Convert back to actual probabilities so we can compute $p_i \cdot \log p_i$.
    """)
    return


@app.cell
def _(log_probs):
    probs = log_probs.exp()

    print(f"probs shape: {probs.shape}")
    print(f"probs:\n{probs}")
    print()
    print("Sum per position (should be ~1.0):", probs.sum(dim=-1))
    return (probs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: `-(probs * log_probs).sum(dim=-1)`

    This is the entropy formula:

    $$H = -\sum_{i} p_i \log p_i$$

    Let's break it into sub-steps.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3a: Element-wise `probs * log_probs`

    Each element is $p_i \cdot \log p_i$. Note these are all **negative** (or zero),
    because $0 \le p_i \le 1$ means $\log p_i \le 0$.
    """)
    return


@app.cell
def _(log_probs, probs):
    p_logp = probs * log_probs

    print(f"probs * log_probs:\n{p_logp}")
    return (p_logp,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 3b: `.sum(dim=-1)` then negate

    Sum across the vocab dimension, then flip the sign to get positive entropy values.
    """)
    return


@app.cell
def _(p_logp):
    entropy = -(p_logp.sum(dim=-1))

    print(f"entropy shape: {entropy.shape}")
    print(f"entropy: {entropy}")
    return (entropy,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: Interpret the results

    For a vocabulary of size $V=4$, the **maximum possible entropy** is $\log(4) \approx 1.386$ (uniform distribution).

    | Position | Pattern | Entropy | Interpretation |
    |----------|---------|---------|---------------|
    | 0 | One token dominates | ~0.0 | Very confident |
    | 1 | Uniform | ~1.386 | Maximum uncertainty |
    | 2 | Two tokens compete | ~0.69 | Moderate uncertainty |
    """)
    return


@app.cell
def _(entropy):
    import math

    max_entropy = math.log(4)
    print(f"Max possible entropy (log 4): {max_entropy:.4f}")
    print()
    _labels = ["confident", "uniform", "two-way"]
    for _i in range(entropy.shape[1]):
        _val = entropy[0, _i].item()
        _pct = _val / max_entropy * 100
        print(f"  pos {_i} ({_labels[_i]:>9}): {_val:.4f}  ({_pct:.1f}% of max)")
    return (math,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: Verify against the full function

    Call the actual `compute_entropy` from `core/utils` and confirm the outputs match.
    """)
    return


@app.cell
def _():
    import pathlib
    import sys

    _project_root = pathlib.Path("/home/xuzhenhua/git/assignment5-alignment")
    sys.path.insert(0, str(_project_root))
    return


@app.cell
def _(entropy, logits, torch):
    from core.utils.entropy import compute_entropy

    result = compute_entropy(logits)

    print(f"Manual:         {entropy}")
    print(f"compute_entropy: {result}")
    print(f"Match: {torch.allclose(entropy, result)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 6: Bar chart

    Visualize the entropy at each position.
    """)
    return


@app.cell
def _(entropy, math):
    # import math
    import matplotlib.pyplot as plt

    _max_ent = math.log(4)
    _vals = [entropy[0, _i].item() for _i in range(entropy.shape[1])]
    _labels = ["pos 0\n(confident)", "pos 1\n(uniform)", "pos 2\n(two-way)"]
    _colors = ["#2ecc71", "#e74c3c", "#f39c12"]

    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _bars = _ax.bar(_labels, _vals, color=_colors, edgecolor="black", linewidth=0.5)
    _ax.axhline(_max_ent, color="gray", linestyle="--", linewidth=1, label=f"max entropy = ln(4) = {_max_ent:.3f}")
    _ax.set_ylabel("Entropy (nats)")
    _ax.set_title("Per-position entropy")
    _ax.legend()
    _ax.set_ylim(0, _max_ent * 1.15)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why numerical stability matters

    A naive implementation would be:

    ```python
    probs = torch.softmax(logits, dim=-1)        # can overflow for large logits
    entropy = -(probs * torch.log(probs)).sum(-1) # log(0) = -inf
    ```

    Problems:
    1. `exp(1000)` overflows to `inf` in `softmax`
    2. `log(0)` produces `-inf`, and `0 * -inf = nan`

    The stable version avoids both by using `log_softmax` (which internally
    subtracts the max before exp) and computing `probs` from `log_probs`
    (so near-zero probabilities stay finite in log-space).
    """)
    return


@app.cell
def _(torch):
    _big_logits = torch.tensor([[[1000.0, 0.0, 0.0]]])

    # Naive — produces nan
    _naive_probs = torch.softmax(_big_logits, dim=-1)
    _naive_entropy = -(_naive_probs * torch.log(_naive_probs)).sum(dim=-1)
    print(f"Naive  (logits=[1000,0,0]): {_naive_entropy.item()}")

    # Stable — correct result
    _log_probs = torch.log_softmax(_big_logits, dim=-1)
    _probs = _log_probs.exp()
    _stable_entropy = -(_probs * _log_probs).sum(dim=-1)
    print(f"Stable (logits=[1000,0,0]): {_stable_entropy.item()}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    ```python
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)   # stable log-probs
        probs = log_probs.exp()                          # recover probs
        entropy = -(probs * log_probs).sum(dim=-1)       # H(p)
        return entropy
    ```

    | Line | What it does | Why |
    |------|-------------|-----|
    | `log_softmax` | logits -> log-probabilities | Uses logsumexp, avoids overflow |
    | `.exp()` | log-probs -> probabilities | Derived from stable log-probs |
    | `-(p * log_p).sum(-1)` | Entropy formula | Sums $-p_i \log p_i$ over vocab |
    """)
    return


if __name__ == "__main__":
    app.run()
