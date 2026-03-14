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
    # `sft_microbatch_train_step` — Line-by-Line Walkthrough

    This notebook steps through `core/trainer/sft.py` using small tensors
    so every intermediate value is visible.

    ## What does this function do?

    Executes a single **microbatch** training step for Supervised Fine-Tuning (SFT):

    1. **Computes cross-entropy loss** from per-token log-probabilities
    2. **Masks** to only include response tokens (not prompt/padding)
    3. **Scales** by gradient accumulation steps and batch size
    4. **Backpropagates** to compute gradients
    5. **Returns** loss and metadata for logging

    $$\text{loss} = -\frac{\sum_{i \in \text{response}} \log p_i}{\text{normalize\_constant} \times \text{grad\_accum} \times \text{batch\_size}}$$

    ## Why do we need it?

    In distributed training, we split a large batch into **microbatches** to fit in GPU memory.
    After processing all microbatches, we average the gradients before the optimizer step.
    This function handles one microbatch's forward and backward pass.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 0: Create toy data

    A `(2, 4)` tensor representing per-token **log-probabilities** for a batch of 2 sequences,
    each with 4 positions. Positive values = confident predictions, negative = uncertain.

    ```
    policy_log_probs:  [[ 0.5, -0.2,  1.0,  0.3],   <- higher is better (log prob)
                         [-0.1,  0.8, -0.5,  0.2]]

    response_mask:      [[ 0,    0,    1,    1  ],   <- response starts at pos 2
                         [ 0,    1,    1,    0  ]]   <- response at pos 1,2
    ```

    We'll use `gradient_accumulation_steps = 2` and `normalize_constant = 1.0`.
    """)
    return


@app.cell
def _():
    import torch

    # Per-token log probabilities from the model
    # Positive = confident prediction, negative = uncertain
    policy_log_probs = torch.tensor([
        [ 0.5, -0.2,  1.0,  0.3],
        [-0.1,  0.8, -0.5,  0.2],
    ], requires_grad=True)

    # 1 = response token, 0 = prompt/padding
    response_mask = torch.tensor([
        [0, 0, 1, 1],
        [0, 1, 1, 0],
    ], dtype=torch.float)

    gradient_accumulation_steps = 2
    normalize_constant = 1.0

    print(f"policy_log_probs shape: {policy_log_probs.shape}")
    print(f"policy_log_probs:\n{policy_log_probs}")
    print(f"\nresponse_mask:\n{response_mask}")
    print(f"\ngradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"normalize_constant: {normalize_constant}")
    return (
        gradient_accumulation_steps,
        normalize_constant,
        policy_log_probs,
        response_mask,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: `batch_size = policy_log_probs.shape[0]`

    Extract the batch dimension for computing the total scaling factor.
    """)
    return


@app.cell
def _(policy_log_probs):
    batch_size = policy_log_probs.shape[0]

    print(f"policy_log_probs.shape: {policy_log_probs.shape}")
    print(f"batch_size: {batch_size}")
    print("\nThis will be used in the scaling formula.")
    return (batch_size,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: Compute the scaling factor

    The normalization constant is scaled by **three factors**:

    $$\text{effective\_norm} = \text{normalize\_constant} \times \text{grad\_accum} \times \text{batch\_size}$$

    Why all three?

    | Factor | Purpose |
    |--------|---------|
    | `normalize_constant` | User-provided scaling (e.g., Dr. GRPO uses fixed value) |
    | `gradient_accumulation_steps` | Each microbatch contributes 1/N of the total gradient |
    | `batch_size` | Normalize across examples in the batch |

    For our toy example:
    $$\text{effective\_norm} = 1.0 \times 2 \times 2 = 4.0$$
    """)
    return


@app.cell
def _(batch_size, gradient_accumulation_steps, normalize_constant):
    effective_norm = normalize_constant * gradient_accumulation_steps * batch_size

    print(f"normalize_constant: {normalize_constant}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"batch_size: {batch_size}")
    print(f"\neffective_norm = {normalize_constant} × {gradient_accumulation_steps} × {batch_size} = {effective_norm}")
    return (effective_norm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: `masked = policy_log_probs * response_mask.float()`

    Zero out the log-probabilities at prompt/padding positions.
    Only response tokens contribute to the loss.
    """)
    return


@app.cell
def _(policy_log_probs, response_mask):
    masked = policy_log_probs * response_mask

    print("policy_log_probs:")
    print(policy_log_probs)
    print("\nresponse_mask:")
    print(response_mask)
    print("\nmasked = policy_log_probs * response_mask:")
    print(masked)
    print("\nOnly response positions have non-zero values")
    return (masked,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: `loss = -masked.sum() / effective_norm`

    Compute the **negative log-likelihood** (cross-entropy) loss:

    1. **Sum** all masked log-probabilities
    2. **Negate** (minimizing loss = maximizing log-prob)
    3. **Divide** by the effective normalization constant

    $$\text{loss} = -\frac{\sum \text{masked\_log\_probs}}{\text{effective\_norm}}$$
    """)
    return


@app.cell
def _(effective_norm, masked):
    masked_sum = masked.sum()
    loss = -masked_sum / effective_norm

    print(f"masked.sum(): {masked_sum.item():.4f}")
    print(f"  = 1.0 + 0.3 + 0.8 + (-0.5)")
    print(f"  = {(1.0 + 0.3 + 0.8 - 0.5):.4f}")
    print(f"\nloss = -masked.sum() / {effective_norm}")
    print(f"     = -{masked_sum.item():.4f} / {effective_norm}")
    print(f"     = {loss.item():.4f}")
    return loss, masked_sum


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: `loss.backward()`

    Compute gradients via backpropagation. The gradient w.r.t. each input is:

    $$\frac{\partial \text{loss}}{\partial \text{log\_prob}_i} = -\frac{\text{mask}_i}{\text{effective\_norm}}$$

    For response tokens: $\frac{-1}{4} = -0.25$
    For prompt/padding: $0$
    """)
    return


@app.cell
def _(effective_norm, loss, policy_log_probs):
    # Before backward, gradients don't exist
    print(f"Gradients before backward: {policy_log_probs.grad}")

    # Perform backward pass
    loss.backward()

    # After backward, check gradients
    print(f"\nGradients after backward:")
    print(policy_log_probs.grad)

    print(f"\nExpected gradient at response positions: -1 / {effective_norm} = {-1/effective_norm}")
    print(f"Expected gradient at masked positions: 0")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 6: Create metadata and return

    The function returns:
    1. **Loss** (detached from graph) for logging
    2. **Metadata dict** with additional info for monitoring

    We use `.detach()` to remove the tensor from the computation graph,
    preventing unnecessary gradient tracking in the logging code.
    """)
    return


@app.cell
def _(loss):
    metadata = {
        "loss": loss.detach(),
    }

    print(f"metadata['loss']: {metadata['loss'].item():.4f}")
    print(f"loss.detach(): {loss.detach().item():.4f}")
    print(f"\nOriginal loss still has grad_fn: {loss.grad_fn}")
    print(f"Detached loss has no grad_fn: {loss.detach().grad_fn}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 7: Verify against the actual function

    Import `sft_microbatch_train_step` from `core/trainer` and confirm
    the loss and gradients match our manual computation.
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
def _(
    effective_norm,
    gradient_accumulation_steps,
    masked_sum,
    normalize_constant,
    policy_log_probs,
    response_mask,
    torch,
):
    # import torch
    from core.trainer.sft import sft_microbatch_train_step as smts

    # Reset gradients for clean test
    policy_log_probs.grad = None

    # Call the actual function
    actual_loss, actual_metadata = smts(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )

    # Compute expected values
    expected_loss = -masked_sum / effective_norm

    print(f"Expected loss: {expected_loss.item():.4f}")
    print(f"Actual loss:   {actual_loss.item():.4f}")
    print(f"Match: {torch.allclose(actual_loss, expected_loss)}")

    print(f"\nExpected gradients (response positions): {-1/effective_norm:.4f}")
    print(f"Actual gradients:\n{policy_log_probs.grad}")

    # Verify gradients
    expected_grad = torch.tensor([
        [ 0.0,   0.0,  -0.25, -0.25],
        [ 0.0,  -0.25, -0.25,  0.0],
    ])
    print(f"\nGradient match: {torch.allclose(policy_log_probs.grad, expected_grad)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 8: Visualization

    Heatmaps showing:
    1. **Original log-probabilities** with response mask overlay
    2. **Masked values** (prompt/padding zeroed out)
    3. **Gradient magnitudes** computed during backward pass
    """)
    return


@app.cell
def _(policy_log_probs, response_mask, torch):
    import matplotlib.pyplot as plt
    import numpy as np

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    # Recompute for visualization (without grad)
    _log_probs = policy_log_probs.detach()
    _masked = _log_probs * response_mask
    _grads = policy_log_probs.grad.detach() if policy_log_probs.grad is not None else torch.zeros_like(_log_probs)

    # 1. Original log-probs with mask overlay
    _im1 = _axes[0].imshow(_log_probs.numpy(), cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    _axes[0].set_title('Policy log-probabilities')
    _axes[0].set_xlabel('Position')
    _axes[0].set_ylabel('Example')
    for _i in range(2):
        for _j in range(4):
            _axes[0].text(_j, _i, f'{_log_probs[_i, _j].item():.1f}', ha='center', va='center', fontsize=11, fontweight='bold')
            # Red border for masked positions
            if response_mask[_i, _j].item() == 0:
                _axes[0].add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
    plt.colorbar(_im1, ax=_axes[0], label='Log prob')

    # 2. Masked values
    _im2 = _axes[1].imshow(_masked.numpy(), cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    _axes[1].set_title('After masking (response only)')
    _axes[1].set_xlabel('Position')
    _axes[1].set_yticks([])
    for _i in range(2):
        for _j in range(4):
            _v = _masked[_i, _j].item()
            _axes[1].text(_j, _i, f'{_v:.1f}', ha='center', va='center', fontsize=11,
                         color='gray' if abs(_v) < 0.01 else 'black')
    plt.colorbar(_im2, ax=_axes[1], label='Log prob')

    # 3. Gradients
    _im3 = _axes[2].imshow(_grads.numpy(), cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
    _axes[2].set_title('Gradients (∂loss/∂input)')
    _axes[2].set_xlabel('Position')
    _axes[2].set_yticks([])
    for _i in range(2):
        for _j in range(4):
            _g = _grads[_i, _j].item()
            _axes[2].text(_j, _i, f'{_g:.2f}', ha='center', va='center', fontsize=11,
                         color='white' if abs(_g) > 0.1 else 'lightgray')
    plt.colorbar(_im3, ax=_axes[2], label='Gradient')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 9: Edge cases & numerical stability

    Why this implementation handles edge cases:

    1. **Zero response tokens**: If `response_mask` is all zeros, `masked.sum() = 0`, so `loss = 0`
       and gradients are all zero. This is mathematically correct (no tokens = no loss).

    2. **Gradient accumulation**: The `batch_size` factor ensures correct scaling when
       accumulating gradients across multiple microbatches.

    3. **Detach in metadata**: Using `.detach()` prevents the logged loss from holding
       references to the computation graph, avoiding memory leaks.

    Let's verify the zero-mask case:
    """)
    return


@app.cell
def _(torch):
    # import torch
    from core.trainer.sft import sft_microbatch_train_step

    # All zeros mask (no response tokens)
    _empty_log_probs = torch.tensor([[1.0, 2.0], [0.5, 1.5]], requires_grad=True)
    _empty_mask = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    _loss, _metadata = sft_microbatch_train_step(
        policy_log_probs=_empty_log_probs,
        response_mask=_empty_mask,
        gradient_accumulation_steps=2,
        normalize_constant=1.0,
    )

    print(f"All-zero mask case:")
    print(f"  loss: {_loss.item():.4f}")
    print(f"  gradients: {_empty_log_probs.grad}")
    print(f"\nZero loss and zero gradients — correct!")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    ```python
    def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    ```

    | Line | What it does | Why |
    |------|-------------|-----|
    | `batch_size = ...` | Extract batch dimension | Used for scaling computation |
    | `masked_normalize(...)` | Sum masked log-probs and normalize | Compute cross-entropy loss on response only |
    | `-loss` | Negate | Minimizing loss = maximizing log probability |
    | `× grad_accum × batch_size` | Scale normalization | Correct gradient accumulation across microbatches |
    | `loss.backward()` | Backpropagate | Compute gradients for optimizer |
    | `loss.detach()` | Remove from graph | Clean logging, no memory leak |

    ### Key formulas

    | Quantity | Formula |
    |----------|---------|
    **Loss** | $-\frac{\sum_{i \in \text{response}} \log p_i}{\text{norm} \times \text{grad\_accum} \times \text{batch}}$ |
    **Gradient** | $\frac{\partial \text{loss}}{\partial \log p_i} = -\frac{\text{mask}_i}{\text{norm} \times \text{grad\_accum} \times \text{batch}}$ |
    """)
    return


if __name__ == "__main__":
    app.run()
