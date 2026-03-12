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
    # `masked_normalize` — Line-by-Line Walkthrough

    This notebook steps through `core/utils/masked_ops.py` using small tensors
    so every intermediate value is visible.

    ## What does this function do?

    Given a tensor, a boolean mask, and a normalization constant, it:

    1. **Zeros out** masked positions (`mask == 0`)
    2. **Sums** along a specified dimension (or all dims)
    3. **Divides** by a constant

    $$\text{result} = \frac{\sum_{i \in \text{mask}} \text{tensor}_i}{\text{normalize\_constant}}$$

    ## Why do we need it?

    In alignment training, we compute loss only on **response tokens** (not prompt
    or padding). The `response_mask` tells us which positions are response tokens.
    This function sums the per-token losses and normalizes — for example, by the
    number of response tokens or by the gradient accumulation steps.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 0: Create toy data

    A `(2, 5)` tensor representing per-token losses for a batch of 2 sequences,
    each with 5 positions. The mask marks response tokens (1) vs prompt/padding (0).

    ```
    tensor:  [[-1.2, -0.5, -3.1, -0.8, -2.0],
              [-0.3, -1.7, -0.4, -0.9, -1.1]]

    mask:    [[ 0,    0,    1,    1,    1  ],    <- response starts at pos 2
              [ 0,    0,    0,    1,    1  ]]    <- response starts at pos 3
    ```
    """)
    return


@app.cell
def _():
    import torch

    tensor = torch.tensor([
        [-1.2, -0.5, -3.1, -0.8, -2.0],
        [-0.3, -1.7, -0.4, -0.9, -1.1],
    ])

    mask = torch.tensor([
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=torch.float)

    normalize_constant = 2.0

    print(f"tensor shape: {tensor.shape}")
    print(f"tensor:\n{tensor}")
    print(f"\nmask:\n{mask}")
    print(f"\nnormalize_constant: {normalize_constant}")
    return mask, normalize_constant, tensor, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: `masked = tensor * mask`

    Element-wise multiplication zeros out positions where `mask == 0`.
    Only the response token values survive.
    """)
    return


@app.cell
def _(mask, tensor):
    masked = tensor * mask

    print("tensor:")
    print(tensor)
    print("\nmask:")
    print(mask)
    print("\nmasked = tensor * mask:")
    print(masked)
    print("\nZeroed positions: prompt/padding values are gone")
    return (masked,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Visual: before vs after masking

    ```
    tensor:  [-1.2, -0.5, -3.1, -0.8, -2.0]     mask: [0, 0, 1, 1, 1]
                                ↓
    masked:  [ 0.0,  0.0, -3.1, -0.8, -2.0]     prompt values erased
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2a: `masked.sum(dim=dim)` — with `dim=None`

    When `dim=None`, sum **all** elements into a single scalar.
    This is useful when you want a single loss value for the entire batch.
    """)
    return


@app.cell
def _(masked, normalize_constant):
    _sum_all = masked.sum()
    _result_none = _sum_all / normalize_constant

    print(f"masked:\n{masked}")
    print(f"\nmasked.sum() = {_sum_all.item():.4f}")
    print(f"  = (-3.1) + (-0.8) + (-2.0) + (-0.9) + (-1.1)")
    print(f"  = {-3.1 + -0.8 + -2.0 + -0.9 + -1.1:.4f}")
    print(f"\n/ normalize_constant ({normalize_constant}) = {_result_none.item():.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2b: `masked.sum(dim=0)` — sum across batch

    Sum each position across examples. Result shape: `(5,)`.
    """)
    return


@app.cell
def _(masked, normalize_constant):
    _sum_dim0 = masked.sum(dim=0)
    _result_dim0 = _sum_dim0 / normalize_constant

    print(f"masked:\n{masked}")
    print(f"\nmasked.sum(dim=0) = {_sum_dim0.tolist()}")
    print("  each column summed across the 2 examples")
    print(f"\n/ {normalize_constant} = {_result_dim0.tolist()}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2c: `masked.sum(dim=1)` — sum across sequence

    Sum each example across positions. Result shape: `(2,)`.
    This is the most common case: **per-example loss** summed over response tokens.
    """)
    return


@app.cell
def _(masked, normalize_constant):
    _sum_dim1 = masked.sum(dim=1)
    _result_dim1 = _sum_dim1 / normalize_constant

    print(f"masked:\n{masked}")
    print(f"\nmasked.sum(dim=1) = {_sum_dim1.tolist()}")
    print(f"  example 0: (-3.1) + (-0.8) + (-2.0) = {-3.1 + -0.8 + -2.0:.4f}")
    print(f"  example 1: (-0.9) + (-1.1)           = {-0.9 + -1.1:.4f}")
    print(f"\n/ {normalize_constant} = {_result_dim1.tolist()}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: `/ normalize_constant`

    The final division. In training, `normalize_constant` is typically:

    - The **number of response tokens** (for mean loss)
    - The **gradient accumulation steps** (for distributed training)
    - A fixed constant like in **Dr. GRPO**

    This is a simple scalar division applied after the sum.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: Verify against the actual function

    Call `masked_normalize` from `core/utils` for all `dim` variants and confirm
    they match our manual computations.
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
def _(mask, normalize_constant, tensor, torch):
    from core.utils.masked_ops import masked_normalize

    # dim=None
    _r_none = masked_normalize(tensor, mask, normalize_constant, dim=None)
    _expected_none = (tensor * mask).sum() / normalize_constant
    print(f"dim=None:  {_r_none.item():.4f}  (expected {_expected_none.item():.4f})  match={torch.allclose(_r_none, _expected_none)}")

    # dim=0
    _r_dim0 = masked_normalize(tensor, mask, normalize_constant, dim=0)
    _expected_dim0 = (tensor * mask).sum(dim=0) / normalize_constant
    print(f"dim=0:     {_r_dim0.tolist()}  match={torch.allclose(_r_dim0, _expected_dim0)}")

    # dim=1
    _r_dim1 = masked_normalize(tensor, mask, normalize_constant, dim=1)
    _expected_dim1 = (tensor * mask).sum(dim=1) / normalize_constant
    print(f"dim=1:     {_r_dim1.tolist()}  match={torch.allclose(_r_dim1, _expected_dim1)}")

    # dim=-1 (same as dim=1 for 2D)
    _r_last = masked_normalize(tensor, mask, normalize_constant, dim=-1)
    print(f"dim=-1:    {_r_last.tolist()}  match={torch.allclose(_r_last, _expected_dim1)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: Heatmap visualization

    Three heatmaps showing the data flow:
    1. **Original tensor** — all values
    2. **After masking** — prompt/padding zeroed out
    3. **Sum (dim=1)** — per-example totals before normalization
    """)
    return


@app.cell
def _(mask, masked, normalize_constant, tensor):
    import matplotlib.pyplot as plt
    import numpy as np

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3), gridspec_kw={'width_ratios': [5, 5, 1.2]})

    # 1. Original tensor
    _im1 = _axes[0].imshow(tensor.numpy(), cmap='RdYlGn', aspect='auto')
    _axes[0].set_title('Original tensor')
    _axes[0].set_xlabel('Position')
    _axes[0].set_ylabel('Example')
    _axes[0].set_yticks([0, 1])
    for _i in range(2):
        for _j in range(5):
            _axes[0].text(_j, _i, f'{tensor[_i, _j].item():.1f}', ha='center', va='center', fontsize=10)
    # Overlay mask borders
    for _i in range(2):
        for _j in range(5):
            if mask[_i, _j].item() == 0:
                _axes[0].add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
    plt.colorbar(_im1, ax=_axes[0], shrink=0.8)

    # 2. After masking
    _im2 = _axes[1].imshow(masked.numpy(), cmap='RdYlGn', aspect='auto')
    _axes[1].set_title('After masking (tensor * mask)')
    _axes[1].set_xlabel('Position')
    _axes[1].set_yticks([0, 1])
    for _i in range(2):
        for _j in range(5):
            _v = masked[_i, _j].item()
            _axes[1].text(_j, _i, f'{_v:.1f}', ha='center', va='center', fontsize=10,
                         color='gray' if _v == 0 else 'black')
    plt.colorbar(_im2, ax=_axes[1], shrink=0.8)

    # 3. Sum per example (dim=1) then normalize
    _sums = masked.sum(dim=1).numpy()
    _normalized = _sums / normalize_constant
    _data = np.column_stack([_sums, _normalized])
    _axes[2].set_title('sum(dim=1) / C')
    _axes[2].barh([0, 1], _normalized, color=['#2ecc71', '#3498db'], edgecolor='black')
    _axes[2].set_yticks([0, 1])
    _axes[2].set_yticklabels(['Ex 0', 'Ex 1'])
    for _i, (_s, _n) in enumerate(zip(_sums, _normalized)):
        _axes[2].text(_n - 0.1, _i, f'{_s:.1f}/{normalize_constant}={_n:.2f}', ha='right', va='center', fontsize=8, color='white', fontweight='bold')
    _axes[2].set_xlabel('Normalized sum')
    _axes[2].invert_xaxis()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    ```python
    def masked_normalize(tensor, mask, normalize_constant, dim=None):
        masked = tensor * mask                        # 1. zero out masked positions
        if dim is None:
            return masked.sum() / normalize_constant  # 2a. sum everything
        return masked.sum(dim=dim) / normalize_constant  # 2b. sum along dim
    ```

    | Line | What it does | Why |
    |------|-------------|-----|
    | `tensor * mask` | Element-wise zero out | Only response tokens contribute |
    | `.sum(dim=...)` | Reduce along dimension | Aggregate per-token values |
    | `/ normalize_constant` | Scale the result | Normalize by token count, grad accum steps, etc. |

    ### `dim` cheat sheet

    | `dim` | Sums over | Output shape (for `(B, L)` input) | Typical use |
    |-------|-----------|-----------------------------------|-------------|
    | `None` | Everything | Scalar `()` | Single batch loss |
    | `0` | Batch | `(L,)` | Per-position stats |
    | `1` / `-1` | Sequence | `(B,)` | Per-example loss |
    """)
    return


if __name__ == "__main__":
    app.run()
