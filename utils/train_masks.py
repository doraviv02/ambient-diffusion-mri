"""Deterministic Cartesian undersampling masks for cross-view fine-tuning.

Columns are undersampled (matching the pretrained model's mask orientation).
``base_and_delta`` returns a base acquisition mask plus a further-corrupted
(delta) submask used as the Ambient network input, so the held-out coefficients
(base minus delta) can supervise an input-consistency term.
"""

from __future__ import annotations

import numpy as np
import torch


def random_cartesian_mask(H, W, R, acs_lines, rng):
    """Return a [1, H, W] float32 column mask: central ACS + uniform outer lines."""
    c0 = (W - acs_lines) // 2
    acs = np.arange(c0, c0 + acs_lines)
    n_total = int(round(W / R))
    n_outer = max(n_total - len(acs), 0)
    pool = np.setdiff1d(np.arange(W), acs)
    chosen = rng.choice(pool, size=min(n_outer, len(pool)), replace=False)
    cols = np.union1d(acs, chosen)
    m = np.zeros((1, H, W), dtype=np.float32)
    m[:, :, cols] = 1.0
    return m, acs, cols


def base_and_delta(H, W, R, acs_lines, delta_fraction, rng):
    """Return (base_mask, delta_mask) with delta ⊂ base (ACS always retained)."""
    base, acs, cols = random_cartesian_mask(H, W, R, acs_lines, rng)
    outer = np.setdiff1d(cols, acs)
    n_drop = int(round(delta_fraction * len(outer)))
    drop = rng.choice(outer, size=min(n_drop, len(outer)), replace=False) if len(outer) else np.array([], int)
    delta = base.copy()
    delta[:, :, drop] = 0.0
    return (torch.from_numpy(base), torch.from_numpy(delta))
