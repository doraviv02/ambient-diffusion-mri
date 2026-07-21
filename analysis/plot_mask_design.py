#!/usr/bin/env python
"""Mask design figure: every condition, its per-view masks and its true coverage.

`tools/generate_project_masks.py` draws a 4-panel figure covering only the original
runbook conditions.  This one covers all eight, including the complementary
designs, and reports the *measured* unique-column coverage rather than the
nominal acceleration -- which is the whole point of the duplicated-vs-
complementary comparison.
"""

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (condition, pretty name, design tag)
ROWS = [
    ("single_r4", "single_r4\none cheap scan", "base"),
    ("joint_fixed", "fixed_split\nfixed budget, 2xR=8", "base"),
    ("joint_extra", "dup2\n2xR=4 DUPLICATED", "dup"),
    ("joint_extra_comp", "comp2\n2xR=4 COMPLEMENTARY (shared ACS)", "comp"),
    ("joint_extra_fullcomp", "fcomp2\n2xR=4 FULLY-COMP (split ACS)", "fcomp"),
    ("joint_quad", "dup4\n4xR=4 DUPLICATED", "dup"),
    ("joint_quad_comp", "comp4\n4xR=4 COMPLEMENTARY (shared ACS)", "comp"),
    ("joint_quad_fullcomp", "fcomp4\n4xR=4 FULLY-COMP (split ACS)", "fcomp"),
    ("single_full", "single_full\none complete measurement", "full"),
]
# union-panel colormap + text colour per design tag
STYLE = {
    "base": ("Blues", "#2a5d8f"),
    "dup": ("Oranges", "#b5561a"),
    "comp": ("Greens", "#2a7f3f"),
    "fcomp": ("Purples", "#6a51a3"),
    "full": ("Reds", "#a83236"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-root", required=True, help="dir with test/*.pt mask bundles")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    f = sorted(glob.glob(os.path.join(args.masks_root, "test", "*.pt")))[0]
    b = torch.load(f, map_location="cpu", weights_only=False)
    conds = b["conditions"]
    meta = b["metadata"]

    max_v = max(conds[c].shape[0] for c, _, _ in ROWS)
    fig, axes = plt.subplots(len(ROWS), max_v + 1,
                             figsize=(2.5 * (max_v + 1) + 2.6, 2.5 * len(ROWS)))
    for r, (cond, name, tag) in enumerate(ROWS):
        m = conds[cond].numpy()          # [V,1,H,W]
        V, _, H, W = m.shape
        cols = [np.where(m[v, 0, 0] > 0)[0] for v in range(V)]
        uniq = np.unique(np.concatenate(cols))
        lines = sum(len(c) for c in cols)
        cmap, txt = STYLE[tag]
        for v in range(max_v):
            ax = axes[r, v]
            if v < V:
                # vmin/vmax pinned: single_full's mask is constant 1 and would
                # otherwise autoscale to the bottom of the colormap (all black).
                ax.imshow(m[v, 0], cmap="gray", aspect="auto", interpolation="nearest",
                          vmin=0, vmax=1)
                ax.set_title(f"view {v}  ({len(cols[v])} cols)", fontsize=9)
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])
        # union panel
        ax = axes[r, max_v]
        u = np.zeros((H, W), dtype=np.float32)
        u[:, uniq] = 1.0
        ax.imshow(u, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1.4)
        ax.set_title(f"UNION: {len(uniq)}/{W} cols = {len(uniq)/W:.0%}", fontsize=10,
                     fontweight="bold", color=txt)
        ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=9.5, rotation=0, ha="right", va="center",
                              labelpad=12,
                              fontweight="bold" if tag in ("comp", "fcomp") else "normal",
                              color=txt)
        # efficiency annotation
        axes[r, max_v].text(1.04, 0.5, f"{lines} lines bought\n-> {len(uniq)} unique\n"
                                       f"({len(uniq)/lines:.0%} efficient)",
                            transform=axes[r, max_v].transAxes, fontsize=8.5,
                            va="center", ha="left")
    fig.suptitle(
        f"Mask design: what each condition buys vs what it learns  "
        f"(subject {meta['subject_id']}, slice {meta['slice_id']})\n"
        "duplicated masks re-buy the same columns; complementary masks partition the outer "
        "lines and share only the ACS",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0, 0.93, 0.95])
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
