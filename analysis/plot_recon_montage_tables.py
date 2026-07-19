#!/usr/bin/env python
"""Reconstruction-image montages grouped to match the results tables.

Columns are exactly the methods of each results table in reports/mvp_summary.md;
rows are a few example slices (spanning input SNR). Grouped per report section:
  recon_montage_2view.png : the two 2-view tables stacked (fixed budget; 2x budget)
  recon_montage_4view.png : the 4-view table

Reference + zero-filled input are shown first for context. Intensity window is
fixed per row (per slice) so methods are directly comparable within a row.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_common import load_results, recon_mag, get_reference, zero_filled_mag
from analysis.experiments import EXPERIMENTS, DESIGN_COLORS, design_of

# the results tables, exactly as presented in mvp_summary.md
T_FIXED = ["classical_adjoint", "classical_l1wav", "single_r4", "fixed_split_merge", "fixed_split_ft"]
T_2X = ["dup2_merge", "dup2_ft", "comp2_merge", "comp2_ft", "fcomp2_merge", "fcomp2_ft"]
T_QUAD = ["single_r4", "dup4_merge", "dup4_ft", "comp4_merge", "comp4_ft",
          "fcomp4_merge", "fcomp4_ft", "full_diffusion", "full_plain",
          "dualfull_merge", "dualfull_ft"]


def _input_snr(d):
    import torch
    mp = d.get("mask_path")
    if not mp:
        return 0.0
    try:
        b = torch.load(mp, map_location="cpu", weights_only=False)
        s = torch.load(b["processed_path"], map_location="cpu", weights_only=False)
        noise = float(s["noise_std"].mean())
        ref = s.get("reference_mag")
        sig = float(ref.abs().flatten().quantile(0.9)) if ref is not None else 1.0
        return sig / max(noise, 1e-8)
    except Exception:
        return 0.0


def pick_cases(res, methods, n=3):
    """(subject,slice) cases present for ALL `methods`, spread across input SNR."""
    cases = {}
    for (m, subj, sl, seed), d in res.items():
        cases.setdefault((subj, sl), {})[m] = d
    good = {k: v for k, v in cases.items() if all(m in v for m in methods)}
    keys = sorted(good, key=lambda k: _input_snr(next(iter(good[k].values()))))
    if len(keys) <= n:
        chosen = keys
    else:                                   # low / middle / high input SNR
        idx = [0, len(keys) // 2, len(keys) - 1][:n]
        chosen = [keys[i] for i in idx]
    return [(k, good[k]) for k in chosen]


def block_title(fig, cell, text):
    """Render a block heading in its OWN gridspec row, so it can never collide
    with the per-column titles of the montage below it."""
    ax = fig.add_subplot(cell)
    ax.axis("off")
    ax.text(0.0, 0.15, text, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", ha="left", va="bottom")


def draw_block(fig, cell, res, methods, cases, zf_cond):
    ncol = 2 + len(methods)                  # reference + zero-filled + methods
    inner = gridspec.GridSpecFromSubplotSpec(len(cases), ncol, subplot_spec=cell,
                                             wspace=0.04, hspace=0.06)
    for r, ((subj, sl), mdict) in enumerate(cases):
        any_d = next(iter(mdict.values()))
        ref = get_reference(any_d)
        vmax = np.percentile(ref, 99) if ref is not None else None
        zf = zero_filled_mag(any_d["mask_path"], zf_cond) if any_d.get("mask_path") else None
        panels = [("Reference", ref, None), ("Zero-filled", zf, None)]
        for m in methods:
            panels.append((m, recon_mag(mdict[m]) if m in mdict else None, design_of(m)))
        for c, (name, img, dz) in enumerate(panels):
            ax = fig.add_subplot(inner[r, c])
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if dz is not None:               # colour-code method columns by design
                for sp in ax.spines.values():
                    sp.set_edgecolor(DESIGN_COLORS[dz]); sp.set_linewidth(2.2)
            if r == 0:
                lab = name if name in ("Reference", "Zero-filled") else \
                    f"{name}\n({EXPERIMENTS[name].cov:.0%})"
                col = "black" if dz is None else DESIGN_COLORS[dz]
                ax.set_title(lab, fontsize=8, color=col,
                             fontweight="bold" if (dz and EXPERIMENTS[name].prior == "finetuned") else "normal")
            if c == 0:
                ax.set_ylabel(f"{subj}\nsl{sl}", fontsize=7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="2-view results dir")
    ap.add_argument("--final-quad", required=True, help="4-view results dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cases", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    res2 = load_results(args.final)
    resq = load_results(args.final_quad)

    # ---- 2-view: two tables stacked into one grouped figure ----
    cases2 = pick_cases(res2, list(set(T_FIXED + T_2X)), n=args.cases)
    ncol2 = max(2 + len(T_FIXED), 2 + len(T_2X))
    fig = plt.figure(figsize=(1.55 * ncol2 + 1, 1.7 * args.cases * 2 + 2.2))
    # dedicated thin rows for the block headings -> no overlap with column titles
    outer = gridspec.GridSpec(4, 1, height_ratios=[0.13, 1, 0.13, 1], hspace=0.30)
    block_title(fig, outer[0], "Table 1 — classical baselines & fixed budget (64 lines)")
    draw_block(fig, outer[1], res2, T_FIXED, cases2, "single_r4")
    block_title(fig, outer[2], "Table 2 — 2x budget (128 lines): duplicated / complementary / fully-comp")
    draw_block(fig, outer[3], res2, T_2X, cases2, "single_r4")
    fig.suptitle("2-view set — reconstructions per results table  (columns coloured by design; "
                 "same intensity window per row)", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.01, left=0.05, right=0.99)
    p2 = os.path.join(args.output_dir, "recon_montage_2view.png")
    fig.savefig(p2, dpi=135, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p2}")

    # ---- 4-view: single table ----
    casesq = pick_cases(resq, T_QUAD, n=args.cases)
    figq = plt.figure(figsize=(1.5 * (2 + len(T_QUAD)) + 1, 1.9 * args.cases + 1.8))
    gsq = gridspec.GridSpec(2, 1, height_ratios=[0.11, 1], hspace=0.30)
    block_title(figq, gsq[0], "4-view table — single_r4 -> full scan -> NEX=2")
    draw_block(figq, gsq[1], resq, T_QUAD, casesq, "joint_quad_comp")
    figq.suptitle("4-view set — reconstructions per results table  (columns coloured by design; "
                  "same intensity window per row)", fontsize=12, fontweight="bold")
    figq.subplots_adjust(top=0.88, bottom=0.02, left=0.05, right=0.99)
    pq = os.path.join(args.output_dir, "recon_montage_4view.png")
    figq.savefig(pq, dpi=135, bbox_inches="tight"); plt.close(figq)
    print(f"Wrote {pq}")


if __name__ == "__main__":
    main()
