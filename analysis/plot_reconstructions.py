#!/usr/bin/env python
"""Reconstruction montage + error maps (runbook Section 17.3)."""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_common import (load_results, zero_filled_mag, recon_mag, get_reference,
                                  METHOD_LABELS)

# Default column order for the 2-view set: classical -> single -> fixed budget ->
# 2x duplicated -> 2x complementary. Override with --methods for the 4-view set.
COLUMN_METHODS = ["classical_l1wav", "single_r4", "fixed_split_merge", "fixed_split_ft",
                  "dup2_merge", "dup2_ft", "comp2_merge", "comp2_ft",
                  "fcomp2_merge", "fcomp2_ft"]
QUAD_METHODS = ["single_r4", "dup4_merge", "dup4_ft", "comp4_merge", "comp4_ft",
                "fcomp4_merge", "fcomp4_ft", "full_diffusion", "full_plain"]


def _input_snr(case_methods):
    """Input SNR of a case from the processed sample (reference signal / noise_std).

    SNR is a property of the *measurement*, independent of any method outcome, so
    the difficult case is not cherry-picked."""
    import torch
    d = next(iter(case_methods.values()))
    mp = d.get("mask_path")
    if not mp:
        return None
    try:
        b = torch.load(mp, map_location="cpu", weights_only=False)
        s = torch.load(b["processed_path"], map_location="cpu", weights_only=False)
        noise = float(s["noise_std"].mean())
        ref = s.get("reference_mag")
        sig = float(ref.abs().flatten().quantile(0.9)) if ref is not None else 1.0
        return sig / max(noise, 1e-8)
    except Exception:
        return None


def pick_cases(res, max_cases=4):
    cases = {}
    for (method, subj, sl, seed), d in res.items():
        cases.setdefault((subj, sl), {})[method] = d
    core = ["single_r4", "comp2_ft"]
    good = {k: v for k, v in cases.items() if all(m in v for m in core)}
    if not good:
        good = cases
    keys = list(good.keys())
    if len(keys) <= max_cases:
        return list(good.items())
    # rank candidates by INPUT SNR (measurement property, not method outcome)
    snr = {k: (_input_snr(good[k]) or 0.0) for k in keys}
    by_snr = sorted(keys, key=lambda k: snr[k])
    # difficult (lowest SNR), median SNR, and two spanning the range
    chosen = [by_snr[0], by_snr[len(by_snr) // 3], by_snr[2 * len(by_snr) // 3], by_snr[-1]]
    seen, out = set(), []
    for k in chosen:
        if k not in seen:
            seen.add(k); out.append((k, good[k]))
    return out[:max_cases]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--methods", default=None,
                    help="comma-separated column order (default: the 2-view set)")
    ap.add_argument("--suffix", default="", help="appended to output filenames")
    ap.add_argument("--zf-condition", default="single_r4",
                    help="condition whose view-0 zero-filled image is shown")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    res = load_results(args.results_root)
    cases = pick_cases(res)
    if not cases:
        print("No cases to plot"); return
    order = args.methods.split(",") if args.methods else COLUMN_METHODS
    methods = [m for m in order if any(k[0] == m for k in res)]
    ncol = 2 + len(methods)  # reference + zero-filled + methods

    # ---- montage ----
    fig, axes = plt.subplots(len(cases), ncol, figsize=(2.1 * ncol, 2.3 * len(cases)),
                             squeeze=False)
    for r, ((subj, sl), mdict) in enumerate(cases):
        any_d = next(iter(mdict.values()))
        ref = get_reference(any_d)
        vmax = np.percentile(ref, 99) if ref is not None else None
        zf = zero_filled_mag(any_d["mask_path"], args.zf_condition) if any_d.get("mask_path") else None
        panels = [("Reference", ref), ("Zero-filled", zf)]
        for m in methods:
            panels.append((METHOD_LABELS.get(m, m), recon_mag(mdict[m]) if m in mdict else None))
        for c, (title, img) in enumerate(panels):
            ax = axes[r, c]
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{subj}\nsl{sl}", fontsize=8)
    fig.suptitle("Reconstruction montage (identical intensity window per row)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"reconstruction_montage{args.suffix}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- error maps ----
    fig, axes = plt.subplots(len(cases), len(methods), figsize=(2.3 * len(methods), 2.5 * len(cases)),
                             squeeze=False)
    for r, ((subj, sl), mdict) in enumerate(cases):
        any_d = next(iter(mdict.values()))
        ref = get_reference(any_d)
        errs = {}
        for m in methods:
            if m in mdict and ref is not None:
                rm = recon_mag(mdict[m])
                errs[m] = np.abs(rm - ref)
        emax = np.percentile(np.concatenate([e.ravel() for e in errs.values()]), 99) if errs else 1.0
        for c, m in enumerate(methods):
            ax = axes[r, c]
            if m in errs:
                ax.imshow(errs[m], cmap="magma", vmin=0, vmax=emax)
                met = mdict[m].get("metrics", {})
                ax.set_xlabel(f"NRMSE {met.get('nrmse', float('nan')):.3f}\nSSIM {met.get('ssim', float('nan')):.3f}",
                              fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(METHOD_LABELS.get(m, m), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{subj}\nsl{sl}", fontsize=8)
    fig.suptitle("Absolute error maps (shared color scale per row)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"error_maps{args.suffix}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote reconstruction_montage.png and error_maps.png ({len(cases)} cases, {len(methods)} methods)")


if __name__ == "__main__":
    main()
