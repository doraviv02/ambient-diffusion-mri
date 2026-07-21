#!/usr/bin/env python
"""Flatten per-reconstruction result files into a per-slice metrics CSV.

Each result .pt (from solve_inverse_mv_adps.py or run_classical_recon.py) already
carries a uniformly computed ``metrics`` dict (NRMSE, PSNR, SSIM, measured &
held-out k-space error) plus runtime and net-eval counts.  This collects them.
"""

import argparse
import glob
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="dir (or comma-list) with result .pt files")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    roots = args.results_root.split(",")
    files = []
    for r in roots:
        files += sorted(glob.glob(os.path.join(r, "*.pt")))
    if not files:
        print(f"No result files under {args.results_root}")
        sys.exit(1)

    import numpy as np
    from analysis.metrics_common import (compute_image_metrics, brain_mask_for,
                                         heldout_kspace_error)
    from utils.multiview_mri import complex_to_channels
    from solve_inverse_mv_adps import _MASK_SOURCE
    _mask_cache = {}
    _sample_cache = {}
    _dev = torch.device("cpu")

    def _heldout(d):
        """Recompute held-out k-space error on coefficients the method never saw."""
        mp, cond, recon = d.get("mask_path"), d.get("condition"), d.get("reconstruction")
        if not mp or not cond or recon is None or not os.path.exists(mp):
            return None
        key = (d.get("subject_id"), d.get("slice_id"))
        if key not in _sample_cache:
            b = torch.load(mp, map_location="cpu", weights_only=False)
            s = torch.load(b["processed_path"], map_location="cpu", weights_only=False)
            _sample_cache[key] = (b, s)
        b, s = _sample_cache[key]
        masks = b["conditions"][_MASK_SOURCE.get(cond, cond)]      # [V,1,H,W]
        heldout = 1.0 - (masks.sum(dim=0) > 0).float()             # [1,H,W]
        if float(heldout.sum()) == 0:
            return float("nan")   # fully-sampled condition: nothing is held out
        rc = complex_to_channels(torch.from_numpy(np.asarray(recon))[None, None])
        return heldout_kspace_error(s, rc.to(_dev), _dev, heldout_mask=heldout)

    rows = []
    for path in files:
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = dict(d.get("metrics", {}))
        # Recompute image metrics with a brain mask derived from the reference
        # The mask is cached per (subject, slice) so every method
        # on a given slice is scored with the identical mask. k-space errors are
        # mask-independent and are reused from the stored metrics.
        ref = d.get("reference")
        recon = d.get("reconstruction")
        if ref is not None and recon is not None:
            key = (d.get("subject_id"), d.get("slice_id"))
            if key not in _mask_cache:
                _mask_cache[key] = brain_mask_for(ref)
            m.update(compute_image_metrics(np.abs(np.asarray(recon)),
                                           np.asarray(ref), _mask_cache[key]))
        # keep the old (all-coefficient) value for transparency, add the corrected one
        m["heldout_kspace_err_allcoef"] = m.get("heldout_kspace_err")
        ho = _heldout(d)
        if ho is not None:
            m["heldout_kspace_err"] = ho
        row = {
            "method": d.get("method"), "condition": d.get("condition"),
            "subject_id": d.get("subject_id"), "slice_id": d.get("slice_id"),
            "seed": d.get("seed"), "num_steps": d.get("num_steps"),
            "l_ss": d.get("l_ss"), "likelihood_type": d.get("likelihood_type"),
            "lambda": d.get("lambda"),
            "runtime_seconds": d.get("runtime_seconds"),
            "num_net_evals": d.get("num_net_evals"),
        }
        for k, v in m.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(df)} rows, methods: {sorted(df['method'].dropna().unique())}")
    # quick summary
    if "ssim" in df.columns:
        print(df.groupby("method")[["ssim", "nrmse", "heldout_kspace_err"]].mean().round(4).to_string())
    else:
        print(df.groupby("method")[["heldout_kspace_err"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
