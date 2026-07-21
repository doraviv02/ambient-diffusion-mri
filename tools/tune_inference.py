#!/usr/bin/env python
"""Tune multi-view A-DPS on a fixed validation subset.

Selection criteria (in order): normalized held-out k-space error, subject-mean
SSIM, stability (no divergence), runtime.  A single global configuration is
chosen (not per-method).  Test subjects are never used here.

The l_ss grid is recentred (log-spaced [3,10,30]) around the validation smoke
test because the normalized fidelity's effective l_ss is ~10x the nominal
grid -- documented in reports/project_notes.md.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import solve_inverse_mv_adps as S
from utils.multiview_mri import MRIViewOperator, channels_to_complex
from torch_utils.ambient_diffusion import create_masks
from analysis.metrics_common import compute_all_metrics


def select_subset(manifest, n_subjects):
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject_id"], []).append(r)
    subjects = sorted(by_subj.keys())[:n_subjects]
    subset = []
    for s in subjects:
        srows = sorted(by_subj[s], key=lambda r: int(r["slice_id"]))
        subset.append(srows[len(srows) // 2])  # central slice
    return subset, os.path.dirname(manifest)


def run_one(net, sample, masks_bundle, condition, device, cfg, l_ss, num_steps, l_type):
    mv_op, y_views, pmaps = S.build_condition_operator(sample, masks_bundle, condition, device,
                                                       cfg.get("min_noise_variance", 1e-8))
    H, W = sample["ksp_views"].shape[-2:]
    corr = create_masks(cfg.get("prior_training_R", 4), cfg.get("prior_delta_prob", 5),
                        cfg.get("prior_acs_lines", 20), H, W).to(device)
    cm = torch.ones(1, 2, H, W, device=device); cm[:, 0] = corr
    prior = MRIViewOperator(mask=cm[:, 0:1], maps=pmaps[0])
    g = torch.Generator(device=device).manual_seed(0)
    lat = torch.randn(1, 2, H, W, device=device, dtype=torch.float64, generator=g)
    import time
    t0 = time.time()
    rec, _ = S.mv_posterior_sample(net, mv_op, y_views, prior, cm, lat, l_type, l_ss, num_steps,
                                   cfg.get("sigma_min", 0.004), cfg.get("sigma_max", 10.0),
                                   cfg.get("rho", 7), cfg.get("S_churn", 0.0))
    runtime = time.time() - t0
    recon_cplx = channels_to_complex(rec.float())[0, 0].cpu().numpy()
    ref = sample.get("reference_mag")
    m = compute_all_metrics(recon_cplx.astype(np.complex64), sample, mv_op, y_views, device,
                            ref.numpy() if ref is not None else None)
    m["runtime"] = runtime
    m["diverged"] = bool(not np.isfinite(recon_cplx).all())
    del rec, mv_op, y_views, prior, lat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True, help="validation mask manifest")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-subjects", type=int, default=8)
    ap.add_argument("--condition", default="single_r4")
    ap.add_argument("--l-ss-grid", default="3,10,30")
    ap.add_argument("--steps-grid", default="100")
    ap.add_argument("--likelihood-grid", default="DPS,ALD")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = S.load_network(cfg["checkpoints"]["original"], device)

    subset, mask_root = select_subset(args.manifest, args.n_subjects)
    l_ss_grid = [float(x) for x in args.l_ss_grid.split(",")]
    steps_grid = [int(x) for x in args.steps_grid.split(",")]
    lik_grid = args.likelihood_grid.split(",")

    rows = []
    for l_type in lik_grid:
        for num_steps in steps_grid:
            for l_ss in l_ss_grid:
                hk, ss, rt, div = [], [], [], 0
                for r in subset:
                    mb = torch.load(os.path.join(mask_root, r["split"], r["mask_file"]),
                                    map_location="cpu", weights_only=False)
                    sample = torch.load(mb["processed_path"], map_location="cpu", weights_only=False)
                    m = run_one(net, sample, mb, args.condition, device, cfg, l_ss, num_steps, l_type)
                    hk.append(m["heldout_kspace_err"]); ss.append(m.get("ssim", np.nan))
                    rt.append(m["runtime"]); div += int(m["diverged"])
                row = {"likelihood": l_type, "num_steps": num_steps, "l_ss": l_ss,
                       "heldout_kspace_err": float(np.mean(hk)), "ssim": float(np.nanmean(ss)),
                       "runtime": float(np.mean(rt)), "diverged": div, "n": len(subset)}
                rows.append(row)
                print(f"[{l_type} steps={num_steps} l_ss={l_ss}] "
                      f"heldoutK={row['heldout_kspace_err']:.4f} ssim={row['ssim']:.4f} "
                      f"div={div}", flush=True)

    # write grid
    grid_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tables", "project", "validation_grid.csv")
    os.makedirs(os.path.dirname(grid_csv), exist_ok=True)
    with open(grid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # select: valid (no divergence) -> min held-out k-space err -> max ssim -> min runtime
    valid = [r for r in rows if r["diverged"] == 0] or rows
    best = sorted(valid, key=lambda r: (round(r["heldout_kspace_err"], 4),
                                        -round(r["ssim"], 4), r["runtime"]))[0]
    selected = dict(cfg)
    selected["num_steps"] = best["num_steps"]
    selected["l_ss"] = best["l_ss"]
    selected["likelihood_type"] = best["likelihood"]
    sel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "project", "selected_inference.yaml")
    with open(sel_path, "w") as f:
        yaml.safe_dump(selected, f, sort_keys=False)
    print(f"\nSelected: l_ss={best['l_ss']} steps={best['num_steps']} likelihood={best['likelihood']}")
    print(f"Wrote {sel_path} and {grid_csv}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for l_type in lik_grid:
            for num_steps in steps_grid:
                sub = [r for r in rows if r["likelihood"] == l_type and r["num_steps"] == num_steps]
                sub = sorted(sub, key=lambda r: r["l_ss"])
                ax.plot([r["l_ss"] for r in sub], [r["heldout_kspace_err"] for r in sub],
                        "o-", label=f"{l_type} steps={num_steps}")
        ax.set_xscale("log"); ax.set_xlabel("l_ss"); ax.set_ylabel("held-out k-space error")
        ax.set_title(f"Validation tuning ({args.condition}, n={len(subset)} subjects)")
        ax.legend(fontsize=8)
        figp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "figures", "project", "validation_grid.png")
        os.makedirs(os.path.dirname(figp), exist_ok=True)
        fig.tight_layout(); fig.savefig(figp, dpi=120, bbox_inches="tight")
        print(f"Wrote {figp}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
