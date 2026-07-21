#!/usr/bin/env python
"""Per-coverage-tier l_ss tuning on validation (test never touched).

Motivation
----------
This project used a single l_ss=30 tuned on sparse single_r4 (25% coverage) for every
condition.  The optimal guidance scale depends on how much the data constrains
the image: sparse data wants the prior to have a strong voice; well-sampled data
wants guidance turned up so the prior stops injecting plausible-but-wrong texture
(the M12 < MF failure).  This script tunes l_ss separately for three tiers.

Tiers and representatives (all runnable on the 2-view validation set):
  sparse  (<=25% coverage) : condition single_r4
  mid     (44%)            : condition joint_extra_comp
  dense   (>=80%)          : synthetic single-view mask keeping 80% of PE columns,
                             scored on the withheld 20% (held-out k-space error is
                             undefined at 100% sampling, so we proxy the dense
                             regime at joint_quad_comp's 81% coverage).

Selection criterion (matches the original tune): normalized held-out k-space
error primary, subject-mean SSIM tiebreak, then stability, then runtime.

Writes configs/project/l_ss_by_tier.yaml, tables/project/lss_tier_grid.csv and
figures/project/lss_tier_tuning.png.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import solve_inverse_mv_adps as S
from utils.multiview_mri import MRIViewOperator, MultiViewMRI, channels_to_complex, complex_to_channels
from torch_utils.ambient_diffusion import create_masks
from analysis.metrics_common import compute_image_metrics, brain_mask_for, heldout_kspace_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def acs_columns(width, acs_lines):
    c0 = (width - acs_lines) // 2
    return np.arange(c0, c0 + acs_lines)


def dense_holdout_masks(H, W, acs_lines, keep_frac, rng):
    """Single-view mask keeping `keep_frac` of PE columns (ACS always kept), plus
    the complementary held-out column mask for scoring."""
    acs = acs_columns(W, acs_lines)
    pool = np.setdiff1d(np.arange(W), acs)
    n_keep_outer = int(round(keep_frac * W)) - len(acs)
    n_keep_outer = max(0, min(n_keep_outer, len(pool)))
    keep_outer = rng.choice(pool, size=n_keep_outer, replace=False)
    keep_cols = np.union1d(acs, keep_outer)
    obs = np.zeros((1, 1, H, W), dtype=np.float32)
    obs[..., keep_cols] = 1.0
    held = np.ones((1, H, W), dtype=np.float32)
    held[..., keep_cols] = 0.0            # score only the withheld columns
    return torch.from_numpy(obs), torch.from_numpy(held), len(keep_cols)


def build_tier_operator(tier, sample, masks_bundle, device, cfg, rng):
    """Return (mv_op, y_views, prior_maps_mod, heldout_mask_or_None)."""
    min_var = cfg.get("min_noise_variance", 1e-8)
    if tier == "dense":
        # synthetic 80%-coverage single view on view 0
        ksp = S.fftmod(sample["ksp_views"][0:1].to(device))
        maps = S.fftmod(sample["s_maps_views"][0:1].to(device))
        noise = sample["noise_std"][0:1].to(device)
        H, W = ksp.shape[-2:]
        obs, held, _ = dense_holdout_masks(H, W, cfg.get("prior_acs_lines", 20),
                                           keep_frac=0.80, rng=rng)
        obs = obs.to(device)
        y = obs * ksp
        mv = MultiViewMRI(obs, maps, noise, min_var)
        return mv, y, maps, held
    # sparse / mid use stored conditions
    cond = {"sparse": "single_r4", "mid": "joint_extra_comp"}[tier]
    mv, y, pmaps = S.build_condition_operator(sample, masks_bundle, cond, device, min_var)
    return mv, y, pmaps, None


def run_one(net, sample, masks_bundle, tier, device, cfg, l_ss, num_steps, l_type, rng):
    mv_op, y_views, pmaps, held = build_tier_operator(tier, sample, masks_bundle, device, cfg, rng)
    H, W = sample["ksp_views"].shape[-2:]
    corr = create_masks(cfg.get("prior_training_R", 4), cfg.get("prior_delta_prob", 5),
                        cfg.get("prior_acs_lines", 20), H, W).to(device)
    cm = torch.ones(1, 2, H, W, device=device); cm[:, 0] = corr
    prior = MRIViewOperator(mask=cm[:, 0:1], maps=pmaps[0])
    g = torch.Generator(device=device).manual_seed(0)
    lat = torch.randn(1, 2, H, W, device=device, dtype=torch.float64, generator=g)
    t0 = time.time()
    rec, _ = S.mv_posterior_sample(net, mv_op, y_views, prior, cm, lat, l_type, l_ss, num_steps,
                                   cfg.get("sigma_min", 0.004), cfg.get("sigma_max", 10.0),
                                   cfg.get("rho", 7), cfg.get("S_churn", 0.0))
    runtime = time.time() - t0
    recon_ch = rec.to(torch.float32)
    recon_cplx = channels_to_complex(recon_ch)[0, 0].cpu().numpy()
    ref = sample.get("reference_mag")
    ref_np = ref.numpy() if ref is not None else None
    im = compute_image_metrics(np.abs(recon_cplx), ref_np, brain_mask_for(ref_np))
    hk = heldout_kspace_error(sample, recon_ch, device, heldout_mask=held)
    diverged = bool(not np.isfinite(recon_cplx).all())
    del rec, mv_op, y_views, prior, lat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"heldout_kspace_err": hk, "ssim": im.get("ssim", np.nan),
            "runtime": runtime, "diverged": diverged}


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
        subset.append(srows[len(srows) // 2])
    return subset, os.path.dirname(manifest)


TIER_GRIDS = {
    "sparse": [3.0, 10.0, 30.0, 100.0],
    "mid": [3.0, 10.0, 30.0, 100.0],
    "dense": [3.0, 10.0, 30.0, 100.0, 300.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True, help="validation mask manifest")
    ap.add_argument("--n-subjects", type=int, default=6)
    ap.add_argument("--tiers", default="sparse,mid,dense")
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--likelihood", default="DPS")
    ap.add_argument("--out-tag", default="", help="suffix for output files (parallel-safe)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = S.load_network(cfg["checkpoints"]["original"], device)
    subset, mask_root = select_subset(args.manifest, args.n_subjects)

    rows, selected = [], {}
    for tier in args.tiers.split(","):
        grid = TIER_GRIDS[tier]
        tier_rows = []
        for l_ss in grid:
            hk, ss, rt, div = [], [], [], 0
            for r in subset:
                mb = torch.load(os.path.join(mask_root, r["split"], r["mask_file"]),
                                map_location="cpu", weights_only=False)
                sample = torch.load(mb["processed_path"], map_location="cpu", weights_only=False)
                # deterministic per-(subject,slice) rng so the dense holdout mask is
                # fixed across l_ss values (fair comparison) and reproducible across
                # runs (hashlib, not the per-process-salted builtin hash()).
                import hashlib
                h = hashlib.sha256(f"{r['subject_id']}_{r['slice_id']}".encode()).hexdigest()
                rng = np.random.RandomState(int(h[:8], 16))
                m = run_one(net, sample, mb, tier, device, cfg, l_ss, args.num_steps,
                            args.likelihood, rng)
                hk.append(m["heldout_kspace_err"]); ss.append(m["ssim"])
                rt.append(m["runtime"]); div += int(m["diverged"])
            row = {"tier": tier, "l_ss": l_ss, "likelihood": args.likelihood,
                   "num_steps": args.num_steps, "heldout_kspace_err": float(np.mean(hk)),
                   "ssim": float(np.nanmean(ss)), "runtime": float(np.mean(rt)),
                   "diverged": div, "n": len(subset)}
            rows.append(row); tier_rows.append(row)
            print(f"[{tier} l_ss={l_ss}] heldoutK={row['heldout_kspace_err']:.4f} "
                  f"ssim={row['ssim']:.4f} div={div}", flush=True)
        valid = [r for r in tier_rows if r["diverged"] == 0] or tier_rows
        best = sorted(valid, key=lambda r: (round(r["heldout_kspace_err"], 4),
                                            -round(r["ssim"], 4), r["runtime"]))[0]
        selected[tier] = best["l_ss"]
        print(f"  -> tier '{tier}' selected l_ss={best['l_ss']}", flush=True)

    # write grid csv
    grid_csv = os.path.join(ROOT, "tables", "project", f"lss_tier_grid{args.out_tag}.csv")
    os.makedirs(os.path.dirname(grid_csv), exist_ok=True)
    with open(grid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # write selected l_ss per tier
    out_yaml = os.path.join(ROOT, "configs", "project", f"l_ss_by_tier{args.out_tag}.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"l_ss_by_tier": selected, "likelihood_type": args.likelihood,
                        "num_steps": args.num_steps,
                        "note": "sparse<=25%, mid=44%, dense>=80% (tuned via 80%-holdout proxy); "
                                "held-out k-space error primary, val subset, test untouched"},
                       f, sort_keys=False)
    print(f"\nSelected per-tier l_ss: {selected}")
    print(f"Wrote {out_yaml} and {grid_csv}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for tier in selected:
            sub = sorted([r for r in rows if r["tier"] == tier], key=lambda r: r["l_ss"])
            axes[0].plot([r["l_ss"] for r in sub], [r["heldout_kspace_err"] for r in sub],
                         "o-", label=f"{tier} (sel {selected[tier]:g})")
            axes[1].plot([r["l_ss"] for r in sub], [r["ssim"] for r in sub], "o-", label=tier)
        for ax, yl in zip(axes, ["held-out k-space error (primary)", "validation SSIM"]):
            ax.set_xscale("log"); ax.set_xlabel("l_ss"); ax.set_ylabel(yl); ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Per-tier l_ss tuning on validation (n={len(subset)} subjects, test untouched)",
                     fontweight="bold")
        fig.tight_layout()
        figp = os.path.join(ROOT, "figures", "project", f"lss_tier_tuning{args.out_tag}.png")
        fig.savefig(figp, dpi=120, bbox_inches="tight")
        print(f"Wrote {figp}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
