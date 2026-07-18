#!/usr/bin/env python
"""Multi-view Ambient Diffusion Posterior Sampling for low-field MRI (Stage 1).

Keeps the pretrained Ambient denoiser unchanged and introduces multi-view
information through a normalized, noise-weighted multi-view likelihood
(utils.multiview_mri.MultiViewMRI).  All four experiment conditions are handled
through one operator abstraction:

    single_r4  -> V=1 (view 0, R=4 mask)
    merge_fixed-> V=1 (analytic noise-weighted merge of two R=8 views)
    joint_fixed-> V=2 (two R=8 views, joint likelihood)
    joint_extra-> V=2 (two R=4 views, joint likelihood)

The Ambient prior input is always the view-0 adjoint plus its artificial
``create_masks`` corruption channel (4 channels), preserving the pretrained
input distribution.  The measurements affect only the likelihood.

Nothing here is hard-coded to a specific matrix size or coil count; all new
tensors take the device/dtype/shape of their inputs.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnnlib
from torch_utils.ambient_diffusion import create_masks, nrmse_np, psnr
from skimage.metrics import structural_similarity as ssim
from utils.multiview_mri import (MRIViewOperator, MultiViewMRI, channels_to_complex,
                                 complex_to_channels, noise_weighted_merge)
from utils.checkpoint_arch import build_network_from_options, load_training_options


# ---------------------------------------------------------------------------
def fftmod(x):
    x = x.clone()
    x[..., ::2, :] *= -1
    x[..., :, ::2] *= -1
    return x


def load_network(checkpoint_dir, device, img_channels_image=2):
    """Reconstruct the architecture from training_options.json and strict-load EMA."""
    opts = load_training_options(os.path.join(checkpoint_dir, "training_options.json"))
    net, _, _ = build_network_from_options(opts, img_channels_image=img_channels_image)
    with dnnlib.util.open_url(os.path.join(checkpoint_dir, "network-snapshot.pkl")) as f:
        data = pickle.load(f)
    ema = data["ema"]
    sd = OrderedDict({k.replace("_orig_mod.", ""): v for k, v in ema.items()})
    missing, unexpected = net.load_state_dict(sd, strict=False)
    trainable = [n for n, _ in net.named_parameters()]
    miss_trainable = [m for m in missing if m in trainable]
    if miss_trainable:
        raise RuntimeError(f"strict load failed: {len(miss_trainable)} trainable tensors missing")
    # Keep the denoiser in fp16 (as trained / as the original repo runs it) to
    # bound memory on 11 GB GPUs; the likelihood/guidance FFTs are computed in
    # fp64 (the sampler casts `denoised` to float64), so accuracy is preserved.
    net.eval().requires_grad_(False).to(device)
    return net


# Which stored mask set each condition draws from. ``merge_extra`` reuses the
# joint_extra masks (two identical R=4 views = a second full repetition) but
# combines them analytically instead of keeping them as separate views.
_MASK_SOURCE = {
    "single_r4": "single_r4",
    "merge_fixed": "merge_fixed",
    "joint_fixed": "joint_fixed",
    "joint_extra": "joint_extra",
    "merge_extra": "joint_extra",
    # 4 x R=4 = one full k-space measurement's budget, spent as four cheap repeats
    "joint_quad": "joint_quad",
    "merge_quad": "joint_quad",
    # one complete k-space measurement (R=1), the comparison target
    "single_full": "single_full",
    # COMPLEMENTARY extra-repetition designs: separate scans measure *different*
    # lines (shared ACS + disjoint outer), so coverage grows with the scan count
    # at the same budget (2 views -> 112/256 cols, 4 views -> 208/256).
    "joint_extra_comp": "joint_extra_comp",
    "merge_extra_comp": "joint_extra_comp",
    "joint_quad_comp": "joint_quad_comp",
    "merge_quad_comp": "joint_quad_comp",
    # FULLY-complementary: ACS also partitioned (no overlap anywhere), so coverage
    # == lines bought (V=2 -> 50%, V=4 -> 100%) at single-repetition noise.
    "joint_extra_fullcomp": "joint_extra_fullcomp",
    "merge_extra_fullcomp": "joint_extra_fullcomp",
    "joint_quad_fullcomp": "joint_quad_fullcomp",
    "merge_quad_fullcomp": "joint_quad_fullcomp",
}
# Condition -> unique-column coverage (measured; see analysis/experiments.py).
# Used only to resolve per-tier l_ss when cfg carries `l_ss_by_tier`.
_CONDITION_COVERAGE = {
    "single_r4": 0.25, "merge_fixed": 0.188, "joint_fixed": 0.188,
    "joint_extra": 0.25, "merge_extra": 0.25, "joint_quad": 0.25, "merge_quad": 0.25,
    "joint_extra_comp": 0.438, "merge_extra_comp": 0.438,
    "joint_quad_comp": 0.812, "merge_quad_comp": 0.812,
    "joint_extra_fullcomp": 0.50, "merge_extra_fullcomp": 0.50,
    "joint_quad_fullcomp": 1.0, "merge_quad_fullcomp": 1.0,
    "single_full": 1.0,
    # two complete measurements of the same slice (NEX=2), 100% coverage each
    "dual_full": 1.0,
}

_MERGE_CONDITIONS = ("merge_fixed", "merge_extra", "merge_quad",
                     "merge_extra_comp", "merge_quad_comp",
                     "merge_extra_fullcomp", "merge_quad_fullcomp")


def build_condition_operator(sample, masks_bundle, condition, device, min_variance):
    """Return (mv_operator, y_views, prior_maps_mod) for a condition.

    ``sample`` is the processed dict; ``masks_bundle`` is the mask dict for the
    same slice.  Stored k-space/maps are natural; we apply ``fftmod`` here (the
    repo convention) before constructing operators.
    """
    ksp = sample["ksp_views"].to(device)          # [Vdata,C,H,W] complex, fully sampled
    maps = sample["s_maps_views"].to(device)      # [Vdata,C,H,W] complex (shared across views)
    noise = sample["noise_std"].to(device)        # [Vdata]
    src = _MASK_SOURCE.get(condition, condition)
    cond_masks = masks_bundle["conditions"][src].to(device)  # [V,1,H,W]

    # Use exactly as many repetitions as the condition asks for (a 4-view dataset
    # can serve 1-, 2- and 4-view conditions).
    V = cond_masks.shape[0]
    if ksp.shape[0] < V:
        raise ValueError(f"condition '{condition}' needs {V} views but the sample has "
                         f"{ksp.shape[0]}; re-preprocess with num_views>={V}")
    ksp_mod = fftmod(ksp[:V])
    maps_mod = fftmod(maps[:V])
    noise = noise[:V]

    if condition in ("single_r4", "single_full"):
        m = cond_masks[0:1]                        # [1,1,H,W]
        y = (m * ksp_mod[0:1])                     # [1,C,H,W] == [V=1,C,H,W]
        mv = MultiViewMRI(m, maps_mod[0:1], noise[0:1], min_variance)
        return mv, y, maps_mod[0:1]

    if condition in _MERGE_CONDITIONS:
        # merge_fixed: two complementary R=8 views. merge_extra: two identical R=4
        # views (a second full repetition) -> inverse-variance average = sqrt(2)
        # noise reduction at the same coverage as single_r4.
        y_masked = cond_masks * ksp_mod            # [V,C,H,W] each masked by its mask
        merged = noise_weighted_merge(y_masked, cond_masks, noise, min_variance)
        m = merged["mask"]                         # [1,1,H,W] == [V=1,1,H,W]
        y = merged["ksp"]                          # [1,C,H,W] == [V=1,C,H,W]
        eff = merged["eff_noise_std"]              # [1] == [V=1]
        mv = MultiViewMRI(m, maps_mod[0:1], eff, min_variance)
        return mv, y, maps_mod[0:1]

    # joint_fixed / joint_extra: keep both views
    y = cond_masks * ksp_mod                       # [2,C,H,W]
    mv = MultiViewMRI(cond_masks, maps_mod, noise, min_variance)
    return mv, y, maps_mod[0:1]


# ---------------------------------------------------------------------------
def edm_sigma_steps(num_steps, sigma_min, sigma_max, rho, device, net):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigma_steps = (sigma_max ** (1 / rho) + step_idx / (num_steps - 1) *
                   (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(sigma_steps), torch.zeros_like(sigma_steps[:1])])
    return t_steps


def mv_posterior_sample(net, mv_op, y_views, prior_op, corruption_mask_ch, latents,
                        l_type, l_ss, num_steps, sigma_min, sigma_max, rho,
                        S_churn, save_trajectory=False):
    """EDM/Euler posterior sampling with a multi-view likelihood guidance term."""
    device = latents.device
    t_steps = edm_sigma_steps(num_steps, sigma_min, sigma_max, rho, device, net)
    x_next = latents.to(torch.float64) * t_steps[0]
    dc_traj = []
    img_channels_out = int(net.img_channels // 2)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next.detach().requires_grad_(True)
        gamma = min(S_churn / num_steps, 2 ** 0.5 - 1) if S_churn > 0 else 0.0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).clamp_min(0).sqrt() * torch.randn_like(x_cur)

        # --- Ambient prior denoiser (unchanged 4-channel input) ---
        masked_x_hat = prior_op.adjoint(prior_op.forward(x_hat.to(torch.float32)))  # [B,1,H,W] cplx
        noisy_image = complex_to_channels(masked_x_hat)                             # [B,2,H,W]
        net_in = torch.cat([noisy_image, corruption_mask_ch.expand(noisy_image.shape[0], -1, -1, -1)], dim=1)
        denoised = net(net_in, t_hat, None).to(torch.float64)[:, :img_channels_out]

        d_cur = (x_hat - denoised) / t_hat

        # --- multi-view likelihood guidance ---
        # DPS: differentiate the data fidelity of the Tweedie estimate through the
        #      denoiser network w.r.t. the noisy state x_cur.
        # ALD: gradient of the data fidelity w.r.t. the clean estimate directly
        #      (through the measurement operator only), applied as a correction.
        if l_type == "DPS":
            dc_per_sample = mv_op.data_fidelity_per_sample(denoised, y_views)  # [B]
            dc = dc_per_sample.sum()
            (likelihood_score,) = torch.autograd.grad(dc, x_cur)
        elif l_type == "ALD":
            x_var = denoised.detach().requires_grad_(True)
            dc_per_sample = mv_op.data_fidelity_per_sample(x_var, y_views)
            dc = dc_per_sample.sum()
            (likelihood_score,) = torch.autograd.grad(dc, x_var)
        else:
            raise ValueError(l_type)
        normalizer = torch.sqrt(dc_per_sample.detach().clamp_min(1e-12))
        h = t_next - t_hat
        x_next = x_hat + h * d_cur - l_ss * likelihood_score / normalizer[:, None, None, None]

        if save_trajectory:
            dc_traj.append(float(dc_per_sample.mean().item()))
        x_next = x_next.detach()
    return x_next.detach(), dc_traj


# ---------------------------------------------------------------------------
def compute_metrics(recon_mag, ref_mag):
    if ref_mag is None:
        return {}
    ref = ref_mag.astype(np.float64); est = recon_mag.astype(np.float64)
    dr = ref.max() - ref.min()
    return {
        "nrmse": float(nrmse_np(ref, est)),
        "psnr": float(psnr(gt=ref, est=est, max_pixel=ref.max())),
        "ssim": float(ssim(ref, est, data_range=dr if dr > 0 else 1.0)),
    }


def measured_kspace_error(mv_op, recon_ch, y_views):
    """Normalized error on the measured coefficients used for the fit."""
    with torch.no_grad():
        pred = mv_op.forward(recon_ch)
        yv = y_views[None] if y_views.dim() == 4 else y_views
        num = (pred - yv).abs().pow(2).sum().sqrt()
        den = yv.abs().pow(2).sum().sqrt().clamp_min(1e-12)
    return float(num / den)


def heldout_kspace_error(sample, recon_ch, device):
    """Normalized error on FULLY-SAMPLED coefficients NOT used by any condition mask.

    Uses view 0's fully sampled k-space as ground-truth measurement; the held-out
    set is everything (a proper generalization check independent of the mask)."""
    ksp0 = fftmod(sample["ksp_views"][0:1].to(device))       # [1,C,H,W] full
    maps0 = fftmod(sample["s_maps_views"][0:1].to(device))
    full_op = MRIViewOperator(mask=torch.ones_like(ksp0[:, :1].real), maps=maps0[0])
    with torch.no_grad():
        pred_full = full_op.forward(recon_ch.to(device))
        num = (pred_full - ksp0).abs().pow(2).sum().sqrt()
        den = ksp0.abs().pow(2).sum().sqrt().clamp_min(1e-12)
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--methods", default=None, help="comma-separated method names from config")
    ap.add_argument("--condition", default=None, help="override condition")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint dir")
    ap.add_argument("--likelihood_type", default=None, choices=["DPS", "ALD"])
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--l_ss", type=float, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--save_dc_trajectory", action="store_true")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_steps = args.num_steps or cfg.get("num_steps", 100)
    l_ss = args.l_ss if args.l_ss is not None else cfg.get("l_ss", 1.0)
    l_type = args.likelihood_type or cfg.get("likelihood_type", "DPS")
    sigma_min = cfg.get("sigma_min", 0.004)
    sigma_max = cfg.get("sigma_max", 10.0)
    rho = cfg.get("rho", 7)
    S_churn = cfg.get("S_churn", 0.0)
    min_variance = cfg.get("min_noise_variance", 1e-8)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else str(cfg.get("seeds", "0")).split(","))]

    # method registry: name -> {checkpoint, condition}
    methods_cfg = cfg.get("methods", {})
    checkpoints_cfg = cfg.get("checkpoints", {})
    if args.methods:
        method_names = args.methods.split(",")
    elif methods_cfg:
        method_names = list(methods_cfg.keys())
    else:
        method_names = ["custom"]

    # per-coverage-tier l_ss (optional). If cfg carries `l_ss_by_tier`, a method's
    # guidance scale is resolved from its condition's coverage tier; otherwise the
    # single global l_ss is used for every method (original behaviour). An explicit
    # per-method `l_ss` in the config always wins. Tuned on validation only --
    # see tools/tune_lss_by_tier.py and reports/mvp_notes.md.
    l_ss_by_tier = cfg.get("l_ss_by_tier") or {}

    def tier_of(cond):
        cov = _CONDITION_COVERAGE.get(cond, 0.25)
        if cov <= 0.25:
            return "sparse"
        return "mid" if cov <= 0.60 else "dense"

    def resolve_l_ss(mdef, cond):
        if mdef.get("l_ss") is not None:
            return float(mdef["l_ss"])
        if l_ss_by_tier:
            return float(l_ss_by_tier.get(tier_of(cond), l_ss))
        return l_ss

    # resolve each method to (checkpoint_dir, condition, l_ss)
    resolved = {}
    net_cache = {}
    for mname in method_names:
        mdef = methods_cfg.get(mname, {})
        ckpt = args.checkpoint or checkpoints_cfg.get(mdef.get("checkpoint", ""), mdef.get("checkpoint"))
        cond = args.condition or mdef.get("condition")
        if ckpt is None or cond is None:
            raise ValueError(f"method {mname}: need checkpoint and condition (got {ckpt}, {cond})")
        resolved[mname] = (ckpt, cond, resolve_l_ss(mdef, cond))
        if ckpt not in net_cache:
            net_cache[ckpt] = load_network(ckpt, device)

    # load manifest, shard
    import csv
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
    if args.max_samples:
        rows = rows[: args.max_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    dist_prior_R = cfg.get("prior_training_R", 4)
    dist_prior_delta = cfg.get("prior_delta_prob", 5)
    acs_prior = cfg.get("prior_acs_lines", 20)

    n_done = 0
    for row in rows:
        mask_file = row.get("mask_file")
        masks_path = os.path.join(os.path.dirname(args.manifest),
                                  row["split"], mask_file) if "split" in row else row.get("mask_path")
        masks_bundle = torch.load(masks_path, map_location="cpu", weights_only=False)
        sample = torch.load(masks_bundle["processed_path"], map_location="cpu", weights_only=False)
        meta = sample["metadata"]
        ref_mag = sample.get("reference_mag")
        ref_np = ref_mag.numpy() if ref_mag is not None else None
        H, W = sample["ksp_views"].shape[-2:]

        for mname in method_names:
            ckpt, cond, l_ss_m = resolved[mname]
            net = net_cache[ckpt]
            mv_op, y_views, prior_maps_mod = build_condition_operator(
                sample, masks_bundle, cond, device, min_variance)

            # prior corruption mask (artificial training corruption; view-0 maps)
            corr = create_masks(dist_prior_R, dist_prior_delta, acs_prior, H, W).to(device)
            corr_mask_ch = torch.ones(1, 2, H, W, device=device, dtype=torch.float32)
            corr_mask_ch[:, 0] = corr
            prior_op = MRIViewOperator(mask=corr_mask_ch[:, 0:1], maps=prior_maps_mod[0])

            for seed in seeds:
                out_name = f"{mname}__{meta['subject_id']}_sl{meta['slice_id']:02d}_seed{seed}.pt"
                out_path = os.path.join(args.output_dir, out_name)
                if os.path.exists(out_path):
                    continue
                torch.manual_seed(seed); np.random.seed(seed)
                gen = torch.Generator(device=device).manual_seed(seed)
                latents = torch.randn(1, 2, H, W, device=device, dtype=torch.float64, generator=gen)
                t0 = time.time()
                recon, dc_traj = mv_posterior_sample(
                    net, mv_op, y_views, prior_op, corr_mask_ch, latents,
                    l_type=l_type, l_ss=l_ss_m, num_steps=num_steps,
                    sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, S_churn=S_churn,
                    save_trajectory=args.save_dc_trajectory)
                runtime = time.time() - t0
                recon_ch = recon.to(torch.float32)
                recon_cplx = channels_to_complex(recon_ch)[0, 0].cpu().numpy()
                recon_mag = np.abs(recon_cplx)
                metrics = compute_metrics(recon_mag, ref_np)
                metrics["measured_kspace_err"] = measured_kspace_error(mv_op, recon_ch, y_views)
                metrics["heldout_kspace_err"] = heldout_kspace_error(sample, recon_ch, device)

                out = {
                    "method": mname, "condition": cond, "checkpoint": ckpt,
                    "reconstruction": recon_cplx, "reference": ref_np,
                    "subject_id": meta["subject_id"], "slice_id": meta["slice_id"],
                    "seed": seed, "num_steps": num_steps, "l_ss": l_ss_m,
                    "likelihood_type": l_type, "dc_trajectory": dc_traj,
                    "runtime_seconds": runtime, "num_net_evals": num_steps,
                    "metrics": metrics, "normalization_scale": meta.get("normalization_scale"),
                    "mask_path": masks_path,
                }
                torch.save(out, out_path)
                del recon, recon_ch, latents
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                n_done += 1
                msg = f"[{mname}/{cond}] {meta['subject_id']} sl{meta['slice_id']} seed{seed} " \
                      f"t={runtime:.1f}s heldoutK={metrics['heldout_kspace_err']:.3f}"
                if "ssim" in metrics:
                    msg += f" ssim={metrics['ssim']:.3f} nrmse={metrics['nrmse']:.3f}"
                print(msg, flush=True)
    print(f"DONE shard {args.shard_id}/{args.num_shards}: {n_done} reconstructions")


if __name__ == "__main__":
    main()
