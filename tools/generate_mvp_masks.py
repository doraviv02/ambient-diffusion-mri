#!/usr/bin/env python
"""Generate deterministic experiment masks for the four MVP conditions (Section 10).

Conditions (Cartesian column undersampling, fully-sampled central ACS):
  single_r4  : view 0 only, one R=4 mask.
  merge_fixed: two ~R=8 complementary masks (noise-weighted merge downstream).
  joint_fixed: the SAME two R=8 masks kept as separate views (joint likelihood).
  joint_extra: both views use the same R=4 mask (an extra full repetition).

Fixed-budget guarantee: acquired columns of merge/joint_fixed (counting the
duplicated ACS in both views) equal single_r4 within ``budget_tolerance``.

Masks are deterministic per subject/slice (seed = global seed + subject + slice)
and never depend on image content.  Writes one mask file per processed sample
plus a manifest and ``figures/mvp/mask_design.png``.
"""

import argparse
import csv
import glob
import hashlib
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def subject_slice_seed(global_seed, subject_id, slice_id):
    h = hashlib.sha256(f"{subject_id}_{slice_id}".encode()).hexdigest()
    return (global_seed + int(h[:8], 16)) % (2 ** 31)


def acs_columns(width, acs_lines):
    c0 = (width - acs_lines) // 2
    return np.arange(c0, c0 + acs_lines)


def make_r_mask(width, R, acs_lines, rng):
    """Single variable-density-uniform R-fold column mask (ACS + uniform outer)."""
    acs = acs_columns(width, acs_lines)
    n_total = int(round(width / R))
    n_outer = max(n_total - len(acs), 0)
    outer_pool = np.setdiff1d(np.arange(width), acs)
    chosen = rng.choice(outer_pool, size=min(n_outer, len(outer_pool)), replace=False)
    cols = np.union1d(acs, chosen)
    return cols


def make_complementary_r8(width, R_each, acs_lines, rng):
    """Two R_each masks sharing ACS with disjoint outer columns."""
    acs = acs_columns(width, acs_lines)
    n_total_each = int(round(width / R_each))
    n_outer_each = max(n_total_each - len(acs), 0)
    outer_pool = np.setdiff1d(np.arange(width), acs)
    chosen = rng.choice(outer_pool, size=min(2 * n_outer_each, len(outer_pool)), replace=False)
    rng.shuffle(chosen)
    out_a = chosen[:n_outer_each]
    out_b = chosen[n_outer_each:2 * n_outer_each]
    cols_a = np.union1d(acs, out_a)
    cols_b = np.union1d(acs, out_b)
    return cols_a, cols_b, acs


def cols_to_mask(cols, H, W):
    m = np.zeros((1, H, W), dtype=np.float32)
    m[:, :, cols] = 1.0
    return m


def build_conditions(H, W, cfg, rng):
    acs_lines = cfg["acs_lines"]
    R1 = cfg["acceleration_single"]
    R8 = cfg["acceleration_fixed_each"]

    r4_cols = make_r_mask(W, R1, acs_lines, rng)
    a_cols, b_cols, acs = make_complementary_r8(W, R8, acs_lines, rng)

    single = np.stack([cols_to_mask(r4_cols, H, W)], axis=0)                 # [1,1,H,W]
    fixed = np.stack([cols_to_mask(a_cols, H, W), cols_to_mask(b_cols, H, W)], axis=0)  # [2,1,H,W]
    extra = np.stack([cols_to_mask(r4_cols, H, W), cols_to_mask(r4_cols, H, W)], axis=0)  # [2,1,H,W]
    quad = np.stack([cols_to_mask(r4_cols, H, W)] * 4, axis=0)               # [4,1,H,W]
    # one complete k-space measurement (R=1), the comparison target
    full = np.stack([np.ones((1, H, W), dtype=np.float32)], axis=0)          # [1,1,H,W]
    # TWO complete measurements of the same slice (realistic low-field NEX=2):
    # both views fully sampled -> merge == classical average (sigma/sqrt2), or joint
    # xview-FT. Deterministic (all-ones), consumes no rng, so existing conditions
    # stay bit-identical.
    dual_full = np.stack([np.ones((1, H, W), dtype=np.float32),
                          np.ones((1, H, W), dtype=np.float32)], axis=0)     # [2,1,H,W]
    acs_mask = cols_to_mask(acs, H, W)

    # --- COMPLEMENTARY extra-repetition designs -------------------------------
    # Separate scans should measure *different* lines. `*_extra`/`*_quad` above
    # duplicate one mask (the runbook's Condition D: "an additional full
    # repetition"), which re-buys the same 64 columns and caps coverage at 25%.
    # The complementary variants share only the ACS (averaged -> sigma/sqrt(V)
    # where the signal energy is) and partition the outer lines disjointly, so
    # coverage grows with the number of scans at the *same* budget.
    # NOTE: these rng draws come AFTER the ones above, so previously generated
    # conditions are bit-identical and earlier results stay valid.
    n_outer_each = max(int(round(W / R1)) - acs_lines, 0)
    pool_c = np.setdiff1d(np.arange(W), acs)
    sel_c = rng.choice(pool_c, size=min(4 * n_outer_each, len(pool_c)), replace=False)
    comp_cols = [np.union1d(acs, sel_c[i * n_outer_each:(i + 1) * n_outer_each])
                 for i in range(4)]
    extra_comp = np.stack([cols_to_mask(c, H, W) for c in comp_cols[:2]], axis=0)  # [2,1,H,W]
    quad_comp = np.stack([cols_to_mask(c, H, W) for c in comp_cols], axis=0)       # [4,1,H,W]
    union_extra_comp = int(len(np.union1d(comp_cols[0], comp_cols[1])))
    union_quad_comp = int(len(np.unique(np.concatenate(comp_cols))))

    # --- FULLY-COMPLEMENTARY designs (no overlap anywhere, split ACS) ----------
    # Unlike *_comp above (which shares the full ACS across views for a sqrt(V)
    # SNR gain in the centre), here the ACS itself is partitioned disjointly
    # (round-robin) across views, and so are the outer lines. Nothing is measured
    # twice, so coverage == lines bought (100% efficient): V=2 -> 128 cols (50%),
    # V=4 -> 256 cols (a *complete* k-space assembled from 4 cheap scans). The
    # trade-off is no averaging anywhere -> single-repetition noise per column.
    # NOTE: these rng draws come AFTER all the ones above, so every previously
    # generated condition is bit-identical and earlier results stay valid.
    def make_fully_complementary(V):
        n_per_view = int(round(W / R1))          # 64 lines/view (== R=4 budget)
        outer_perm = pool_c.copy()               # non-ACS columns
        rng.shuffle(outer_perm)
        cols, o = [], 0
        for v in range(V):
            acs_v = acs[v::V]                    # interleaved round-robin ACS split
            n_outer_v = n_per_view - len(acs_v)
            outer_v = outer_perm[o:o + n_outer_v]; o += n_outer_v
            cols.append(np.union1d(acs_v, outer_v))
        return cols
    fcomp2_cols = make_fully_complementary(2)
    fcomp4_cols = make_fully_complementary(4)
    extra_fullcomp = np.stack([cols_to_mask(c, H, W) for c in fcomp2_cols], axis=0)  # [2,1,H,W]
    quad_fullcomp = np.stack([cols_to_mask(c, H, W) for c in fcomp4_cols], axis=0)   # [4,1,H,W]
    union_extra_fullcomp = int(len(np.unique(np.concatenate(fcomp2_cols))))
    union_quad_fullcomp = int(len(np.unique(np.concatenate(fcomp4_cols))))

    # per-condition acquired-coefficient accounting (columns * H rows)
    def col_count(cols):
        return int(len(cols)) * H
    counts = {
        "single_r4_view0": col_count(r4_cols),
        "fixed_view0": col_count(a_cols),
        "fixed_view1": col_count(b_cols),
        "fixed_sum_dupACS": col_count(a_cols) + col_count(b_cols),
        "fixed_union": int(len(np.union1d(a_cols, b_cols))) * H,
        "extra_sum": 2 * col_count(r4_cols),
        "quad_sum": 4 * col_count(r4_cols),          # == a full k-space measurement
        "single_full": W * H,
        "acs_cols": int(len(acs)),
        # complementary variants: same budget, far more unique coverage
        "extra_comp_union": union_extra_comp * H,
        "quad_comp_union": union_quad_comp * H,
        # fully-complementary variants: no overlap, coverage == lines bought
        "extra_fullcomp_union": union_extra_fullcomp * H,
        "quad_fullcomp_union": union_quad_fullcomp * H,
    }
    conditions = {
        "single_r4": torch.from_numpy(single),
        "merge_fixed": torch.from_numpy(fixed),
        "joint_fixed": torch.from_numpy(fixed.copy()),
        "joint_extra": torch.from_numpy(extra),
        "joint_quad": torch.from_numpy(quad),
        "single_full": torch.from_numpy(full),
        "joint_extra_comp": torch.from_numpy(extra_comp),
        "joint_quad_comp": torch.from_numpy(quad_comp),
        "joint_extra_fullcomp": torch.from_numpy(extra_fullcomp),
        "joint_quad_fullcomp": torch.from_numpy(quad_fullcomp),
        "dual_full": torch.from_numpy(dual_full),
    }
    return conditions, torch.from_numpy(acs_mask), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True, help="processed dataset root (with train/val/test)")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.output_root, exist_ok=True)

    budget_report = {}
    example_masks = None
    for split in ("train", "val", "test"):
        split_in = os.path.join(args.data_root, split)
        if not os.path.isdir(split_in):
            continue
        split_out = os.path.join(args.output_root, split)
        os.makedirs(split_out, exist_ok=True)
        rows = []
        for pt in sorted(glob.glob(os.path.join(split_in, "*.pt"))):
            s = torch.load(pt, map_location="cpu", weights_only=False)
            meta = s["metadata"]
            H, W = s["ksp_views"].shape[-2:]
            rng = np.random.RandomState(subject_slice_seed(cfg["seed"], meta["subject_id"], meta["slice_id"]))
            conditions, acs_mask, counts = build_conditions(H, W, cfg, rng)
            out = {
                "conditions": conditions,
                "acs_mask": acs_mask,
                "counts": counts,
                "processed_path": os.path.abspath(pt),
                "metadata": meta,
            }
            fn = os.path.basename(pt)
            torch.save(out, os.path.join(split_out, fn))
            rows.append({
                "split": split, "subject_id": meta["subject_id"], "slice_id": meta["slice_id"],
                "mask_file": fn, "processed_path": os.path.abspath(pt),
                "single_r4_count": counts["single_r4_view0"],
                "fixed_sum_dupACS": counts["fixed_sum_dupACS"],
                "fixed_union": counts["fixed_union"], "extra_sum": counts["extra_sum"],
                "H": H, "W": W,
            })
            if example_masks is None:
                example_masks = (conditions, acs_mask, counts, meta)
        if rows:
            with open(os.path.join(args.output_root, f"{split}_manifest.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            # budget check on the first row
            r0 = rows[0]
            rel = abs(r0["fixed_sum_dupACS"] - r0["single_r4_count"]) / max(r0["single_r4_count"], 1)
            budget_report[split] = {
                "single_r4": r0["single_r4_count"], "fixed_sum_dupACS": r0["fixed_sum_dupACS"],
                "fixed_union": r0["fixed_union"], "extra_sum": r0["extra_sum"],
                "rel_budget_diff": rel, "within_tolerance": rel <= cfg["budget_tolerance"],
                "n_samples": len(rows),
            }
        print(f"[{split}] {len(rows)} mask files")

    print("=== budget report ===")
    import json
    print(json.dumps(budget_report, indent=2))
    with open(os.path.join(args.output_root, "budget_report.json"), "w") as f:
        json.dump(budget_report, f, indent=2)

    # visualization
    fig_path = args.figure or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "mvp", "mask_design.png")
    if example_masks is not None:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conditions, acs_mask, counts, meta = example_masks
        panels = [
            ("single_r4 v0", conditions["single_r4"][0, 0]),
            ("merge/joint_fixed v0", conditions["merge_fixed"][0, 0]),
            ("merge/joint_fixed v1", conditions["merge_fixed"][1, 0]),
            ("joint_extra v0=v1", conditions["joint_extra"][0, 0]),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        for ax, (title, m) in zip(axes, panels):
            ax.imshow(m.numpy(), cmap="gray", aspect="auto")
            ax.set_title(title, fontsize=11); ax.set_xlabel("kx (FE)"); ax.set_ylabel("ky (PE)")
        fig.suptitle(
            f"MVP mask design (subj {meta['subject_id']} sl{meta['slice_id']})  |  "
            f"single_r4={counts['single_r4_view0']}  fixed_sum(dupACS)={counts['fixed_sum_dupACS']}  "
            f"fixed_union={counts['fixed_union']}  extra_sum={counts['extra_sum']}", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=110, bbox_inches="tight")
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
