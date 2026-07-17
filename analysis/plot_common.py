"""Shared helpers for the MVP figures."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from utils.multiview_mri import MRIViewOperator

# Canonical method identity/order/labels live in analysis.experiments -- the single
# source of truth shared with the report builder. Re-exported here so the existing
# figure scripts keep working.
from analysis.experiments import (  # noqa: F401
    EXPERIMENTS, METHOD_ORDER, METHOD_LABELS, MAIN_ORDER, QUAD_ORDER,
    BUDGET_COEFFS, DESIGN_COLORS, design_of, methods_for,
)


def fftmod(x):
    x = x.clone()
    x[..., ::2, :] *= -1
    x[..., :, ::2] *= -1
    return x


def load_results(roots):
    """roots: comma-separated dirs. Returns dict[(method,subject,slice,seed)] -> result."""
    files = []
    for r in roots.split(","):
        files += sorted(glob.glob(os.path.join(r, "*.pt")))
    res = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        key = (d.get("method"), d.get("subject_id"), int(d.get("slice_id")), d.get("seed", 0))
        res[key] = d
    return res


def zero_filled_mag(mask_path, condition="single_r4"):
    """Zero-filled (adjoint) magnitude of the given condition's view 0."""
    b = torch.load(mask_path, map_location="cpu", weights_only=False)
    s = torch.load(b["processed_path"], map_location="cpu", weights_only=False)
    ksp = fftmod(s["ksp_views"][0:1])
    maps = fftmod(s["s_maps_views"][0:1])
    m = b["conditions"][condition][0:1]
    op = MRIViewOperator(mask=m, maps=maps[0])
    zf = op.adjoint(m * ksp)[0, 0].abs().numpy()
    return zf


def recon_mag(result):
    r = result.get("reconstruction")
    if r is None:
        return None
    return np.abs(np.asarray(r))


def get_reference(result):
    ref = result.get("reference")
    return None if ref is None else np.asarray(ref)
