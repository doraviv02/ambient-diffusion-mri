"""Single source of truth for the MVP experiment grid.

Every method's identity (which prior, which combiner, which acquisition) and its
acquisition accounting live here so the report, the tables and the figures cannot
drift apart.  Coverage numbers are *measured* from the stored mask tensors, not
design intent -- see `tools/generate_mvp_masks.py` and the check in the notes.

Families
--------
classical : no learned prior (adjoint / L1-wavelet)
merge     : views collapsed into one measurement before sampling (analytic merge)
joint     : views kept separate, summed in the per-step data fidelity
single    : one view only
"""

from __future__ import annotations

from collections import namedtuple

Exp = namedtuple("Exp", "label prior combiner condition views lines uniq cov budget set_")

# budget tags: "1x" (64 lines), "fixed" (64, split), "2x" (128), "4x" (256), "full" (256)
EXPERIMENTS = {
    # ---- classical baselines (2-view set) --------------------------------
    "M0":  Exp("Noise-wtd adjoint",       "none",      "merge",  "merge_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    "M1":  Exp("L1-wavelet SENSE",        "none",      "merge",  "merge_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    # ---- single view -----------------------------------------------------
    "M2":  Exp("Single-view R=4",         "original",  "single", "single_r4",        1,  64,  64, 0.250, "1x",    "both"),
    # ---- fixed budget (2xR=8, split one scan's worth) --------------------
    "M3":  Exp("Merged fixed",            "original",  "merge",  "merge_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    "M4":  Exp("Joint fixed",             "original",  "joint",  "joint_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    "M5":  Exp("Joint fixed + xview-FT",  "finetuned", "joint",  "joint_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    # ---- 2x budget, DUPLICATED mask (25% coverage) -----------------------
    "M7":  Exp("Merged 2xR=4 dup",        "original",  "merge",  "merge_extra",      2, 128,  64, 0.250, "2x",    "main"),
    "M8":  Exp("Joint 2xR=4 dup",         "original",  "joint",  "joint_extra",      2, 128,  64, 0.250, "2x",    "main"),
    "M6":  Exp("Joint 2xR=4 dup + FT",    "finetuned", "joint",  "joint_extra",      2, 128,  64, 0.250, "2x",    "main"),
    # ---- 2x budget, COMPLEMENTARY (44% coverage) -------------------------
    "M15": Exp("Merged 2xR=4 COMP",       "original",  "merge",  "merge_extra_comp", 2, 128, 112, 0.438, "2x",    "main"),
    "M14": Exp("Joint 2xR=4 COMP",        "original",  "joint",  "joint_extra_comp", 2, 128, 112, 0.438, "2x",    "main"),
    "M13": Exp("Joint 2xR=4 COMP + FT",   "finetuned", "joint",  "joint_extra_comp", 2, 128, 112, 0.438, "2x",    "main"),
    # ---- 4x budget, DUPLICATED (25% coverage) ----------------------------
    "M11": Exp("Merged 4xR=4 dup",        "original",  "merge",  "merge_quad",       4, 256,  64, 0.250, "4x",    "quad"),
    "M10": Exp("Joint 4xR=4 dup",         "original",  "joint",  "joint_quad",       4, 256,  64, 0.250, "4x",    "quad"),
    "M9":  Exp("Joint 4xR=4 dup + FT",    "finetuned", "joint",  "joint_quad",       4, 256,  64, 0.250, "4x",    "quad"),
    # ---- 4x budget, COMPLEMENTARY (81% coverage) -------------------------
    "M18": Exp("Merged 4xR=4 COMP",       "original",  "merge",  "merge_quad_comp",  4, 256, 208, 0.812, "4x",    "quad"),
    "M17": Exp("Joint 4xR=4 COMP",        "original",  "joint",  "joint_quad_comp",  4, 256, 208, 0.812, "4x",    "quad"),
    "M16": Exp("Joint 4xR=4 COMP + FT",   "finetuned", "joint",  "joint_quad_comp",  4, 256, 208, 0.812, "4x",    "quad"),
    # ---- one complete measurement (the ceiling) --------------------------
    "M12": Exp("Diffusion, FULL k-space", "original",  "single", "single_full",      1, 256, 256, 1.000, "full",  "quad"),
    "MF":  Exp("Plain recon, FULL",       "none",      "single", "single_full",      1, 256, 256, 1.000, "full",  "quad"),
}

# Display order for the 2-view set: by budget, then merge -> joint -> joint+FT.
MAIN_ORDER = ["M0", "M1", "M2", "M3", "M4", "M5", "M7", "M8", "M6", "M15", "M14", "M13"]
# Display order for the 4-view set: worst -> best.
QUAD_ORDER = ["M2", "M11", "M10", "M9", "M18", "M17", "M16", "M12", "MF"]

METHOD_ORDER = MAIN_ORDER  # back-compat for existing figure scripts
METHOD_LABELS = {k: v.label for k, v in EXPERIMENTS.items()}

# Acquired-coefficient budget (columns x 256 rows), dup-ACS accounting.
BUDGET_COEFFS = {k: v.lines * 256 for k, v in EXPERIMENTS.items()}

# Colour per acquisition design -- the study's main axis.
DESIGN_COLORS = {
    "fixed": "#9e9e9e",   # split one scan
    "1x":    "#4c72b0",   # one cheap scan
    "dup":   "#dd8452",   # duplicated masks
    "comp":  "#55a868",   # complementary masks
    "full":  "#c44e52",   # one complete measurement
}


def design_of(m):
    """Coarse acquisition-design tag used for colouring."""
    e = EXPERIMENTS[m]
    if e.budget in ("full", "1x", "fixed"):
        return e.budget
    return "comp" if e.condition.endswith("_comp") else "dup"


def methods_for(set_):
    """Methods belonging to an evaluation set ('main' or 'quad')."""
    order = MAIN_ORDER if set_ == "main" else QUAD_ORDER
    return [m for m in order if EXPERIMENTS[m].set_ in (set_, "both")]
