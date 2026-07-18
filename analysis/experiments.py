"""Single source of truth for the MVP experiment grid.

Every method's identity (which prior, which combiner, which acquisition) and its
acquisition accounting live here so the report, the tables and the figures cannot
drift apart.  Coverage numbers are *measured* from the stored mask tensors, not
design intent -- see `tools/generate_mvp_masks.py`.

Naming
------
Methods use descriptive slugs (single_r4, comp2_ft, full_plain, ...) instead of
opaque M-numbers.  `LEGACY_IDS` maps the old M#/MF ids to the current slugs for
migrating stored result files and for reading historical notes.

The redundant "joint likelihood, original prior" methods (old M4/M8/M10/M14/M17)
were removed: for aligned identical-mask views the joint likelihood is provably
equal to the analytic noise-weighted merge, and they showed no dramatic
difference from it.  Each acquisition family therefore keeps one analytic-combiner
representative (`*_merge`) plus the cross-view fine-tuned method (`*_ft`), which is
the project's contribution.

Families
--------
classical : no learned prior (adjoint / L1-wavelet)
merge     : views collapsed into one measurement before sampling (analytic merge)
ft        : cross-view fine-tuned prior, joint multi-view likelihood (contribution)
single    : one view only
plain     : direct least-squares recon of a complete measurement
"""

from __future__ import annotations

from collections import namedtuple

Exp = namedtuple("Exp", "label prior combiner condition views lines uniq cov budget set_")

# budget tags: "1x" (64 lines), "fixed" (64, split into 2 views), "2x" (128),
# "4x" (256), "full" (256).
EXPERIMENTS = {
    # ---- classical baselines (2-view set) --------------------------------
    "classical_adjoint": Exp("Classical adjoint",        "none",      "merge",  "merge_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    "classical_l1wav":   Exp("Classical L1-wavelet",     "none",      "merge",  "merge_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    # ---- single view -----------------------------------------------------
    "single_r4":         Exp("Single scan (R=4)",        "original",  "single", "single_r4",        1,  64,  64, 0.250, "1x",    "both"),
    # ---- fixed budget (2xR=8, one scan's budget split across two views) ---
    "fixed_split_merge": Exp("Fixed budget, merged",     "original",  "merge",  "merge_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    "fixed_split_ft":    Exp("Fixed budget, xview-FT",   "finetuned", "ft",     "joint_fixed",      2,  64,  48, 0.188, "fixed", "main"),
    # ---- 2x budget, DUPLICATED mask (25% coverage) -----------------------
    "dup2_merge":        Exp("2x duplicated, merged",    "original",  "merge",  "merge_extra",      2, 128,  64, 0.250, "2x",    "main"),
    "dup2_ft":           Exp("2x duplicated, xview-FT",  "finetuned", "ft",     "joint_extra",      2, 128,  64, 0.250, "2x",    "main"),
    # ---- 2x budget, COMPLEMENTARY (44% coverage) -------------------------
    "comp2_merge":       Exp("2x complementary, merged", "original",  "merge",  "merge_extra_comp", 2, 128, 112, 0.438, "2x",    "main"),
    "comp2_ft":          Exp("2x complementary, xview-FT","finetuned","ft",     "joint_extra_comp", 2, 128, 112, 0.438, "2x",    "main"),
    # ---- 4x budget, DUPLICATED (25% coverage) ----------------------------
    "dup4_merge":        Exp("4x duplicated, merged",    "original",  "merge",  "merge_quad",       4, 256,  64, 0.250, "4x",    "quad"),
    "dup4_ft":           Exp("4x duplicated, xview-FT",  "finetuned", "ft",     "joint_quad",       4, 256,  64, 0.250, "4x",    "quad"),
    # ---- 4x budget, COMPLEMENTARY (81% coverage) -------------------------
    "comp4_merge":       Exp("4x complementary, merged", "original",  "merge",  "merge_quad_comp",  4, 256, 208, 0.812, "4x",    "quad"),
    "comp4_ft":          Exp("4x complementary, xview-FT","finetuned","ft",     "joint_quad_comp",  4, 256, 208, 0.812, "4x",    "quad"),
    # ---- one complete measurement (the ceiling) --------------------------
    "full_diffusion":    Exp("Full scan, diffusion",     "original",  "single", "single_full",      1, 256, 256, 1.000, "full",  "quad"),
    "full_plain":        Exp("Full scan, plain recon",   "none",      "plain",  "single_full",      1, 256, 256, 1.000, "full",  "quad"),
}

# Old M#/MF id -> current slug (for migrating result files and reading old notes).
# The five removed "joint, original prior" methods map to None.
LEGACY_IDS = {
    "M0": "classical_adjoint", "M1": "classical_l1wav", "M2": "single_r4",
    "M3": "fixed_split_merge", "M4": None,               "M5": "fixed_split_ft",
    "M6": "dup2_ft",           "M7": "dup2_merge",       "M8": None,
    "M9": "dup4_ft",           "M10": None,              "M11": "dup4_merge",
    "M12": "full_diffusion",   "M13": "comp2_ft",        "M14": None,
    "M15": "comp2_merge",      "M16": "comp4_ft",        "M17": None,
    "M18": "comp4_merge",      "MF": "full_plain",
}

# Display order for the 2-view set: by budget, then merge -> fine-tuned.
MAIN_ORDER = ["classical_adjoint", "classical_l1wav", "single_r4",
              "fixed_split_merge", "fixed_split_ft",
              "dup2_merge", "dup2_ft", "comp2_merge", "comp2_ft"]
# Display order for the 4-view set: worst -> best.
QUAD_ORDER = ["single_r4", "dup4_merge", "dup4_ft", "comp4_merge", "comp4_ft",
              "full_diffusion", "full_plain"]

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
