#!/usr/bin/env python
"""Select the published Ambient checkpoint by inspecting ``training_options.json``.

Do NOT choose a checkpoint directory by its name alone.  This scans every
``training_options.json`` under ``--models-root`` and returns the single
directory whose loss / corruption setting matches the requested method and
acceleration.  Prints exactly one directory on success (and a human-readable
architecture summary to stderr), or fails with an explanation.

Example::

    python tools/select_ambient_checkpoint.py \
        --models-root "$MODEL_ROOT/ambient_models" --method ambient --acceleration 4
"""

import argparse
import json
import os
import sys


def _iter_option_files(models_root):
    for root, _dirs, files in os.walk(models_root):
        if "__MACOSX" in root or ".ipynb_checkpoints" in root:
            continue
        if "training_options.json" in files and "network-snapshot.pkl" in files:
            yield root


def find_candidates(models_root, method, acceleration):
    candidates = []
    for root in _iter_option_files(models_root):
        with open(os.path.join(root, "training_options.json")) as f:
            opts = json.load(f)
        loss = opts.get("loss_kwargs", {}).get("class_name", "")
        is_ambient = "Ambient" in loss
        corr = opts.get("dataset_kwargs", {}).get("corruption_probability", None)
        if method == "ambient" and not is_ambient:
            continue
        if method == "edm" and is_ambient:
            continue
        if acceleration is not None and corr is not None:
            if int(round(float(corr))) != int(acceleration):
                continue
        candidates.append((root, opts))
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--method", default="ambient", choices=["ambient", "edm"])
    ap.add_argument("--acceleration", type=int, default=4)
    ap.add_argument("--describe", action="store_true",
                    help="Print the architecture summary to stderr.")
    args = ap.parse_args()

    cands = find_candidates(args.models_root, args.method, args.acceleration)
    if len(cands) == 0:
        sys.stderr.write(
            f"ERROR: no checkpoint with method={args.method} acceleration={args.acceleration} "
            f"found under {args.models_root}\n")
        sys.exit(2)
    if len(cands) > 1:
        sys.stderr.write("ERROR: ambiguous selection, multiple matches:\n")
        for root, _ in cands:
            sys.stderr.write(f"  {root}\n")
        sys.exit(3)

    root, opts = cands[0]
    if args.describe:
        # Local import so the tool works without a full environment for listing.
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils.checkpoint_arch import summarize_architecture
            sys.stderr.write(summarize_architecture(opts) + "\n")
        except Exception as e:  # pragma: no cover
            sys.stderr.write(f"(could not import architecture summary: {e})\n")
    print(root)


if __name__ == "__main__":
    main()
