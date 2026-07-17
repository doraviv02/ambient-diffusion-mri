#!/usr/bin/env python
"""Select the fine-tuned checkpoint with the best validation held-out error.

Reads ``validation.csv`` from a training run directory (the loop writes one row
per validation pass with ``subject_mean_cross``), picks the kimg with the lowest
value, matches the nearest ``network-snapshot-*.pkl``, and materializes a
``best/`` directory with ``network-snapshot.pkl`` + ``training_options.json`` so
the inference script can load it directly.  Selection uses validation subjects
only (never the test set).
"""

import argparse
import csv
import glob
import os
import re
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="training_loop run_dir (the NNNNN-... dir)")
    ap.add_argument("--out", default=None, help="best/ output dir (default <run-dir>/../best)")
    args = ap.parse_args()

    run_dir = args.run_dir
    # if run_dir is the outdir parent, descend into the single run subdir
    if not os.path.exists(os.path.join(run_dir, "validation.csv")):
        subs = sorted(glob.glob(os.path.join(run_dir, "?????-*")))
        if subs:
            run_dir = subs[-1]

    vcsv = os.path.join(run_dir, "validation.csv")
    snaps = sorted(glob.glob(os.path.join(run_dir, "network-snapshot-*.pkl")))
    if not snaps:
        raise SystemExit(f"No snapshots in {run_dir}")

    def snap_kimg(p):
        m = re.search(r"network-snapshot-(\d+)\.pkl", p)
        return int(m.group(1)) if m else -1

    best_kimg = None
    if os.path.exists(vcsv):
        rows = list(csv.DictReader(open(vcsv)))
        rows = [r for r in rows if r.get("subject_mean_cross") not in (None, "", "nan")]
        if rows:
            best = min(rows, key=lambda r: float(r["subject_mean_cross"]))
            best_kimg = int(float(best["kimg"]))
            print(f"Best validation: kimg={best_kimg} subject_mean_cross={best['subject_mean_cross']}")
    if best_kimg is None:
        # fall back to the last snapshot
        chosen = snaps[-1]
        print("No validation rows; using last snapshot.")
    else:
        chosen = min(snaps, key=lambda p: abs(snap_kimg(p) - best_kimg))
    print(f"Chosen snapshot: {chosen} (kimg={snap_kimg(chosen)})")

    out = args.out or os.path.join(os.path.dirname(run_dir.rstrip("/")), "best")
    os.makedirs(out, exist_ok=True)
    shutil.copy2(chosen, os.path.join(out, "network-snapshot.pkl"))
    topt = os.path.join(run_dir, "training_options.json")
    if os.path.exists(topt):
        shutil.copy2(topt, os.path.join(out, "training_options.json"))
    print(f"Wrote {out}/network-snapshot.pkl and training_options.json")


if __name__ == "__main__":
    main()
