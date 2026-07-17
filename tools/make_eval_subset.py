#!/usr/bin/env python
"""Build the final test evaluation subset: all subjects x N central slices.

Bounds the final experiment compute while keeping every eligible test subject
(subject-level bootstrap uses subjects as units)."""

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-subjects", type=int, default=1000)
    ap.add_argument("--n-slices", type=int, default=3)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject_id"], []).append(r)
    subjects = sorted(by_subj.keys())[: args.n_subjects]
    out = []
    for s in subjects:
        srows = sorted(by_subj[s], key=lambda r: int(r["slice_id"]))
        mid = len(srows) // 2
        half = args.n_slices // 2
        lo = max(0, mid - half)
        picked = srows[lo: lo + args.n_slices]
        out.extend(picked)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"Wrote {args.output}: {len(subjects)} subjects x <= {args.n_slices} slices = {len(out)} rows")


if __name__ == "__main__":
    main()
