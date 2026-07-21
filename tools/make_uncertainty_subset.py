#!/usr/bin/env python
"""Select the uncertainty subset: 10 test subjects, one central slice each,
chosen deterministically BEFORE inspecting method outcomes."""

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="test mask manifest")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-subjects", type=int, default=10)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject_id"], []).append(r)
    subjects = sorted(by_subj.keys())[: args.n_subjects]
    out_rows = []
    for s in subjects:
        srows = sorted(by_subj[s], key=lambda r: int(r["slice_id"]))
        out_rows.append(srows[len(srows) // 2])  # central slice

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"Wrote {args.output} with {len(out_rows)} subjects: {subjects}")


if __name__ == "__main__":
    main()
