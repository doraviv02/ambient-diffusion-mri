#!/usr/bin/env python
"""Migrate stored result files from legacy M#/MF ids to descriptive slugs.

- Deleted methods (LEGACY_IDS[...] is None) are MOVED to a backup dir (reversible),
  not hard-deleted.
- Kept methods have their stored "method" field rewritten to the slug and the file
  renamed `<slug>__<subject>_sl<slice>_seed<seed>.pt`.

Idempotent: files already named with a slug are left alone.
"""

import argparse
import glob
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiments import LEGACY_IDS, EXPERIMENTS


def migrate_dir(results_dir, backup_dir, dry_run=False):
    moved, renamed, kept = 0, 0, 0
    os.makedirs(backup_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.pt"))):
        fn = os.path.basename(path)
        legacy = fn.split("__", 1)[0]
        if legacy in EXPERIMENTS:          # already a slug
            kept += 1
            continue
        if legacy not in LEGACY_IDS:
            print(f"  ?? unknown method id in {fn}; leaving as-is")
            kept += 1
            continue
        slug = LEGACY_IDS[legacy]
        if slug is None:                   # a removed method -> back up
            if not dry_run:
                shutil.move(path, os.path.join(backup_dir, fn))
            moved += 1
            continue
        # kept: rewrite internal method field + rename file
        rest = fn.split("__", 1)[1]
        new_path = os.path.join(results_dir, f"{slug}__{rest}")
        if not dry_run:
            d = torch.load(path, map_location="cpu", weights_only=False)
            d["method"] = slug
            torch.save(d, new_path)
            if os.path.abspath(new_path) != os.path.abspath(path):
                os.remove(path)
        renamed += 1
    print(f"[{os.path.basename(results_dir)}] renamed={renamed} moved_to_backup={moved} "
          f"already_slug={kept}")
    return moved, renamed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="dir containing *.pt result files")
    ap.add_argument("--backup-root", required=True, help="where removed-method files are moved")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    migrate_dir(args.results_root, args.backup_root, args.dry_run)


if __name__ == "__main__":
    main()
