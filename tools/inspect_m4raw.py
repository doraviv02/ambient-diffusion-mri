#!/usr/bin/env python
"""Inspect the raw M4Raw dataset before writing the converter.

Enumerates all H5 files under the train/val/test roots, reads their keys,
k-space shape/dtype, ``reconstruction_rss`` shape, and the ISMRMRD header, then
records subject / contrast / repetition structure and cross-checks the expected
repetition counts (3 T2 for train/val, 6 for test).  Writes:

    <output-dir>/m4raw_inventory.csv
    <output-dir>/m4raw_inventory.json

Nothing here loads a training/tuning target; it only reports structure.
"""

import argparse
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import h5py
import numpy as np


def _localname(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def parse_ismrmrd_header(header_bytes):
    """Return a flat dict of a few useful ISMRMRD header fields (ns-agnostic)."""
    out = {}
    if header_bytes is None:
        return out
    try:
        if isinstance(header_bytes, bytes):
            header = header_bytes.decode("utf-8", "ignore")
        else:
            header = str(header_bytes)
        root = ET.fromstring(header)
    except Exception as e:  # pragma: no cover
        out["header_parse_error"] = str(e)
        return out

    # Collect all leaf elements by local tag name.
    values = defaultdict(list)
    for el in root.iter():
        ln = _localname(el.tag)
        if el.text and el.text.strip():
            values[ln].append(el.text.strip())

    def first(name, cast=str, default=None):
        if name in values and values[name]:
            try:
                return cast(values[name][0])
            except Exception:
                return values[name][0]
        return default

    out["field_strength_t"] = first("systemFieldStrength_T", float)
    out["receiver_channels"] = first("receiverChannels", int)
    out["acquisition_type"] = first("acquisitionType")
    out["protocol_name"] = first("protocolName")
    out["system_model"] = first("systemModel")
    out["TR"] = first("TR", float)
    out["TE"] = first("TE", float)
    out["study_uid"] = first("studyUID")
    # matrix size: the encodedSpace matrixSize has x,y,z children -> collect x/y
    out["matrix_x"] = first("x", int)
    out["matrix_y"] = first("y", int)
    # sibling repetitions listed in the header
    reps = []
    for el in root.iter():
        if _localname(el.tag) == "MeasurementID" and el.text:
            reps.append(el.text.strip())
    out["header_measurement_ids"] = sorted(set(reps))
    return out


CONTRAST_FROM_ACQ = {
    "AXT1": "T1", "AXT2": "T2", "AXFLAIR": "FLAIR",
    "T1": "T1", "T2": "T2", "FLAIR": "FLAIR",
}


def parse_subject_contrast_rep(stem, acquisition):
    """Return (subject_id, contrast, repetition_id) from filename + acquisition attr."""
    contrast = None
    if acquisition:
        contrast = CONTRAST_FROM_ACQ.get(str(acquisition).upper().strip())
    subject_id, suffix = (stem.rsplit("_", 1) + [""])[:2] if "_" in stem else (stem, "")
    # derive contrast from suffix if attr missing
    if contrast is None:
        m = re.match(r"^(T1|T2|FLAIR|F|GRE)", suffix.upper())
        if m:
            token = m.group(1)
            contrast = {"F": "FLAIR"}.get(token, token)
    # repetition: strip a leading contrast token, keep trailing digits
    rep_str = suffix
    for token in ("FLAIR", "T1", "T2", "GRE", "F"):
        if rep_str.upper().startswith(token):
            rep_str = rep_str[len(token):]
            break
    dm = re.search(r"(\d+)$", rep_str)
    rep_id = int(dm.group(1)) if dm else None
    return subject_id, contrast, rep_id


def inspect_root(root, split, rows, errors):
    if not root or not os.path.isdir(root):
        errors.append(f"[{split}] root not found: {root}")
        return
    h5_files = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".h5"):
                h5_files.append(os.path.join(dirpath, fn))
    h5_files.sort()
    for path in h5_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        row = {"split": split, "path": path, "filename": os.path.basename(path)}
        try:
            with h5py.File(path, "r") as hf:
                keys = list(hf.keys())
                row["keys"] = "|".join(keys)
                attrs = dict(hf.attrs)
                acquisition = attrs.get("acquisition")
                row["acquisition"] = acquisition
                row["attr_max"] = float(attrs["max"]) if "max" in attrs else None
                row["patient_id"] = attrs.get("patient_id")
                if "kspace" in hf:
                    ks = hf["kspace"]
                    row["kspace_shape"] = "x".join(map(str, ks.shape))
                    row["kspace_dtype"] = str(ks.dtype)
                    row["is_complex"] = np.iscomplexobj(np.empty(0, dtype=ks.dtype))
                    if ks.ndim == 4:
                        row["n_slices"], row["n_coils"] = int(ks.shape[0]), int(ks.shape[1])
                        row["H"], row["W"] = int(ks.shape[2]), int(ks.shape[3])
                else:
                    errors.append(f"[{split}] {path}: no 'kspace' key")
                if "reconstruction_rss" in hf:
                    row["rss_shape"] = "x".join(map(str, hf["reconstruction_rss"].shape))
                hdr = hf["ismrmrd_header"][()] if "ismrmrd_header" in hf else None
                meta = parse_ismrmrd_header(hdr)
        except Exception as e:
            errors.append(f"[{split}] {path}: OPEN FAILED: {e}")
            row["error"] = str(e)
            rows.append(row)
            continue

        subj, contrast, rep = parse_subject_contrast_rep(stem, acquisition)
        row["subject_id"] = subj
        row["contrast"] = contrast
        row["repetition_id"] = rep
        row["field_strength_t"] = meta.get("field_strength_t")
        row["header_channels"] = meta.get("receiver_channels")
        row["TE"] = meta.get("TE")
        row["TR"] = meta.get("TR")
        row["n_header_reps"] = len(meta.get("header_measurement_ids", []))
        rows.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root")
    ap.add_argument("--val-root")
    ap.add_argument("--test-root")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows, errors = [], []
    inspect_root(args.train_root, "train", rows, errors)
    inspect_root(args.val_root, "val", rows, errors)
    inspect_root(args.test_root, "test", rows, errors)

    # Group by split/subject/contrast to count repetitions.
    groups = defaultdict(list)
    for r in rows:
        if r.get("subject_id") and r.get("contrast"):
            groups[(r["split"], r["subject_id"], r["contrast"])].append(r.get("repetition_id"))

    # Subject overlap check across splits.
    subj_by_split = defaultdict(set)
    for r in rows:
        if r.get("subject_id"):
            subj_by_split[r["split"]].add(r["subject_id"])
    overlaps = {}
    splits = [s for s in ["train", "val", "test"] if subj_by_split.get(s)]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = subj_by_split[splits[i]] & subj_by_split[splits[j]]
            if inter:
                overlaps[f"{splits[i]}&{splits[j]}"] = sorted(inter)

    # T2 repetition-count checks.
    t2_counts = {"train": [], "val": [], "test": []}
    for (split, subj, contrast), reps in groups.items():
        if contrast == "T2":
            t2_counts.setdefault(split, []).append(len(reps))

    # coil / slice distributions
    coil_counts = defaultdict(int)
    slice_counts = defaultdict(int)
    field_strengths = defaultdict(int)
    contrasts = defaultdict(int)
    for r in rows:
        if r.get("n_coils") is not None:
            coil_counts[r["n_coils"]] += 1
        if r.get("n_slices") is not None:
            slice_counts[r["n_slices"]] += 1
        if r.get("field_strength_t") is not None:
            field_strengths[r["field_strength_t"]] += 1
        if r.get("contrast"):
            contrasts[r["contrast"]] += 1

    summary = {
        "n_files": len(rows),
        "n_errors": len(errors),
        "errors": errors,
        "contrasts": dict(contrasts),
        "coil_count_distribution": {str(k): v for k, v in coil_counts.items()},
        "slice_count_distribution": {str(k): v for k, v in slice_counts.items()},
        "field_strength_distribution": {str(k): v for k, v in field_strengths.items()},
        "subject_overlap_across_splits": overlaps,
        "n_subjects_per_split": {k: len(v) for k, v in subj_by_split.items()},
        "t2_groups_per_split": {
            k: {
                "n_subjects_with_T2": len(v),
                "rep_count_min": (min(v) if v else None),
                "rep_count_max": (max(v) if v else None),
                "rep_count_hist": {str(c): v.count(c) for c in sorted(set(v))},
            } for k, v in t2_counts.items()
        },
    }

    # Acceptance checks (report, do not crash).
    checks = {}
    checks["train_val_no_overlap"] = not any(
        k in overlaps for k in ("train&val", "train&test", "val&test"))
    checks["train_T2_reps_ge3"] = all(c >= 3 for c in t2_counts.get("train", []) or [0])
    checks["val_T2_reps_ge3"] = all(c >= 3 for c in t2_counts.get("val", []) or [0])
    checks["test_T2_reps_ge6"] = all(c >= 6 for c in t2_counts.get("test", []) or [0]) \
        if t2_counts.get("test") else None
    summary["acceptance_checks"] = checks

    # write outputs
    csv_path = os.path.join(args.output_dir, "m4raw_inventory.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path = os.path.join(args.output_dir, "m4raw_inventory.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "files": rows}, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {csv_path}\nWrote {json_path}")
    if errors:
        print(f"\n{len(errors)} errors (see json).", file=sys.stderr)


if __name__ == "__main__":
    main()
