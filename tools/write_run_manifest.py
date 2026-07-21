#!/usr/bin/env python
"""Write a reproducibility manifest for a completed run."""

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
    except Exception:
        return None


def gpu_names():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
        return [l.strip() for l in out.splitlines()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--upstream-commit", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset-manifest", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--status", default="complete")
    ap.add_argument("--components", default=None, help="comma-separated component manifest paths")
    args = ap.parse_args()

    upstream = args.upstream_commit
    if upstream and os.path.exists(upstream):
        upstream = open(upstream).read().strip()
    try:
        import torch
        torch_v = torch.__version__
        cuda_v = torch.version.cuda
    except Exception:
        torch_v = cuda_v = None

    manifest = {
        "git_commit": git_commit(args.repo),
        "upstream_commit": upstream,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "config": args.config,
        "seeds": [int(s) for s in args.seeds.split(",") if s != ""],
        "hostname": socket.gethostname(),
        "gpu_names": gpu_names(),
        "torch_version": torch_v,
        "cuda_version": cuda_v,
        "timestamp": datetime.datetime.now().isoformat(),
        "status": args.status,
        "component_manifests": (args.components.split(",") if args.components else []),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
