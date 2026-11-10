#!/usr/bin/env python3
"""Collect the multi-seed summary files into one zip, from the repo root.

Run from the repo root (the directory containing wholerunner.py and the
results_seed*/ results_multiseed/ folders):

    python collect_multiseed_results.py

It finds every results_seed<seed>/ folder and results_multiseed/, pulls
out the summary files the paper needs (per-seed summaries, emissions and
sweep tables, delay summary, the multi-seed aggregate, and the LinkMOE
files used for the Test 6 accuracy figure), and writes them into
"Full Results Mutliseed.zip" in the repo root. Missing files are reported
and skipped, never fatal.
"""
import os
import sys
import zipfile
from pathlib import Path

ZIP_NAME = "Full Results Mutliseed.zip"

# Files collected from each results_seed<seed>/ folder, relative to it.
PER_SEED_FILES = [
    "manifest.txt",
    "table4_emissions.csv",
    "linkmoe_summary.txt",
    "test1/proposed_summary.txt",
    "test2/realtime_summary.txt",
    "test2/reaction_summary.txt",
    "test2/predicted_LinkMOE.csv",
    "test2/full_results/output_LinkMOE.csv",
    "test3/realtime_summary.txt",
    "test3/reaction_summary.txt",
    "test3/predicted_LinkMOE.csv",
    "test3/full_results/output_LinkMOE.csv",
    "test4/ue_summary.txt",
    "test4/reaction_summary.txt",
    "test4/predicted_LinkMOE.csv",
    "test4/full_results/output_LinkMOE.csv",
    "test1/full_results/output_LinkMOE.csv",
    "test5/mpr_summary.txt",
    "test5/mpr_table5.csv",
    "test5/mpr_table6.csv",
]

# Files collected from results_multiseed/, relative to it.
AGGREGATE_FILES = [
    "multiseed_summary.csv",
    "multiseed_summary.txt",
]


def collect(root):
    root = Path(root)
    seed_dirs = sorted(p for p in root.iterdir()
                       if p.is_dir() and p.name.startswith("results_seed"))
    agg_dir = root / "results_multiseed"
    if not seed_dirs and not agg_dir.is_dir():
        print("No results_seed*/ or results_multiseed/ folders found in %s" % root)
        print("Run this script from the repo root.")
        return None

    added, missing = [], []
    zip_path = root / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for seed_dir in seed_dirs:
            for rel in PER_SEED_FILES:
                src = seed_dir / rel
                if src.is_file():
                    z.write(src, "%s/%s" % (seed_dir.name, rel))
                    added.append("%s/%s" % (seed_dir.name, rel))
                else:
                    missing.append("%s/%s" % (seed_dir.name, rel))
        if agg_dir.is_dir():
            for rel in AGGREGATE_FILES:
                src = agg_dir / rel
                if src.is_file():
                    z.write(src, "%s/%s" % (agg_dir.name, rel))
                    added.append("%s/%s" % (agg_dir.name, rel))
                else:
                    missing.append("%s/%s" % (agg_dir.name, rel))
    return zip_path, seed_dirs, added, missing


def main():
    root = Path.cwd()
    result = collect(root)
    if result is None:
        sys.exit(1)
    zip_path, seed_dirs, added, missing = result
    print("Seeds found: %s" % ", ".join(d.name for d in seed_dirs))
    print("Wrote %s (%d files, %d bytes)"
          % (zip_path.name, len(added), zip_path.stat().st_size))
    if missing:
        print("Missing (skipped, %d):" % len(missing))
        for m in missing:
            print("  - %s" % m)
    else:
        print("No files missing.")


if __name__ == "__main__":
    main()
