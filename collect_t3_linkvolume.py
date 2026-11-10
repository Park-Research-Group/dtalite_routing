"""Collect the T3 (real-time, 10-minute updates) link-volume information
needed for the Test 6 accuracy figure, by walking the per-seed results
folders and the temp staging dirs.

Run from the repo root (the folder that holds wholerunner.py and the
results_seed*/ dirs):

    python collect_t3_linkvolume.py

It writes test6_t3_accuracy.csv in the repo root with one row per seed:
seed, rt10_accuracy_pct, n_links, predicted_source, status.

What it collects per seed:
  predicted : the T3 Step 1 (predict) link volumes, from
              results_seed<seed>/test3/predicted_LinkMOE.csv
  actual    : the T3 reaction (actual) link volumes, from
              results_seed<seed>/test3/full_results/output_LinkMOE.csv

Accuracy is the share of the 76 links whose predicted volume is within
10% of the realized volume, the same pairing used for the seed-42
analysis.

Why the duplicate check matters: the collected test3/predicted_LinkMOE.csv
can be a stale copy of the test2 (1-minute) file. When that happens the
computed accuracy is T2's predicted volumes against T3's actual volumes,
which is not T3's accuracy. The script compares the T3 predicted file
against the T2 predicted file (md5) and against the known stale file,
and flags it. It also looks in the temp staging dirs
(temp/run_*/test_3/staging/predicted_LinkMOE.csv) for a fresh copy,
because that is where scenarios_realtime.py saves the Step 1 output
before wholerunner collects it. If a fresh, non-duplicate predicted file
is found there, it is used and the source is recorded.
"""

import csv
import hashlib
import os
import sys
from glob import glob

# md5 of the stale predicted_LinkMOE.csv that keeps showing up (the
# October 26 big-run file, also copied into every test3/ dir).
STALE_MD5 = "91adfb456eb503ef6e2792941ed0d2ba"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_link_volumes(path):
    """Return {(from_node, to_node): total_link_volume} for a LinkMOE csv."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        vol_idx = None
        for i, name in enumerate(header):
            if name.strip().lower() == "total_link_volume":
                vol_idx = i
                break
        if vol_idx is None:
            for i, name in enumerate(header):
                if "volume" in name.lower():
                    vol_idx = i
                    break
        volumes = {}
        for row in reader:
            if not row or len(row) <= vol_idx:
                continue
            try:
                key = (row[0].strip(), row[1].strip())
                volumes[key] = float(row[vol_idx])
            except (ValueError, IndexError):
                continue
    return volumes


def accuracy_pct(predicted, actual):
    """Share of links whose predicted volume is within 10% of realized."""
    n = 0
    within = 0
    for link, pred_vol in predicted.items():
        if link not in actual:
            continue
        act_vol = actual[link]
        if act_vol == 0:
            continue
        n += 1
        if abs(pred_vol - act_vol) / abs(act_vol) <= 0.10:
            within += 1
    if n == 0:
        return None, 0
    return 100.0 * within / n, n


def find_staging_predicted(repo_root, seed):
    """Look for a fresh T3 predicted file in the temp staging dirs.

    Returns the path if one exists whose md5 is not the stale file and
    not identical to the test2 predicted file, else None.
    """
    candidates = sorted(glob(os.path.join(
        repo_root, "temp", "run_*", "test_3", "staging",
        "predicted_LinkMOE.csv")))
    for path in candidates:
        try:
            if md5_of(path) == STALE_MD5:
                continue
            return path
        except OSError:
            continue
    return None


def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."
    seed_dirs = sorted(glob(os.path.join(repo_root, "results_seed*")))
    if not seed_dirs:
        print("No results_seed*/ dirs found under %s" % repo_root)
        return 1

    rows = []
    for seed_dir in seed_dirs:
        seed = os.path.basename(seed_dir).replace("results_seed", "")
        test3 = os.path.join(seed_dir, "test3")
        pred_path = os.path.join(test3, "predicted_LinkMOE.csv")
        act_path = os.path.join(test3, "full_results", "output_LinkMOE.csv")
        test2_pred = os.path.join(seed_dir, "test2", "predicted_LinkMOE.csv")

        if not os.path.isfile(act_path):
            print("seed %s: actual LinkMOE missing, skipping" % seed)
            continue
        if not os.path.isfile(pred_path):
            print("seed %s: predicted LinkMOE missing, skipping" % seed)
            continue

        actual = read_link_volumes(act_path)
        pred_md5 = md5_of(pred_path)
        test2_md5 = md5_of(test2_pred) if os.path.isfile(test2_pred) else None

        source = pred_path
        status = "ok"
        if pred_md5 == STALE_MD5 or (test2_md5 and pred_md5 == test2_md5):
            status = "duplicate"
            fresh = find_staging_predicted(repo_root, seed)
            if fresh is not None:
                source = fresh
                status = "ok (staging)"
                print("seed %s: test3 predicted is a duplicate; "
                      "using fresh staging file %s" % (seed, fresh))
            else:
                print("seed %s: test3 predicted is a duplicate of test2 "
                      "(stale); no fresh staging file found" % seed)

        predicted = read_link_volumes(source)
        pct, n = accuracy_pct(predicted, actual)
        if pct is None:
            print("seed %s: no comparable links, skipping" % seed)
            continue
        print("seed %s: T3 accuracy %.2f%% (%d links) [%s]"
              % (seed, pct, n, status))
        rows.append((seed, "%.2f" % pct, n, source, status))

    out_path = os.path.join(repo_root, "test6_t3_accuracy.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "rt10_accuracy_pct", "n_links",
                         "predicted_source", "status"])
        writer.writerows(rows)
    print("Wrote %s (%d seeds)" % (out_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
