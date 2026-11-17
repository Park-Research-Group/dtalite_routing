#!/usr/bin/env python3
"""Collect the most recent wholerunner v2 run into one zip for figures.

A run is a set of per-seed results dirs sharing a common <results-dir>
prefix (<results-dir>_seed<S>/, plus <results-dir>_multiseed/ when the
aggregate has been run), or a single <results-dir>/ holding a
summary.csv directly (single-seed run). The most recent run is the one
whose files were modified last (by mtime). It works for a run of any
size or demand level: nothing here hardcodes agent counts, levels,
MPR points, or seed numbers.

The zip holds everything needed to make the graphs:
  - the per-seed summary.csv files and the aggregate summary.csv /
    summary.txt (when present)
  - the per-test/per-level summary files (*.txt) and LinkMOE files
    (predicted + actual, per test / level / method)
  - the per-link V/C tables (*_vc.csv): included from disk when the
    run wrote them; otherwise generated here from each realized
    LinkMOE file (output_LinkMOE.csv / *actual_LinkMOE.csv) using
    congestion.py, so the zip is complete even for older runs

Agent files (*agent.csv) are NOT collected: they are large and the
figures do not need them.

Usage (run from the directory holding the results folders, or pass
--results-root):
    python collect_latest_run.py
    python collect_latest_run.py --results-root C:\\path\\to\\results_parent
    python collect_latest_run.py --list-runs
    python collect_latest_run.py --zip-name my_run.zip

The zip is named <results-dir>_latest_<YYYYMMDD-HHMM>.zip by default,
with the timestamp taken from the run's newest file (local time), and
written next to the results folders unless --out-dir is given.

Stdlib only. Read-only on the results folders (the zip is the only
thing written, plus nothing is modified in place).
"""

import os
import re
import sys
import time
import zipfile

try:
    import congestion
except ImportError:  # allow running from another directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import congestion

SEED_RE = re.compile(r"^(.*)_seed(\d+)$")
AGG_RE = re.compile(r"^(.*)_multiseed$")
TEST_DIR_RE = re.compile(r"^test[1-4]$")

# File names collected from a seed dir (relative to it). Agent files
# are excluded by collect_seed_dir below, not by this list.
KEEP_EXTENSIONS = (".csv", ".txt")


def dir_mtime(path):
    """Newest mtime of any file under path (dir mtime as fallback)."""
    newest = 0.0
    try:
        newest = os.path.getmtime(path)
    except OSError:
        return 0.0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                m = os.path.getmtime(os.path.join(dirpath, name))
            except OSError:
                continue
            if m > newest:
                newest = m
    return newest


def find_runs(root):
    """All runs under root, newest first.

    Returns a list of dicts: base, seed_dirs, agg_dir (or None),
    single (bool), mtime. base is the <results-dir> name the run
    belongs to.
    """
    runs = {}
    try:
        names = sorted(os.listdir(root))
    except OSError as e:
        print("ERROR: cannot list results root %s: %s" % (root, e),
              file=sys.stderr)
        return []
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        m = SEED_RE.match(name)
        if m:
            base = m.group(1)
            run = runs.setdefault(base, {"base": base, "seed_dirs": [],
                                         "agg_dir": None, "single": False})
            run["seed_dirs"].append(path)
            continue
        m = AGG_RE.match(name)
        if m:
            base = m.group(1)
            run = runs.setdefault(base, {"base": base, "seed_dirs": [],
                                         "agg_dir": None, "single": False})
            run["agg_dir"] = path
            continue
        if os.path.isfile(os.path.join(path, "summary.csv")):
            run = runs.setdefault(name, {"base": name, "seed_dirs": [path],
                                         "agg_dir": None, "single": True})
    out = []
    for run in runs.values():
        dirs = list(run["seed_dirs"])
        if run["agg_dir"]:
            dirs.append(run["agg_dir"])
        run["mtime"] = max((dir_mtime(d) for d in dirs), default=0.0)
        run["seed_dirs"].sort()
        out.append(run)
    out.sort(key=lambda r: (r["mtime"], r["base"]), reverse=True)
    return out


def is_agent_file(name):
    return name.lower().endswith("agent.csv")


def collect_seed_dir(z, seed_dir, arc_prefix):
    """Add one seed dir's graph inputs to the zip.

    Returns (added, generated_vc) where added is a list of arc names
    written and generated_vc a list of (source, arc name) for V/C
    tables computed on the fly. Realized LinkMOE files are remembered
    so their V/C tables can be generated afterwards.
    """
    added = []
    realized = []  # (disk path, arc name) of realized LinkMOE files
    vc_on_disk = set()  # directories (relative) that already have a *_vc.csv
    for dirpath, _dirnames, filenames in os.walk(seed_dir):
        rel_dir = os.path.relpath(dirpath, seed_dir)
        arc_dir = arc_prefix if rel_dir == "." else \
            arc_prefix + "/" + rel_dir.replace(os.sep, "/")
        for name in sorted(filenames):
            if not name.lower().endswith(KEEP_EXTENSIONS):
                continue
            if is_agent_file(name):
                continue
            src = os.path.join(dirpath, name)
            arc = arc_dir + "/" + name
            z.write(src, arc)
            added.append(arc)
            if name.lower().endswith("_vc.csv"):
                vc_on_disk.add(rel_dir)
            if name == "output_LinkMOE.csv" or \
                    name.endswith("actual_LinkMOE.csv"):
                realized.append((src, arc, rel_dir))
    generated = []
    for src, arc, rel_dir in realized:
        if rel_dir in vc_on_disk:
            continue  # this directory already ships a V/C table
        stem = os.path.splitext(os.path.basename(arc))[0]
        vc_arc = os.path.dirname(arc) + "/" + stem + "_vc.csv"
        try:
            res = congestion.compute_congestion(src)
        except (OSError, ValueError) as e:
            print("WARNING: could not compute V/C table for %s: %s"
                  % (src, e), file=sys.stderr)
            continue
        if res["n_links"] == 0:
            continue
        import io
        buf = io.StringIO()
        import csv
        w = csv.writer(buf, lineterminator="\r\n")
        w.writerow(["from_node_id", "to_node_id", "total_link_volume",
                    "link_capacity_in_vhc_per_hour", "vc"])
        for t in res["table"]:
            w.writerow([t["from_node"], t["to_node"],
                        "%.4f" % t["volume"], "%.4f" % t["capacity"],
                        "%.4f" % t["vc"]])
        z.writestr(vc_arc, buf.getvalue())
        added.append(vc_arc)
        generated.append((arc, vc_arc))
    return added, generated


def collect(run, out_dir):
    """Write the zip for one run. Returns (zip_path, added, generated)."""
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(run["mtime"]))
    zip_name = "%s_latest_%s.zip" % (run["base"], stamp)
    zip_path = os.path.join(out_dir, zip_name)
    added = []
    generated = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for seed_dir in run["seed_dirs"]:
            arc_prefix = os.path.basename(seed_dir)
            a, g = collect_seed_dir(z, seed_dir, arc_prefix)
            added += a
            generated += g
        if run["agg_dir"]:
            arc_prefix = os.path.basename(run["agg_dir"])
            for name in sorted(os.listdir(run["agg_dir"])):
                src = os.path.join(run["agg_dir"], name)
                if os.path.isfile(src) and \
                        name.lower().endswith(KEEP_EXTENSIONS):
                    z.write(src, arc_prefix + "/" + name)
                    added.append(arc_prefix + "/" + name)
    return zip_path, added, generated


def main(argv):
    root = os.getcwd()
    out_dir = None
    zip_name = None
    list_runs = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--results-root":
            if i + 1 >= len(argv):
                print("ERROR: --results-root needs a value", file=sys.stderr)
                return 2
            root = argv[i + 1]
            i += 2
            continue
        if tok == "--out-dir":
            if i + 1 >= len(argv):
                print("ERROR: --out-dir needs a value", file=sys.stderr)
                return 2
            out_dir = argv[i + 1]
            i += 2
            continue
        if tok == "--zip-name":
            if i + 1 >= len(argv):
                print("ERROR: --zip-name needs a value", file=sys.stderr)
                return 2
            zip_name = argv[i + 1]
            i += 2
            continue
        if tok == "--list-runs":
            list_runs = True
            i += 1
            continue
        if tok in ("-h", "--help"):
            print("Usage: python collect_latest_run.py "
                  "[--results-root <dir>] [--out-dir <dir>] "
                  "[--zip-name <name>] [--list-runs]")
            return 0
        print("ERROR: unknown argument: %s" % tok, file=sys.stderr)
        return 2
    if not os.path.isdir(root):
        print("ERROR: results root not found: %s" % root, file=sys.stderr)
        return 1
    runs = find_runs(root)
    if not runs:
        print("ERROR: no runs found under %s (looked for "
              "<results-dir>_seed<S>/, <results-dir>_multiseed/, or a "
              "directory holding summary.csv)" % root, file=sys.stderr)
        return 1
    if list_runs:
        print("Runs under %s (newest first):" % root)
        for run in runs:
            seeds = [os.path.basename(d) for d in run["seed_dirs"]]
            print("  %-28s %s  seeds: %s%s" % (
                run["base"],
                time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(run["mtime"])),
                ", ".join(seeds),
                "  [aggregate present]" if run["agg_dir"] else
                "  [no aggregate yet]"))
        return 0
    run = runs[0]
    if out_dir is None:
        out_dir = root
    os.makedirs(out_dir, exist_ok=True)
    zip_path, added, generated = collect(run, out_dir)
    if zip_name is not None:
        custom = os.path.join(out_dir, zip_name)
        os.replace(zip_path, custom)
        zip_path = custom
    seeds = [os.path.basename(d) for d in run["seed_dirs"]]
    print("Most recent run: %s (modified %s)"
          % (run["base"],
             time.strftime("%Y-%m-%d %H:%M", time.localtime(run["mtime"]))))
    print("Seeds: %s%s" % (", ".join(seeds),
                            "" if run["single"] else
                            "  (aggregate: %s)"
                            % ("present" if run["agg_dir"]
                               else "MISSING - run aggregate_v2.py first")))
    print("Wrote %s (%d files, %d bytes)"
          % (zip_path, len(added), os.path.getsize(zip_path)))
    if generated:
        print("V/C tables generated from realized LinkMOE files (%d):"
              % len(generated))
        for src_arc, vc_arc in generated:
            print("  %s -> %s" % (src_arc, vc_arc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
