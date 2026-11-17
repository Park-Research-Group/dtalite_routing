#!/usr/bin/env python3
"""Aggregate wholerunner v2 results across seeds (new Test 1-4).

Reads the per-seed master summaries produced by wholerunner v2,
    <results-dir>_seed<S>/summary.csv
with the pinned columns (in this order):
    seed, test, level, method, mpr, satisfaction, veh_hrs, delay,
    energy, co2, nox, co, hc, linkvol_acc, max_vc, p95_vc, over_cap, heavy
and writes the multiseed aggregate to <results-dir>_multiseed/:
    summary.csv   one row per (test, level, method, mpr) group: the mean
                  of each numeric metric across seeds, plus a _sd column
                  per metric (sample SD, ddof=1) and n, the number of
                  seeds that reported that group. A metric cell is blank
                  when no seed reported it.
    summary.txt   human-readable mean +/- SD report.

Mean + sample SD (ddof=1) across seeds, the same convention as the v1
multiseed_aggregate.py; with a single seed the SD is 0.0. A row or a
metric missing for some seeds is skipped for those seeds (n is smaller)
and never fails the whole aggregation.

Congestion backfill: when a per-seed row lacks max_vc / p95_vc /
over_cap / heavy, the aggregator looks for that run's realized
output_LinkMOE.csv under the seed dir (see find_realized_linkmoe for
the locations tried) and fills the missing values from it using
congestion.py, which reuses the congestion_metric.md formulas. Values
already present in summary.csv always win; use --no-backfill to turn
the fill off.

Usage:
    python aggregate_v2.py --seed-dirs results_seed42 results_seed43 --seeds 42 43
    python aggregate_v2.py --results-dir results --seeds 42 43 44
    python aggregate_v2.py --results-dir results   (discovers results_seed*/)

Stdlib only. Read-only on the per-seed results dirs.
"""

import csv
import glob
import os
import re
import statistics
import sys
import time

try:
    import congestion
except ImportError:  # allow running from another directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import congestion

# Pinned per-seed summary.csv columns, in order (design doc).
SUMMARY_COLUMNS = ["seed", "test", "level", "method", "mpr",
                   "satisfaction", "veh_hrs", "delay", "energy", "co2",
                   "nox", "co", "hc", "linkvol_acc", "max_vc", "p95_vc",
                   "over_cap", "heavy"]
NUMERIC_METRICS = SUMMARY_COLUMNS[5:]
CONGESTION_METRICS = ["max_vc", "p95_vc", "over_cap", "heavy"]

# Canonical method order for the output; anything unrecognised sorts
# after these, alphabetically.
METHOD_ORDER = ["proposed", "rt1", "rt10", "ue"]


def method_slug(method):
    """Normalise a method label to a short slug for file matching.

    Accepts the slugs themselves ("proposed", "rt1", "rt10", "ue") and
    the display names ("Proposed method", "Real-time 1-min",
    "Real-time 10-min", "UE guidance").
    """
    m = (method or "").strip().lower()
    if "propos" in m:
        return "proposed"
    if "1-min" in m or "1min" in m or "sigma=1" in m or m in ("rt1", "rt-1"):
        return "rt1"
    if "10-min" in m or "10min" in m or "sigma=10" in m or m in ("rt10", "rt-10"):
        return "rt10"
    if m.startswith("ue") or "ue guidance" in m or "ue as" in m:
        return "ue"
    return re.sub(r"[^a-z0-9]+", "_", m).strip("_") or "unknown"


def mean_sd(values):
    """Mean and sample standard deviation (ddof=1) of a list of floats.

    Returns (mean, sd). sd is 0.0 when only one value is present and
    (None, None) when the list is empty. Same as v1.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def parse_float(text):
    """Float from a summary.csv cell; None for blank / non-numeric."""
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s.lower() in ("n/a", "na", "none", "nan"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v == v else None  # NaN -> None


def num_key(text, blank_value=-1):
    """Sort key for a test/level/mpr cell: numeric when possible."""
    if text is None or str(text).strip() == "":
        return blank_value
    try:
        return int(str(text).strip())
    except ValueError:
        try:
            return float(str(text).strip())
        except ValueError:
            return str(text).strip()


def group_sort_key(group):
    test, level, method, mpr = group
    slug = method_slug(method)
    method_rank = (METHOD_ORDER.index(slug) if slug in METHOD_ORDER
                   else len(METHOD_ORDER))
    return (num_key(test), num_key(level), method_rank,
            method.lower(), num_key(mpr))


def read_seed_summary(path):
    """Read one per-seed summary.csv.

    Returns {group: {metric: value or None}} keyed by
    (test, level, method, mpr) with the key cells as written (stripped).
    Duplicate rows for a group: the last one wins, with a warning.
    Missing numeric columns are treated as all-None for that metric.
    Raises OSError / csv.Error on unreadable files.
    """
    groups = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or []) if c]
        missing = [c for c in SUMMARY_COLUMNS if c not in fieldnames]
        if missing:
            print("WARNING: %s: missing column(s): %s (treated as blank)"
                  % (path, ", ".join(missing)), file=sys.stderr)
        for row in reader:
            def cell(name):
                for key, val in row.items():
                    if key is not None and key.strip() == name:
                        return (val or "").strip()
                return ""
            test = cell("test")
            level = cell("level")
            method = cell("method")
            mpr = cell("mpr")
            if test == "" and method == "":
                continue  # blank line
            group = (test, level, method, mpr)
            if group in groups:
                print("WARNING: %s: duplicate row for test=%s level=%s "
                      "method=%s mpr=%s; last row wins"
                      % (path, test, level, method, mpr), file=sys.stderr)
            groups[group] = {m: parse_float(cell(m))
                             for m in NUMERIC_METRICS}
    return groups


def find_realized_linkmoe(seed_dir, test, level, method, mpr):
    """Locate the realized LinkMOE file for one summary row.

    Tries the conventional v2 locations first (see RESULTS_FORMAT.md),
    then falls back to a recursive search under the test (and level)
    directory for output_LinkMOE.csv / *actual_LinkMOE.csv files whose
    path mentions the method slug (and the MPR point, when given).
    Predicted files (predicted_LinkMOE.csv) are never returned: V/C
    comes from realized volumes only. Returns a path or None.
    """
    slug = method_slug(method)
    test_dir = os.path.join(seed_dir, "test%s" % test) \
        if str(test).strip().isdigit() else os.path.join(seed_dir, test)
    level_dir = test_dir
    if str(level).strip() != "":
        level_dir = os.path.join(test_dir, "level_%s" % str(level).strip())
    mpr_s = str(mpr).strip()
    bigloop = None
    if mpr_s.isdigit() and int(mpr_s) % 10 == 0:
        bigloop = int(mpr_s) // 10

    candidates = []
    if str(test).strip() == "1":
        candidates += [
            os.path.join(test_dir, slug, "output_LinkMOE.csv"),
            os.path.join(test_dir, slug, "full_results",
                         "output_LinkMOE.csv"),
        ]
        if slug == "proposed":
            candidates.append(os.path.join(
                test_dir, "full_results", "output_LinkMOE.csv"))
            # Proposed system jobs collect the bigloop-10 actual file
            # as {n}actual_LinkMOE.csv (no output_LinkMOE.csv exists
            # for them); the system job always runs --bigloops 10.
            candidates.append(os.path.join(
                test_dir, "proposed", "full_results",
                "10actual_LinkMOE.csv"))
    elif str(test).strip() == "2":
        if mpr_s != "":
            candidates += [
                os.path.join(test_dir, "mpr_%s" % mpr_s, slug,
                             "output_LinkMOE.csv"),
                os.path.join(test_dir, slug, "mpr_%s" % mpr_s,
                             "output_LinkMOE.csv"),
            ]
        if slug in ("proposed", "ue") and bigloop is not None:
            sweep = ("proposed_sweep" if slug == "proposed"
                     else "ue_sweep")
            candidates.append(os.path.join(
                test_dir, sweep, "bigloop_%d" % bigloop,
                "%dactual_LinkMOE.csv" % bigloop))
    elif str(test).strip() == "3":
        if mpr_s != "":
            candidates += [
                os.path.join(level_dir, "mpr_%s" % mpr_s, slug,
                             "output_LinkMOE.csv"),
                os.path.join(level_dir, slug, "mpr_%s" % mpr_s,
                             "output_LinkMOE.csv"),
            ]
        # Sweep rows: the sweep file is this row's own output, so try
        # it BEFORE the generic system-job files below, which belong
        # to the bigloop-10 system job (p=1.0), not the sweep point.
        # (Same candidate set as before, only the order changes.)
        if slug in ("proposed", "ue") and bigloop is not None:
            sweep = ("proposed_sweep" if slug == "proposed"
                     else "ue_sweep")
            candidates.append(os.path.join(
                level_dir, sweep, "bigloop_%d" % bigloop,
                "%dactual_LinkMOE.csv" % bigloop))
        if slug in ("proposed", "ue") and mpr_s.isdigit():
            # Test 3 sweep: the bigloop index IS the MPR point
            # (bigloop_33 / 33actual_LinkMOE.csv), not mpr//10.
            sweep = ("proposed_sweep" if slug == "proposed"
                     else "ue_sweep")
            candidates.append(os.path.join(
                level_dir, sweep, "bigloop_%s" % mpr_s,
                "%sactual_LinkMOE.csv" % mpr_s))
        candidates += [
            os.path.join(level_dir, slug, "output_LinkMOE.csv"),
            os.path.join(level_dir, slug, "full_results",
                         "output_LinkMOE.csv"),
        ]
        if slug == "proposed" and mpr_s == "":
            # Proposed system row (not a sweep row): collect writes the
            # bigloop-10 actual file as {n}actual_LinkMOE.csv.
            candidates.append(os.path.join(
                level_dir, "proposed", "full_results",
                "10actual_LinkMOE.csv"))
    elif str(test).strip() == "4":
        candidates += [
            os.path.join(test_dir, slug, "output_LinkMOE.csv"),
            os.path.join(test_dir, slug, "full_results",
                         "output_LinkMOE.csv"),
        ]
        if slug == "ue":
            candidates.append(os.path.join(
                test_dir, "full_results", "output_LinkMOE.csv"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Fallback: recursive search under the test / level directory.
    roots = []
    if os.path.isdir(level_dir):
        roots.append(level_dir)
    elif os.path.isdir(test_dir):
        # Tests 1, 2, 4 have no level_<N>/ subdir: search the test dir.
        roots.append(test_dir)
    hits = []
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name == "output_LinkMOE.csv" or \
                        name.endswith("actual_LinkMOE.csv"):
                    hits.append(os.path.join(dirpath, name))
    if not hits:
        return None
    rel_hits = []
    for path in hits:
        rel = os.path.relpath(path, seed_dir).lower().replace(os.sep, "/")
        if slug not in rel and slug != "unknown":
            continue
        if mpr_s != "":
            tokens = ("mpr_%s" % mpr_s,
                      "mpr%s" % mpr_s)
            if bigloop is not None:
                tokens += ("bigloop_%d" % bigloop,)
            if mpr_s.isdigit():
                # Test 3 sweep: the bigloop index IS the MPR point.
                tokens += ("bigloop_%s" % mpr_s,)
            if not any(t in rel for t in tokens):
                continue
        rel_hits.append(path)
    if not rel_hits:
        return None
    pool = sorted(rel_hits)
    if len(pool) > 1:
        print("WARNING: %d realized LinkMOE candidates for test=%s "
              "level=%s method=%s mpr=%s; using %s"
              % (len(pool), test, level, method, mpr, pool[0]),
              file=sys.stderr)
    return pool[0]


def collect_seed(seed_dir, do_backfill=True):
    """Collect one seed's summary rows, with congestion backfill.

    Returns (groups, stats) where groups is {group: {metric: value}}
    and stats counts rows read, rows backfilled, and rows whose
    congestion could not be backfilled (no LinkMOE file found).
    """
    path = os.path.join(seed_dir, "summary.csv")
    if not os.path.isfile(path):
        raise OSError("summary.csv not found in %s" % seed_dir)
    groups = read_seed_summary(path)
    stats = {"rows": len(groups), "backfilled": 0, "no_linkmoe": 0}
    if not do_backfill:
        return groups, stats
    for group, metrics in groups.items():
        test, level, method, mpr = group
        missing = [m for m in CONGESTION_METRICS if metrics.get(m) is None]
        if not missing:
            continue
        linkmoe = find_realized_linkmoe(seed_dir, test, level, method, mpr)
        if linkmoe is None:
            stats["no_linkmoe"] += 1
            continue
        try:
            res = congestion.compute_congestion(linkmoe)
        except (OSError, ValueError) as e:
            print("WARNING: congestion backfill failed for %s: %s"
                  % (linkmoe, e), file=sys.stderr)
            stats["no_linkmoe"] += 1
            continue
        if res["n_links"] == 0:
            stats["no_linkmoe"] += 1
            continue
        for m in missing:
            metrics[m] = float(res[m]) if res[m] is not None else None
        stats["backfilled"] += 1
    return groups, stats


def aggregate(seeds, seed_data):
    """Mean + SD per (test, level, method, mpr) group across seeds.

    Returns a list of row dicts with keys test, level, method, mpr, n
    plus each numeric metric and its _sd. n is the number of seeds that
    reported the group; a metric's mean/SD use only the seeds that
    reported that metric (per-metric n is in the text report).
    """
    groups = sorted({g for s in seeds for g in seed_data[s]},
                    key=group_sort_key)
    rows = []
    for group in groups:
        test, level, method, mpr = group
        present = [s for s in seeds if group in seed_data[s]]
        row = {"test": test, "level": level, "method": method,
               "mpr": mpr, "n": len(present)}
        for metric in NUMERIC_METRICS:
            vals = [seed_data[s][group][metric] for s in present]
            mean, sd = mean_sd(vals)
            row[metric] = mean
            row[metric + "_sd"] = sd
            row[metric + "_n"] = len([v for v in vals if v is not None])
        rows.append(row)
    return rows


def fmt(v, nd=2):
    if v is None:
        return "n/a"
    return "%.*f" % (nd, v)


def fmt_mean_sd(mean, sd, n):
    if mean is None:
        return "n/a"
    if n <= 1:
        return "%s (n=%d)" % (fmt(mean), n)
    return "%s +/- %s (n=%d)" % (fmt(mean), fmt(sd), n)


def write_summary_csv(path, rows):
    """Aggregate summary.csv: same columns as the per-seed file, with
    seed replaced by n (seeds reporting the group) and a _sd column
    after each numeric metric."""
    header = ["test", "level", "method", "mpr", "n"]
    for m in NUMERIC_METRICS:
        header += [m, m + "_sd"]
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            out = [row["test"], row["level"], row["method"],
                   row["mpr"], row["n"]]
            for m in NUMERIC_METRICS:
                out.append("" if row[m] is None else "%.4f" % row[m])
                out.append("" if row[m + "_sd"] is None
                           else "%.4f" % row[m + "_sd"])
            w.writerow(out)


def write_summary_txt(path, seeds, seed_dirs, rows):
    """Human-readable mean +/- SD report, grouped by test and level."""
    lines = []
    lines.append("Multi-seed summary (wholerunner v2, Tests 1-4)")
    lines.append("Seeds: %s (n=%d)"
                 % (", ".join(str(s) for s in seeds), len(seeds)))
    lines.append("Per-seed results dirs:")
    for s, d in zip(seeds, seed_dirs):
        lines.append("  seed %d: %s" % (s, d))
    lines.append("Generated: %s" % time.strftime("%Y-%m-%d %H:%M"))
    lines.append("Mean and sample standard deviation (ddof=1) across seeds;")
    lines.append("a metric with n<seeds was missing for some seeds (see n).")
    lines.append("The aggregate summary.csv replaces the per-seed 'seed'")
    lines.append("column with n (seeds reporting the group).")
    lines.append("")
    by_test = {}
    for row in rows:
        by_test.setdefault(row["test"], []).append(row)
    for test in sorted(by_test, key=num_key):
        lines.append("Test %s" % test)
        lines.append("-" * 70)
        by_level = {}
        for row in by_test[test]:
            by_level.setdefault(row["level"], []).append(row)
        for level in sorted(by_level, key=num_key):
            if str(level).strip() != "":
                lines.append("Level %s agents:" % level)
            for row in sorted(
                    by_level[level],
                    key=lambda r: group_sort_key(
                        (r["test"], r["level"], r["method"],
                         r["mpr"]))):
                label = row["method"]
                if str(row["mpr"]).strip() != "":
                    label += " (MPR %s%%)" % row["mpr"]
                lines.append("  %s" % label)
                lines.append("    satisfaction %s%%, veh-hrs %s, "
                             "delay %s veh-hrs"
                             % (fmt_mean_sd(row["satisfaction"],
                                           row["satisfaction_sd"],
                                           row["satisfaction_n"]),
                                fmt_mean_sd(row["veh_hrs"],
                                            row["veh_hrs_sd"],
                                            row["veh_hrs_n"]),
                                fmt_mean_sd(row["delay"], row["delay_sd"],
                                            row["delay_n"])))
                acc = fmt_mean_sd(row["linkvol_acc"],
                                  row["linkvol_acc_sd"],
                                  row["linkvol_acc_n"])
                acc = acc + "%" if acc != "n/a" else acc
                lines.append("    link-volume accuracy %s, max V/C %s, "
                             "p95 V/C %s, V/C>1.0 %s, V/C>0.8 %s"
                             % (acc,
                                fmt_mean_sd(row["max_vc"], row["max_vc_sd"],
                                            row["max_vc_n"]),
                                fmt_mean_sd(row["p95_vc"], row["p95_vc_sd"],
                                            row["p95_vc_n"]),
                                fmt_mean_sd(row["over_cap"],
                                            row["over_cap_sd"],
                                            row["over_cap_n"]),
                                fmt_mean_sd(row["heavy"], row["heavy_sd"],
                                            row["heavy_n"])))
                em = ", ".join(
                    "%s %s" % (pol.upper(), fmt_mean_sd(
                        row[pol], row[pol + "_sd"], row[pol + "_n"]))
                    for pol in ("energy", "co2", "nox", "co", "hc"))
                lines.append("    emissions: %s" % em)
        lines.append("")
    thin = [r for r in rows if r["n"] < len(seeds)]
    if thin:
        lines.append("Notes")
        lines.append("-" * 70)
        lines.append("Groups reported by fewer than all seeds:")
        for row in thin:
            lines.append("  test=%s level=%s method=%s mpr=%s: n=%d"
                         % (row["test"], row["level"], row["method"],
                            row["mpr"], row["n"]))
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text)
    return text


def discover_seed_dirs(results_dir):
    """Find <base>_seed<S>/ dirs next to <base> (results_dir)."""
    parent = os.path.dirname(os.path.abspath(results_dir))
    base = os.path.basename(os.path.abspath(results_dir))
    found = []
    for name in sorted(os.listdir(parent)):
        m = re.match(r"^%s_seed(\d+)$" % re.escape(base), name)
        if m and os.path.isdir(os.path.join(parent, name)):
            found.append((int(m.group(1)), os.path.join(parent, name)))
    found.sort()
    return [d for _s, d in found], [s for s, _d in found]


def default_out_dir(seed_dirs):
    """<results-dir>_multiseed next to the first seed dir."""
    first = os.path.abspath(seed_dirs[0])
    parent = os.path.dirname(first)
    name = os.path.basename(first)
    m = re.match(r"^(.*)_seed\d+$", name)
    base = m.group(1) if m else name
    return os.path.join(parent, base + "_multiseed")


def main(argv):
    seed_dirs = []
    seeds = []
    out_dir = None
    results_dir = None
    do_backfill = True
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--seed-dirs":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                seed_dirs.append(argv[i])
                i += 1
            continue
        if tok == "--seeds":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                try:
                    seeds.append(int(argv[i]))
                except ValueError:
                    print("ERROR: --seeds values must be integers",
                          file=sys.stderr)
                    return 2
                i += 1
            continue
        if tok == "--out-dir":
            if i + 1 >= len(argv):
                print("ERROR: --out-dir needs a value", file=sys.stderr)
                return 2
            out_dir = argv[i + 1]
            i += 2
            continue
        if tok == "--results-dir":
            if i + 1 >= len(argv):
                print("ERROR: --results-dir needs a value", file=sys.stderr)
                return 2
            results_dir = argv[i + 1]
            i += 2
            continue
        if tok == "--no-backfill":
            do_backfill = False
            i += 1
            continue
        if tok in ("-h", "--help"):
            print("Usage: python aggregate_v2.py --seed-dirs <dir>... "
                  "--seeds <int>... [--out-dir <dir>] [--no-backfill]")
            print("   or: python aggregate_v2.py --results-dir <base> "
                  "[--seeds <int>...] [--out-dir <dir>]")
            return 0
        print("ERROR: unknown argument: %s" % tok, file=sys.stderr)
        return 2
    if results_dir is not None:
        found_dirs, found_seeds = discover_seed_dirs(results_dir)
        if not found_dirs:
            print("ERROR: no <base>_seed<S>/ dirs found next to %s"
                  % results_dir, file=sys.stderr)
            return 2
        if seeds:
            keep = [(s, d) for s, d in zip(found_seeds, found_dirs)
                    if s in seeds]
            if not keep:
                print("ERROR: none of --seeds %s found next to %s"
                      % (seeds, results_dir), file=sys.stderr)
                return 2
            seeds = [s for s, _d in keep]
            seed_dirs = [d for _s, d in keep]
        else:
            seeds = found_seeds
            seed_dirs = found_dirs
        if out_dir is None:
            parent = os.path.dirname(os.path.abspath(results_dir))
            base = os.path.basename(os.path.abspath(results_dir))
            out_dir = os.path.join(parent, base + "_multiseed")
    if not seed_dirs:
        print("ERROR: no --seed-dirs given (or no --results-dir)",
              file=sys.stderr)
        return 2
    if seeds and len(seeds) != len(seed_dirs):
        print("ERROR: --seeds and --seed-dirs must have the same length",
              file=sys.stderr)
        return 2
    if not seeds:
        seeds = []
        for d in seed_dirs:
            m = re.search(r"_seed(\d+)$",
                          os.path.basename(os.path.normpath(d)))
            seeds.append(int(m.group(1)) if m else len(seeds) + 1)
    good_dirs, good_seeds = [], []
    for s, d in zip(seeds, seed_dirs):
        if not os.path.isdir(d):
            print("WARNING: seed results dir not found, skipped: %s" % d,
                  file=sys.stderr)
            continue
        good_dirs.append(d)
        good_seeds.append(s)
    if not good_dirs:
        print("ERROR: no usable seed dirs", file=sys.stderr)
        return 2
    seeds, seed_dirs = good_seeds, good_dirs
    if out_dir is None:
        out_dir = default_out_dir(seed_dirs)
    os.makedirs(out_dir, exist_ok=True)
    seed_data = {}
    total_backfilled = 0
    total_no_linkmoe = 0
    for s, d in zip(seeds, seed_dirs):
        print("seed %d: reading %s" % (s, d))
        try:
            seed_data[s], stats = collect_seed(d, do_backfill)
        except (OSError, csv.Error) as e:
            print("WARNING: seed %d: could not read %s: %s"
                  % (s, d, e), file=sys.stderr)
            seed_data[s] = {}
            stats = {"rows": 0, "backfilled": 0, "no_linkmoe": 0}
        print("  %d rows%s" % (stats["rows"],
              (", %d congestion-backfilled, %d rows without a LinkMOE file"
               % (stats["backfilled"], stats["no_linkmoe"]))
              if do_backfill else ""))
        total_backfilled += stats["backfilled"]
        total_no_linkmoe += stats["no_linkmoe"]
    rows = aggregate(seeds, seed_data)
    csv_path = os.path.join(out_dir, "summary.csv")
    txt_path = os.path.join(out_dir, "summary.txt")
    write_summary_csv(csv_path, rows)
    write_summary_txt(txt_path, seeds, seed_dirs, rows)
    print("Wrote %s (%d groups)" % (csv_path, len(rows)))
    print("Wrote %s" % txt_path)
    if do_backfill:
        print("Congestion backfill: %d rows filled from LinkMOE files, "
              "%d rows had no LinkMOE file" % (total_backfilled,
                                               total_no_linkmoe))
    thin = [r for r in rows if r["n"] < len(seeds)]
    if thin:
        print("WARNING: %d of %d groups reported by fewer than all seeds"
              % (len(thin), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
