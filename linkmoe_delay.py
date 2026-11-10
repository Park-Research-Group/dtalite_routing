#!/usr/bin/env python3
"""Total-delay analysis for DTALite/wholerunner results (Track A paper).

Computes TOTAL DELAY per scenario from LinkMOE output files using the
established formula from runner_bingrong.py compute_delay:

    delay (vehicle-hours) = sum over links of
        (travel_time - free-flow_travel_time) * total_link_volume / 60

Read-only: never writes into the results tree. Writes its report to
linkmoe_summary.txt next to this script and prints it to the console.

Usage (Windows):
    python linkmoe_delay.py C:\\Users\\ucg8nb\\Downloads\\results_dtalite

    python linkmoe_delay.py C:\\Users\\ucg8nb\\Downloads\\results_dtalite --agents 54045

Layout expected under the results root (standard wholerunner layout):
    test1/full_results/<N>actual_LinkMOE.csv   (N = last complete bigloop)
    test2/full_results/output_LinkMOE.csv
    test3/full_results/output_LinkMOE.csv
    test4/full_results/output_LinkMOE.csv
    test5/proposed_sweep/bigloop_<N>/<N>actual_LinkMOE.csv  (per MPR)
    test5/ue_sweep/bigloop_<N>/<N>actual_LinkMOE.csv       (per MPR)

Bingrong (2018) Ch.4 total-delay targets (vehicle-hours):
    Proposed            7,860.4
    Real-time sigma=1   9,831.9
    Real-time sigma=10 10,809.5
    UE               13,857.4

Stdlib only. No simulation re-runs.
"""

import csv
import glob
import os
import re
import sys

# Bingrong dissertation Ch.4 total-delay targets (vehicle-hours).
BINGRONG_DELAY = {
    "test1": 7860.4,   # Proposed
    "test2": 9831.9,   # Real-time sigma=1
    "test3": 10809.5,  # Real-time sigma=10
    "test4": 13857.4,  # UE as information
}

SCENARIO_NAMES = {
    "test1": "Proposed (Test 1, 100% MPR)",
    "test2": "Real-time sigma=1 (Test 2)",
    "test3": "Real-time sigma=10 (Test 3)",
    "test4": "UE as information (Test 4)",
}

# Test 5 MPR mapping: bigloop N -> N*10% MPR.
MPR_LABEL = {n: "%d%%" % (n * 10) for n in range(11)}

TAG_RE = re.compile(r"^(\d+)(UOinfor|predict|actual)_agent\.csv$")
BIGLOOP_DIR_RE = re.compile(r"^bigloop_(\d+)$")
AGENTS_RE = re.compile(r"agents[^0-9]*(\d{4,6})", re.IGNORECASE)


def norm_headers(fieldnames):
    """DTALite headers carry stray leading spaces; normalize for lookup."""
    return {c.strip().lower(): c for c in fieldnames if c and c.strip()}


def compute_delay(linkmoe_path):
    """sum((travel_time - free-flow_travel_time) * total_link_volume)/60.

    Assumes LinkMOE travel times are in minutes. Returns None when the
    file or the needed columns are absent, or when no rows parse.
    Rows that fail float conversion (or are NaN) are skipped.
    """
    if not os.path.isfile(linkmoe_path):
        return None
    total = 0.0
    n = 0
    with open(linkmoe_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        need = ("travel_time", "free-flow_travel_time", "total_link_volume")
        if any(k not in cols for k in need):
            return None
        tt_c, fft_c, vol_c = (cols[k] for k in need)
        for row in reader:
            try:
                tt = float(row[tt_c])
                fft = float(row[fft_c])
                vol = float(row[vol_c])
            except (TypeError, ValueError):
                continue
            if tt == tt and fft == fft and vol == vol:  # not NaN
                total += (tt - fft) * vol
                n += 1
    if n == 0:
        return None
    return total / 60.0, n


def find_last_bigloop(d):
    """Last bigloop number with both actual and UOinfor agent files.

    Same logic as wholerunner.find_last_bigloop: scans for
    <N>(UOinfor|predict|actual)_agent.csv, keeps bigloops that have
    both 'actual' and 'UOinfor', returns the max (or None).
    """
    found = {}
    try:
        names = os.listdir(d)
    except OSError:
        return None
    for name in names:
        m = TAG_RE.match(name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(m.group(2))
    cands = [b for b, tags in found.items()
             if "actual" in tags and "UOinfor" in tags]
    return max(cands) if cands else None


def find_test1_linkmoe(test1_dir):
    """Return (path, bigloop) for Test 1's actual LinkMOE file.

    Prefers bigloop 10 if present; otherwise auto-detects the last
    complete bigloop (same logic as the analyzer).
    """
    full = os.path.join(test1_dir, "full_results")
    if not os.path.isdir(full):
        return None, None
    # Prefer bigloop 10.
    p10 = os.path.join(full, "10actual_LinkMOE.csv")
    if os.path.isfile(p10):
        return p10, 10
    # Fall back to last complete bigloop.
    last = find_last_bigloop(full)
    if last is None:
        return None, None
    p = os.path.join(full, "%dactual_LinkMOE.csv" % last)
    if os.path.isfile(p):
        return p, last
    return None, None


def find_actual_linkmoe_in_dir(d, bigloop=None):
    """Find an actual LinkMOE file inside a directory.

    Tries <N>actual_LinkMOE.csv (if bigloop given), then any
    *actual_LinkMOE.csv, then a bare actual_LinkMOE.csv.
    """
    if bigloop is not None:
        p = os.path.join(d, "%dactual_LinkMOE.csv" % bigloop)
        if os.path.isfile(p):
            return p
    for pat in ("*actual_LinkMOE.csv", "actual_LinkMOE.csv"):
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    return None


def discover_bigloop_dirs(root):
    """bigloop -> directory holding its CSVs.

    Accepts bigloop_<n>/ subdirectories, or a flat layout with
    <n>UOinfor/predict/actual_agent.csv directly in root.
    """
    out = {}
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in sorted(names):
        m = BIGLOOP_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            out[int(m.group(1))] = os.path.join(root, name)
    if not out:
        for name in sorted(names):
            m = re.match(r"^(\d+)(?:UOinfor|predict|actual)_agent\.csv$", name)
            if m:
                out.setdefault(int(m.group(1)), root)
    return out


def read_agent_count(test_dir, default=54045):
    """Best-effort agent count from a test's summary file."""
    for fname in ("proposed_summary.txt", "realtime_summary.txt",
                  "ue_summary.txt", "reaction_summary.txt"):
        p = os.path.join(test_dir, fname)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8-sig", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        m = AGENTS_RE.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return default


def fmt(v, nd=1):
    return "%s" % ("%.{}f".format(nd) % v) if v is not None else "n/a"


def main(argv):
    if len(argv) < 2:
        print("Usage: python linkmoe_delay.py <results_root> [--agents N] [--out PATH]")
        print("Example: python linkmoe_delay.py "
              r"C:\Users\ucg8nb\Downloads\results_dtalite")
        return 2
    root = argv[1]
    agents_override = None
    if "--agents" in argv:
        i = argv.index("--agents")
        if i + 1 < len(argv):
            try:
                agents_override = int(argv[i + 1])
            except ValueError:
                print("warning: --agents value not an integer; ignoring")
    out_override = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 < len(argv):
            out_override = argv[i + 1]
        else:
            print("warning: --out needs a value; ignoring")
    if not os.path.isdir(root):
        print("ERROR: results root not found: %s" % root)
        return 1

    lines = []
    lines.append("=" * 70)
    lines.append("Total delay from LinkMOE outputs (Track A paper)")
    lines.append("Formula: sum((travel_time - free-flow_travel_time) * "
                 "total_link_volume) / 60  [vehicle-hours]")
    lines.append("Results root: %s" % root)
    lines.append("=" * 70)
    lines.append("")

    # ---- Tests 1-4 ------------------------------------------------
    lines.append("Tests 1-4: total delay vs Bingrong (2018) Ch.4")
    lines.append("-" * 70)
    header = ("%-32s %14s %14s %10s %12s" %
              ("Scenario", "Delay (veh-h)", "Bingrong", "Diff %",
               "Delay/agent (h)"))
    lines.append(header)
    for test in ("test1", "test2", "test3", "test4"):
        tdir = os.path.join(root, test)
        if test == "test1":
            path, bigloop = find_test1_linkmoe(tdir)
            src_note = ("bigloop %d" % bigloop) if bigloop is not None else "?"
        else:
            path = os.path.join(tdir, "full_results", "output_LinkMOE.csv")
            if not os.path.isfile(path):
                # Some layouts keep output_LinkMOE.csv directly in test dir.
                alt = os.path.join(tdir, "output_LinkMOE.csv")
                path = alt if os.path.isfile(alt) else path
            src_note = "output_LinkMOE.csv"
        n_agents = agents_override or read_agent_count(tdir)
        if path is None or not os.path.isfile(path):
            lines.append("%-32s %14s %14s %10s %12s" %
                         (SCENARIO_NAMES[test], "NOT FOUND",
                          fmt(BINGRONG_DELAY[test]), "n/a", "n/a"))
            lines.append("    (looked for: %s)" % path)
            continue
        res = compute_delay(path)
        if res is None:
            lines.append("%-32s %14s %14s %10s %12s" %
                         (SCENARIO_NAMES[test], "NO COLUMNS/ROWS",
                          fmt(BINGRONG_DELAY[test]), "n/a", "n/a"))
            lines.append("    (file: %s)" % path)
            continue
        delay_vh, n_links = res
        target = BINGRONG_DELAY[test]
        diff_pct = (delay_vh - target) / target * 100.0
        per_agent = delay_vh / n_agents if n_agents else float("nan")
        lines.append("%-32s %14s %14s %+9.1f%% %12s" %
                     (SCENARIO_NAMES[test], fmt(delay_vh),
                      fmt(target), diff_pct, "%.4f" % per_agent))
        lines.append("    (source: %s, %s, %d links, %d agents)" %
                     (os.path.basename(path), src_note, n_links, n_agents))
    lines.append("")

    # ---- Test 5: delay vs MPR -------------------------------------
    lines.append("Test 5: total delay vs MPR (Proposed sweep)")
    lines.append("-" * 70)
    lines.append("%-10s %14s %12s" % ("MPR", "Delay (veh-h)", "Delay/agent (h)"))
    prop_root = os.path.join(root, "test5", "proposed_sweep")
    prop_dirs = discover_bigloop_dirs(prop_root)
    n_agents5 = agents_override or read_agent_count(
        os.path.join(root, "test5")) or 54045
    # Fall back to test1 agent count if test5 has no summary.
    if n_agents5 == 54045 and agents_override is None:
        n_agents5 = read_agent_count(os.path.join(root, "test1"))
    if not prop_dirs:
        lines.append("  (no proposed_sweep bigloop dirs found under %s)" % prop_root)
    else:
        for bl in sorted(prop_dirs):
            d = prop_dirs[bl]
            p = find_actual_linkmoe_in_dir(d, bl)
            if p is None:
                lines.append("%-10s %14s %12s" % (MPR_LABEL.get(bl, str(bl)),
                                                 "NOT FOUND", "n/a"))
                continue
            res = compute_delay(p)
            if res is None:
                lines.append("%-10s %14s %12s" % (MPR_LABEL.get(bl, str(bl)),
                                                 "NO COLUMNS", "n/a"))
                continue
            delay_vh, n_links = res
            per_agent = delay_vh / n_agents5 if n_agents5 else float("nan")
            lines.append("%-10s %14s %12.4f" %
                         (MPR_LABEL.get(bl, str(bl)), fmt(delay_vh), per_agent))
    lines.append("")
    ue_root = os.path.join(root, "test5", "ue_sweep")
    ue_dirs = discover_bigloop_dirs(ue_root)
    if ue_dirs:
        lines.append("Test 5: total delay vs MPR (UE sweep)")
        lines.append("-" * 70)
        lines.append("%-10s %14s %12s" % ("MPR", "Delay (veh-h)", "Delay/agent (h)"))
        for bl in sorted(ue_dirs):
            d = ue_dirs[bl]
            p = find_actual_linkmoe_in_dir(d, bl)
            if p is None:
                lines.append("%-10s %14s %12s" % (MPR_LABEL.get(bl, str(bl)),
                                                 "NOT FOUND", "n/a"))
                continue
            res = compute_delay(p)
            if res is None:
                lines.append("%-10s %14s %12s" % (MPR_LABEL.get(bl, str(bl)),
                                                 "NO COLUMNS", "n/a"))
                continue
            delay_vh, n_links = res
            per_agent = delay_vh / n_agents5 if n_agents5 else float("nan")
            lines.append("%-10s %14s %12.4f" %
                         (MPR_LABEL.get(bl, str(bl)), fmt(delay_vh), per_agent))
        lines.append("")

    lines.append("Notes:")
    lines.append("  - Delay uses actual (experienced) LinkMOE volumes and times.")
    lines.append("  - Per-agent hours = total vehicle-hours / number of agents.")
    lines.append("  - Bingrong targets are dissertation Ch.4 totals (vehicle-hours).")
    lines.append("  - Rows that fail float conversion or are NaN are skipped.")
    text = "\n".join(lines) + "\n"

    print(text)
    if out_override:
        out_path = out_override
    else:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "linkmoe_summary.txt")
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text)
    print("Wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
