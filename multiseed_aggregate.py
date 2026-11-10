#!/usr/bin/env python3
"""Aggregate wholerunner results across seeds (Track A paper).

Reads the per-seed results dirs produced by
    python wholerunner.py --tests 1-6 --seeds 42 43 44 --engine dtalite
and reports the mean and sample standard deviation (ddof=1) across
seeds for the headline metrics: satisfaction and vehicle-hours for the
guidance comparison (Tests 1-4), emissions, delay, and the Test 5
market-penetration sweep (per-MPR satisfaction, vehicle-hours, gap,
emissions, and delay).

Inputs (per seed dir, the standard wholerunner layout):
    test1/proposed_summary.txt, test2/reaction_summary.txt,
    test3/reaction_summary.txt, test4/reaction_summary.txt
    table4_emissions.csv (results root)
    test5/mpr_summary.txt, test5/mpr_table5.csv, test5/mpr_table6.csv
    test5/ue_sweep/bigloop_<N>/<N>actual_agent.csv (UE sweep veh-hrs)
    linkmoe_summary.txt (written by linkmoe_delay.py; if a seed dir
    has no linkmoe_summary.txt, delay is reported as unavailable
    for that seed instead of failing the whole aggregation)

Outputs (into --out-dir, default <first-seed-dir>_multiseed):
    multiseed_summary.txt   human-readable mean +/- SD report
    multiseed_summary.csv   one row per metric, with mean, SD, n, and
                            the per-seed values

Stdlib only. Read-only on the per-seed results dirs.

Usage:
    python multiseed_aggregate.py --seed-dirs results_seed42 results_seed43 results_seed44 --seeds 42 43 44 --out-dir results_multiseed
"""

import csv
import glob
import os
import re
import statistics
import sys
import time

SYSTEMS = [
    ("proposed", "Proposed method", 1),
    ("rt1", "Real-time, 1-min updates", 2),
    ("rt10", "Real-time, 10-min updates", 3),
    ("ue", "UE guidance", 4),
]
SYSTEM_BY_TEST = {num: key for key, _label, num in SYSTEMS}
SYSTEM_LABEL = {key: label for key, label, _num in SYSTEMS}
POLLUTANTS = ["energy", "co2", "nox", "co", "hc"]
POLLUTANT_LABEL = {
    "energy": "Energy (10^9 KJ)",
    "co2": "CO2 (ton)",
    "nox": "NOX (kg)",
    "co": "CO (kg)",
    "hc": "HC (kg)",
}


def mean_sd(values):
    """Mean and sample standard deviation (ddof=1) of a list of floats.

    Returns (mean, sd). sd is 0.0 when only one value is present and
    (None, None) when the list is empty.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt(v):
    if v is None:
        return "n/a"
    return "%.2f" % v


def fmt_mean_sd(mean, sd, n):
    if mean is None:
        return "n/a"
    if n <= 1:
        return "%s (n=%d)" % (fmt(mean), n)
    return "%s +/- %s (n=%d)" % (fmt(mean), fmt(sd), n)


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_guidance_summary(path):
    """Satisfaction pct and vehicle-hours from a per-test summary file.

    Handles both the Test 1 form ("vehicle-hours: 25705.15") and the
    Tests 2-4 form ("vehicle-hours (actual run): 33644.21").
    Returns (sat_pct, veh_hrs) with None for anything not found.
    """
    text = read_text(path)
    sat = None
    m = re.search(r"satisfaction:\s*\d+\s*/\s*\d+\s*=\s*([\d.]+)%", text)
    if m:
        sat = float(m.group(1))
    veh = None
    m = re.search(r"vehicle-hours(?: \(actual run\))?:\s*([\d.]+)", text)
    if m:
        veh = float(m.group(1))
    return sat, veh


def parse_table4_csv(path):
    """Emissions per system from table4_emissions.csv.

    Returns {system_key: {pollutant: value}}.
    """
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or []) if c}
        def col(*keys):
            for k in keys:
                if k in cols:
                    return cols[k]
            return None
        c_energy = col("energy (10^9 kj)", "energy")
        c_co2 = col("co2 (ton)", "co2")
        c_nox = col("nox (kg)", "nox")
        c_co = col("co (kg)", "co")
        c_hc = col("hc (kg)", "hc")
        c_scen = col("scenario")
        for row in reader:
            scen = (row.get(c_scen) or "").strip()
            m = re.search(r"Test\s*(\d+)", scen)
            if not m:
                continue
            key = SYSTEM_BY_TEST.get(int(m.group(1)))
            if key is None:
                continue
            def num(c):
                if c is None:
                    return None
                try:
                    return float((row.get(c) or "").strip())
                except ValueError:
                    return None
            out[key] = {
                "energy": num(c_energy),
                "co2": num(c_co2),
                "nox": num(c_nox),
                "co": num(c_co),
                "hc": num(c_hc),
            }
    return out


def parse_mpr_summary(path):
    """Proposed sweep per-MPR satisfaction and vehicle-hours.

    Reads the "Per-bigloop results (proposed)" table in
    mpr_summary.txt. Returns {mpr_int: {"sat": sat_pct, "veh": veh_hrs}}.
    """
    text = read_text(path)
    out = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("Per-bigloop results (proposed)"):
            in_table = True
            continue
        if in_table:
            if line.strip() == "" or line.startswith("Sustainability"):
                break
            parts = line.split()
            if len(parts) < 9 or not parts[0].isdigit():
                continue
            try:
                mpr = int(parts[1].rstrip("%"))
                sat = float(parts[4])
                veh = float(parts[7])
            except ValueError:
                continue
            out[mpr] = {"sat": sat, "veh": veh}
    return out


def parse_mpr_table5(path):
    """Per-MPR unsatisfied pct for both sweeps, from mpr_table5.csv.

    Returns {mpr_int: {"prop_unsat": pct, "ue_unsat": pct}}.
    """
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or []) if c}
        def col(*keys):
            for k in keys:
                if k in cols:
                    return cols[k]
            return None
        c_mpr = col("mpr")
        c_prop = col("proposed unsatisfied %", "proposed unsatisfied")
        c_ue = col("ue unsatisfied %", "ue unsatisfied")
        for row in reader:
            try:
                mpr = int((row.get(c_mpr) or "").strip().rstrip("%"))
            except ValueError:
                continue
            def pct(c):
                if c is None:
                    return None
                s = (row.get(c) or "").strip().rstrip("%")
                try:
                    return float(s)
                except ValueError:
                    return None
            out[mpr] = {"prop_unsat": pct(c_prop), "ue_unsat": pct(c_ue)}
    return out


def parse_mpr_table6(path):
    """Per-MPR emissions for both sweeps, from mpr_table6.csv.

    Returns {mpr_int: {"prop": {pollutant: value}, "ue": {...}}}.
    "N/A" cells become None.
    """
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or []) if c}
        def col(name):
            return cols.get(name.lower())
        for row in reader:
            try:
                mpr = int((row.get(col("MPR")) or "").strip().rstrip("%"))
            except ValueError:
                continue
            def num(prefix, pol):
                c = col("%s %s" % (prefix, POLLUTANT_LABEL[pol]))
                if c is None:
                    return None
                s = (row.get(c) or "").strip()
                if s.upper() == "N/A" or s == "":
                    return None
                try:
                    return float(s)
                except ValueError:
                    return None
            entry = {}
            for who, prefix in (("prop", "Proposed"), ("ue", "UE")):
                entry[who] = {pol: num(prefix, pol) for pol in POLLUTANTS}
            out[mpr] = entry
    return out


def norm_headers(fieldnames):
    return {c.strip().lower(): c for c in fieldnames if c and c.strip()}


def find_tt_col(cols):
    for key, orig in cols.items():
        if "travel_time" in key and "min" in key:
            return orig
    return None


def sum_travel_minutes(path):
    """Sum of the travel-time-minutes column over all rows of an agent file."""
    total = 0.0
    n = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        tt_c = find_tt_col(cols)
        if tt_c is None:
            raise ValueError("no travel-time column in %s" % path)
        for row in reader:
            try:
                v = float(row[tt_c])
            except (TypeError, ValueError):
                continue
            if v == v:
                total += v
                n += 1
    return total, n


def ue_sweep_vehhrs(ue_sweep_dir):
    """UE sweep vehicle-hours per MPR, computed from the agent files.

    Mirrors ue_vehhrs.py: sums travel-time-minutes over all rows of
    each bigloop's <N>actual_agent.csv, divided by 60. Returns
    {mpr_int: veh_hrs}.
    """
    out = {}
    if not os.path.isdir(ue_sweep_dir):
        return out
    for name in sorted(os.listdir(ue_sweep_dir)):
        m = re.match(r"^bigloop_(\d+)$", name)
        if not m:
            continue
        bl = int(m.group(1))
        d = os.path.join(ue_sweep_dir, name)
        cand = os.path.join(d, "%dactual_agent.csv" % bl)
        if not os.path.isfile(cand):
            hits = sorted(glob.glob(os.path.join(d, "*actual_agent.csv")))
            cand = hits[0] if hits else None
        if cand is None:
            continue
        try:
            tot_min, _n = sum_travel_minutes(cand)
        except ValueError:
            continue
        out[bl * 10] = tot_min / 60.0
    return out


def parse_linkmoe_summary(path):
    """Delay values from a linkmoe_summary.txt.

    Returns (guidance_delay, sweep_delay) where guidance_delay is
    {test_num: delay_veh_hrs} for Tests 1-4 and sweep_delay is
    {(sweep, mpr_int): delay_veh_hrs} with sweep "prop" or "ue".
    Missing sections give empty dicts.
    """
    text = read_text(path)
    guidance = {}
    sweep = {}
    lines = text.splitlines()
    section = None
    for line in lines:
        if line.startswith("Tests 1-4:"):
            section = "guidance"
            continue
        if "Test 5: total delay vs MPR (Proposed sweep)" in line:
            section = "prop"
            continue
        if "Test 5: total delay vs MPR (UE sweep)" in line:
            section = "ue"
            continue
        if line.startswith("Notes:"):
            section = None
            continue
        if section == "guidance":
            m = re.search(r"Test\s*(\d+)", line)
            if not m:
                continue
            close = line.find(")", m.start())
            if close < 0:
                continue
            rest = line[close + 1:]
            m2 = re.search(r"([\d.]+)", rest)
            if m2:
                guidance[int(m.group(1))] = float(m2.group(1))
        elif section in ("prop", "ue"):
            m = re.match(r"\s*(\d+)%\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                sweep[(section, int(m.group(1)))] = float(m.group(2))
    return guidance, sweep


def collect_seed(seed_dir):
    """Collect all headline metrics available in one per-seed results dir.

    Returns a dict with keys:
      "guidance": {system_key: {"sat": pct, "veh": veh_hrs,
                                "emissions": {pol: val}, "delay": veh_hrs}}
      "sweep": {mpr: {"prop_sat": pct, "prop_veh": veh_hrs,
                      "ue_sat": pct, "ue_veh": veh_hrs,
                      "gap": pp,
                      "prop_emissions": {pol: val}, "ue_emissions": {pol: val},
                      "prop_delay": veh_hrs, "ue_delay": veh_hrs}}
    Missing files leave the corresponding values as None.
    """
    data = {"guidance": {}, "sweep": {}}
    summary_file = {
        1: os.path.join(seed_dir, "test1", "proposed_summary.txt"),
        2: os.path.join(seed_dir, "test2", "reaction_summary.txt"),
        3: os.path.join(seed_dir, "test3", "reaction_summary.txt"),
        4: os.path.join(seed_dir, "test4", "reaction_summary.txt"),
    }
    for num, key in SYSTEM_BY_TEST.items():
        entry = {"sat": None, "veh": None, "emissions": {}, "delay": None}
        p = summary_file[num]
        if os.path.isfile(p):
            try:
                sat, veh = parse_guidance_summary(p)
                entry["sat"] = sat
                entry["veh"] = veh
            except (OSError, ValueError):
                pass
        data["guidance"][key] = entry
    t4 = os.path.join(seed_dir, "table4_emissions.csv")
    if os.path.isfile(t4):
        try:
            for key, em in parse_table4_csv(t4).items():
                if key in data["guidance"]:
                    data["guidance"][key]["emissions"] = em
        except (OSError, ValueError, csv.Error):
            pass
    mpr_sum = os.path.join(seed_dir, "test5", "mpr_summary.txt")
    prop_sweep = {}
    if os.path.isfile(mpr_sum):
        try:
            prop_sweep = parse_mpr_summary(mpr_sum)
        except (OSError, ValueError):
            prop_sweep = {}
    t5 = os.path.join(seed_dir, "test5", "mpr_table5.csv")
    table5 = {}
    if os.path.isfile(t5):
        try:
            table5 = parse_mpr_table5(t5)
        except (OSError, ValueError, csv.Error):
            table5 = {}
    t6 = os.path.join(seed_dir, "test5", "mpr_table6.csv")
    table6 = {}
    if os.path.isfile(t6):
        try:
            table6 = parse_mpr_table6(t6)
        except (OSError, ValueError, csv.Error):
            table6 = {}
    ue_veh = ue_sweep_vehhrs(os.path.join(seed_dir, "test5", "ue_sweep"))
    delay_g, delay_s = {}, {}
    lm = os.path.join(seed_dir, "linkmoe_summary.txt")
    if os.path.isfile(lm):
        try:
            delay_g, delay_s = parse_linkmoe_summary(lm)
        except (OSError, ValueError):
            delay_g, delay_s = {}, {}
    for num, key in SYSTEM_BY_TEST.items():
        if num in delay_g:
            data["guidance"][key]["delay"] = delay_g[num]
    mprs = set(prop_sweep) | set(table5) | set(table6) | set(ue_veh)
    for mpr in sorted(mprs):
        entry = {
            "prop_sat": None, "prop_veh": None,
            "ue_sat": None, "ue_veh": None, "gap": None,
            "prop_emissions": {}, "ue_emissions": {},
            "prop_delay": None, "ue_delay": None,
        }
        if mpr in prop_sweep:
            entry["prop_sat"] = prop_sweep[mpr]["sat"]
            entry["prop_veh"] = prop_sweep[mpr]["veh"]
        if mpr in table5:
            pu = table5[mpr]["prop_unsat"]
            uu = table5[mpr]["ue_unsat"]
            if pu is not None:
                entry["prop_sat"] = 100.0 - pu
            if uu is not None:
                entry["ue_sat"] = 100.0 - uu
            if pu is not None and uu is not None:
                entry["gap"] = uu - pu
        if mpr in ue_veh:
            entry["ue_veh"] = ue_veh[mpr]
        if mpr in table6:
            entry["prop_emissions"] = table6[mpr]["prop"]
            entry["ue_emissions"] = table6[mpr]["ue"]
        if ("prop", mpr) in delay_s:
            entry["prop_delay"] = delay_s[("prop", mpr)]
        if ("ue", mpr) in delay_s:
            entry["ue_delay"] = delay_s[("ue", mpr)]
        data["sweep"][mpr] = entry
    return data


def aggregate(seeds, seed_data):
    """Mean and SD per metric across the seeds that have it.

    Returns a list of rows: (section, metric, group, mean, sd, n, values)
    where values is the list of per-seed values in seed order.
    """
    rows = []

    def add(section, metric, group, values):
        mean, sd = mean_sd(values)
        rows.append((section, metric, group, mean, sd,
                     len([v for v in values if v is not None]), values))

    for key, _label, _num in SYSTEMS:
        vals = [seed_data[s]["guidance"].get(key, {}).get("sat")
                for s in seeds]
        add("guidance", "satisfaction_pct", key, vals)
        vals = [seed_data[s]["guidance"].get(key, {}).get("veh")
                for s in seeds]
        add("guidance", "vehicle_hours", key, vals)
        for pol in POLLUTANTS:
            vals = [seed_data[s]["guidance"].get(key, {}).get("emissions", {}).get(pol)
                    for s in seeds]
            add("guidance", "emissions_" + pol, key, vals)
        vals = [seed_data[s]["guidance"].get(key, {}).get("delay")
                for s in seeds]
        add("guidance", "delay_veh_hours", key, vals)
    mprs = sorted({m for s in seeds for m in seed_data[s]["sweep"]})
    for mpr in mprs:
        group = "%d%%" % mpr
        for metric, field in (("prop_sat_pct", "prop_sat"),
                              ("prop_veh_hours", "prop_veh"),
                              ("ue_sat_pct", "ue_sat"),
                              ("ue_veh_hours", "ue_veh"),
                              ("gap_pp", "gap"),
                              ("prop_delay_veh_hours", "prop_delay"),
                              ("ue_delay_veh_hours", "ue_delay")):
            vals = [seed_data[s]["sweep"].get(mpr, {}).get(field)
                    for s in seeds]
            add("sweep", metric, group, vals)
        for who, field in (("prop", "prop_emissions"), ("ue", "ue_emissions")):
            for pol in POLLUTANTS:
                vals = [seed_data[s]["sweep"].get(mpr, {}).get(field, {}).get(pol)
                        for s in seeds]
                add("sweep", "%s_emissions_%s" % (who, pol), group, vals)
    return rows


def write_summary_txt(path, seeds, seed_dirs, rows):
    """Human-readable mean +/- SD report."""
    by_key = {}
    for section, metric, group, mean, sd, n, _vals in rows:
        by_key[(section, metric, group)] = (mean, sd, n)

    def cell(section, metric, group):
        mean, sd, n = by_key.get((section, metric, group), (None, None, 0))
        return fmt_mean_sd(mean, sd, n)

    lines = []
    lines.append("Multi-seed summary (wholerunner --seeds)")
    lines.append("Seeds: %s (n=%d)" % (", ".join(str(s) for s in seeds), len(seeds)))
    lines.append("Per-seed results dirs:")
    for s, d in zip(seeds, seed_dirs):
        lines.append("  seed %d: %s" % (s, d))
    lines.append("Generated: %s" % time.strftime("%Y-%m-%d %H:%M"))
    lines.append("Mean and sample standard deviation (ddof=1) across seeds;")
    lines.append("a metric with n<seeds was missing for some seeds (see n).")
    lines.append("")
    lines.append("Guidance comparison (Tests 1-4)")
    lines.append("-" * 70)
    lines.append("%-28s %22s %22s" % ("System", "Satisfaction %", "Vehicle-hours"))
    for key, label, _num in SYSTEMS:
        lines.append("%-28s %22s %22s" % (
            label, cell("guidance", "satisfaction_pct", key),
            cell("guidance", "vehicle_hours", key)))
    lines.append("")
    lines.append("Emissions, Tests 1-4 (summed over agents)")
    lines.append("-" * 70)
    for pol in POLLUTANTS:
        lines.append(POLLUTANT_LABEL[pol] + ":")
        for key, label, _num in SYSTEMS:
            lines.append("  %-26s %s" % (
                label, cell("guidance", "emissions_" + pol, key)))
    lines.append("")
    lines.append("Delay, Tests 1-4 (vehicle-hours, from LinkMOE)")
    lines.append("-" * 70)
    for key, label, _num in SYSTEMS:
        lines.append("  %-26s %s" % (
            label, cell("guidance", "delay_veh_hours", key)))
    lines.append("")
    lines.append("Test 5 market-penetration sweep (per MPR)")
    lines.append("-" * 70)
    lines.append("%-6s %20s %20s %10s %20s %20s" % (
        "MPR", "Prop sat %", "UE sat %", "Gap (pp)",
        "Prop veh-hrs", "UE veh-hrs"))
    mprs = sorted({int(g.rstrip("%")) for (sec, met, g) in by_key
                   if sec == "sweep" and met == "prop_sat_pct"})
    for mpr in mprs:
        group = "%d%%" % mpr
        lines.append("%-6s %20s %20s %10s %20s %20s" % (
            group,
            cell("sweep", "prop_sat_pct", group),
            cell("sweep", "ue_sat_pct", group),
            cell("sweep", "gap_pp", group),
            cell("sweep", "prop_veh_hours", group),
            cell("sweep", "ue_veh_hours", group)))
    lines.append("")
    lines.append("Test 5 emissions at 10% and 100% MPR (mean +/- SD)")
    lines.append("-" * 70)
    for mpr in (10, 100):
        group = "%d%%" % mpr
        if ("sweep", "prop_emissions_energy", group) not in by_key:
            continue
        lines.append("%s MPR, proposed:" % group)
        for pol in POLLUTANTS:
            lines.append("  %-26s %s" % (
                POLLUTANT_LABEL[pol],
                cell("sweep", "prop_emissions_" + pol, group)))
        lines.append("%s MPR, UE:" % group)
        for pol in POLLUTANTS:
            lines.append("  %-26s %s" % (
                POLLUTANT_LABEL[pol],
                cell("sweep", "ue_emissions_" + pol, group)))
    lines.append("")
    lines.append("Test 5 delay per MPR (vehicle-hours, mean +/- SD)")
    lines.append("-" * 70)
    for mpr in mprs:
        group = "%d%%" % mpr
        lines.append("%-6s prop %-22s ue %s" % (
            group,
            cell("sweep", "prop_delay_veh_hours", group),
            cell("sweep", "ue_delay_veh_hours", group)))
    lines.append("")
    lines.append("Notes")
    lines.append("-" * 70)
    lines.append("- Satisfaction is the share of guided agents whose actual")
    lines.append("  path equals the assigned path; vehicle-hours sum travel")
    lines.append("  time over all agents in the reaction run.")
    lines.append("- Delay comes from linkmoe_summary.txt in each seed dir;")
    lines.append("  if a seed has none, its delay is n/a and n is smaller.")
    lines.append("- UE sweep vehicle-hours are computed from the UE sweep")
    lines.append("  agent files, the same way ue_vehhrs.py computes them.")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text)
    return text


def write_summary_csv(path, seeds, rows):
    """One row per metric: mean, SD, n, and the per-seed values."""
    header = (["section", "metric", "group", "mean", "sd", "n"]
              + ["seed_%d" % s for s in seeds])
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        w = csv.writer(f)
        w.writerow(header)
        for section, metric, group, mean, sd, n, values in rows:
            w.writerow([section, metric, group,
                        "" if mean is None else "%.4f" % mean,
                        "" if sd is None else "%.4f" % sd, n]
                       + ["" if v is None else "%.4f" % v for v in values])


def main(argv):
    seed_dirs = []
    seeds = []
    out_dir = None
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
                    print("ERROR: --seeds values must be integers", file=sys.stderr)
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
        if tok in ("-h", "--help"):
            print("Usage: python multiseed_aggregate.py --seed-dirs <dir>... "
                  "--seeds <int>... [--out-dir <dir>]")
            return 0
        print("ERROR: unknown argument: %s" % tok, file=sys.stderr)
        return 2
    if not seed_dirs:
        print("ERROR: no --seed-dirs given", file=sys.stderr)
        return 2
    if seeds and len(seeds) != len(seed_dirs):
        print("ERROR: --seeds and --seed-dirs must have the same length",
              file=sys.stderr)
        return 2
    if not seeds:
        seeds = []
        for d in seed_dirs:
            m = re.search(r"_seed(\d+)$", os.path.basename(os.path.normpath(d)))
            seeds.append(int(m.group(1)) if m else len(seeds) + 1)
    for d in seed_dirs:
        if not os.path.isdir(d):
            print("ERROR: seed results dir not found: %s" % d, file=sys.stderr)
            return 2
    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(seed_dirs[0])), "multiseed")
    os.makedirs(out_dir, exist_ok=True)
    seed_data = {}
    for s, d in zip(seeds, seed_dirs):
        print("seed %d: reading %s" % (s, d))
        try:
            seed_data[s] = collect_seed(d)
        except Exception as e:
            print("WARNING: seed %d: could not read %s: %s" % (s, d, e),
                  file=sys.stderr)
            seed_data[s] = {"guidance": {}, "sweep": {}}
    rows = aggregate(seeds, seed_data)
    txt_path = os.path.join(out_dir, "multiseed_summary.txt")
    csv_path = os.path.join(out_dir, "multiseed_summary.csv")
    write_summary_txt(txt_path, seeds, seed_dirs, rows)
    write_summary_csv(csv_path, seeds, rows)
    print("Wrote %s" % txt_path)
    print("Wrote %s" % csv_path)
    missing = [r for r in rows if r[3] is None]
    if missing:
        print("WARNING: %d metrics had no data in any seed" % len(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
