#!/usr/bin/env python3
"""Congestion (V/C) metrics from a realized DTALite output_LinkMOE.csv.

Per link, V/C = total_link_volume / link_capacity_in_vhc_per_hour; both
columns are in output_LinkMOE.csv. System-level numbers per run:
  - max V/C (peak congestion) and p95 V/C (robust peak, not driven by
    a single link)
  - over-capacity count: links with V/C > 1.0
  - heavily-loaded count: links with V/C > 0.8
  - mean V/C (tracks the demand level more than the method)
  - total delay (veh-hrs) = sum((travel_time - free-flow_travel_time)
    * total_link_volume) / 60  [the established linkmoe_delay.py formula]
  - links used: links with volume > 0

The formulas are the ones in congestion_metric.md. The p95 convention
is the one in the reference compute_congestion.py: sort the V/C values
and take index int(0.95 * n). The V/C is period-aggregated over the
whole 1-hour demand window, not time-resolved.

Usable two ways: imported by aggregate_v2.py, which fills the max_vc /
p95_vc / over_cap / heavy columns of a per-seed summary.csv from the
run's LinkMOE files when a row lacks them, and standalone:

    python congestion.py <output_LinkMOE.csv> [--out vc_table.csv]

With --out, the per-link V/C table (from_node, to_node, volume,
capacity, V/C) is written to the given CSV path.

Stdlib only. Read-only on the input file. Handles the DTALite
trailing-comma quirk (duplicated headers keep their first occurrence,
the same as volume_estimation.py) and headers with stray spaces.
"""

import csv
import os
import sys

# Column names as they appear in output_LinkMOE.csv (matched case- and
# whitespace-insensitively; DTALite pads some names, e.g. " speed_limit").
COL_FROM = "from_node_id"
COL_TO = "to_node_id"
COL_VOLUME = "total_link_volume"
COL_CAPACITY = "link_capacity_in_vhc_per_hour"
COL_TT = "travel_time"
COL_FFTT = "free-flow_travel_time"

OVER_CAPACITY = 1.0   # V/C above this counts as over capacity
HEAVILY_LOADED = 0.8  # V/C above this counts as heavily loaded


def warn(msg):
    sys.stderr.write("WARNING: " + msg + "\n")


def load_linkmoe(path):
    """Load the rows of a LinkMOE CSV as a list of dicts.

    Each dict has from_node, to_node, volume, capacity, travel_time,
    free_flow_time. travel_time / free_flow_time are None when those
    columns are absent (delay then comes back as None). Rows that fail
    float conversion, have a blank node id, or a non-positive capacity
    are skipped with a warning. Raises ValueError when a required
    column (nodes, volume, capacity) is missing, OSError when the
    file cannot be read.
    """
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        col = {}
        for raw in reader:
            if not col:
                for i, name in enumerate(raw):
                    norm = name.strip().lower()
                    if norm and norm not in col:
                        col[norm] = i
                missing = [c for c in (COL_FROM, COL_TO, COL_VOLUME,
                                       COL_CAPACITY) if c not in col]
                if missing:
                    raise ValueError(
                        "%s: missing required column(s): %s"
                        % (path, ", ".join(missing)))
                continue
            if not raw or not any(cell.strip() for cell in raw):
                continue
            try:
                from_node = raw[col[COL_FROM]].strip()
                to_node = raw[col[COL_TO]].strip()
                volume = float(raw[col[COL_VOLUME]].strip())
                capacity = float(raw[col[COL_CAPACITY]].strip())
            except (IndexError, ValueError):
                warn("skipping malformed row in %s" % path)
                continue
            if not from_node or not to_node:
                warn("skipping row with blank node id in %s" % path)
                continue
            if capacity <= 0:
                warn("skipping row with non-positive capacity in %s "
                     "(link %s->%s)" % (path, from_node, to_node))
                continue
            tt = fft = None
            if COL_TT in col and COL_FFTT in col:
                try:
                    tt = float(raw[col[COL_TT]].strip())
                    fft = float(raw[col[COL_FFTT]].strip())
                except (IndexError, ValueError):
                    tt = fft = None
            if volume != volume or capacity != capacity:
                continue  # NaN volume or capacity
            rows.append({"from_node": from_node, "to_node": to_node,
                         "volume": volume, "capacity": capacity,
                         "travel_time": tt, "free_flow_time": fft})
    return rows


def compute_congestion(rows_or_path):
    """Congestion metrics for one realized LinkMOE file.

    Accepts either a path to the CSV or a list of row dicts as returned
    by load_linkmoe(). Returns a dict with:
      n_links, links_used, max_vc, p95_vc, mean_vc, over_cap, heavy,
      delay (veh-hrs, or None when travel-time columns are absent),
      table: per-link rows {from_node, to_node, volume, capacity, vc}.
    max_vc / p95_vc / mean_vc are None when the file has no usable rows.
    """
    if isinstance(rows_or_path, str):
        rows = load_linkmoe(rows_or_path)
    else:
        rows = rows_or_path
    table = []
    for r in rows:
        table.append({"from_node": r["from_node"], "to_node": r["to_node"],
                      "volume": r["volume"], "capacity": r["capacity"],
                      "vc": r["volume"] / r["capacity"]})
    vcs = sorted(t["vc"] for t in table)
    delay = None
    if rows and all(r["travel_time"] is not None
                    and r["free_flow_time"] is not None for r in rows):
        delay = sum((r["travel_time"] - r["free_flow_time"]) * r["volume"]
                    for r in rows) / 60.0
    elif rows:
        # Travel-time columns present on some rows only: use the rows
        # that have them rather than dropping delay entirely.
        usable = [r for r in rows if r["travel_time"] is not None
                  and r["free_flow_time"] is not None]
        if usable:
            delay = sum((r["travel_time"] - r["free_flow_time"])
                        * r["volume"] for r in usable) / 60.0
    result = {
        "n_links": len(table),
        "links_used": sum(1 for t in table if t["volume"] > 0),
        "max_vc": max(vcs) if vcs else None,
        "p95_vc": vcs[int(0.95 * len(vcs))] if vcs else None,
        "mean_vc": (sum(vcs) / len(vcs)) if vcs else None,
        "over_cap": sum(1 for v in vcs if v > OVER_CAPACITY),
        "heavy": sum(1 for v in vcs if v > HEAVILY_LOADED),
        "delay": delay,
        "table": table,
    }
    return result


def vc_rows(path):
    """Per-link V/C rows for a LinkMOE file, as a list of dicts."""
    return compute_congestion(path)["table"]


def write_vc_table(path, table):
    """Write per-link V/C rows to CSV (CRLF, csv module default)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from_node_id", "to_node_id", "total_link_volume",
                    "link_capacity_in_vhc_per_hour", "vc"])
        for t in table:
            w.writerow([t["from_node"], t["to_node"],
                        "%.4f" % t["volume"], "%.4f" % t["capacity"],
                        "%.4f" % t["vc"]])


def fmt(v, nd=2):
    return "n/a" if v is None else ("%.*f" % (nd, v))


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print("Usage: python congestion.py <output_LinkMOE.csv> "
              "[--out vc_table.csv]")
        return 0 if len(argv) >= 2 else 2
    path = argv[1]
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            print("ERROR: --out needs a value", file=sys.stderr)
            return 2
        out = argv[i + 1]
    if not os.path.isfile(path):
        print("ERROR: LinkMOE file not found: %s" % path, file=sys.stderr)
        return 1
    try:
        res = compute_congestion(path)
    except (OSError, ValueError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    if res["n_links"] == 0:
        print("ERROR: no usable link rows in %s" % path, file=sys.stderr)
        return 1
    print("%s: %d links (%d used), delay %s veh-hrs, max V/C %s, "
          "p95 V/C %s, mean V/C %s, V/C>1.0 %d, V/C>0.8 %d"
          % (os.path.basename(path), res["n_links"], res["links_used"],
             fmt(res["delay"], 1), fmt(res["max_vc"]), fmt(res["p95_vc"]),
             fmt(res["mean_vc"]), res["over_cap"], res["heavy"]))
    if out:
        write_vc_table(out, res["table"])
        print("Wrote %s (%d rows)" % (out, len(res["table"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
