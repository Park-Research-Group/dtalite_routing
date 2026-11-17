"""Build a bootstrapped agent pool for the DTALite wholerunner v2 pipeline.

Resamples rows with replacement from the 54k Sioux Falls agent file to make
a pool of N agents, assigns fresh agent_ids 0..N-1, and draws one preference
index per pool agent uniformly from the dataset-28 half of the weights sheet
(rows 1..16650). Both draws come from a single numpy default_rng(seed), row
resampling first, then preference draws, so the same seed and inputs always
produce byte-identical outputs.

Usage:
    python3 build_pool.py [--source PATH] [--pool-size N] [--seed S]
                          [--weights PATH] [--out-dir DIR]
"""
import argparse
import csv
import os
import sys

import numpy as np
import openpyxl

DEFAULT_SOURCE = os.path.expanduser(
    "~/workspace/user/files/input_agent_initial.csv")
DEFAULT_WEIGHTS = os.path.expanduser(
    "~/workspace/projects/phd-dtalite/DTALite_Files/SiouxFalls_net.xlsx")
WEIGHTS_SHEET = "weights"
DATASET28_ROWS = 16650  # rows 1..16650 of the weights sheet are dataset 28
POOL_CSV = "input_agent_pool.csv"
WEIGHTS_CSV = "agent_weights.csv"


def fail(message):
    print("build_pool: error: " + message, file=sys.stderr)
    sys.exit(1)


def count_weights_data_rows(workbook_path):
    """Return the number of data rows in the weights sheet.

    The sheet has a few preamble rows (a note, scale rows, a blank row) and
    then a header row whose first cell is 'distance'; the data rows follow
    it. Only rows with a value in the first column are counted.
    """
    try:
        workbook = openpyxl.load_workbook(
            workbook_path, read_only=True, data_only=True)
    except Exception as exc:
        fail("cannot open weights workbook %r: %s" % (workbook_path, exc))
    if WEIGHTS_SHEET not in workbook.sheetnames:
        fail("weights workbook %r has no sheet named %r"
             % (workbook_path, WEIGHTS_SHEET))
    sheet = workbook[WEIGHTS_SHEET]
    header_seen = False
    data_rows = 0
    for row in sheet.iter_rows(min_col=1, max_col=1, values_only=True):
        value = row[0]
        if not header_seen:
            if value == "distance":
                header_seen = True
            continue
        if value is not None:
            data_rows += 1
    workbook.close()
    if not header_seen:
        fail("weights sheet %r has no 'distance' header row; cannot locate "
             "the data rows" % WEIGHTS_SHEET)
    return data_rows


def read_source_agents(source_path):
    """Read the source agent CSV, returning (raw header line, data rows).

    Rows are kept as plain strings so every field is written back exactly
    as it appeared in the source; only agent_id (column 0) is replaced.
    The raw header line is kept verbatim, including the leading space in
    the path_node_sequence column name.
    """
    with open(source_path, "r", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            fail("source agent CSV %r is empty" % source_path)
        header = next(csv.reader([header_line.rstrip("\r\n")]))
        if len(header) != 15:
            fail("source agent CSV %r has %d columns, expected 15"
                 % (source_path, len(header)))
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        fail("source agent CSV %r has no agent rows" % source_path)
    for index, row in enumerate(rows):
        if len(row) != 15:
            fail("source agent CSV %r row %d has %d columns, expected 15"
                 % (source_path, index + 2, len(row)))
    return header_line.rstrip("\r\n"), rows


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap a pool of N agents from the 54k agent file "
                    "and draw one preference index per pool agent.")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="source agent CSV (default: %(default)s)")
    parser.add_argument("--pool-size", type=int, default=200000,
                        help="number of agents in the pool (default: "
                             "%(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed; same seed + same inputs gives "
                             "byte-identical outputs (default: %(default)s)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="weights workbook, used only to validate that "
                             "the dataset-28 half exists "
                             "(default: %(default)s)")
    parser.add_argument("--out-dir", default=".",
                        help="directory for the two output CSVs "
                             "(default: %(default)s)")
    args = parser.parse_args()

    if args.pool_size <= 0:
        fail("--pool-size must be positive, got %d" % args.pool_size)
    if not os.path.isfile(args.source):
        fail("source agent CSV not found: %s" % args.source)
    if not os.path.isfile(args.weights):
        fail("weights workbook not found: %s" % args.weights)

    data_rows = count_weights_data_rows(args.weights)
    if data_rows < DATASET28_ROWS:
        fail("weights sheet %r has only %d data rows, fewer than the %d "
             "needed for the dataset-28 half"
             % (WEIGHTS_SHEET, data_rows, DATASET28_ROWS))

    header_line, source_rows = read_source_agents(args.source)
    n_source = len(source_rows)

    rng = np.random.default_rng(args.seed)
    # Draw order is fixed: row resampling first, preference draws second.
    sampled = rng.integers(0, n_source, size=args.pool_size)
    prefs = rng.integers(1, DATASET28_ROWS + 1, size=args.pool_size)

    os.makedirs(args.out_dir, exist_ok=True)
    pool_path = os.path.join(args.out_dir, POOL_CSV)
    weights_path = os.path.join(args.out_dir, WEIGHTS_CSV)

    with open(pool_path, "w", newline="") as handle:
        handle.write(header_line + "\r\n")
        writer = csv.writer(handle, lineterminator="\r\n")
        for new_id, source_index in enumerate(sampled):
            row = list(source_rows[source_index])
            row[0] = str(new_id)
            writer.writerow(row)

    with open(weights_path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(["agent_id", "pref"])
        for new_id, pref in enumerate(prefs):
            writer.writerow([new_id, int(pref)])

    print("pool: %d agents from %d source rows (seed %d)"
          % (args.pool_size, n_source, args.seed))
    print("wrote %s" % pool_path)
    print("wrote %s" % weights_path)


if __name__ == "__main__":
    main()
