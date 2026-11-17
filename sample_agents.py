"""Deterministic stratified sampler for the wholerunner v2 agent pool.

Samples N agents from a pool file (one input_agents CSV, same format as
DTALite_Files/input_agent_initial.csv) and writes exactly N rows (+ header)
to the output file, same columns and order, no column changes.

Stratification is within OD cells (from_zone_id, to_zone_id): each cell
contributes its proportional share of N, so the demand pattern is preserved
in expectation and the per-seed jitter from sampling is the OD randomness.

Determinism: the RNG is derived from (seed, level) as
np.random.default_rng([seed, level]), so the same (seed, level, pool file)
always gives a byte-identical output, and each demand level is an
independent draw rather than a reuse of the same RNG stream.

The pool file must exist. A missing pool file is a loud failure: this
module never falls back to another file.
"""

import argparse
import sys

import numpy as np
import pandas as pd

OD_COLUMNS = ["from_zone_id", "to_zone_id"]

EXPECTED_COLUMNS = [
    "agent_id",
    "tour_id",
    "from_zone_id",
    "to_zone_id",
    "from_origin_node_id",
    "to_destination_node_id",
    "departure_time_in_min",
    "demand_type",
    "PCE",
    "information_type",
    "vehicle_age",
    " path_node_sequence",
    "vehicle_type",
    "pricing_type",
    "value_of_time",
]


def _sort_key(cell):
    """Sort OD cells numerically when the zone ids are integers."""
    try:
        return (int(cell[0]), int(cell[1]))
    except (TypeError, ValueError):
        return (str(cell[0]), str(cell[1]))


def _largest_remainder_counts(cell_sizes, n_agents):
    """Proportional per-cell counts that sum to exactly n_agents.

    Each cell gets floor(quota) rows, then the leftover rows go one each
    to the cells with the largest fractional remainders (ties broken by
    cell order, which is fixed, so the result is deterministic).
    """
    total = sum(cell_sizes)
    quotas = [n_agents * size / total for size in cell_sizes]
    counts = [int(q) for q in quotas]
    leftover = n_agents - sum(counts)
    order = sorted(
        range(len(cell_sizes)),
        key=lambda i: (-(quotas[i] - counts[i]), i),
    )
    for i in order[:leftover]:
        counts[i] += 1
    return counts


def sample_pool(pool_csv, seed, n_agents, out_csv, level=0):
    """Sample n_agents rows from pool_csv and write them to out_csv.

    pool_csv: path to the agent pool file (must exist).
    seed: the run seed S; the one seed driving all randomness.
    n_agents: number of rows to sample (must be 1..pool size).
    out_csv: path of the sampled output file.
    level: demand level (e.g. 20000); combined with seed to derive the
        per-level RNG, so each level is an independent draw.

    Returns the number of rows written (== n_agents).
    Raises FileNotFoundError if the pool file is missing, ValueError for
    a bad n_agents or a malformed pool file.
    """
    try:
        with open(pool_csv, "rb") as f:
            header_line = f.readline()
    except FileNotFoundError:
        raise FileNotFoundError(
            "Pool file not found: %s\n"
            "Generate the agent pool and place it at %s "
            "(default location: <data-dir>/input_agent_pool.csv). "
            "Sampling cannot proceed without the pool file."
            % (pool_csv, pool_csv)
        ) from None
    if not header_line:
        raise ValueError("Pool file is empty: %s" % pool_csv)
    lineterm = "\r\n" if header_line.endswith(b"\r\n") else "\n"
    header = header_line.decode("utf-8").rstrip("\r\n")

    pool = pd.read_csv(pool_csv, dtype=str, keep_default_na=False)
    if list(pool.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Pool file %s does not have the expected 15 columns in order; "
            "got: %s" % (pool_csv, list(pool.columns))
        )
    n_pool = len(pool)
    if n_pool == 0:
        raise ValueError("Pool file has no data rows: %s" % pool_csv)
    if not isinstance(n_agents, (int, np.integer)) or isinstance(n_agents, bool):
        raise ValueError("n_agents must be an integer, got %r" % (n_agents,))
    n_agents = int(n_agents)
    if n_agents < 1:
        raise ValueError("n_agents must be at least 1, got %d" % n_agents)
    if n_agents > n_pool:
        raise ValueError(
            "Cannot sample %d agents from a pool of %d (no replacement); "
            "generate a larger pool or request fewer agents." % (n_agents, n_pool)
        )

    grouped = pool.groupby(OD_COLUMNS, sort=False).indices
    cells = sorted(grouped.keys(), key=_sort_key)
    cell_sizes = [len(grouped[cell]) for cell in cells]
    counts = _largest_remainder_counts(cell_sizes, n_agents)

    rng = np.random.default_rng([int(seed), int(level)])
    picked = []
    for cell, k in zip(cells, counts):
        if k == 0:
            continue
        positions = grouped[cell]
        picked.extend(rng.choice(positions, size=k, replace=False).tolist())
    picked.sort()

    sample = pool.iloc[picked]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write(header + lineterm)
        sample.to_csv(f, index=False, header=False, lineterminator=lineterm)
    return n_agents


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample N agents from the wholerunner v2 agent pool."
    )
    parser.add_argument("--pool-file", required=True,
                        help="path to the agent pool CSV (input_agent_pool.csv)")
    parser.add_argument("--seed", required=True, type=int,
                        help="run seed S; drives all randomness")
    parser.add_argument("--n", required=True, type=int,
                        help="number of agents to sample")
    parser.add_argument("--out", required=True,
                        help="path of the sampled output CSV")
    parser.add_argument("--level", type=int, default=0,
                        help="demand level, combined with --seed to derive "
                             "the per-level RNG (default: 0)")
    args = parser.parse_args(argv)
    try:
        n = sample_pool(args.pool_file, args.seed, args.n, args.out,
                        level=args.level)
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    print("Sampled %d agents (seed=%d, level=%d) from %s -> %s"
          % (n, args.seed, args.level, args.pool_file, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
