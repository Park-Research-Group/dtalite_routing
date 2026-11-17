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

Preference weights travel with the agent, keyed by agent_id, never
by row position. When a full-pool agent_weights.csv (header
agent_id,pref) sits next to the pool file, sampling also writes the
sampled rows -- one (agent_id, pref) row per sampled agent, in the
same order as the sampled agent rows -- to a weights CSV derived
from the output path (same stem with a _weights suffix), unless an
explicit output path is given. msa.py then looks the preference up
by agent_id, so the same agent_id always gets the same pref.
"""

import argparse
import os
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


def default_weights_path(pool_csv):
    """agent_weights.csv sitting next to the pool file."""
    return os.path.join(os.path.dirname(os.path.abspath(pool_csv)),
                        "agent_weights.csv")


def weights_out_path(out_csv):
    """Sampled weights path derived from the sampled agent CSV path:
    same directory and stem, with a _weights suffix."""
    stem, ext = os.path.splitext(out_csv)
    return stem + "_weights" + ext


def _write_sample_weights(pool_csv, weights_csv, out_weights_csv,
                          out_csv, sample, lineterm):
    """Write the sampled agent_weights rows for the sampled agents.

    weights_csv: the full-pool agent_weights.csv (header
        agent_id,pref). Defaults to agent_weights.csv next to the
        pool file. When it does not exist, warn on stderr and write
        nothing: old pools have no weights file, and the sampler
        must still work for them (msa.py then falls back to the
        users sheet). When it exists, every sampled agent_id must
        have exactly one pref row; a missing or duplicated id is a
        loud failure, never a silent default.
    out_weights_csv: where the sampled rows go. Defaults to the
        sampled agent CSV path with a _weights suffix.
    The sampled rows follow the sampled agent rows in order, one
    (agent_id, pref) row per sampled agent, keyed by agent_id.
    Returns the weights path written, or None when skipped.
    """
    if weights_csv is None:
        weights_csv = default_weights_path(pool_csv)
    if not os.path.isfile(weights_csv):
        print("WARNING: no agent_weights file at %s; sampled agents "
              "will have no weights file (msa.py falls back to the "
              "users sheet)" % weights_csv, file=sys.stderr)
        return None
    if out_weights_csv is None:
        out_weights_csv = weights_out_path(out_csv)
    wdf = pd.read_csv(weights_csv, dtype=str, keep_default_na=False)
    if list(wdf.columns) != ["agent_id", "pref"]:
        raise ValueError(
            "Weights file %s must have header agent_id,pref; got: %s"
            % (weights_csv, list(wdf.columns)))
    if wdf["agent_id"].duplicated().any():
        raise ValueError(
            "Weights file %s has duplicated agent_ids; one agent must "
            "map to exactly one pref" % weights_csv)
    pref_by_agent = dict(zip(wdf["agent_id"], wdf["pref"]))
    missing = [aid for aid in sample["agent_id"]
               if aid not in pref_by_agent]
    if missing:
        raise ValueError(
            "Weights file %s has no pref for %d sampled agent(s), "
            "e.g. agent_id %s" % (weights_csv, len(missing), missing[0]))
    with open(out_weights_csv, "w", encoding="utf-8", newline="") as f:
        f.write("agent_id,pref" + lineterm)
        for aid in sample["agent_id"]:
            f.write("%s,%s%s" % (aid, pref_by_agent[aid], lineterm))
    return out_weights_csv


def sample_pool(pool_csv, seed, n_agents, out_csv, level=0,
                weights_csv=None, out_weights_csv=None):
    """Sample n_agents rows from pool_csv and write them to out_csv.

    pool_csv: path to the agent pool file (must exist).
    seed: the run seed S; the one seed driving all randomness.
    n_agents: number of rows to sample (must be 1..pool size).
    out_csv: path of the sampled output file.
    level: demand level (e.g. 20000); combined with seed to derive the
        per-level RNG, so each level is an independent draw.
    weights_csv: path to the full-pool agent_weights.csv (header
        agent_id,pref); defaults to agent_weights.csv next to the
        pool file. When it exists, the sampled weights are written
        too (see _write_sample_weights); when it does not exist,
        sampling still succeeds and only the agents are written.
    out_weights_csv: path of the sampled weights output file;
        defaults to the sampled agent CSV path with a _weights
        suffix (see weights_out_path).

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
    _write_sample_weights(pool_csv, weights_csv, out_weights_csv,
                          out_csv, sample, lineterm)
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
    parser.add_argument("--weights-file", default=None,
                        help="path to the full-pool agent_weights.csv "
                             "(default: agent_weights.csv next to "
                             "--pool-file)")
    parser.add_argument("--out-weights", default=None,
                        help="path of the sampled agent_weights CSV "
                             "(default: <out> stem + _weights)")
    args = parser.parse_args(argv)
    try:
        n = sample_pool(args.pool_file, args.seed, args.n, args.out,
                        level=args.level, weights_csv=args.weights_file,
                        out_weights_csv=args.out_weights)
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    print("Sampled %d agents (seed=%d, level=%d) from %s -> %s"
          % (n, args.seed, args.level, args.pool_file, args.out))
    wout = weights_out_path(args.out) if args.out_weights is None \
        else args.out_weights
    if os.path.isfile(wout):
        print("Sampled weights -> %s" % wout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
