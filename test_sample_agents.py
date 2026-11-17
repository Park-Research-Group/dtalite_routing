"""Tests for sample_agents.py.

Run: python3 test_sample_agents.py
Uses the clone's input_agent_initial.csv as a stand-in pool (read-only)
plus a tiny synthetic pool built in a temp dir. No DTALite run.
"""

import math
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import sample_agents
from sample_agents import sample_pool

STANDIN_POOL = os.path.join(
    os.path.expanduser("~"),
    "workspace/projects/phd-dtalite/DTALite_Files/input_agent_initial.csv",
)

HEADER = (
    "agent_id,tour_id,from_zone_id,to_zone_id,from_origin_node_id,"
    "to_destination_node_id,departure_time_in_min,demand_type,PCE,"
    "information_type,vehicle_age, path_node_sequence,vehicle_type,"
    "pricing_type,value_of_time"
)


def _synthetic_pool(path, rows_per_cell=((1, 2, 10), (1, 3, 7), (2, 1, 5))):
    """Tiny pool with known OD cells: (from_zone, to_zone, n_rows)."""
    lines = [HEADER]
    agent = 0
    for fz, tz, n in rows_per_cell:
        for _ in range(n):
            lines.append(
                "%d,-1,%d,%d,%d,%d,360.0,1,1,4,5,%d;%d;,1,1,10"
                % (agent, fz, tz, fz, tz, fz, tz)
            )
            agent += 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")
    return agent


class TestSampleAgents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = lambda name: os.path.join(self.tmp.name, name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_deterministic_same_seed_level_pool(self):
        """Same (seed, level, pool) -> byte-identical output."""
        a, b = self.out("a.csv"), self.out("b.csv")
        sample_pool(STANDIN_POOL, 42, 5000, a, level=50000)
        sample_pool(STANDIN_POOL, 42, 5000, b, level=50000)
        with open(a, "rb") as f:
            bytes_a = f.read()
        with open(b, "rb") as f:
            bytes_b = f.read()
        self.assertEqual(bytes_a, bytes_b)

    def test_different_level_different_sample(self):
        """Per-level RNG: same seed, different level -> different sample."""
        a, b = self.out("a.csv"), self.out("b.csv")
        sample_pool(STANDIN_POOL, 42, 5000, a, level=50000)
        sample_pool(STANDIN_POOL, 42, 5000, b, level=30000)
        with open(a, "rb") as f:
            bytes_a = f.read()
        with open(b, "rb") as f:
            bytes_b = f.read()
        self.assertNotEqual(bytes_a, bytes_b)

    def test_different_seed_different_sample(self):
        """Different seed -> different sample."""
        a, b = self.out("a.csv"), self.out("b.csv")
        sample_pool(STANDIN_POOL, 42, 5000, a, level=50000)
        sample_pool(STANDIN_POOL, 423, 5000, b, level=50000)
        with open(a, "rb") as f:
            bytes_a = f.read()
        with open(b, "rb") as f:
            bytes_b = f.read()
        self.assertNotEqual(bytes_a, bytes_b)

    def test_exact_n_rows_and_header(self):
        """Exactly N rows + header, header byte-identical to the pool."""
        out = self.out("s.csv")
        n = sample_pool(STANDIN_POOL, 42, 5000, out, level=50000)
        self.assertEqual(n, 5000)
        with open(out, "rb") as f:
            raw = f.read()
        lines = raw.split(b"\r\n")
        self.assertEqual(lines[0].decode("utf-8"), HEADER)
        self.assertEqual(len([ln for ln in lines if ln]), 5001)
        df = pd.read_csv(out, dtype=str, keep_default_na=False)
        self.assertEqual(len(df), 5000)
        self.assertEqual(list(df.columns), sample_agents.EXPECTED_COLUMNS)

    def test_sample_is_row_subset_no_duplicates(self):
        """Sampled rows exist verbatim in the pool; no agent sampled twice."""
        out = self.out("s.csv")
        sample_pool(STANDIN_POOL, 42, 5000, out, level=50000)
        pool = pd.read_csv(STANDIN_POOL, dtype=str, keep_default_na=False)
        sample = pd.read_csv(out, dtype=str, keep_default_na=False)
        self.assertEqual(sample["agent_id"].nunique(), 5000)
        self.assertTrue(set(sample["agent_id"]) <= set(pool["agent_id"]))
        merged = sample.merge(pool, on=list(pool.columns), how="left",
                              indicator=True)
        self.assertTrue((merged["_merge"] == "both").all())

    def test_od_proportions_preserved(self):
        """Each OD cell contributes its proportional share (within 1 row
        of the exact quota), so sampled shares track pool shares."""
        out = self.out("s.csv")
        n = 5000
        sample_pool(STANDIN_POOL, 42, n, out, level=50000)
        pool = pd.read_csv(STANDIN_POOL, dtype=str, keep_default_na=False,
                           usecols=["from_zone_id", "to_zone_id"])
        sample = pd.read_csv(out, dtype=str, keep_default_na=False,
                             usecols=["from_zone_id", "to_zone_id"])
        pool_counts = pool.groupby(["from_zone_id", "to_zone_id"]).size()
        sample_counts = sample.groupby(["from_zone_id", "to_zone_id"]).size()
        total = len(pool)
        max_share_diff = 0.0
        for cell, pool_n in pool_counts.items():
            quota = n * pool_n / total
            got = int(sample_counts.get(cell, 0))
            self.assertIn(got, (math.floor(quota), math.ceil(quota)),
                          "cell %s: got %d, quota %.3f" % (cell, got, quota))
            max_share_diff = max(max_share_diff,
                                 abs(got / n - pool_n / total))
        # Largest-remainder allocation keeps every share within one row
        # of its quota; 1/5000 = 0.0002 bounds the share error.
        self.assertLessEqual(max_share_diff, 1.0 / n + 1e-12)
        print("max OD share deviation: %.6f (bound %.6f)"
              % (max_share_diff, 1.0 / n))

    def test_full_pool_sample_reproduces_pool(self):
        """Sampling the whole synthetic pool returns it byte-identical."""
        pool_path = self.out("pool.csv")
        total = _synthetic_pool(pool_path)
        out = self.out("s.csv")
        sample_pool(pool_path, 7, total, out, level=20000)
        with open(pool_path, "rb") as f:
            pool_bytes = f.read()
        with open(out, "rb") as f:
            out_bytes = f.read()
        self.assertEqual(out_bytes, pool_bytes)

    def test_small_synthetic_stratification(self):
        """Tiny pool: N=11 of 22 rows keeps the 10/7/5 split at 5/3-4/2-3."""
        pool_path = self.out("pool.csv")
        _synthetic_pool(pool_path)
        out = self.out("s.csv")
        sample_pool(pool_path, 1, 11, out, level=20000)
        sample = pd.read_csv(out, dtype=str, keep_default_na=False,
                             usecols=["from_zone_id", "to_zone_id"])
        counts = sample.groupby(["from_zone_id", "to_zone_id"]).size()
        self.assertEqual(int(counts[("1", "2")]), 5)   # quota 11*10/22 = 5
        self.assertEqual(int(counts[("2", "1")]), 2)   # quota 11*5/22 = 2.5
        self.assertEqual(int(counts[("1", "3")]), 4)   # quota 11*7/22 = 3.5

    def test_missing_pool_file_fails_loudly(self):
        """Missing pool file -> FileNotFoundError naming the pool path."""
        missing = os.path.join(self.tmp.name, "input_agent_pool.csv")
        with self.assertRaises(FileNotFoundError) as ctx:
            sample_pool(missing, 42, 100, self.out("s.csv"), level=50000)
        self.assertIn("input_agent_pool.csv", str(ctx.exception))
        self.assertIn("Pool file not found", str(ctx.exception))
        self.assertFalse(os.path.exists(self.out("s.csv")))

    def test_missing_pool_file_cli_fails_loudly(self):
        """CLI with a missing pool file exits non-zero with a clear error."""
        proc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "sample_agents.py"),
             "--pool-file", os.path.join(self.tmp.name, "input_agent_pool.csv"),
             "--seed", "42", "--n", "100",
             "--out", self.out("s.csv"), "--level", "50000"],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Pool file not found", proc.stderr)
        self.assertIn("input_agent_pool.csv", proc.stderr)

    def test_bad_n_fails_loudly(self):
        """n=0 and n > pool size both fail instead of writing a bad file."""
        pool_path = self.out("pool.csv")
        total = _synthetic_pool(pool_path)
        with self.assertRaises(ValueError):
            sample_pool(pool_path, 1, 0, self.out("s.csv"))
        with self.assertRaises(ValueError):
            sample_pool(pool_path, 1, total + 1, self.out("s.csv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
