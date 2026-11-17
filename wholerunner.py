#!/usr/bin/env python3
"""
wholerunner.py -- Run the new Tests 1-4 in parallel sandboxes (v2 revamp).

Tests at exactly 50,000 agents sampled per seed from a ~200k-agent pool
(--pool-file; --baseline-level overrides the 50,000 default), plus a
demand-level sweep:

  Test 1 -- Baseline at 50k: all four systems, each in its own sandbox:
            Proposed (v9 pipeline, bigloop 10), Real-time 1-min,
            Real-time 10-min, UE guidance (scenarios + preference_reaction).
  Test 2 -- MPR sweep at the 50k baseline (old Test 5): Proposed bigloops
            0-10 plus UE-guidance bigloops 1-10, then mpr_aggregate.py.
  Test 3 -- Demand levels (default 20k, 30k, 40k, 50k, 60k, 70k, 80k).
            At EACH level: the four systems plus the MPR sweep at 33% and
            66% (proposed AND UE). The 50k level is re-run, not reused.
  Test 4 -- Link-volume validation (old Test 6) at the 50k baseline only:
            volume_estimation.py over Test 1's four systems. Per-level
            prediction accuracy comes from Test 3's saved predicted/actual
            LinkMOE files, not from a separate Test 4 per level.

Each job gets its own sandbox under temp/run_<pid>/<name>/: a private copy
of DTALite_Files/ (the real one stays read-only) plus the scripts that job
needs, with the per-(seed, level) sampled agent file injected as
input_agent_initial.csv. Jobs run concurrently as separate processes --
no shared writable state -- and all outputs are collected under
<results-dir>/.

Every job appends its row(s) to the per-seed master summary at
<results-dir>_seed<S>/summary.csv (see SUMMARY_COLUMNS below). That file
is the contract with the results/aggregation agent: exactly those
columns, in that order.

While the jobs run, a live progress display shows one bar per job,
refreshed every 2 seconds (main thread). Subprocess stdout is captured
via run.log files, parsed for progress markers. Use --no-progress to
disable the display (e.g. when redirecting output to a file).

Usage:
  python wholerunner.py --seed 42                 # single seed, all tests
  python wholerunner.py                            # all 10 default seeds
  python wholerunner.py --tests 1                  # Test 1 only
  python wholerunner.py --tests 1,3 --levels 20000,50000
  python wholerunner.py --tests 2 --skip-aggregation
  python wholerunner.py --data-dir DTALite_Files --results-dir results
  python wholerunner.py --pool-file pool.csv --seeds 42 43

Outputs per test under <results-dir>/ (per seed; see summary.csv):
  test1/proposed/  proposed_summary.txt, full_results/ (bigloop-10 CSVs)
  test1/rt1/       realtime_routes.csv, reaction_summary.txt,
                   predicted_LinkMOE.csv, full_results/
  test1/rt10/      (same as rt1)
  test1/ue/        ue_routes.csv, reaction_summary.txt,
                   predicted_LinkMOE.csv, full_results/
  test2/proposed_sweep/bigloop_0..10/, ue_sweep/bigloop_1..10/,
                   mpr_table5.csv, mpr_table6.csv, mpr_summary.txt
  test3/level_<N>/proposed|rt1|rt10|ue/  (same layout as test1/)
  test3/level_<N>/proposed_sweep/bigloop_33|66/,
                   ue_sweep/bigloop_33|66/
  test4/           volume_estimation.csv, volume_estimation_summary.txt
  (root)           summary.csv, manifest.txt

Sandboxes live in temp/run_<pid>/<name>/ and are deleted on success unless
--keep-temp is given. On failure the sandbox is kept and its path is
printed for debugging; other jobs keep running.

Seeds: bare invocation runs the 10 default seeds (42, 423, 454, 162, 904,
916, 91, 962, 201, 624), one subprocess per seed via run_multiseed.
Pass --seed <S> for a single seed, or --seeds to override the list.
--seed and --seeds together is an error.

Python 3.9 compatible, stdlib only.
"""

import argparse
import concurrent.futures
import csv
import glob
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# Default seed list (the established 10-seed set). --seeds overrides this.
DEFAULT_SEEDS = [42, 423, 454, 162, 904, 916, 91, 962, 201, 624]

# Baseline demand: every test except Test 3 samples exactly this many
# agents per seed from the pool. Overridable with --baseline-level
# (default 50000); main() applies the flag to this global before any
# sampling or job setup runs, so every BASELINE_LEVEL use below follows
# the flag. Test 3 levels still come from --levels.
BASELINE_LEVEL = 50000

# Bingrong's satisfaction targets, used by preference_reaction
TARGETS = {
    "realtime_1min": 70.49,   # real-time 1min
    "realtime_10min": 70.19,  # real-time 10min
    "ue": 66.54,              # UE as information
}

# Test 1 (Proposed) reference targets, Sun (2018) Ch.4 Tables 8 & 9.
# Vehicle-hours target is for 54,045 agents; at 50k it is a rough
# reference only, not a pass/fail bar.
T1_SAT_RATE = 88.64      # % of agents whose actual choice matched UO guidance
T1_VEH_HRS = 24719.3     # system vehicle-hours (54,045 agents)
T1_MIN_PER_AGENT = 27.44

# v9 pipeline support scripts (shared by the Proposed jobs and sweeps)
V9_SCRIPTS = ["runner_parallel.py", "parallel_msa.py", "msa.py",
              "assignment.py", "realtimeassignment.py",
              "traveltimecal.py", "progress.py", "choiceset.py",
              "netconfig.py"]

# UE-guidance sweep scripts (the UE trio + the 5 v9 support scripts +
# netconfig.py), all next to wholerunner.py.
UE_SWEEP_SCRIPTS = ["runner_parallel_ue.py", "parallel_msa_ue.py",
                    "msa_ue.py", "assignment.py",
                    "realtimeassignment.py", "traveltimecal.py",
                    "progress.py", "choiceset.py", "netconfig.py"]

# Simulation-engine files staged into EVERY sandbox alongside the scripts
# above (setup_sandbox copies them for all jobs, so the engine is
# available wherever a script runs -- including inside RunX/ workers,
# which resolve run_pysim.py and pysim/ from engine.py's own directory).
ENGINE_FILES = ["engine.py", "run_pysim.py"]
ENGINE_DIRS = ["pysim"]

# Method labels used in summary.csv and in result directory names.
METHODS = ("proposed", "realtime_1min", "realtime_10min", "ue")
METHOD_DIRS = {"proposed": "proposed", "realtime_1min": "rt1",
               "realtime_10min": "rt10", "ue": "ue"}

# Per-seed master summary: the column list is the contract with the
# results/aggregation agent. Exactly these columns, in this order.
# Empty string marks a metric that does not apply to a row (mpr is empty
# for a non-sweep method row; linkvol_acc is empty where no
# predicted/actual LinkMOE pair was available to compute it from).
SUMMARY_COLUMNS = ["seed", "test", "level", "method", "mpr",
                   "satisfaction", "veh_hrs", "delay", "energy", "co2",
                   "nox", "co", "hc", "linkvol_acc", "max_vc", "p95_vc",
                   "over_cap", "heavy"]

# Bigloop result files collected for Proposed jobs: the 9 result CSVs
# ({b}UOinfor/predict/actual x agent/LinkMOE/LinkTDMOE).
T1_KINDS = ("agent", "LinkMOE", "LinkTDMOE")
T1_TAGS = ("UOinfor", "predict", "actual")
TAG_RE = re.compile(r"^(\d+)(UOinfor|predict|actual)_agent\.csv$")

# Test 1's UE job MSA iteration cap (matches scenarios_ue.py --max-iters).
UE_MAX_ITERS = 50

# LinkMOE filename suffixes (mirrored from volume_estimation.py).
T4_PREDICT_SUFFIX = "predict_LinkMOE.csv"
T4_ACTUAL_SUFFIX = "actual_LinkMOE.csv"


def log(msg):
    print("[wholerunner] %s" % msg, flush=True)


def fmt_min(sec):
    return "%.1f min" % (sec / 60.0)


# ----------------------------------------------------------------------
# --tests / --levels / --mpr-points spec parsing
# ----------------------------------------------------------------------
def parse_tests_spec(spec):
    """'1-3' -> [1, 2, 3]. Raises ValueError on bad input or unknown
    test numbers. Valid tests are 1-4 only; the old 1-6 numbering is gone."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
            except ValueError:
                raise ValueError("invalid range %r in --tests" % part)
            if lo > hi:
                raise ValueError("invalid range %r in --tests" % part)
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(part))
            except ValueError:
                raise ValueError("invalid test %r in --tests" % part)
    tests = sorted(set(out))
    bad = [t for t in tests if t not in (1, 2, 3, 4)]
    if bad:
        raise ValueError("unknown test(s) %s (valid: 1-4)" % bad)
    if not tests:
        raise ValueError("--tests selected no tests")
    return tests


def parse_int_list(spec, flag, lo=None, hi=None):
    """'20000,30000' -> [20000, 30000]. Raises ValueError on bad input
    or values outside [lo, hi] when bounds are given."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            raise ValueError("invalid value %r in %s" % (part, flag))
        if lo is not None and v < lo:
            raise ValueError("%s value %d is below %d" % (flag, v, lo))
        if hi is not None and v > hi:
            raise ValueError("%s value %d is above %d" % (flag, v, hi))
        out.append(v)
    out = sorted(set(out))
    if not out:
        raise ValueError("%s selected no values" % flag)
    return out

# ----------------------------------------------------------------------
# Progress tracking (thread-safe; one entry per job)
# ----------------------------------------------------------------------
class ProgressTracker:
    """Thread-safe per-job progress: fraction 0.0-1.0 plus a short
    status string. Fractions are monotonic (never move backwards)."""

    def __init__(self, keys):
        self._lock = threading.Lock()
        self._frac = {t: 0.0 for t in keys}
        self._status = {t: "starting..." for t in keys}
        self._done = {t: False for t in keys}
        self._t0 = {t: None for t in keys}

    def start(self, key):
        with self._lock:
            self._t0[key] = time.time()
            self._status[key] = "starting..."

    def update(self, key, frac, status=None):
        """frac=None keeps the current fraction (status-only update)."""
        with self._lock:
            if key not in self._frac:
                return
            if frac is not None and frac > self._frac[key]:
                self._frac[key] = min(frac, 1.0)
            if status is not None:
                self._status[key] = status

    def finish(self, key, ok, status=None):
        """status overrides the default "done"/"FAILED" text (e.g. "skipped")."""
        with self._lock:
            if key not in self._frac:
                return
            self._frac[key] = 1.0
            self._done[key] = True
            self._status[key] = (status if status is not None
                                 else ("done" if ok else "FAILED"))

    def snapshot(self):
        with self._lock:
            return {t: (self._frac[t], self._status[t],
                       self._done[t], self._t0[t])
                    for t in self._frac}


# ----------------------------------------------------------------------
# Progress-line parsers (fuzzy: each returns True if it handled the line)
# ----------------------------------------------------------------------
_BIGLOOP_RE = re.compile(r"bigloop\s+(\d+)", re.IGNORECASE)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_ITER_RE = re.compile(r"\biter\s+(\d+)", re.IGNORECASE)
_GAP_RE = re.compile(r"gap=([0-9.eE+-]+)")
# v9 sweep launch: "Launching 11 MSA iterations in parallel ..." and the
# UE-guidance variant "Launching 10 UE-guidance MSA iterations in parallel".
_T1_LAUNCH_RE = re.compile(r"launching\s+(\d+)\s+.*?msa\s+iterations",
                           re.IGNORECASE)
# v9 sweep per-bigloop tqdm lines, e.g.
# "Bigloop 3 itr 12/18: :  60%|####    | 12/20 [05:23<03:41]".
_T1_ITR_RE = re.compile(r"bigloop\s+(\d+)\s+itr\s+(\d+)\s*/\s*(\d+)",
                        re.IGNORECASE)
# v9 sweep per-bigloop completion, e.g. "Bigloop 10 done: : 100%|...".
_T1_DONE_RE = re.compile(r"bigloop\s+(\d+)\s+done", re.IGNORECASE)


class SweepProgressState:
    """Per-sweep progress state for the v9 sweep line parser: per-bigloop
    high-water iteration marks plus the run dimensions parsed from the log.
    Only one thread ever touches a given state object, so no lock needed."""

    def __init__(self, n_bigloops=11, iters_per=18):
        self.iters = {}
        self.n_bigloops = max(1, n_bigloops)
        self.iters_per = iters_per

    def reset(self, n_bigloops):
        self.iters.clear()
        self.n_bigloops = max(1, n_bigloops)
        self.iters_per = 18


def parse_v9_sweep_line(line, tracker, key, state):
    """v9 sweep line parser. `key` is the tracker key; `state` is a
    SweepProgressState. Returns True if the line was handled."""
    m = _T1_LAUNCH_RE.search(line)
    if m:
        try:
            state.reset(int(m.group(1)))
        except ValueError:
            state.reset(11)
        tracker.update(key, None,
                       "launching %d bigloops" % state.n_bigloops)
        return True
    m = _T1_ITR_RE.search(line)
    if m:
        try:
            b = int(m.group(1))
            it = int(m.group(2))
            per = int(m.group(3))
        except ValueError:
            return False
        if per > 0:
            state.iters_per = per
        cur = state.iters.get(b, 0)
        if it > cur:
            state.iters[b] = it
            cur = it
        total = sum(state.iters.values())
        grand = state.n_bigloops * state.iters_per
        frac = (total / grand) if grand > 0 else 0.0
        tracker.update(key, frac,
                       "iter %d/%d (bigloop %d: %d/%d)"
                       % (total, grand, b, cur, state.iters_per))
        return True
    m = _T1_DONE_RE.search(line)
    if m:
        try:
            b = int(m.group(1))
        except ValueError:
            return False
        if state.iters.get(b, 0) < state.iters_per:
            state.iters[b] = state.iters_per
        total = sum(state.iters.values())
        grand = state.n_bigloops * state.iters_per
        frac = (total / grand) if grand > 0 else 0.0
        tracker.update(key, frac,
                       "iter %d/%d (bigloop %d done)"
                       % (total, grand, b))
        return True
    # Legacy fallback: a bare "bigloop N" with no iteration info.
    # Only use if we haven't seen any itr data yet AND N is a valid
    # bigloop index. Otherwise ignore -- the itr/done patterns are
    # authoritative, and N may be a bigloop ID (e.g. 10 or 33) rather
    # than a progress count.
    m = _BIGLOOP_RE.search(line)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            return False
        if state.iters or n >= state.n_bigloops:
            return True  # consume; don't guess progress
        tracker.update(key, n / state.n_bigloops,
                       "bigloop %d/%d" % (n, state.n_bigloops))
        return True
    m = _PCT_RE.search(line)
    if m:
        try:
            tracker.update(key, float(m.group(1)) / 100.0,
                           line.strip()[:44])
            return True
        except ValueError:
            pass
    return False


# Per-job parser state: one SweepProgressState per v9 job key, created
# lazily (the launch line corrects the bigloop count when the run starts).
_SWEEP_STATES = {}

# Parser kind per job key, filled in when jobs are built:
#   "v9" -- runner_parallel.py jobs (Proposed job and Proposed sweeps)
#   "rt" -- scenarios_realtime.py + preference_reaction.py
#   "ue" -- scenarios_ue.py + preference_reaction.py
JOB_KINDS = {}


def _parse_reaction_line(key, line, tracker, base):
    """Shared preference_reaction.py markers. base is where the reaction
    phase starts (0.50 for real-time jobs, 0.80 for UE jobs); spans to 1.0."""
    low = line.lower()
    span = 1.0 - base
    if "loading network" in low:
        tracker.update(key, base + span * 0.10, "reaction: loading data")
        return True
    if "tournament run for" in low:
        tracker.update(key, base + span * 0.40, "reaction: tournament done")
        return True
    if "writing" in low and "input_agent" in low:
        tracker.update(key, base + span * 0.50, "reaction: writing input")
        return True
    if "running dtalite" in low or "running pysim" in low:
        tracker.update(key, base + span * 0.70, "reaction: engine")
        return True
    if "actual_routes" in low and "wrote" in low:
        tracker.update(key, base + span * 0.90, "reaction: routes written")
        return True
    if "reaction_summary" in low or ("wrote" in low and "summary" in low):
        tracker.update(key, 1.0, "done")
        return True
    if low.strip().startswith("satisfaction:"):
        tracker.update(key, 1.0, line.strip()[:44])
        return True
    return False


def parse_rt_line(cmd_idx, line, tracker, key):
    low = line.lower()
    if cmd_idx == 0:  # scenarios_realtime.py (guidance), spans 0.0-0.50
        if "step 1:" in low:
            if "running dtalite" in low:
                tracker.update(key, 0.20, "step 1: DTALite baseline")
            else:
                tracker.update(key, 0.10, "step 1: free-flow assignment")
            return True
        if "step 2:" in low:
            if "running dtalite" in low:
                tracker.update(key, 0.40, "step 2: DTALite reactive")
            else:
                tracker.update(key, 0.30, "step 2: reactive assignment")
            return True
        if "realtime_routes" in low and "wrote" in low:
            tracker.update(key, 0.50, "guidance complete")
            return True
        return False
    # cmd_idx == 1: preference_reaction.py, spans 0.50-1.0
    return _parse_reaction_line(key, line, tracker, 0.50)


def parse_ue_line(cmd_idx, line, tracker, key):
    low = line.lower()
    if cmd_idx == 0:  # scenarios_ue.py (guidance), spans 0.0-0.80
        m = _ITER_RE.search(low)
        if m:
            n = int(m.group(1))
            frac = 0.05 + (n / UE_MAX_ITERS) * 0.75
            status = "iter %d/%d" % (n, UE_MAX_ITERS)
            gm = _GAP_RE.search(low)
            if gm:
                status += " gap=%s" % gm.group(1)
            tracker.update(key, frac, status)
            return True
        if "initial gap" in low:
            tracker.update(key, 0.05, "MSA starting")
            return True
        if "converged" in low:
            tracker.update(key, 0.80, "MSA converged")
            return True
        if "ue_routes" in low and "wrote" in low:
            tracker.update(key, 0.80, "guidance complete")
            return True
        return False
    # cmd_idx == 1: preference_reaction.py, spans 0.80-1.0
    return _parse_reaction_line(key, line, tracker, 0.80)


def parse_progress_line(key, cmd_idx, line, tracker):
    """Dispatch one captured stdout line to the right parser. `key` is
    the job's tracker key. Unknown keys are ignored."""
    if tracker is None or not line or not line.strip():
        return
    try:
        kind = JOB_KINDS.get(key)
        if kind == "v9":
            state = _SWEEP_STATES.setdefault(key, SweepProgressState(11))
            parse_v9_sweep_line(line, tracker, key, state)
        elif kind == "rt":
            parse_rt_line(cmd_idx, line, tracker, key)
        elif kind == "ue":
            parse_ue_line(cmd_idx, line, tracker, key)
    except Exception:
        # Parsing must never break a run; markers are best-effort.
        pass


# ----------------------------------------------------------------------
# Progress display (main thread; redraws every 2 seconds)
# ----------------------------------------------------------------------
def _bar_chars():
    """Block characters if the console can encode them, else ASCII."""
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    try:
        ("\u2588\u2591").encode(enc)
        return "\u2588", "\u2591"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


_FULL_CH, _EMPTY_CH = _bar_chars()

# Display labels per job key, filled in when jobs are built.
DISPLAY_LABELS = {}


def render_bars(tracker, keys):
    snap = tracker.snapshot()
    lines = []
    for t in keys:
        frac, status, done, t0 = snap[t]
        pct = int(frac * 100)
        filled = int(frac * 20)
        bar = _FULL_CH * filled + _EMPTY_CH * (20 - filled)
        if not done and t0 is not None:
            status = "%s (%s)" % (status, fmt_min(time.time() - t0))
        lines.append("%-26s [%s] %3d%%  %s"
                     % (DISPLAY_LABELS.get(t, str(t)),
                        bar, pct, status))
    return lines


def display_loop(tracker, keys, stop_event, interval=2.0):
    """Redraw the progress block until stop_event is set."""
    n_lines = len(keys) + 1  # bars plus trailing blank line
    first = True
    try:
        while not stop_event.is_set():
            if not first:
                sys.stdout.write("\033[%dA" % n_lines)
            for line in render_bars(tracker, keys):
                sys.stdout.write("\r\033[K" + line + "\n")
            sys.stdout.write("\r\033[K\n")
            sys.stdout.flush()
            first = False
            if stop_event.wait(interval):
                break
    except Exception:
        # Display must never break a run (e.g. dumb terminal).
        pass
    # Final render so the terminal is left in a clean state.
    try:
        if not first:
            sys.stdout.write("\033[%dA" % n_lines)
        for line in render_bars(tracker, keys):
            sys.stdout.write("\r\033[K" + line + "\n")
        sys.stdout.write("\r\033[K\n")
        sys.stdout.flush()
    except Exception:
        pass

# ----------------------------------------------------------------------
# Sampling: per-(seed, level) agent file from the pool
# ----------------------------------------------------------------------
def sample_level(pool_file, seed, level, out_csv):
    """Sample exactly `level` agents from the pool into out_csv, plus
    the matching sampled agent_weights CSV.

    Calls sample_pool() from sample_agents.py (built by the sampling
    layer; it must sit next to this file), passing the run seed and the
    demand level: the sampler derives its per-level RNG from
    (seed, level), so each level is an independent draw. Tries an
    import first, then falls back to invoking sample_agents.py as a
    subprocess with the same arguments as flags. Any failure --
    missing pool file, missing sampler, wrong row count -- raises
    RuntimeError with a clear message; the caller treats that as fatal.
    Nothing here ever falls back to the old unsampled input file.

    The sampled weights file is out_csv with a _weights suffix
    (sample_agents.weights_out_path), written from the pool's
    agent_weights.csv next to the pool file. When the pool has no
    weights file, sampling still succeeds and a warning is logged:
    msa.py then falls back to the users sheet (only correct up to
    36,030 agents). Returns the sampled weights path, or None.
    """
    if not os.path.isfile(pool_file):
        raise RuntimeError(
            "pool file not found: %s -- generate/place the pool file "
            "(see the run README) and rerun; wholerunner will not fall "
            "back to the old input_agent_initial.csv" % pool_file)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pool_weights = os.path.join(
        os.path.dirname(os.path.abspath(pool_file)), "agent_weights.csv")
    out_weights = (os.path.splitext(out_csv)[0] + "_weights"
                   + os.path.splitext(out_csv)[1])
    log("sampling %d agents from %s (seed %d, level %d) -> %s"
        % (level, os.path.basename(pool_file), seed, level, out_csv))
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import sample_agents
    except ImportError:
        script = os.path.join(HERE, "sample_agents.py")
        if not os.path.isfile(script):
            raise RuntimeError(
                "sampling layer not found: sample_agents.py is missing "
                "next to wholerunner.py (expected at %s). The sampling "
                "layer must be delivered before this run can proceed."
                % script)
        cmd = [PY, script, "--pool-file", pool_file,
               "--seed", str(seed), "--n", str(level),
               "--out", out_csv, "--level", str(level),
               "--weights-file", pool_weights,
               "--out-weights", out_weights]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(
                "sample_agents.py failed (rc=%d) for seed %d level %d; "
                "see its output above" % (rc, seed, level))
    else:
        try:
            sample_agents.sample_pool(pool_file, seed, level, out_csv,
                                      level=level,
                                      weights_csv=pool_weights,
                                      out_weights_csv=out_weights)
        except Exception as e:
            raise RuntimeError(
                "sample_pool() failed for seed %d level %d: %s"
                % (seed, level, e))
    if not os.path.isfile(out_csv):
        raise RuntimeError(
            "sampler produced no output file for seed %d level %d: %s"
            % (seed, level, out_csv))
    n = count_csv_rows(out_csv)
    if n != level:
        raise RuntimeError(
            "sampler produced %d agent rows for level %d (seed %d); "
            "expected exactly %d" % (n, level, seed, level))
    log("sampled %d agents (seed %d, level %d)" % (n, seed, level))
    if os.path.isfile(pool_weights):
        if not os.path.isfile(out_weights):
            raise RuntimeError(
                "sampler wrote no weights file for seed %d level %d: %s"
                % (seed, level, out_weights))
        log("sampled %d agent weights (seed %d, level %d)"
            % (level, seed, level))
        return out_weights
    log("WARNING: no pool weights file at %s; sampled agents will "
        "have no weights file (msa.py falls back to the users sheet)"
        % pool_weights)
    return None


def count_csv_rows(path):
    """Data-row count of a CSV file (header excluded)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    return max(0, len(rows) - 1)


# ----------------------------------------------------------------------
# Sandbox setup (fail-fast: any error here aborts before anything runs)
# ----------------------------------------------------------------------
def setup_sandbox(name, data_dir, temp_root, scripts, sample_csv=None,
                  sample_weights_csv=None):
    """Create temp/run_<pid>/<name>/ with a private DTALite_Files/ copy and
    the job's scripts. `scripts` is a list of (src_dir, filename) tuples;
    src_dir None means next to wholerunner.py, otherwise a subdir of HERE.
    When sample_csv is given, it overwrites the sandbox copy's
    input_agent_initial.csv -- that is how the per-(seed, level) sample
    becomes the run's demand. When sample_weights_csv is given, it
    overwrites the sandbox copy's agent_weights.csv the same way, so
    the preference travels with the sampled agents (matched by
    agent_id in msa.py). When it is None, the sandbox keeps whatever
    agent_weights.csv the copytree brought in, if any. Raises on any
    error."""
    sandbox = os.path.join(temp_root, name)
    if os.path.isdir(sandbox):
        log("removing stale sandbox %s" % sandbox)
        shutil.rmtree(sandbox)
    os.makedirs(sandbox)

    sandbox_data = os.path.join(sandbox, "DTALite_Files")
    log("%s: copying DTALite_Files -> sandbox ..." % name)
    shutil.copytree(data_dir, sandbox_data)

    if sample_csv is not None:
        dst = os.path.join(sandbox_data, "input_agent_initial.csv")
        shutil.copy2(sample_csv, dst)
        log("%s: injected sampled agents (%s)"
            % (name, os.path.basename(sample_csv)))

    if sample_weights_csv is not None:
        dst = os.path.join(sandbox_data, "agent_weights.csv")
        shutil.copy2(sample_weights_csv, dst)
        log("%s: injected sampled agent weights (%s)"
            % (name, os.path.basename(sample_weights_csv)))

    for src_dir, fn in scripts:
        if src_dir:
            src = os.path.join(HERE, src_dir, fn)
        else:
            src = os.path.join(HERE, fn)
        if not os.path.isfile(src):
            raise RuntimeError("required script not found: %s" % src)
        shutil.copy2(src, sandbox)
    for fn in ENGINE_FILES:
        src = os.path.join(HERE, fn)
        if not os.path.isfile(src):
            raise RuntimeError("required engine file not found: %s" % src)
        shutil.copy2(src, sandbox)
    for dirname in ENGINE_DIRS:
        src = os.path.join(HERE, dirname)
        if not os.path.isdir(src):
            raise RuntimeError("required engine dir not found: %s" % src)
        shutil.copytree(src, os.path.join(sandbox, dirname))
    log("%s: sandbox ready (%s)" % (name, sandbox))
    return sandbox


def scripts_for(scripts):
    """[(src_dir, filename)] with src_dir None (next to wholerunner.py)."""
    return [(None, fn) for fn in scripts]


# ----------------------------------------------------------------------
# Per-job command lines
# ----------------------------------------------------------------------
def proposed_commands(sandbox, seed, network="siouxfalls", engine="pysim"):
    """Proposed system: v9 pipeline, bigloop 10 only (100% MPR)."""
    cmd = [PY, os.path.join(sandbox, "runner_parallel.py"),
           "--seed", str(seed), "--network", network, "--engine", engine,
           "--bigloops", "10"]
    return [(cmd, "v9 pipeline (runner_parallel.py --bigloops 10)")]


def scenario_commands(method, sandbox, seed, network="siouxfalls",
                      gap_threshold=0.001, engine="pysim"):
    """Guidance scenario followed by the preference-reaction step.

    method is "realtime_1min", "realtime_10min" or "ue"."""
    staging = os.path.join(sandbox, "staging")
    os.makedirs(staging, exist_ok=True)
    data = os.path.join(sandbox, "DTALite_Files")

    if method in ("realtime_1min", "realtime_10min"):
        sigma = 1 if method == "realtime_1min" else 10
        scenario = [PY, os.path.join(sandbox, "scenarios_realtime.py"),
                    "--sigma", str(sigma),
                    "--data-dir", data, "--result-dir", staging,
                    "--seed", str(seed), "--network", network,
                    "--engine", engine]
        scenario_desc = "real-time sigma=%d (guidance)" % sigma
        guidance_csv = os.path.join(staging, "realtime_routes.csv")
    else:  # ue
        scenario = [PY, os.path.join(sandbox, "scenarios_ue.py"),
                    "--data-dir", data, "--result-dir", staging,
                    "--seed", str(seed), "--network", network,
                    "--engine", engine,
                    "--gap-threshold", str(gap_threshold)]
        scenario_desc = "UE baseline (MSA guidance)"
        guidance_csv = os.path.join(staging, "ue_routes.csv")

    reaction = [PY, os.path.join(sandbox, "preference_reaction.py"),
                "--guidance-csv", guidance_csv,
                "--data-dir", data,
                "--result-dir", staging,
                "--target-satisfaction", str(TARGETS[method]),
                "--network", network, "--engine", engine]
    return [(scenario, scenario_desc),
            (reaction, "preference reaction")]


def sweep_commands(sub, sandbox, seed, bigloops=None, mpr_points=None,
                   ue_routes_path=None, network="siouxfalls",
                   engine="pysim"):
    """One [(cmd, desc)] entry per sweep job. sub is "P" (proposed) or
    "U" (UE-guidance). Exactly one of bigloops / mpr_points is used:
    bigloops is a list of ints (Test 2 full sweep), mpr_points a list
    of MPR percentages (Test 3 33/66 points)."""
    if sub == "P":
        cmd = [PY, os.path.join(sandbox, "runner_parallel.py"),
               "--seed", str(seed), "--network", network,
               "--engine", engine]
        if mpr_points is not None:
            cmd += ["--mpr-points", ",".join(str(p) for p in mpr_points)]
            desc = ("v9 pipeline (runner_parallel.py --mpr-points %s, "
                    "proposed MPR sweep)"
                    % ",".join(str(p) for p in mpr_points))
        else:
            cmd += ["--bigloops", ",".join(str(b) for b in bigloops)]
            desc = ("v9 pipeline (runner_parallel.py --bigloops %s, "
                    "proposed MPR sweep)"
                    % ",".join(str(b) for b in bigloops))
    else:
        if not ue_routes_path:
            raise ValueError("ue_routes_path is required for the UE-guidance sweep")
        cmd = [PY, os.path.join(sandbox, "runner_parallel_ue.py"),
               "--ue-routes", ue_routes_path,
               "--seed", str(seed), "--network", network,
               "--engine", engine]
        if mpr_points is not None:
            cmd += ["--mpr-points", ",".join(str(p) for p in mpr_points)]
            desc = ("UE-guidance MPR sweep (runner_parallel_ue.py "
                    "--mpr-points %s)"
                    % ",".join(str(p) for p in mpr_points))
        else:
            cmd += ["--bigloops", ",".join(str(b) for b in bigloops)]
            desc = ("UE-guidance MPR sweep (runner_parallel_ue.py "
                    "--bigloops %s)"
                    % ",".join(str(b) for b in bigloops))
    return [(cmd, desc)]

# ----------------------------------------------------------------------
# Execution (shared Popen engine; one thread per job)
# ----------------------------------------------------------------------
# Subprocess I/O strategy: the child writes DIRECTLY to run.log (a file);
# the parent polls the file for new lines. Pipes are deliberately NOT used.
# On Windows, if the child exits while one of its own descendants still
# holds the pipe open, the reader never sees EOF; worse, if the child
# itself hangs (e.g. ProcessPoolExecutor.shutdown(wait=True) waiting on a
# stuck worker), proc.poll() stays None forever, so no pipe-EOF grace
# period can ever fire. A file cannot deadlock in either case.
#
# Two timeouts guarantee forward progress:
#   _RUN_STALL_S: no new output for this long -> kill the process tree and
#                 mark the job failed. (All jobs print tqdm/line progress
#                 every few seconds in normal operation.)
#   _RUN_ABS_S:   absolute backstop per _run_commands call.
_RUN_POLL_S = 2.0
_RUN_STALL_S = 30 * 60  # overridden by --stall-timeout-min in main()
_RUN_ABS_S = 18 * 3600


def _kill_proc_tree(proc):
    """Best-effort kill of a subprocess and its children."""
    if sys.platform == "win32":
        # /T kills the whole tree (ProcessPoolExecutor workers too).
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def _poll_process(label, key, cmd_idx, proc, log_path, parse_line,
                  tracker=None):
    """Poll log_path for new output lines and proc for exit.

    Returns (returncode, killed, kill_reason). `killed` is True when the
    process had to be killed (stall or absolute timeout); otherwise the
    process exited on its own.

    The stall detector is progress-aware: last_progress is reset only when
    the tracker's fraction for `key` actually advances. A chatty-but-stuck
    child that emits bytes without progress will still be killed.
    """
    pos = 0  # header lines ("### ...") match no progress regex; harmless
    last_out = time.monotonic()
    last_progress = time.monotonic()

    def _get_frac():
        if tracker is None:
            return None
        try:
            snap = tracker.snapshot()
            if key in snap:
                return snap[key][0]
        except Exception:
            pass
        return None

    last_frac = _get_frac()
    abs_deadline = last_out + _RUN_ABS_S
    killed = False
    kill_why = ""
    rc = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as rf:
        while True:
            rf.seek(pos)
            try:
                chunk = rf.read()
            except Exception:
                chunk = ""
            if chunk:
                pos = rf.tell()
                # tqdm writes \r-separated bar updates with no \n; split
                # on both so progress markers are seen promptly.
                for piece in re.split(r"[\r\n]+", chunk):
                    if piece.strip():
                        parse_line(cmd_idx, piece)
                # Progress-aware: only reset timers if fraction advanced.
                # Bytes without progress do not reset the stall detector.
                new_frac = _get_frac()
                if new_frac is not None and last_frac is not None:
                    if new_frac > last_frac:
                        last_progress = time.monotonic()
                        last_frac = new_frac
                # If no tracker, fall back to byte-based (original behavior)
                elif new_frac is None and last_frac is None:
                    last_out = time.monotonic()
                    last_progress = time.monotonic()
            rc = proc.poll()
            if rc is not None:
                # Final drain: the child may have written between our
                # last read and its exit.
                rf.seek(pos)
                try:
                    chunk = rf.read()
                except Exception:
                    chunk = ""
                if chunk:
                    for piece in re.split(r"[\r\n]+", chunk):
                        if piece.strip():
                            parse_line(cmd_idx, piece)
                break
            now = time.monotonic()
            if now - last_progress > _RUN_STALL_S:
                kill_why = ("no progress for %s; killed stuck process"
                            % fmt_min(_RUN_STALL_S))
                log("ERROR: %s: %s" % (label, kill_why))
                _kill_proc_tree(proc)
                killed = True
                rc = proc.poll()
                break
            if now > abs_deadline:
                kill_why = ("exceeded absolute timeout %s; killed process"
                            % fmt_min(_RUN_ABS_S))
                log("ERROR: %s: %s" % (label, kill_why))
                _kill_proc_tree(proc)
                killed = True
                rc = proc.poll()
                break
            time.sleep(_RUN_POLL_S)
    if rc is None:
        # Kill did not yield a returncode; treat as failure regardless.
        rc = -1
    return rc, killed, kill_why


def _run_commands(label, sandbox, cmds, tracker, key, parse_line):
    """Run cmds sequentially in sandbox; new run.log lines are passed to
    parse_line(cmd_idx, line). Returns (ok, elapsed).

    Never hangs indefinitely: a process that produces no output for
    _RUN_STALL_S is killed, and _RUN_ABS_S bounds the whole call.
    """
    log_path = os.path.join(sandbox, "run.log")
    quiet = tracker is not None
    if tracker:
        tracker.start(key)
    t0 = time.time()
    if not quiet:
        log("%s: started (log: %s)" % (label, log_path))
    else:
        tracker.update(key, 0.0, "starting...")
    try:
        for cmd_idx, (cmd, desc) in enumerate(cmds):
            step0 = time.time()
            if not quiet:
                log("%s: %s ..." % (label, desc))
            else:
                tracker.update(key, None, desc)
            # The child writes straight to the log file. No pipe, so no
            # pipe-EOF deadlock is possible, ever.
            with open(log_path, "w", encoding="utf-8",
                       errors="replace") as logfh:
                logfh.write("### %s\n### cmd: %s\n"
                            % (desc, " ".join(cmd)))
                logfh.flush()
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=sandbox,
                        stdout=logfh, stderr=subprocess.STDOUT)
                except OSError as e:
                    if tracker:
                        tracker.finish(key, False)
                    log("%s: could not launch process: %s" % (label, e))
                    return False, time.time() - t0
            # Parent's handle is closed; the child holds its own duplicate.
            # Poll the file for progress and the process for exit.
            rc, killed, kill_why = _poll_process(
                label, key, cmd_idx, proc, log_path, parse_line,
                tracker=tracker)
            if killed or rc != 0:
                if tracker:
                    tracker.finish(key, False)
                log("%s: FAILED at '%s' (rc=%s%s); sandbox kept: %s"
                    % (label, desc, rc,
                       "; %s" % kill_why if killed else "", sandbox))
                return False, time.time() - t0
            if not quiet:
                log("%s: %s done (%s)"
                    % (label, desc, fmt_min(time.time() - step0)))
        # All commands succeeded; make sure the bar reaches 100%.
        if tracker:
            tracker.finish(key, True)
    except OSError as e:
        if tracker:
            tracker.finish(key, False)
        log("%s: could not launch process: %s" % (label, e))
        return False, time.time() - t0
    dt = time.time() - t0
    if not quiet:
        log("%s: finished (%s)" % (label, fmt_min(dt)))
    return True, dt


def run_job(job, sandbox, tracker, args):
    """Run one job's command sequence inside its sandbox. Returns
    (ok, elapsed). `job` is a job dict built by build_jobs()."""
    cmds = job["commands"](sandbox, args)
    key = job["key"]
    return _run_commands(job["label"], sandbox, cmds, tracker, key,
                         lambda i, line: parse_progress_line(key, i, line,
                                                            tracker))

# ----------------------------------------------------------------------
# Metrics (formulas reused from the established scripts, not reinvented)
# ----------------------------------------------------------------------
def find_last_bigloop(d):
    """Last bigloop number with both actual and UOinfor agent files."""
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


def norm_headers(fieldnames):
    return {c.strip().lower(): c for c in fieldnames if c and c.strip()}


def norm_path(s):
    """Identical normalization on both sides: strip, split ';', drop
    empty segments, rejoin."""
    return ";".join(seg.strip() for seg in s.strip().split(";")
                    if seg.strip())


def read_agent_paths(path):
    """agent_id -> normalized path_node_sequence."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        if "agent_id" not in cols or "path_node_sequence" not in cols:
            raise ValueError(
                "missing agent_id/path_node_sequence in %s" % path)
        aid_c = cols["agent_id"]
        path_c = cols["path_node_sequence"]
        for row in reader:
            aid = (row.get(aid_c) or "").strip()
            if aid:
                out[aid] = norm_path(row.get(path_c) or "")
    return out


def satisfaction(actual_paths, ref_paths):
    """(n_satisfied, n_common, rate) on the intersection of agent_ids."""
    common = set(actual_paths) & set(ref_paths)
    n_sat = sum(1 for a in common if actual_paths[a] == ref_paths[a])
    rate = (n_sat / len(common)) if common else 0.0
    return n_sat, len(common), rate


def find_tt_col(cols):
    for key, orig in cols.items():
        if "travel_time" in key and "min" in key:
            return orig
    return None


def sum_travel_minutes(path):
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
            if v == v:  # not NaN
                total += v
                n += 1
    return total, n


def proposed_metrics(f_uo, f_ac):
    """(satisfaction_pct, veh_hrs) for a Proposed run from its UOinfor
    and actual agent files."""
    actual = read_agent_paths(f_ac)
    uo = read_agent_paths(f_uo)
    n_sat, n_common, rate = satisfaction(actual, uo)
    tot_min, _n_tt = sum_travel_minutes(f_ac)
    return 100.0 * rate, tot_min / 60.0


def generate_proposed_summary(bigloop, full_output_dir, summary_path):
    """Write proposed_summary.txt mirroring the baseline format:
    satisfaction vs Bingrong's 88.64% and vehicle-hours vs 24,719.3.
    Returns (satisfaction_pct, veh_hrs) for the summary.csv row."""
    f_uo = os.path.join(full_output_dir, "%dUOinfor_agent.csv" % bigloop)
    f_ac = os.path.join(full_output_dir, "%dactual_agent.csv" % bigloop)
    if not os.path.isfile(f_uo):
        raise RuntimeError("missing %s" % f_uo)
    if not os.path.isfile(f_ac):
        raise RuntimeError("missing %s" % f_ac)

    actual = read_agent_paths(f_ac)
    uo = read_agent_paths(f_uo)
    n_sat, n_common, rate = satisfaction(actual, uo)

    tot_min, n_tt = sum_travel_minutes(f_ac)
    veh_hrs = tot_min / 60.0
    per_agent = tot_min / n_tt if n_tt else float("nan")

    rate_pct = 100.0 * rate
    pp = rate_pct - T1_SAT_RATE

    with open(summary_path, "w", newline="") as f:
        f.write("Proposed system (Test 1) -- v9 pipeline\n")
        f.write("bigloop: %d\n" % bigloop)
        f.write("agents: %d\n" % n_common)
        f.write("\n")
        f.write("satisfaction: %d / %d = %.2f%%\n"
                % (n_sat, n_common, rate_pct))
        f.write("Bingrong target: %.2f%% (diff %+.2f pp)\n"
                % (T1_SAT_RATE, pp))
        f.write("\n")
        f.write("vehicle-hours: %.2f\n" % veh_hrs)
        f.write("per-agent: %.2f min/agent\n" % per_agent)
        f.write("Bingrong target: %.1f veh-hrs (%.2f min/agent)\n"
                % (T1_VEH_HRS, T1_MIN_PER_AGENT))
    return rate_pct, veh_hrs


_SAT_RE = re.compile(r"satisfaction:\s*\d+\s*/\s*\d+\s*=\s*([0-9.]+)%")
_VH_RE = re.compile(r"vehicle-hours \(actual run\):\s*([0-9.]+)")


def parse_reaction_summary(path):
    """(satisfaction_pct, veh_hrs) from a reaction_summary.txt, or
    (None, None) when the file or the lines are missing."""
    if not os.path.isfile(path):
        return None, None
    sat = None
    veh = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _SAT_RE.search(line)
            if m:
                try:
                    sat = float(m.group(1))
                except ValueError:
                    pass
            m = _VH_RE.search(line)
            if m:
                try:
                    veh = float(m.group(1))
                except ValueError:
                    pass
    return sat, veh


def compute_linkmoe_metrics(path):
    """Delay and congestion from one realized output_LinkMOE.csv.

    Delay (veh-hrs) = sum over links of
        (travel_time - free-flow_travel_time) * total_link_volume / 60
    (linkmoe_delay.py). V/C = total_link_volume /
    link_capacity_in_vhc_per_hour (congestion_metric.md); max V/C is the
    peak, p95 V/C the 95th-percentile link (same index rule as
    compute_congestion.py: sorted values, index int(0.95 * n)),
    over_cap counts V/C > 1.0, heavy counts V/C > 0.8.
    Returns a dict with None values when the file or columns are absent.
    """
    out = {"delay": None, "max_vc": None, "p95_vc": None,
           "over_cap": None, "heavy": None}
    if not os.path.isfile(path):
        return out
    total = 0.0
    vcs = []
    n = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = norm_headers(reader.fieldnames or [])
        need = ("travel_time", "free-flow_travel_time",
                "total_link_volume", "link_capacity_in_vhc_per_hour")
        if any(k not in cols for k in need):
            return out
        tt_c, fft_c, vol_c, cap_c = (cols[k] for k in need)
        for row in reader:
            try:
                tt = float(row[tt_c])
                fft = float(row[fft_c])
                vol = float(row[vol_c])
                cap = float(row[cap_c])
            except (TypeError, ValueError):
                continue
            if not (tt == tt and fft == fft and vol == vol
                    and cap == cap):
                continue
            total += (tt - fft) * vol
            n += 1
            if cap > 0:
                vcs.append(vol / cap)
    if n == 0:
        return out
    out["delay"] = total / 60.0
    if vcs:
        vcs.sort()
        out["max_vc"] = vcs[-1]
        out["p95_vc"] = vcs[min(int(0.95 * len(vcs)), len(vcs) - 1)]
        out["over_cap"] = sum(1 for v in vcs if v > 1.0)
        out["heavy"] = sum(1 for v in vcs if v > 0.8)
    return out


# Emission column sums: identical definitions to table4_emissions.py /
# mpr_aggregate.py SUST_SPECS (agent-file sums; no correction factor).
# (metric, candidate canonical column names, divisor to table unit)
SUST_SPECS = [
    ("energy", ("totalenergykj", "energykj", "energyinkj"), 1e9),
    ("co2", ("co2g",), 1e6),
    ("nox", ("noxg",), 1e3),
    ("co", ("cog",), 1e3),
    ("hc", ("hcg",), 1e3),
]


def _canon_col(col):
    return re.sub(r"[^a-z0-9]", "", col.strip().lower())


def sum_emissions(path):
    """Sum per-agent emission columns in one agent file.

    Returns a dict metric -> float in table units (energy 10^9 KJ, CO2
    ton, NOX/CO/HC kg), or None per metric when the column is absent.
    A present-but-zero column sums to 0.0, not None. Returns all-None
    when the file is missing."""
    out = {name: None for name, _c, _d in SUST_SPECS}
    if not os.path.isfile(path):
        return out
    totals = {name: 0.0 for name, _c, _d in SUST_SPECS}
    found = {name: False for name, _c, _d in SUST_SPECS}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {}
        for c in (reader.fieldnames or []):
            if c and c.strip():
                cols.setdefault(_canon_col(c), c)
        use = {}
        for name, cands, _div in SUST_SPECS:
            for cand in cands:
                if cand in cols:
                    use[name] = cols[cand]
                    break
        if not use:
            return out
        for row in reader:
            for name, col in use.items():
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    continue
                if v == v:  # not NaN
                    totals[name] += v
                    found[name] = True
    for name, _c, div in SUST_SPECS:
        if name in use:
            out[name] = totals[name] / div
    return out


def _load_link_volumes(path):
    """{(from_node, to_node): volume} from a LinkMOE CSV. Keeps the
    FIRST occurrence of each duplicated header (DTALite trailing-comma
    quirk); mirrors volume_estimation.load_link_volumes."""
    volumes = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        col = {}
        for row in reader:
            if not col:
                for i, name in enumerate(row):
                    norm = name.strip().lower()
                    if norm and norm not in col:
                        col[norm] = i
                missing = [c for c in ("from_node_id", "to_node_id",
                                       "total_link_volume")
                           if c not in col]
                if missing:
                    raise ValueError(
                        "%s: missing required column(s): %s"
                        % (path, ", ".join(missing)))
                continue
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                from_node = row[col["from_node_id"]].strip()
                to_node = row[col["to_node_id"]].strip()
                volume = float(row[col["total_link_volume"]].strip())
            except (IndexError, ValueError):
                continue
            if from_node and to_node:
                volumes[(from_node, to_node)] = volume
    return volumes


def linkvol_accuracy(predict_path, actual_path):
    """Link-volume accuracy: 100 * (#links with |pred-act|/|act|*100 < 10)
    / (#links with nonzero actual volume). Same binning rule as
    volume_estimation.analyze_pair (its "<10%" bin). Returns None when
    either file is missing or no comparable link has nonzero volume."""
    if not os.path.isfile(predict_path) or not os.path.isfile(actual_path):
        return None
    try:
        predicted = _load_link_volumes(predict_path)
        actual = _load_link_volumes(actual_path)
    except (ValueError, OSError):
        return None
    common = set(predicted) & set(actual)
    denom = 0
    within = 0
    for key in common:
        act = actual[key]
        if act == 0.0:
            continue
        denom += 1
        if abs(predicted[key] - act) / abs(act) * 100.0 < 10.0:
            within += 1
    if denom == 0:
        return None
    return 100.0 * within / denom


# ----------------------------------------------------------------------
# Per-seed master summary (summary.csv)
# ----------------------------------------------------------------------
def _fmt_metric(name, value):
    """Format one metric value for summary.csv; None -> empty string."""
    if value is None:
        return ""
    if name in ("over_cap", "heavy"):
        return str(int(value))
    if name in ("co2", "nox", "co", "hc"):
        return "%.1f" % value
    if name in ("max_vc", "p95_vc"):
        return "%.3f" % value
    return "%.2f" % value


def append_summary_row(results_root, seed, test, level, method, mpr,
                       metrics):
    """Append one row to <results_root>/summary.csv, creating it with
    the pinned SUMMARY_COLUMNS header when it does not exist yet.
    metrics maps metric name -> value (None = empty cell)."""
    path = os.path.join(results_root, "summary.csv")
    exists = os.path.isfile(path)
    os.makedirs(results_root, exist_ok=True)
    row = {"seed": seed, "test": test, "level": level,
           "method": method,
           "mpr": "" if mpr is None else str(mpr)}
    for name in ("satisfaction", "veh_hrs", "delay", "energy", "co2",
                 "nox", "co", "hc", "linkvol_acc", "max_vc", "p95_vc",
                 "over_cap", "heavy"):
        row[name] = _fmt_metric(name, metrics.get(name))
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS,
                                extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def collect_metrics(agent_path, linkmoe_path, uo_path=None,
                    predict_linkmoe_path=None, sat_veh=None):
    """Assemble the metrics dict for one summary.csv row.

    agent_path: actual-run agent file (emissions + veh_hrs fallback).
    linkmoe_path: realized LinkMOE file (delay + congestion).
    uo_path: UOinfor agent file (Proposed runs: satisfaction computed
      directly from UOinfor vs actual paths).
    predict_linkmoe_path: predicted LinkMOE file (link-volume accuracy).
    sat_veh: (satisfaction_pct, veh_hrs) from a reaction summary, used
      when uo_path is None (scenario jobs).
    Missing pieces stay None (empty cells), never fabricated."""
    metrics = {"satisfaction": None, "veh_hrs": None, "delay": None,
               "energy": None, "co2": None, "nox": None, "co": None,
               "hc": None, "linkvol_acc": None, "max_vc": None,
               "p95_vc": None, "over_cap": None, "heavy": None}
    if uo_path is not None and os.path.isfile(uo_path) \
            and os.path.isfile(agent_path):
        try:
            sat, veh = proposed_metrics(uo_path, agent_path)
            metrics["satisfaction"] = sat
            metrics["veh_hrs"] = veh
        except (ValueError, OSError) as e:
            log("WARNING: could not compute Proposed metrics from %s: %s"
                % (agent_path, e))
    elif sat_veh is not None:
        sat, veh = sat_veh
        metrics["satisfaction"] = sat
        if veh is None and os.path.isfile(agent_path):
            try:
                tot_min, _n = sum_travel_minutes(agent_path)
                metrics["veh_hrs"] = tot_min / 60.0
            except (ValueError, OSError):
                pass
        else:
            metrics["veh_hrs"] = veh
    elif os.path.isfile(agent_path):
        try:
            tot_min, _n = sum_travel_minutes(agent_path)
            metrics["veh_hrs"] = tot_min / 60.0
        except (ValueError, OSError):
            pass
    if os.path.isfile(agent_path):
        metrics.update(sum_emissions(agent_path))
    if linkmoe_path is not None:
        metrics.update(compute_linkmoe_metrics(linkmoe_path))
    if predict_linkmoe_path is not None and linkmoe_path is not None:
        metrics["linkvol_acc"] = linkvol_accuracy(predict_linkmoe_path,
                                                  linkmoe_path)
    return metrics

# ----------------------------------------------------------------------
# Result collection (one function per job kind; each also appends the
# job's summary.csv row(s))
# ----------------------------------------------------------------------
def _collect_engine_log(sandbox, out_dir):
    """Copy the sandbox's engine_calls.log (PySim audit trail) into the
    results, when one exists. Under --engine dtalite there is no such
    file and this is a no-op."""
    src = os.path.join(sandbox, "engine_calls.log")
    if os.path.isfile(src):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(out_dir, "engine_calls.log"))


def collect_proposed(job, sandbox, results_root, seed):
    """Copy one Proposed job's bigloop result CSVs, write its summary
    file, and append its summary.csv row. Returns the bigloop number."""
    out = job["out_dir"]
    full = os.path.join(out, "full_results")
    os.makedirs(full, exist_ok=True)
    data = os.path.join(sandbox, "DTALite_Files")

    n = find_last_bigloop(data)
    if n is None:
        raise RuntimeError(
            "%s: no completed bigloop files found in %s"
            % (job["key"], data))
    copied = []
    for tag in T1_TAGS:
        for kind in T1_KINDS:
            fn = "%d%s_%s.csv" % (n, tag, kind)
            src = os.path.join(data, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(full, fn))
                copied.append(fn)
            else:
                log("%s: warning: expected file missing: %s"
                    % (job["key"], fn))
    log("%s: collected %d bigloop-%d CSVs -> %s"
        % (job["key"], len(copied), n, full))

    summary_path = os.path.join(out, "proposed_summary.txt")
    generate_proposed_summary(n, full, summary_path)
    _collect_engine_log(sandbox, out)
    log("%s: wrote %s" % (job["key"], summary_path))

    metrics = collect_metrics(
        os.path.join(full, "%dactual_agent.csv" % n),
        os.path.join(full, "%dactual_LinkMOE.csv" % n),
        uo_path=os.path.join(full, "%dUOinfor_agent.csv" % n),
        predict_linkmoe_path=os.path.join(
            full, "%dpredict_LinkMOE.csv" % n))
    append_summary_row(results_root, seed, job["test"], job["level"],
                       "proposed", None, metrics)
    return n


def collect_scenario(job, sandbox, results_root, seed):
    """Copy a scenario job's staging summaries/routes, predicted_LinkMOE
    and output_*.csv, and append its summary.csv row."""
    out = job["out_dir"]
    staging = os.path.join(sandbox, "staging")
    data = os.path.join(sandbox, "DTALite_Files")

    os.makedirs(out, exist_ok=True)
    copied = []
    # predicted_LinkMOE.csv is written into staging by scenarios_realtime.py
    # and scenarios_ue.py; it feeds Test 4 and the per-level accuracy.
    for pat in ("*_summary.txt", "*_routes.csv", "predicted_LinkMOE.csv"):
        for src in sorted(glob.glob(os.path.join(staging, pat))):
            dst = os.path.join(out, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(os.path.basename(src))

    full = os.path.join(out, "full_results")
    os.makedirs(full, exist_ok=True)
    n_csv = 0
    for src in sorted(glob.glob(os.path.join(data, "output_*.csv"))):
        shutil.copy2(src, os.path.join(full, os.path.basename(src)))
        n_csv += 1
    _collect_engine_log(sandbox, out)
    log("%s: collected %d files -> %s, %d output_*.csv -> full_results/"
        % (job["key"], len(copied), out, n_csv))

    sat_veh = parse_reaction_summary(
        os.path.join(out, "reaction_summary.txt"))
    metrics = collect_metrics(
        os.path.join(full, "output_agent.csv"),
        os.path.join(full, "output_LinkMOE.csv"),
        predict_linkmoe_path=os.path.join(out, "predicted_LinkMOE.csv"),
        sat_veh=sat_veh)
    append_summary_row(results_root, seed, job["test"], job["level"],
                       job["method"], None, metrics)


def collect_sweep(job, sub, sandbox, results_root, seed):
    """Copy per-bigloop result CSVs for a sweep job and append one
    summary.csv row per bigloop.

    job["bigloops"] is a list of (bigloop, mpr) pairs: Test 2 uses
    bigloops 0..10 (proposed) / 1..10 (UE) with mpr = bigloop * 10
    (bigloop 0 is the nominal 0% point); Test 3 uses bigloops 33/66
    with mpr 33/66. For each bigloop with BOTH {n}actual_agent.csv and
    {n}UOinfor_agent.csv present, copy the 9 CSVs. Missing/incomplete
    bigloops are skipped with a warning, and a short sweep raises
    RuntimeError (a missing sweep point is a failed test, never
    a silent OK). Returns the list of bigloops actually collected.
    """
    sweep_name = "proposed_sweep" if sub == "P" else "ue_sweep"
    method = "proposed" if sub == "P" else "ue"
    data = os.path.join(sandbox, "DTALite_Files")
    collected = []
    for n, mpr in job["bigloops"]:
        f_act = os.path.join(data, "%dactual_agent.csv" % n)
        f_uo = os.path.join(data, "%dUOinfor_agent.csv" % n)
        if not (os.path.isfile(f_act) and os.path.isfile(f_uo)):
            log("%s: warning: bigloop %d missing actual/UOinfor agent "
                "files -- skipped" % (job["key"], n))
            continue
        out = os.path.join(job["out_dir"], "bigloop_%d" % n)
        os.makedirs(out, exist_ok=True)
        copied = 0
        for tag in T1_TAGS:
            for kind in T1_KINDS:
                fn = "%d%s_%s.csv" % (n, tag, kind)
                src = os.path.join(data, fn)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out, fn))
                    copied += 1
                else:
                    log("%s: warning: expected file missing: %s"
                        % (job["key"], fn))
        log("%s: bigloop %d (mpr %d%%) -> %s (%d files)"
            % (job["key"], n, mpr, out, copied))
        metrics = collect_metrics(
            os.path.join(out, "%dactual_agent.csv" % n),
            os.path.join(out, "%dactual_LinkMOE.csv" % n),
            uo_path=os.path.join(out, "%dUOinfor_agent.csv" % n),
            predict_linkmoe_path=os.path.join(
                out, "%dpredict_LinkMOE.csv" % n))
        append_summary_row(results_root, seed, job["test"], job["level"],
                           method, mpr, metrics)
        collected.append(n)
    log("%s: collected %d/%d bigloops"
        % (job["key"], len(collected), len(job["bigloops"])))
    _collect_engine_log(sandbox, job["out_dir"])
    if len(collected) < len(job["bigloops"]):
        _missing = [n for n, _m in job["bigloops"]
                    if n not in collected]
        raise RuntimeError(
            "%s: sweep incomplete: collected %d/%d bigloops, "
            "missing %s"
            % (job["key"], len(collected), len(job["bigloops"]),
               _missing))
    return collected


# ----------------------------------------------------------------------
# Test 4: link-volume validation (old Test 6) at the 50k baseline
# ----------------------------------------------------------------------
def _proposed_pair_exists(d):
    """True if some filename prefix in d has BOTH *predict_LinkMOE.csv and
    *actual_LinkMOE.csv. Mirrors volume_estimation.find_proposed_pair."""
    preds = set()
    for path in glob.glob(os.path.join(d, "*" + T4_PREDICT_SUFFIX)):
        preds.add(os.path.basename(path)[:-len(T4_PREDICT_SUFFIX)])
    acts = set()
    for path in glob.glob(os.path.join(d, "*" + T4_ACTUAL_SUFFIX)):
        acts.add(os.path.basename(path)[:-len(T4_ACTUAL_SUFFIX)])
    return bool(preds & acts)


def _scenario_pair_exists(method_dir):
    """True if a scenario method dir has both predicted_LinkMOE.csv and
    the reaction run's actual LinkMOE file."""
    pred = os.path.join(method_dir, "predicted_LinkMOE.csv")
    act = os.path.join(method_dir, "full_results", "output_LinkMOE.csv")
    return os.path.isfile(pred) and os.path.isfile(act)


def test1_results_exist(results_root):
    """True if Test 1's four systems all have complete predicted/actual
    LinkMOE pairs under results/test1/ -- i.e. Test 4 can run without
    rerunning Test 1."""
    base = os.path.join(results_root, "test1")
    if not _proposed_pair_exists(os.path.join(base, "proposed",
                                              "full_results")):
        return False
    for m in ("rt1", "rt10", "ue"):
        if not _scenario_pair_exists(os.path.join(base, m)):
            return False
    return True


def run_test4(results_root, seed, tracker=None):
    """Run volume_estimation.py over Test 1's four systems (Test 4).

    volume_estimation.py still expects the old test1..test4 layout, so
    this builds a small compatibility view (copies of the LinkMOE files
    only -- the files are 76 links each) mapping the new layout onto
    it: view/test1/full_results <- test1/proposed/full_results,
    view/test2 <- test1/rt1, view/test3 <- test1/rt10,
    view/test4 <- test1/ue. Outputs go to results/test4/. Appends one
    summary.csv row per system (test=4, level=BASELINE_LEVEL) carrying only
    linkvol_acc; the other metrics live on the Test 1 rows. Returns
    (ok, elapsed). Raises RuntimeError if no system is usable.
    """
    import tempfile
    base = os.path.join(results_root, "test1")
    out_dir = os.path.join(results_root, "test4")
    os.makedirs(out_dir, exist_ok=True)

    view = tempfile.mkdtemp(prefix="test4_view_")
    try:
        # Proposed system -> view/test1/full_results (predict/actual pair).
        src_full = os.path.join(base, "proposed", "full_results")
        dst_full = os.path.join(view, "test1", "full_results")
        os.makedirs(dst_full, exist_ok=True)
        for pat in ("*" + T4_PREDICT_SUFFIX, "*" + T4_ACTUAL_SUFFIX):
            for src in glob.glob(os.path.join(src_full, pat)):
                shutil.copy2(src, dst_full)
        # Scenario systems -> view/testN (predicted + actual).
        for n, m in ((2, "rt1"), (3, "rt10"), (4, "ue")):
            src_dir = os.path.join(base, m)
            dst_dir = os.path.join(view, "test%d" % n)
            os.makedirs(os.path.join(dst_dir, "full_results"),
                        exist_ok=True)
            pred = os.path.join(src_dir, "predicted_LinkMOE.csv")
            if os.path.isfile(pred):
                shutil.copy2(pred, dst_dir)
            act = os.path.join(src_dir, "full_results",
                               "output_LinkMOE.csv")
            if os.path.isfile(act):
                shutil.copy2(act, os.path.join(dst_dir, "full_results"))
        systems = ["test%d" % n for n in (1, 2, 3, 4)]
        cmd = [PY, os.path.join(HERE, "volume_estimation.py"),
               "--results-dir", view,
               "--output-dir", out_dir,
               "--systems", ",".join(systems)]
        if tracker:
            tracker.start("t4")
            tracker.update("t4", None, "running volume estimation...")
        log("Test 4: estimating future-condition volumes (%s) ..."
            % ",".join(systems))
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, timeout=4 * 3600).returncode
        except subprocess.TimeoutExpired:
            log("ERROR: volume_estimation.py (Test 4) timed out after 4h")
            if tracker:
                tracker.finish("t4", False)
            return False, time.time() - t0
        dt = time.time() - t0
        ok = rc == 0
        if tracker:
            tracker.finish("t4", ok)
        log("Test 4: %s (%s)" % ("done" if ok else "FAILED", fmt_min(dt)))
        if not ok:
            return False, dt
        # One summary.csv row per system, carrying only linkvol_acc.
        # Accuracy is recomputed here from volume_estimation.csv with
        # the same "<10%" rule as linkvol_accuracy(); rows with an
        # empty discrepancy (zero actual volume) are excluded.
        acc = _test4_accuracies(os.path.join(out_dir,
                                             "volume_estimation.csv"))
        sys_method = {1: "proposed", 2: "realtime_1min",
                      3: "realtime_10min", 4: "ue"}
        for n in (1, 2, 3, 4):
            metrics = {"satisfaction": None, "veh_hrs": None,
                       "delay": None, "energy": None, "co2": None,
                       "nox": None, "co": None, "hc": None,
                       "linkvol_acc": acc.get(n),
                       "max_vc": None, "p95_vc": None, "over_cap": None,
                       "heavy": None}
            append_summary_row(results_root, seed, 4, BASELINE_LEVEL,
                               sys_method[n], None, metrics)
        return True, dt
    finally:
        shutil.rmtree(view, ignore_errors=True)


def _test4_accuracies(csv_path):
    """{test number: link-volume accuracy %} from volume_estimation.csv.
    Systems are named test1..test4 in that file."""
    acc = {}
    if not os.path.isfile(csv_path):
        return acc
    counts = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int((row.get("system") or "").strip()[-1])
            except (ValueError, IndexError):
                continue
            disc = (row.get("discrepancy_pct") or "").strip()
            bin_ = (row.get("bin") or "").strip()
            denom, within = counts.get(n, (0, 0))
            if disc != "":
                denom += 1
                if bin_ == "<10%":
                    within += 1
            counts[n] = (denom, within)
    for n, (denom, within) in counts.items():
        acc[n] = (100.0 * within / denom) if denom else None
    return acc


# ----------------------------------------------------------------------
# Test 2 aggregation (mpr_aggregate.py over the collected sweeps)
# ----------------------------------------------------------------------
def run_aggregation(results_root, seed, tracker=None):
    """Run mpr_aggregate.py in-place on the collected Test 2 sweeps
    (no sandbox). Writes mpr_table5.csv, mpr_table6.csv and
    mpr_summary.txt into results/test2/. Returns (ok, elapsed). Raises
    RuntimeError if there is nothing to aggregate."""
    sweep_dir = os.path.join(results_root, "test2", "proposed_sweep")
    ue_dir = os.path.join(results_root, "test2", "ue_sweep")
    out_dir = os.path.join(results_root, "test2")
    os.makedirs(out_dir, exist_ok=True)
    if not glob.glob(os.path.join(sweep_dir, "bigloop_*")):
        raise RuntimeError(
            "Test 2: nothing to aggregate (no bigloop dirs in %s)"
            % sweep_dir)
    cmd = [PY, os.path.join(HERE, "mpr_aggregate.py"),
           "--input-dir", sweep_dir,
           "--ue-dir", ue_dir,
           "--seed", str(seed),
           "--output-dir", out_dir]
    if tracker:
        tracker.start("t2-agg")
        tracker.update("t2-agg", None, "aggregating...")
    log("Test 2: aggregating MPR sweep (mpr_aggregate.py) ...")
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, timeout=4 * 3600).returncode
    except subprocess.TimeoutExpired:
        log("ERROR: mpr_aggregate.py timed out after 4h; marking aggregation failed")
        if tracker:
            tracker.finish("t2-agg", False)
        return False, time.time() - t0
    dt = time.time() - t0
    ok = rc == 0
    if tracker:
        tracker.finish("t2-agg", ok)
    log("Test 2: aggregation %s (%s)" % ("done" if ok else "FAILED",
                                        fmt_min(dt)))
    return ok, dt

# ----------------------------------------------------------------------
# Run manifest (big-run package)
# ----------------------------------------------------------------------
# results/manifest.txt: methods setup facts for the paper, read at
# runtime from the actual input files -- never hardcoded Sioux Falls
# numbers, so the same code serves Baltimore (--network baltimore).
# Written once at run start (status: started) and rewritten at run end
# (status: completed/failed) with outcomes. A missing or unreadable
# input file is recorded as "unknown", never fatal: the manifest must
# not kill a run.
MANIFEST_NAME = "manifest.txt"
_MANIFEST_UNKNOWN = "unknown"


def _manifest_ts(t):
    if t is None:
        return _MANIFEST_UNKNOWN
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t))
    except Exception:
        return _MANIFEST_UNKNOWN


def _manifest_csv_rows(path):
    """Data-row count of a CSV file, or None if it cannot be read."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0
        return max(0, len(rows) - 1)
    except Exception:
        return None


def _manifest_agent_stats(pool_file, data_dir):
    """(agents, dep_min, dep_max, od_cells) from the pool file (or
    input_agent_initial.csv when no pool file is given). Any element is
    None when the file is missing or unreadable."""
    path = pool_file or os.path.join(data_dir, "input_agent_initial.csv")
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        cols = {}
        for i, name in enumerate(header):
            cols[name.strip()] = i
        need = ("from_zone_id", "to_zone_id", "departure_time_in_min")
        if any(c not in cols for c in need):
            return None, None, None, None
        agents = 0
        dep_min = None
        dep_max = None
        ods = set()
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) <= max(cols[c] for c in need):
                    continue
                agents += 1
                try:
                    dep = float(row[cols["departure_time_in_min"]])
                except (ValueError, TypeError):
                    dep = None
                if dep is not None:
                    dep_min = dep if dep_min is None else min(dep_min, dep)
                    dep_max = dep if dep_max is None else max(dep_max, dep)
                ods.add((row[cols["from_zone_id"]].strip(),
                         row[cols["to_zone_id"]].strip()))
        return agents, dep_min, dep_max, len(ods)
    except Exception:
        return None, None, None, None


def _manifest_demand_od_cells(data_dir):
    """Nonzero cells of input_demand.csv (OD matrix), or None."""
    path = os.path.join(data_dir, "input_demand.csv")
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None
        count = 0
        for row in rows[1:]:
            for val in row[1:]:
                try:
                    if float(val) != 0:
                        count += 1
                except (ValueError, TypeError):
                    continue
        return count
    except Exception:
        return None


def _manifest_choice_set(data_dir):
    """(source_file, od_cells, total_routes) for the choice set."""
    pkl = os.path.join(data_dir, "choice_sets.pkl")
    if os.path.isfile(pkl):
        try:
            import pickle
            with open(pkl, "rb") as f:
                choice_dict = pickle.load(f)
            return ("choice_sets.pkl", len(choice_dict),
                    sum(len(v) for v in choice_dict.values()))
        except Exception:
            return ("choice_sets.pkl", None, None)
    mats = sorted(glob.glob(os.path.join(data_dir, "choice*.mat")))
    if mats:
        src = os.path.basename(mats[0])
        try:
            from scipy.io import loadmat
            mat = loadmat(mats[0])
            final = mat.get("finallist")
            if final is None:
                return (src, None, None)
            ods = final.shape[1] if len(final.shape) == 2 else final.shape[0]
            routes = 0
            for cell in final.flat:
                try:
                    if cell is not None and len(cell):
                        routes += 1
                except TypeError:
                    routes += 1
            return (src, ods, routes)
        except Exception:
            return (src, None, None)
    return (None, None, None)


def _manifest_val(value):
    return _MANIFEST_UNKNOWN if value is None else str(value)


def write_run_manifest(results_root, args, tests, data_dir, status,
                       outcomes=None, agg_status=None, started=None,
                       ended=None, cmdline=None, pool_file=None,
                       levels=None, sample_counts=None):
    """Write (or rewrite) results/manifest.txt. Never raises."""
    try:
        agents, dep_min, dep_max, agent_ods = _manifest_agent_stats(
            pool_file, data_dir)
        demand_ods = _manifest_demand_od_cells(data_dir)
        links = _manifest_csv_rows(os.path.join(data_dir, "input_link.csv"))
        nodes = _manifest_csv_rows(os.path.join(data_dir, "input_node.csv"))
        choice_src, choice_ods, choice_routes = _manifest_choice_set(data_dir)
        lines = []
        lines.append("Run manifest -- Tests 1-4 (wholerunner.py v2)")
        lines.append("status: %s" % status)
        lines.append("timestamp_start: %s" % _manifest_ts(started))
        lines.append("timestamp_end: %s" % _manifest_ts(ended))
        lines.append("command_line: %s" % (cmdline or _MANIFEST_UNKNOWN))
        lines.append("network: %s" % args.network)
        lines.append("data_dir: %s" % data_dir)
        lines.append("results_dir: %s" % results_root)
        lines.append("engine: %s" % args.engine)
        lines.append("seed: %s" % args.seed)
        lines.append("pool_file: %s" % (pool_file or _MANIFEST_UNKNOWN))
        lines.append("")
        lines.append("Demand and network (read from the input files):")
        lines.append("pool_agents: %s" % _manifest_val(agents))
        lines.append("od_cells_pool_file: %s" % _manifest_val(agent_ods))
        lines.append("od_cells_demand_file: %s" % _manifest_val(demand_ods))
        lines.append("choice_set_file: %s" % _manifest_val(choice_src))
        lines.append("choice_set_od_cells: %s" % _manifest_val(choice_ods))
        lines.append("choice_set_routes_total: %s"
                     % _manifest_val(choice_routes))
        lines.append("links: %s" % _manifest_val(links))
        lines.append("nodes: %s" % _manifest_val(nodes))
        if dep_min is not None and dep_max is not None:
            lines.append("departure_window_min: %s (minutes)" % dep_min)
            lines.append("departure_window_max: %s (minutes)" % dep_max)
        else:
            lines.append("departure_window_min: %s" % _MANIFEST_UNKNOWN)
            lines.append("departure_window_max: %s" % _MANIFEST_UNKNOWN)
        if sample_counts:
            lines.append("sampled_levels: %s"
                         % ", ".join("%d agents (level %d)" % (c, lv)
                                     for lv, c in sorted(sample_counts.items())))
        lines.append("")
        lines.append("Methods setup:")
        lines.append("standardization: meanstd28 "
                     "(weights sheet cols A:E, rows 2-3; "
                     "UE standardization per msa.m / msa.py)")
        lines.append("preference_weights_predict: estimated weights, "
                     "cols A:F of the weights sheet")
        lines.append("preference_weights_reaction: true weights, "
                     "cols J:O of the weights sheet")
        lines.append("ue_gap_threshold: %s" % args.gap_threshold)
        lines.append("ue_max_iters: %s" % UE_MAX_ITERS)
        lines.append("baseline_level: %s agents" % BASELINE_LEVEL)
        lines.append("")
        lines.append("Per-test configuration (selected tests only):")
        lines.append("tests_selected: %s"
                     % ", ".join(str(t) for t in tests))
        if 1 in tests:
            lines.append("test1: baseline %d agents, four systems "
                         "(proposed bigloop 10, real-time 1-min, "
                         "real-time 10-min, UE)" % BASELINE_LEVEL)
        if 2 in tests:
            lines.append("test2: MPR sweep at %d agents, proposed "
                         "bigloops 0-10, UE-guidance bigloops 1-10"
                         % BASELINE_LEVEL)
        if 3 in tests:
            lines.append("test3_levels: %s"
                         % ", ".join(str(lv) for lv in (levels or [])))
            lines.append("test3_mpr_points: %s"
                         % ", ".join(str(p) for p in args.mpr_points))
            lines.append("test3: at each level, four systems + MPR sweep "
                         "at the listed points (proposed AND UE); "
                         "the 50k level is re-run, not reused")
        if 4 in tests:
            lines.append("test4: link-volume validation at the %d-agent "
                         "baseline (volume_estimation.py over Test 1's "
                         "four systems)" % BASELINE_LEVEL)
        if outcomes is not None:
            lines.append("")
            lines.append("Outcomes:")
            for t in tests:
                if t == 2:
                    lines.append("test2_proposed_sweep: %s"
                                 % ("OK" if outcomes.get("t2-P")
                                    else "FAILED"))
                    lines.append("test2_ue_sweep: %s"
                                 % ("OK" if outcomes.get("t2-U")
                                    else "FAILED"))
                    lines.append("test2_aggregation: %s"
                                 % (agg_status or _MANIFEST_UNKNOWN))
                else:
                    lines.append("test%d: %s"
                                 % (t, "OK" if outcomes.get(t) else "FAILED"))
        path = os.path.join(results_root, MANIFEST_NAME)
        with open(path, "w", newline="") as f:
            f.write("\r\n".join(lines) + "\r\n")
        log("Run manifest: %s" % path)
    except Exception as e:
        log("WARNING: could not write run manifest: %s" % e)


# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------
def print_summary(results_root, tests, outcomes, agg_status):
    """outcomes maps test number -> bool (test 2 covers both sweeps
    plus aggregation through agg_status)."""
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for t in tests:
        if t == 2:
            log("Test 2 (MPR sweep): proposed=%s ue=%s aggregation=%s"
                % ("OK" if outcomes.get("t2-P") else "FAILED",
                   "OK" if outcomes.get("t2-U") else "FAILED",
                   agg_status))
            for fn in ("mpr_table5.csv", "mpr_table6.csv", "mpr_summary.txt"):
                p = os.path.join(results_root, "test2", fn)
                log("  %s: %s" % (fn, "present" if os.path.isfile(p)
                                  else "MISSING"))
            continue
        if t == 4:
            log("Test 4 (link-volume validation): %s"
                % ("OK" if outcomes.get(4) else "FAILED"))
            p = os.path.join(results_root, "test4",
                             "volume_estimation_summary.txt")
            log("  volume_estimation_summary.txt: %s"
                % ("present" if os.path.isfile(p) else "MISSING"))
            continue
        log("Test %d: %s" % (t, "OK" if outcomes.get(t) else "FAILED"))
    summ = os.path.join(results_root, "summary.csv")
    if os.path.isfile(summ):
        try:
            n_rows = count_csv_rows(summ)
        except Exception:
            n_rows = "?"
        log("summary.csv: present (%s rows)" % n_rows)
    else:
        log("summary.csv: MISSING")
    log("=" * 60)


# ----------------------------------------------------------------------
# Auto-clear (per-test / per-level scope only, never the results root)
# ----------------------------------------------------------------------
def auto_clear_results(results_root, tests, levels):
    """Delete and recreate the results subdirectories for the tests in
    `tests` -- per-test scope only, never the results root itself. For
    Test 3 only the selected levels' directories are cleared, so a
    re-run with --levels 20000 leaves other levels' results alone.
    Also removes the per-seed summary.csv: it belongs to this run,
    so a re-run starts with an empty summary instead of appending
    duplicate rows onto the previous run's.

    Why: the collect_* functions only overwrite files, never delete
    them, so a stale file from a partial/failed run would otherwise
    survive into the new run's tables. Clearing before a test runs
    makes every run start from an empty per-test dir.

    A removal failure is fatal (sys.exit(1)): proceeding with a
    half-cleared dir would silently mix stale and fresh results. The
    usual cause on Windows is a locked file (stale python.exe / open
    file); unlock and rerun.
    """
    _summ = os.path.join(results_root, "summary.csv")
    if os.path.isfile(_summ):
        try:
            os.remove(_summ)
        except OSError as e:
            print("ERROR: Auto-clear failed to remove %s: %s"
                  % (_summ, e), file=sys.stderr)
            sys.exit(1)
        log("Auto-clear: removed %s" % _summ)
    else:
        log("Auto-clear: %s did not exist, nothing to remove" % _summ)

    def _clear_dir(out):
        # Safety: delete only a direct child of the results root (or a
        # level_<N> dir directly under test3/), never the root itself.
        root_abs = os.path.abspath(results_root)
        out_abs = os.path.abspath(out)
        parent = os.path.dirname(out_abs)
        if out_abs == root_abs or parent not in (
                root_abs, os.path.join(root_abs, "test3")):
            return
        if os.path.isdir(out):
            try:
                shutil.rmtree(out)
            except OSError as e:
                print("ERROR: Auto-clear failed to remove %s: %s"
                      % (out, e), file=sys.stderr)
                sys.exit(1)
            log("Auto-clear: removed %s" % out)
        elif os.path.lexists(out):
            try:
                os.remove(out)
            except OSError as e:
                print("ERROR: Auto-clear failed to remove %s: %s"
                      % (out, e), file=sys.stderr)
                sys.exit(1)
            log("Auto-clear: removed file %s" % out)
        else:
            log("Auto-clear: %s did not exist, nothing to remove" % out)
        os.makedirs(out, exist_ok=True)

    for t in sorted(set(tests)):
        if t == 3:
            # Only the selected levels' dirs are cleared; other
            # level_<N>/ dirs from earlier runs are left alone.
            for lv in levels:
                _clear_dir(os.path.join(results_root, "test3",
                                        "level_%d" % lv))
        else:
            _clear_dir(os.path.join(results_root, "test%d" % t))


# ----------------------------------------------------------------------
# Multiseed driver
# ----------------------------------------------------------------------
def _seed_argv(base_argv, seed, results_dir):
    """Build the argv for one seed's subprocess run.

    Removes --seeds (and its values), --seed and --results-dir (both
    the "--opt value" and "--opt=value" spellings), then appends
    --seed <seed> and --results-dir <results_dir>. Every other option
    is passed through unchanged.
    """
    out = []
    i = 0
    n = len(base_argv)
    while i < n:
        tok = base_argv[i]
        if tok == "--seeds":
            i += 1
            while i < n and not base_argv[i].startswith("-"):
                i += 1
            continue
        if tok.startswith("--seeds="):
            i += 1
            continue
        if tok == "--seed":
            i += 2
            continue
        if tok.startswith("--seed="):
            i += 1
            continue
        if tok == "--results-dir":
            i += 2
            continue
        if tok.startswith("--results-dir="):
            i += 1
            continue
        out.append(tok)
        i += 1
    out.append("--seed")
    out.append(str(seed))
    out.append("--results-dir")
    out.append(results_dir)
    return out


def run_multiseed(args, argv):
    """Run the whole pipeline once per seed, then aggregate.

    Each seed runs as its own subprocess (python wholerunner.py with
    --seed <seed> and a per-seed --results-dir), so seeds are fully
    independent: a failure in one seed does not stop the others, and
    each seed gets its own temp dir, run manifest and summary.csv.
    aggregate_v2.py then combines the per-seed summary.csv
    files into mean and standard deviation. Returns a process exit
    code (0 when every seed and the aggregation succeeded).
    """
    seeds = list(args.seeds)
    raw = list(argv) if argv is not None else sys.argv[1:]
    if any(t == "--seed" or t.startswith("--seed=") for t in raw):
        print("ERROR: --seed and --seeds cannot be combined", file=sys.stderr)
        return 2
    if len(set(seeds)) != len(seeds):
        print("ERROR: --seeds must not repeat a seed", file=sys.stderr)
        return 2
    if any(s < 0 for s in seeds):
        print("ERROR: --seeds values must be non-negative", file=sys.stderr)
        return 2
    t_start = time.time()
    base = os.path.abspath(args.results_dir)
    seed_dirs = {s: base + "_seed%d" % s for s in seeds}
    out_dir = base + "_multiseed"
    os.makedirs(out_dir, exist_ok=True)
    log("Multi-seed: seeds %s" % seeds)
    log("Multi-seed: per-seed results under %s_seed<seed>" % base)
    log("Multi-seed: aggregate output in %s" % out_dir)
    script = os.path.abspath(__file__)
    ok = {}
    for s in seeds:
        per_seed_argv = _seed_argv(raw, s, seed_dirs[s])
        cmd = [PY, script] + per_seed_argv
        log("Multi-seed: seed %d starting -> %s" % (s, seed_dirs[s]))
        t0 = time.time()
        try:
            rc = subprocess.run(cmd).returncode
        except Exception as e:
            log("Multi-seed: seed %d failed to start: %s" % (s, e))
            rc = 1
        dt = time.time() - t0
        ok[s] = (rc == 0)
        log("Multi-seed: seed %d %s (%s, rc=%d)"
            % (s, "done" if rc == 0 else "FAILED", fmt_min(dt), rc))
    good_dirs = [seed_dirs[s] for s in seeds if ok.get(s)]
    good_seeds = [s for s in seeds if ok.get(s)]
    if not good_dirs:
        print("ERROR: Multi-seed: no seed completed successfully; "
              "nothing to aggregate", file=sys.stderr)
        return 1
    agg_script = os.path.join(HERE, "aggregate_v2.py")
    if not os.path.isfile(agg_script):
        print("ERROR: Multi-seed: aggregate_v2.py not found next to "
              "wholerunner.py", file=sys.stderr)
        return 1
    agg_cmd = [PY, agg_script,
               "--seed-dirs"] + good_dirs + [
               "--seeds"] + [str(s) for s in good_seeds] + [
               "--out-dir", out_dir]
    log("Multi-seed: aggregating %d seed(s) ..." % len(good_dirs))
    try:
        arc = subprocess.run(agg_cmd).returncode
    except Exception as e:
        print("ERROR: Multi-seed: aggregation failed to start: %s" % e,
              file=sys.stderr)
        return 1
    if arc != 0:
        print("ERROR: Multi-seed: aggregation FAILED (rc=%d)" % arc,
              file=sys.stderr)
        return 1
    failed = [s for s in seeds if not ok.get(s)]
    if failed:
        print("Multi-seed: completed with failed seed(s): %s "
              "(aggregate covers the successful seeds only)" % failed)
        return 1
    log("Multi-seed: all done (%s total)" % fmt_min(time.time() - t_start))
    return 0

# ----------------------------------------------------------------------
# Job construction
# ----------------------------------------------------------------------
def _job(key, label, kind, test, level, method, scripts, out_dir,
         collect, commands, bigloops=None, sub=None, depends_on=None):
    """One job dict. `commands` is a callable (sandbox, args) -> cmds.
    `bigloops` (sweeps only) is a list of (bigloop, mpr) pairs.
    `depends_on` is the key of the UE job whose ue_routes.csv this job
    needs (UE-guidance sweeps only)."""
    JOB_KINDS[key] = kind
    DISPLAY_LABELS[key] = label
    return {"key": key, "label": label, "kind": kind, "test": test,
            "level": level, "method": method,
            "scripts": scripts_for(scripts), "out_dir": out_dir,
            "collect": collect, "commands": commands,
            "bigloops": bigloops or [], "sub": sub,
            "depends_on": depends_on,
            "sandbox_name": key.replace("-", "_")}


def build_jobs(args, tests, levels, mpr_points, results_root):
    """Build the job dict for the selected tests. Returns (jobs, extra):
    jobs maps key -> job dict; extra is the set of keys that are
    dependency-only jobs (not part of the selected tests) -- currently
    just the Test 1 UE job when Test 2 needs it but Test 1 was not
    selected and no usable ue_routes.csv exists yet."""
    jobs = {}
    extra = set()
    t1 = os.path.join(results_root, "test1")

    def add_system_jobs(test, level, base_dir):
        """The four system jobs for one (test, level)."""
        prefix = "t%d" % test if test == 1 else "t3-L%d" % level
        # Proposed (v9, bigloop 10).
        jobs[prefix + "-proposed"] = _job(
            prefix + "-proposed",
            "T%d L%d Proposed" % (test, level), "v9", test, level,
            "proposed", V9_SCRIPTS,
            os.path.join(base_dir, "proposed"), "proposed",
            lambda sb, a: proposed_commands(sb, a.seed, a.network,
                                           a.engine))
        # Real-time 1-min and 10-min.
        for method, mlabel in (("realtime_1min", "RT-1min"),
                               ("realtime_10min", "RT-10min")):
            jobs[prefix + "-" + METHOD_DIRS[method]] = _job(
                prefix + "-" + METHOD_DIRS[method],
                "T%d L%d %s" % (test, level, mlabel), "rt", test, level,
                method,
                ["scenarios_realtime.py", "preference_reaction.py",
                 "netconfig.py"],
                os.path.join(base_dir, METHOD_DIRS[method]), "scenario",
                lambda sb, a, m=method: scenario_commands(
                    m, sb, a.seed, a.network, a.gap_threshold, a.engine))
        # UE guidance.
        jobs[prefix + "-ue"] = _job(
            prefix + "-ue", "T%d L%d UE" % (test, level), "ue", test,
            level, "ue",
            ["scenarios_ue.py", "preference_reaction.py", "netconfig.py"],
            os.path.join(base_dir, "ue"), "scenario",
            lambda sb, a: scenario_commands(
                "ue", sb, a.seed, a.network, a.gap_threshold, a.engine))

    if 1 in tests:
        add_system_jobs(1, BASELINE_LEVEL, t1)
    if 3 in tests:
        for lv in levels:
            add_system_jobs(3, lv, os.path.join(
                results_root, "test3", "level_%d" % lv))

    # Test 2 sweeps (MPR sweep at the 50k baseline).
    if 2 in tests:
        t2 = os.path.join(results_root, "test2")
        p_loops = [(n, 0 if n == 0 else n * 10) for n in range(0, 11)]
        jobs["t2-P"] = _job(
            "t2-P", "T2 Proposed sweep", "v9", 2, BASELINE_LEVEL,
            "proposed", V9_SCRIPTS,
            os.path.join(t2, "proposed_sweep"), "sweep",
            lambda sb, a: sweep_commands(
                "P", sb, a.seed,
                bigloops=[n for n, _m in p_loops],
                network=a.network, engine=a.engine),
            bigloops=p_loops, sub="P")
        u_loops = [(n, n * 10) for n in range(1, 11)]
        ue_job_key = "t1-ue"
        if 1 not in tests:
            # Test 2's UE sweep needs Test 1's ue_routes.csv. Prefer an
            # existing one; otherwise run the Test 1 UE job as a
            # dependency-only job (its results land in test1/ue/).
            existing = os.path.join(t1, "ue", "ue_routes.csv")
            if os.path.isfile(existing):
                log("Test 2: using existing results/test1/ue/ue_routes.csv")
                ue_job_key = None
                jobs["t2-U"] = _job(
                    "t2-U", "T2 UE sweep", "v9", 2, BASELINE_LEVEL, "ue",
                    UE_SWEEP_SCRIPTS, os.path.join(t2, "ue_sweep"),
                    "sweep",
                    lambda sb, a: sweep_commands(
                        "U", sb, a.seed,
                        bigloops=[n for n, _m in u_loops],
                        ue_routes_path=os.path.join(sb, "ue_routes.csv"),
                        network=a.network, engine=a.engine),
                    bigloops=u_loops, sub="U")
                jobs["t2-U"]["ue_routes_src"] = existing
            else:
                log("Test 2: no usable ue_routes.csv -- running the "
                    "Test 1 UE job as a dependency")
                add_system_jobs(1, BASELINE_LEVEL, t1)
                # Only the UE job is a dependency; drop the other three
                # Test 1 jobs (Test 1 was not selected).
                for k in ("t1-proposed", "t1-rt1", "t1-rt10"):
                    jobs.pop(k, None)
                    JOB_KINDS.pop(k, None)
                    DISPLAY_LABELS.pop(k, None)
                extra.add("t1-ue")
        if ue_job_key is not None:
            jobs["t2-U"] = _job(
                "t2-U", "T2 UE sweep", "v9", 2, BASELINE_LEVEL, "ue",
                UE_SWEEP_SCRIPTS, os.path.join(t2, "ue_sweep"), "sweep",
                lambda sb, a: sweep_commands(
                    "U", sb, a.seed,
                    bigloops=[n for n, _m in u_loops],
                    ue_routes_path=os.path.join(sb, "ue_routes.csv"),
                    network=a.network, engine=a.engine),
                bigloops=u_loops, sub="U", depends_on="t1-ue")

    # Test 3 sweeps at the --mpr-points points (proposed AND UE, per level).
    if 3 in tests:
        for lv in levels:
            base = os.path.join(results_root, "test3", "level_%d" % lv)
            pts = [(p, p) for p in mpr_points]
            prefix = "t3-L%d" % lv
            jobs[prefix + "-P"] = _job(
                prefix + "-P", "T3 L%d Proposed sweep" % lv, "v9", 3, lv,
                "proposed", V9_SCRIPTS,
                os.path.join(base, "proposed_sweep"), "sweep",
                lambda sb, a, pp=mpr_points: sweep_commands(
                    "P", sb, a.seed, mpr_points=pp,
                    network=a.network, engine=a.engine),
                bigloops=pts, sub="P")
            jobs[prefix + "-U"] = _job(
                prefix + "-U", "T3 L%d UE sweep" % lv, "v9", 3, lv, "ue",
                UE_SWEEP_SCRIPTS, os.path.join(base, "ue_sweep"), "sweep",
                lambda sb, a, pp=mpr_points: sweep_commands(
                    "U", sb, a.seed, mpr_points=pp,
                    ue_routes_path=os.path.join(sb, "ue_routes.csv"),
                    network=a.network, engine=a.engine),
                bigloops=pts, sub="U", depends_on=prefix + "-ue")
    return jobs, extra


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run Tests 1-4 (v2 revamp) in parallel sandboxes: "
                    "50k-agent baseline, MPR sweep, demand-level sweep, "
                    "link-volume validation.")
    ap.add_argument("--tests", default=None,
                    help="tests to run: comma-separated ints and ranges, e.g. "
                         "1 | 1,3 | 1-4 (default: 1,2,3,4)")
    ap.add_argument("--data-dir", default="DTALite_Files",
                    help="read-only DTALite_Files source (default: "
                         "DTALite_Files)")
    ap.add_argument("--pool-file", default=None,
                    help="agent pool CSV to sample from (default: "
                         "<data-dir>/input_agent_pool.csv). Missing pool "
                         "file is a fatal error; there is no fallback.")
    ap.add_argument("--levels", default="20000,30000,40000,50000,60000,"
                                        "70000,80000",
                    help="Test 3 demand levels in agents, comma-separated "
                         "(default: 20000,30000,40000,50000,60000,70000,"
                         "80000)")
    ap.add_argument("--baseline-level", type=int, default=BASELINE_LEVEL,
                    help="baseline demand level in agents for Tests 1, 2 "
                         "and 4 (default: 50000). Test 3 levels still come "
                         "from --levels. Use a small value (e.g. 200) for "
                         "smoke runs against a small pool.")
    ap.add_argument("--mpr-points", default="33,66",
                    help="Test 3 MPR sweep points in percent, "
                         "comma-separated (default: 33,66)")
    ap.add_argument("--gap-threshold", type=float, default=0.001,
                    help="UE relative gap convergence threshold (UE jobs; "
                         "the UE sweeps reuse the UE job's ue_routes.csv "
                         "and so inherit this threshold; default: 0.001)")
    ap.add_argument("--network", default="siouxfalls",
                    choices=["siouxfalls", "baltimore"],
                    help="network: siouxfalls (default) or baltimore "
                         "(use with --data-dir baltimore_files)")
    ap.add_argument("--engine", default="pysim",
                    choices=["pysim", "dtalite"],
                    help="simulation engine: pysim (default) or dtalite "
                         "(dtalite runs the DTALite executable exactly as "
                         "before; pysim runs the Python port)")
    ap.add_argument("--results-dir", default="results",
                    help="output root (default: results)")
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep temp/run_<pid>/ sandboxes after success")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="parallel workers (default: min(4, jobs))")
    ap.add_argument("--stall-timeout-min", type=float, default=30,
                    help="kill a job if no progress for this many minutes "
                         "(default: 30; 0 disables)")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable the live progress display")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for all jobs (single-seed run). "
                         "Cannot be combined with --seeds. When neither "
                         "--seed nor --seeds is given, all default seeds "
                         "run (see --seeds).")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="run all tests for each of these seeds (e.g. "
                         "--seeds 42 43 44); each seed's results go to "
                         "<results-dir>_seed<seed> and the headline "
                         "metrics are aggregated into <results-dir>_multiseed. "
                         "Default: 42 423 454 162 904 916 91 962 201 624. "
                         "Cannot be combined with --seed.")
    ap.add_argument("--test4-force-rerun", action="store_true",
                    help="Test 4: ignore existing results/test1/ outputs and "
                         "rerun Test 1 before validating")
    ap.add_argument("--skip-aggregation", action="store_true",
                    help="Test 2: run both MPR sweeps but skip the "
                         "mpr_aggregate.py step")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    global _RUN_STALL_S, BASELINE_LEVEL
    if args.baseline_level < 1:
        print("ERROR: --baseline-level must be a positive integer "
              "(got %s)" % args.baseline_level, file=sys.stderr)
        sys.exit(2)
    BASELINE_LEVEL = args.baseline_level
    if args.stall_timeout_min <= 0:
        _RUN_STALL_S = float('inf')
    else:
        _RUN_STALL_S = args.stall_timeout_min * 60

    # Couple --network to data/results dirs: --network baltimore implies
    # --data-dir baltimore_files and --results-dir results_baltimore,
    # unless the user explicitly overrode them.
    if args.network == "baltimore":
        import sys as _sys
        _argv = _sys.argv[1:] if argv is None else argv
        _argv_str = " ".join(_argv) if _argv else ""
        if "--data-dir" not in _argv_str and args.data_dir == "DTALite_Files":
            args.data_dir = "baltimore_files"
        if "--results-dir" not in _argv_str and args.results_dir == "results":
            args.results_dir = "results_baltimore"

    raw = list(argv) if argv is not None else sys.argv[1:]
    seed_given = any(t == "--seed" or t.startswith("--seed=")
                     for t in raw)
    seeds_given = any(t == "--seeds" or t.startswith("--seeds=")
                      for t in raw)
    if seed_given and seeds_given:
        print("ERROR: --seed and --seeds cannot be combined",
              file=sys.stderr)
        sys.exit(2)

    try:
        tests = parse_tests_spec(
            args.tests if args.tests is not None else "1,2,3,4")
        levels = parse_int_list(args.levels, "--levels", lo=1)
        mpr_points = parse_int_list(args.mpr_points, "--mpr-points",
                                    lo=0, hi=100)
    except ValueError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        sys.exit(2)
    tests = sorted(set(tests))

    data_dir = os.path.abspath(args.data_dir)
    results_root = os.path.abspath(args.results_dir)
    temp_root = os.path.abspath(os.path.join(os.getcwd(), "temp", "run_%d" % os.getpid()))
    if not os.path.isdir(data_dir):
        print("ERROR: data dir not found: %s" % data_dir, file=sys.stderr)
        sys.exit(2)

    pool_file = (os.path.abspath(args.pool_file) if args.pool_file
                 else os.path.join(data_dir, "input_agent_pool.csv"))

    # Multi-seed mode: run the whole pipeline once per seed (each seed
    # in its own subprocess) and aggregate. Returns here; the
    # single-seed body below never runs in this mode. Bare invocation
    # (no --seed) runs the default seed list.
    if not seed_given:
        args.seeds = list(args.seeds)
        sys.exit(run_multiseed(args, argv))
    args.seed = args.seed if args.seed is not None else 42

    # The pool file is a hard dependency: fail loudly before anything
    # runs. There is no fallback to the old input file.
    if not os.path.isfile(pool_file):
        print("ERROR: pool file not found: %s\n"
              "Generate/place the agent pool file (same 15-column format "
              "as input_agent_initial.csv) or pass --pool-file <path>."
              % pool_file, file=sys.stderr)
        sys.exit(2)

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    # Test 4 needs Test 1's four systems. If Test 1 was not selected,
    # reuse existing results when usable; otherwise fold Test 1 in
    # (it runs in Phase A like any other selected test).
    if 4 in tests and 1 not in tests:
        if args.test4_force_rerun or not test1_results_exist(results_root):
            log("Test 4: no usable Test 1 results -- running Test 1 as a "
                "prerequisite")
            tests = sorted(set(tests) | {1})
        else:
            log("Test 4: using existing results/test1/ outputs")

    # Levels that need a sample: the 50k baseline for Tests 1/2/4
    # (including a Test 1 folded in for Test 4, and the Test 2 UE
    # dependency job), plus every --levels level for Test 3.
    needed_levels = set()
    if any(t in tests for t in (1, 2, 4)):
        needed_levels.add(BASELINE_LEVEL)
    if 3 in tests:
        needed_levels.update(levels)
    # Test 2's UE dependency job (Test 1 not selected, no existing
    # ue_routes.csv) also needs the 50k sample; build_jobs decides
    # that, so sample 50k whenever Test 2 is selected at all.
    if 2 in tests:
        needed_levels.add(BASELINE_LEVEL)

    # Sample once per (seed, level); every job at a level shares the
    # same sampled file. A sampling failure is fatal, before any job
    # is set up.
    samples = {}
    sample_weights = {}
    sample_counts = {}
    try:
        for lv in sorted(needed_levels):
            out_csv = os.path.join(temp_root, "samples",
                                   "sample_seed%d_level%d.csv"
                                   % (args.seed, lv))
            w_csv = sample_level(pool_file, args.seed, lv, out_csv)
            samples[lv] = out_csv
            sample_weights[lv] = w_csv
            sample_counts[lv] = lv
    except RuntimeError as e:
        print("ERROR: sampling failed: %s" % e, file=sys.stderr)
        sys.exit(1)

    jobs, extra = build_jobs(args, tests, levels, mpr_points, results_root)

    # A dependency-only Test 1 UE job writes into test1/ue/, which
    # auto-clear (scoped to selected tests) would not clear. Clear
    # just that method dir: it is about to be rerun.
    if "t1-ue" in extra:
        _dep_out = jobs["t1-ue"]["out_dir"]
        if os.path.isdir(_dep_out):
            shutil.rmtree(_dep_out)
            log("Auto-clear: removed dependency dir %s" % _dep_out)

    # Auto-clear: delete + recreate ONLY the results dirs of the tests
    # about to run (Test 3: only the selected levels), so stale files
    # from a partial/failed run can never survive into this run's
    # tables. Placed AFTER the Test 4 prereq fold on purpose: clearing
    # before the fold would make test1_results_exist() false and turn
    # a bare --tests 4 analysis run into a full rerun.
    auto_clear_results(results_root, tests, levels)

    # Run manifest (start): rewritten at run end with outcomes.
    _run_started = time.time()
    _run_cmdline = (" ".join(sys.argv) if argv is None
                    else " ".join([sys.argv[0]] + list(argv)))
    write_run_manifest(results_root, args, tests, data_dir, "started",
                       None, None, _run_started, None, _run_cmdline,
                       pool_file, levels, sample_counts)

    # Fail fast if a sweep job's scripts are missing next to this file.
    for key, job in jobs.items():
        if job["collect"] == "sweep":
            missing = []
            for src_dir, fn in job["scripts"]:
                src = (os.path.join(HERE, src_dir, fn) if src_dir
                       else os.path.join(HERE, fn))
                if not os.path.isfile(src):
                    missing.append(src)
            if missing:
                print("ERROR: %s is missing required script(s): %s"
                      % (key, ", ".join(missing)), file=sys.stderr)
                sys.exit(1)

    if args.max_workers is not None and args.max_workers < 1:
        print("ERROR: --max-workers must be >= 1", file=sys.stderr)
        sys.exit(2)

    # Tracker keys: one per job, plus the Test 2 aggregation and
    # Test 4 pseudo-jobs when selected.
    keys = list(jobs.keys())
    if 2 in tests:
        keys.append("t2-agg")
        DISPLAY_LABELS["t2-agg"] = "T2 (aggregate)"
    if 4 in tests:
        keys.append("t4")
        DISPLAY_LABELS["t4"] = "T4 (volume est.)"

    # Live progress display unless disabled or stdout is not a terminal.
    use_progress = (not args.no_progress) and sys.stdout.isatty()
    tracker = ProgressTracker(keys) if use_progress else None
    stop_event = threading.Event()
    display_thread = None

    log("Engine: %s" % args.engine)
    log("Tests: %s (jobs: %s)" % (tests, sorted(jobs.keys())))
    log("Seed: %d (all jobs)" % args.seed)
    log("Pool: %s" % pool_file)
    log("Levels: %s" % sorted(samples.keys()))
    log("Data dir: %s (read-only)" % data_dir)
    log("Results dir: %s" % results_root)
    log("Temp root: %s" % temp_root)
    if use_progress:
        log("Live progress display enabled (refresh every 2s)")
        display_thread = threading.Thread(
            target=display_loop, args=(tracker, keys, stop_event),
            daemon=True)
        display_thread.start()
    elif not args.no_progress:
        log("Progress display disabled (stdout is not a terminal)")

    # ---- Phase A setup (fail-fast) ----
    # Set up sandboxes for every job that does not wait on a UE job.
    # UE-guidance sweeps are set up later, when their ue_routes.csv
    # is ready (see _maybe_start_sweep).
    sandboxes = {}
    pending = {}  # job key -> job dict, waiting on its UE dependency
    try:
        for key, job in jobs.items():
            if job.get("depends_on"):
                pending[key] = job
                continue
            sb = setup_sandbox(job["sandbox_name"], data_dir, temp_root,
                               job["scripts"],
                               sample_csv=samples[job["level"]],
                               sample_weights_csv=sample_weights[job["level"]])
            if job.get("ue_routes_src"):
                shutil.copy2(job["ue_routes_src"],
                             os.path.join(sb, "ue_routes.csv"))
                log("%s: using pre-resolved ue_routes.csv" % key)
            sandboxes[key] = sb
    except Exception as e:
        print("ERROR: sandbox setup failed: %s" % e, file=sys.stderr)
        if display_thread:
            stop_event.set()
            display_thread.join(timeout=5)
        sys.exit(1)

    # ---- Phase A run (all independent jobs in parallel) ----
    t_start = time.time()
    outcomes = {key: False for key in jobs}
    test4_done = False
    test4_ok = False
    agg_status = "not-run"
    n_jobs = len(sandboxes)
    max_workers = args.max_workers or min(4, max(1, n_jobs))

    if n_jobs or pending:
        _PHASE_A_TIMEOUT_S = 18 * 3600
        _POLL_INTERVAL_S = 10.0

        def _collect_now(key):
            """Collect results immediately for a finished job.

            Returns True on success; False if collection itself failed
            (which marks the job as failed). Never hangs: pure file
            copies with per-call exception handling.
            """
            job = jobs[key]
            sandbox = sandboxes[key]
            try:
                if job["collect"] == "proposed":
                    collect_proposed(job, sandbox, results_root, args.seed)
                elif job["collect"] == "scenario":
                    collect_scenario(job, sandbox, results_root, args.seed)
                else:  # sweep
                    collect_sweep(job, job["sub"], sandbox, results_root,
                                  args.seed)
            except Exception as e:
                print("ERROR: result collection failed for %s: %s"
                      % (key, e), file=sys.stderr)
                return False
            if not args.keep_temp:
                try:
                    shutil.rmtree(sandbox)
                except Exception as e:
                    log("WARNING: could not remove sandbox %s: %s"
                        % (sandbox, e))
                else:
                    log("%s: sandbox removed" % job["label"])
            else:
                log("%s: sandbox kept (--keep-temp): %s"
                    % (job["label"], sandbox))
            return True

        def _maybe_start_sweep(key):
            """Start a pending UE-guidance sweep whose dependency UE job
            has been collected. Returns True if the sweep was submitted
            (or failed loudly and was marked as such)."""
            job = pending.pop(key, None)
            if job is None:
                return False
            dep_key = job["depends_on"]
            if not outcomes.get(dep_key):
                log("%s: dependency %s failed -- sweep cannot run"
                    % (key, dep_key))
                outcomes[key] = False
                if tracker:
                    tracker.finish(key, False)
                return True
            dep_job = jobs[dep_key]
            ue_routes = os.path.join(dep_job["out_dir"], "ue_routes.csv")
            if not os.path.isfile(ue_routes):
                log("%s: dependency %s succeeded but %s is missing -- "
                    "sweep cannot run" % (key, dep_key, ue_routes))
                outcomes[key] = False
                if tracker:
                    tracker.finish(key, False)
                return True
            try:
                sb = setup_sandbox(job["sandbox_name"], data_dir,
                                   temp_root, job["scripts"],
                                   sample_csv=samples[job["level"]],
                                   sample_weights_csv=sample_weights[job["level"]])
            except Exception as e:
                print("ERROR: sandbox setup failed for %s: %s"
                      % (key, e), file=sys.stderr)
                outcomes[key] = False
                if tracker:
                    tracker.finish(key, False)
                return True
            sandboxes[key] = sb
            shutil.copy2(ue_routes, os.path.join(sb, "ue_routes.csv"))
            log("%s: starting -- ue_routes.csv ready" % key)
            fut = ex.submit(run_job, job, sb, tracker, args)
            futures[fut] = key
            return True

        def _maybe_run_test4():
            """Run Test 4 as soon as Test 1's four jobs are collected,
            without waiting for the sweeps to finish."""
            nonlocal test4_done, test4_ok
            if test4_done or 4 not in tests:
                return
            t1_keys = [k for k in jobs if jobs[k]["test"] == 1
                       and k not in extra]
            if not t1_keys:
                return
            if not all(outcomes.get(k) for k in t1_keys):
                return
            test4_done = True
            try:
                ok4, _dt4 = run_test4(results_root, args.seed, tracker)
                test4_ok = ok4
            except RuntimeError as e:
                log("Test 4: %s" % e)
                test4_ok = False
            except Exception as e:  # pragma: no cover - defensive
                log("Test 4: unexpected error: %s" % e)
                test4_ok = False

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as ex:
            futures = {}
            for key in list(sandboxes.keys()):
                fut = ex.submit(run_job, jobs[key], sandboxes[key],
                                tracker, args)
                futures[fut] = key

            # Polling loop (replaces as_completed). as_completed() blocks
            # until a future completes; if a worker thread never releases
            # its future (e.g. a stuck pipe reader), it hangs forever.
            # Polling fut.done() with a hard deadline guarantees forward
            # progress no matter what the workers do.
            _deadline = time.monotonic() + _PHASE_A_TIMEOUT_S
            _processed_keys = set()
            while True:
                # Process every completed-but-unprocessed future. New
                # futures (UE sweeps via _maybe_start_sweep) may be added
                # to the dict while processing; list() snapshots the dict
                # so the mutation is safe, and the new future is picked
                # up on the next pass.
                for fut, key in list(futures.items()):
                    if key in _processed_keys:
                        continue
                    if not fut.done():
                        continue
                    _processed_keys.add(key)
                    try:
                        # fut is done, so result() returns immediately;
                        # the timeout is a safety net only.
                        ok, dt = fut.result(timeout=120)
                    except Exception as e:  # pragma: no cover - defensive
                        log("job %s: unexpected runner error: %s" % (key, e))
                        if tracker:
                            tracker.finish(key, False)
                        ok, dt = False, 0.0
                    if ok:
                        # Collect immediately; a collection failure marks
                        # the job as failed (never hangs: file copies).
                        if not _collect_now(key):
                            ok = False
                        elif tracker:
                            tracker.update(key, None, "done (collected)")
                    else:
                        sb = sandboxes.get(key, "?")
                        print("%s FAILED -- sandbox kept for debugging: %s"
                              % (key, sb))
                    outcomes[key] = ok
                    # A collected UE job unblocks its sweep(s).
                    for pkey in list(pending.keys()):
                        if jobs[pkey].get("depends_on") == key:
                            _maybe_start_sweep(pkey)
                    # Test 4 triggers as soon as Test 1 is collected,
                    # without waiting for the sweeps to finish.
                    _maybe_run_test4()

                # Done when every future currently in the dict has been
                # processed. If _maybe_start_sweep just added a sweep,
                # it is not in _processed_keys, so we loop again.
                _all_keys = set(futures.values())
                if _all_keys and _all_keys <= _processed_keys \
                        and not pending:
                    break

                # Hard deadline: never hang silently at 100%.
                if time.monotonic() > _deadline:
                    _stuck = sorted(
                        str(key) for key in futures.values()
                        if key not in _processed_keys)
                    log("ERROR: Phase A timed out after %s; stuck jobs: %s"
                        % (fmt_min(_PHASE_A_TIMEOUT_S), _stuck))
                    print("ERROR: Phase A timed out; stuck jobs: %s"
                          % _stuck, file=sys.stderr)
                    for key in futures.values():
                        if key in _processed_keys:
                            continue
                        if tracker:
                            tracker.finish(key, False)
                        outcomes[key] = False
                    # Don't let the `with` block's shutdown(wait=True)
                    # hang on stuck workers.
                    ex.shutdown(wait=False, cancel_futures=True)
                    break

                time.sleep(_POLL_INTERVAL_S)
    else:
        log("Phase A: no jobs to run")

    # Any pending sweep whose dependency never completed (e.g. the
    # dependency job itself failed) is marked failed here if
    # _maybe_start_sweep was never reached for it.
    for pkey in list(pending.keys()):
        if not outcomes.get(pkey):
            log("%s: dependency %s did not complete -- sweep cannot run"
                % (pkey, jobs[pkey].get("depends_on")))
            outcomes[pkey] = False
            if tracker:
                tracker.finish(pkey, False)

    # Test 4 fallback: if it wasn't triggered inside the loop (e.g. Test 1
    # was not selected and existing results are used), try it now.
    if 4 in tests and not test4_done:
        try:
            ok4, _dt4 = run_test4(results_root, args.seed, tracker)
            test4_ok = ok4
        except RuntimeError as e:
            log("Test 4: %s" % e)
            test4_ok = False
        except Exception as e:  # pragma: no cover - defensive
            log("Test 4: unexpected error: %s" % e)
            test4_ok = False
        test4_done = True

    # ---- Test 2 aggregation (barrier on collected sweep OUTPUTS) ----
    # Filesystem-based: verifies the per-bigloop dirs actually exist under
    # results/test2/, instead of trusting the in-memory outcomes dict.
    if 2 in tests:
        if args.skip_aggregation:
            log("Test 2: aggregation skipped (--skip-aggregation)")
            agg_status = "skipped"
            if tracker:
                tracker.finish("t2-agg", True, "skipped")
        else:
            _p_dirs = sorted(glob.glob(os.path.join(
                results_root, "test2", "proposed_sweep", "bigloop_*")))
            _u_dirs = sorted(glob.glob(os.path.join(
                results_root, "test2", "ue_sweep", "bigloop_*")))
            if not _p_dirs:
                print("ERROR: Cannot aggregate: "
                      "results/test2/proposed_sweep/ has no bigloop outputs "
                      "(t2-P sweep failed or was not collected)",
                      file=sys.stderr)
                log("Test 2: aggregation not run "
                    "(proposed_sweep outputs missing)")
                agg_status = "FAILED"
                if tracker:
                    tracker.finish("t2-agg", False)
            elif not _u_dirs:
                print("ERROR: Cannot aggregate: "
                      "results/test2/ue_sweep/ has no bigloop outputs "
                      "(t2-U sweep failed or was not collected)",
                      file=sys.stderr)
                log("Test 2: aggregation not run (ue_sweep outputs missing)")
                agg_status = "FAILED"
                if tracker:
                    tracker.finish("t2-agg", False)
            else:
                log("Test 2: aggregating %d proposed + %d UE bigloops ..."
                    % (len(_p_dirs), len(_u_dirs)))
                try:
                    ok_agg, _ = run_aggregation(results_root, args.seed,
                                                tracker)
                    agg_status = "ok" if ok_agg else "FAILED"
                except RuntimeError as e:
                    print("ERROR: %s" % e, file=sys.stderr)
                    agg_status = "FAILED"
                    if tracker:
                        tracker.finish("t2-agg", False)

    # ---- Summary (display stays live until here) ----
    if display_thread:
        stop_event.set()
        display_thread.join(timeout=10)

    # Per-test outcomes for the manifest and the exit code. A test is OK
    # only when all of its jobs succeeded (Test 2 additionally needs the
    # aggregation to be ok or skipped; Test 4 is the run_test4 outcome).
    test_outcomes = {}
    for t in tests:
        if t == 2:
            test_outcomes[2] = bool(outcomes.get("t2-P")
                                    and outcomes.get("t2-U")
                                    and agg_status in ("ok", "skipped"))
        elif t == 4:
            test_outcomes[4] = test4_ok
        else:
            tkeys = [k for k in jobs
                     if jobs[k]["test"] == t and k not in extra]
            test_outcomes[t] = bool(tkeys) and all(
                outcomes.get(k) for k in tkeys)
    # Manifest wants the sweep outcomes under their job keys too.
    manifest_outcomes = dict(test_outcomes)
    manifest_outcomes["t2-P"] = outcomes.get("t2-P", False)
    manifest_outcomes["t2-U"] = outcomes.get("t2-U", False)

    print_summary(results_root, tests, manifest_outcomes, agg_status)
    log("All done (%s total)" % fmt_min(time.time() - t_start))

    failed = [t for t in tests if not test_outcomes.get(t)]
    _run_status = "failed" if failed else "completed"
    write_run_manifest(results_root, args, tests, data_dir, _run_status,
                       manifest_outcomes, agg_status,
                       _run_started, time.time(), _run_cmdline,
                       pool_file, levels, sample_counts)
    if failed:
        print("FAILED tests: %s" % failed, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
