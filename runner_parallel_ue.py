"""
runner_parallel_ue.py
=====================

UE-guidance variant of ``runner_parallel.py``: runs all MSA iterations
concurrently, one sandbox (``RunX/``) per iteration, with guided members
receiving their routes from a UE reference solution (``ue_routes.csv``)
instead of the tournament-selected UO route.

Usage
-----
    python runner_parallel_ue.py --ue-routes ue_routes.csv   # all 11, auto workers
    python runner_parallel_ue.py --ue-routes ue_routes.csv --bigloops 10
    python runner_parallel_ue.py --ue-routes ue_routes.csv --workers 4 --seed 123

``ue_routes.csv`` must be generated from the SAME demand file as
``DTALite_Files/input_agent_initial.csv`` (36k vs 54k alignment is by
``agent_id``); see ``msa_ue.msa_ue``.

Each iteration's result files are copied back to ``DTALite_Files/`` and every
``RunX/`` folder is deleted afterwards (failed runs keep their sandbox for
inspection).

The ``if __name__ == "__main__":`` guard is required on Windows, where
multiprocessing uses the *spawn* start method.
"""

import argparse
import os
import sys
import threading
import time

from parallel_msa_ue import run_parallel_msa_ue, runner_schedule


def parse_bigloops(spec):
    """Parse a bigloop selection spec into a sorted list of ints.

    Accepts comma-separated values and ranges, e.g.:
      "10"     -> [10]
      "0-10"   -> [0, 1, ..., 10]
      "5,10"   -> [5, 10]
      "0-3,10" -> [0, 1, 2, 3, 10]
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if lo > hi:
                raise ValueError("invalid range %r in --bigloops" % part)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    seen = sorted(set(out))
    if not seen:
        raise ValueError("empty --bigloops selection")
    return seen



def mpr_points_schedule(points):
    """Build a (bigloop, p, p_fix, p_realtime) schedule for arbitrary MPR
    points (percentages, e.g. [33, 66]).

    The bigloop index IS the MPR point, so result files are prefixed with
    it (33UOinfor_agent.csv, ...) and stay self-describing. The p
    schedule is the existing one: point 0 is special (p=0.001,
    p_fix=0.6, p_realtime=0.4, absolute shares); any other point uses
    p=point/100 with the non-member share split 60/40 fix/realtime.
    """
    schedule = []
    for point in points:
        if point < 0 or point > 100:
            raise ValueError(
                "--mpr-points value %d out of range 0-100" % point)
        if point == 0:
            schedule.append((0, 0.001, 0.6, 0.4))
        else:
            p = point / 100.0
            schedule.append((point, p, (1 - p) * 0.6, (1 - p) * 0.4))
    return schedule


def parse_mpr_points(spec):
    """Parse an MPR-points spec into a sorted list of ints.

    Accepts comma-separated percentages, e.g. "33,66" -> [33, 66].
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    seen = sorted(set(out))
    if not seen:
        raise ValueError("empty --mpr-points selection")
    return seen

def main():
    parser = argparse.ArgumentParser(
        description="Run all MSA iterations in parallel sandboxes, "
                    "UE-guidance mode (members get UE routes from "
                    "--ue-routes instead of the tournament-selected "
                    "UO route)."
    )
    parser.add_argument(
        "--iterations", type=int, default=11,
        help="Number of iterations (bigloop = 0 .. iterations-1). Default: 11.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes. Default: min(iterations, cpu_count).",
    )
    parser.add_argument(
        "--no-keep-on-failure", action="store_true",
        help="Delete RunX sandboxes even when a run fails.",
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="RNG seed for dividerand (user-type assignment). Default: 1.",
    )
    parser.add_argument(
        "--ue-routes", required=True,
        help="Path to ue_routes.csv (agent_id, path_node_sequence). "
             "UE-guidance mode: guided members receive their UE route from "
             "this file instead of the tournament-selected UO route. The "
             "CSV must be generated from the SAME demand file as "
             "DTALite_Files/input_agent_initial.csv (36k vs 54k alignment "
             "by agent_id).",
    )
    parser.add_argument(
        "--bigloops", default="0-10",
        help="Bigloops to run: e.g. '10' (Test 1, 100%% MPR only), "
             "'0-10' (full MPR sweep, Test 2), '5,10'. Default: '0-10'. "
             "Cannot be combined with --mpr-points.",
    )
    parser.add_argument(
        "--mpr-points", default=None,
        help="MPR points in percent to run instead of --bigloops, e.g. "
             "'33,66' (Test 3 demand-level sweep). The bigloop index is "
             "the MPR point itself, so result files are prefixed 33/66. "
             "Point 0 uses the special p=0.001 schedule; other points "
             "use p=point/100 with the non-member share split 60/40 "
             "fix/realtime. Cannot be combined with --bigloops.",
    )
    parser.add_argument(
        "--network", default="siouxfalls", choices=["siouxfalls", "baltimore"],
        help="Network: siouxfalls (default) or baltimore.",
    )
    parser.add_argument(
        "--engine", default="pysim", choices=["pysim", "dtalite"],
        help="Simulation engine: pysim (default) or dtalite.",
    )
    args = parser.parse_args()

    # Run from the repo root so relative paths resolve, wherever the script
    # is launched from.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    if args.mpr_points is not None:
        if any(t == "--bigloops" or t.startswith("--bigloops=")
               for t in sys.argv[1:]):
            print("ERROR: --mpr-points cannot be combined with --bigloops",
                  file=sys.stderr)
            sys.exit(2)
        try:
            points = parse_mpr_points(args.mpr_points)
            schedule = mpr_points_schedule(points)
        except ValueError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            sys.exit(2)
        print(f"Launching {len(schedule)} UE-guidance MSA iterations in parallel "
              f"(mpr_points={[s[0] for s in schedule]}, "
              f"workers={args.workers or 'auto'}, seed={args.seed}, "
              f"ue_routes={args.ue_routes}).", flush=True)
    else:
        try:
            wanted = parse_bigloops(args.bigloops)
        except ValueError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            sys.exit(2)
        full_schedule = runner_schedule(args.iterations)
        schedule = [s for s in full_schedule if s[0] in wanted]
        missing = [b for b in wanted if b not in
                   {s[0] for s in full_schedule}]
        if missing:
            print("ERROR: --bigloops selected %s, but --iterations %d only "
                  "defines bigloops 0..%d" % (missing, args.iterations,
                                              args.iterations - 1),
                  file=sys.stderr)
            sys.exit(2)
        print(f"Launching {len(schedule)} UE-guidance MSA iterations in parallel "
              f"(bigloops={[s[0] for s in schedule]}, "
              f"workers={args.workers or 'auto'}, seed={args.seed}, "
              f"ue_routes={args.ue_routes}).", flush=True)

    start = time.time()
    results = run_parallel_msa_ue(
        schedule,
        ue_routes_path=args.ue_routes,
        max_workers=args.workers,
        repo_root=repo_root,
        keep_on_failure=not args.no_keep_on_failure,
        seed=args.seed,
        network=args.network,
        engine=args.engine,
    )
    elapsed = time.time() - start

    # Watchdog: work is done; if we haven't exited in 60s something is
    # stuck (e.g. atexit handlers, lingering threads). Force-exit.
    def _watchdog():
        time.sleep(60)
        try:
            sys.stderr.write(
                "\nWATCHDOG: main did not exit 60s after work completed; "
                "forcing os._exit(1)\n")
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(1)
    _wd = threading.Thread(target=_watchdog, daemon=True)
    _wd.start()

    failures = getattr(results, "failures", {})
    if failures and not results:
        print(f"ERROR: all {len(failures)} iterations failed: "
              f"{sorted(failures)}", file=sys.stderr, flush=True)
        for bl, err in sorted(failures.items()):
            print(f"  bigloop {bl}: {err}", file=sys.stderr, flush=True)
        return 1
    if failures:
        # Partial success: the succeeded bigloops' result files were
        # harvested to DTALite_Files/ by their workers and all sandboxes
        # were kept. Exit 0 so wholerunner collects the partial sweep;
        # the missing bigloops are loudly reported here and noted by
        # mpr_aggregate.py in mpr_summary.txt.
        print(f"WARNING: sweep partial: {len(results)} succeeded, "
              f"{len(failures)} failed (bigloops={sorted(failures)}).",
              file=sys.stderr, flush=True)
        for bl, err in sorted(failures.items()):
            print(f"  bigloop {bl}: {err}", file=sys.stderr, flush=True)
    print(f"All {len(results)} iterations finished in {elapsed:.1f}s.", flush=True)
    for r in sorted(results, key=lambda d: d["bigloop"]):
        print(f"  bigloop={r['bigloop']}: "
              f"{len(r['results'])} result files -> DTALite_Files/")
    return 0


if __name__ == "__main__":
    # os._exit (not sys.exit): after main() returns, all results are
    # collected and sandboxes cleaned up. sys.exit() would run the
    # concurrent.futures atexit handler, which re-joins the executor's
    # management thread with wait=True -- hanging forever if any worker
    # was killed or is stuck. os._exit skips atexit entirely.
    try:
        _rc = main()
    except SystemExit as _e:
        _rc = _e.code
    except BaseException:
        import traceback
        traceback.print_exc()
        _rc = 1
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(_rc if isinstance(_rc, int) else (0 if _rc is None else 1))
