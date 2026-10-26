#!/usr/bin/env python3
"""Temporary helper: run every test in its OWN pytest process, time it, and
write the results sorted by wall-clock runtime to a text file.

Usage:
    .venv/bin/python dev/time_tests.py [-j JOBS] [-o OUTFILE] [pytest path args...]

Examples:
    .venv/bin/python dev/time_tests.py                       # whole suite
    .venv/bin/python dev/time_tests.py tests/ui -o /tmp/ui.txt
    .venv/bin/python dev/time_tests.py -j 4                  # 4 parallel workers

Notes:
- Startup overhead (interpreter + torch import + fixtures) is measured with a
  no-op baseline test and reported separately; "test_time" is the measured
  runtime minus that baseline (floored at 0).
- Results are appended to a scratch CSV as they come in, so a partial run can
  still be inspected / resumed (already-measured node ids are skipped).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTEST = [str(ROOT / ".venv/bin/python"), "-m", "pytest"]
BASE_ARGS = ["-p", "no:cacheprovider", "--no-header", "-q", "--tb=no"]
# extra args for profiling plugins, e.g. EXTRA_PYTEST_ARGS="-p prof_plugin"
BASE_ARGS += [a for a in os.environ.get("EXTRA_PYTEST_ARGS", "").split() if a]
COLLECT_RE = re.compile(r"^[^\s]+::[^\s\[\]]+(?:\[[^\]]*\])?$")


def run_pytest(args: list[str], timeout: int = 900) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        PYTEST + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, time.perf_counter() - t0, proc.stdout + proc.stderr


def collect(paths: list[str]) -> list[str]:
    code, _elapsed, out = run_pytest(["--collect-only", "-q"] + paths, timeout=600)
    nodes = [line.strip() for line in out.splitlines() if COLLECT_RE.match(line.strip())]
    if not nodes:
        sys.exit(f"no tests collected:\n{out[-2000:]}")
    return nodes


def measure(node: str) -> tuple[str, float, int]:
    try:
        code, elapsed, _out = run_pytest(BASE_ARGS + [node])
    except subprocess.TimeoutExpired:
        return node, 900.0, -1
    return node, elapsed, code


def baseline(jobs: int) -> float:
    """Runtime of a trivial passing test == pure per-process overhead."""
    tmp = ROOT / "dev" / "timing_baseline_tmp_test.py"
    tmp.write_text("def test_noop():\n    assert True\n")
    try:
        runs = [
            r[1]
            for r in (measure("dev/timing_baseline_tmp_test.py::test_noop") for _ in range(3))
            if r[2] == 0
        ]
    finally:
        tmp.unlink(missing_ok=True)
    if not runs:
        return 0.0
    return sorted(runs)[len(runs) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-o", "--out", default="/tmp/test_timings.txt")
    ap.add_argument("--csv", default="/tmp/test_timings.csv")
    ap.add_argument("paths", nargs="*", default=["tests"])
    args = ap.parse_args()

    paths = args.paths or ["tests"]
    nodes = collect(paths)
    done = {}
    csvpath = Path(args.csv)
    if csvpath.exists():
        with csvpath.open() as fh:
            for row in csv.DictReader(fh):
                done[row["node"]] = (float(row["seconds"]), int(row["exit"]))
    todo = [n for n in nodes if n not in done]
    print(f"collected {len(nodes)} tests ({len(done)} already measured)", flush=True)

    base = baseline(args.jobs)
    print(f"per-run overhead baseline: {base:.2f}s", flush=True)

    write_header = not csvpath.exists() or csvpath.stat().st_size == 0
    new = csvpath.open("a", newline="")
    w = csv.writer(new)
    if write_header:
        w.writerow(["node", "seconds", "exit"])

    started = time.perf_counter()
    count = 0

    def report(node: str, secs: float, code: int) -> None:
        nonlocal count
        done[node] = (secs, code)
        w.writerow([node, f"{secs:.3f}", code])
        new.flush()
        count += 1
        if count % 25 == 0 or count == len(todo):
            print(
                f"  {count}/{len(todo)}  elapsed {time.perf_counter() - started:.0f}s"
                f"  last: {node} {secs:.2f}s",
                flush=True,
            )

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for node, secs, code in ex.map(lambda n: measure(n), todo):
                report(node, secs, code)
    else:
        for node in todo:
            report(*measure(node))
    new.close()

    rows = sorted(done.items(), key=lambda kv: kv[1][0], reverse=True)
    total = sum(s for s, _ in done.values())
    lines = [
        "per-test wall-clock runtime (each test in its own pytest process)",
        f"measured {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"tests: {len(rows)}   jobs: {args.jobs}   sum of runtimes: {total:.1f}s",
        f"bare pytest-process startup baseline: {base:.2f}s "
        f"(no test module imported; 'net' = seconds - baseline)",
        "",
        f"{'seconds':>9}  {'net':>8}  {'exit':>4}  node",
        "-" * 78,
    ]
    for node, (secs, code) in rows:
        lines.append(f"{secs:9.2f}  {max(0.0, secs - base):8.2f}  {code:>4}  {node}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} (raw csv: {args.csv})", flush=True)


if __name__ == "__main__":
    main()
