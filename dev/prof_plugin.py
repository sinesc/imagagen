"""Temporary pytest plugin: find where the time goes in the UI tests.

Run it through dev/time_tests.py or directly:

    PYTHONPATH=dev .venv/bin/python -m pytest -p prof_plugin -s <node id>
    EXTRA_PYTEST_ARGS="-p prof_plugin" .venv/bin/python dev/time_tests.py tests/ui

Env: PROF_REPORT (default /tmp/prof_report.txt), PROF_SLOW (default 0.5,
the wait_until threshold in seconds), PW_SLOW (log individual Playwright
calls above this, 0 = off).

What it reports (appended to PROF_REPORT):
- `TIMEOUT`  a wait_until that ran out its full (10 s default) timeout --
  the return value is usually discarded, so such a check silently never
  ran AND burned 10 s of wall clock. These are bugs, not slow tests.
- `slow`     a wait_until that took PROF_SLOW or longer (typically a
  server-state assertion waiting for canvas.js's 2 s SYNC_IDLE_MS timer --
  use conftest sync()/flush() instead).
- `sleep`    any time.sleep >= 0.15 s in a test module.
- `pw-totals` the six slowest Playwright call kinds per test (tells you
  whether the time is in the browser round trips or in waiting).
"""

import inspect
import os
import sys
import time
import types

REPORT = os.environ.get("PROF_REPORT", "/tmp/prof_report.txt")
SLOW = float(os.environ.get("PROF_SLOW", "0.5"))
PW_SLOW = float(os.environ.get("PW_SLOW", "0") or 0)
_PW_TOTALS = {}


def _log(line):
    with open(REPORT, "a") as fh:
        fh.write(line + "\n")


def _callsite(depth=2):
    """The innermost frame that lives in a tests/ui file."""
    frame = sys._getframe(depth)
    info = inspect.getframeinfo(frame)
    caller = frame.f_back
    while caller is not None:
        cand = inspect.getframeinfo(caller)
        if "/tests/" in cand.filename:
            info = cand
            break
        caller = caller.f_back
    src = info.code_context[0].strip() if info.code_context else ""
    return f"{os.path.basename(info.filename)}:{info.lineno} {src}"


def _wrap_wait(mod):
    orig = mod.wait_until
    if getattr(orig, "_prof", False):
        return

    def wrapper(fn, timeout=10.0, interval=0.1):
        t0 = time.time()
        result = orig(fn, timeout, interval)
        dt = time.time() - t0
        if dt >= SLOW or not result:
            tag = "TIMEOUT" if not result else "slow"
            _log(f"{tag}  {dt:5.2f}s  {_callsite()}")
            if not result:
                _log(f"        lambda: {inspect.getsource(fn).strip()}")
        return result

    wrapper._prof = True
    mod.wait_until = wrapper


_real_sleep = time.sleep


def _sleep_probe(seconds, *args, **kw):
    if seconds >= 0.15:
        _log(f"sleep {seconds}s  {_callsite()}")
    return _real_sleep(seconds, *args, **kw)


_PW_METHODS = (
    "click fill press evaluate evaluate_all wait_for_function goto screenshot "
    "locator hover check select_option wait_for_selector query_selector_all "
    "eval_on_selector_all input_value text_content count reload"
).split()
_PW_CLASSES = ("Page", "Frame", "Locator", "Mouse", "Keyboard", "JSHandle", "ElementHandle")


def _patch_playwright():
    try:
        import playwright.sync_api as pw
    except Exception:
        return
    for cls_name in _PW_CLASSES:
        cls = getattr(pw, cls_name, None)
        if cls is None:
            continue
        for name in _PW_METHODS:
            fn = getattr(cls, name, None)
            if not callable(fn) or getattr(fn, "_pw_prof", False):
                continue

            def make(fn=fn, name=name, cls_name=cls_name):
                def wrapper(*a, **k):
                    t0 = time.perf_counter()
                    try:
                        return fn(*a, **k)
                    finally:
                        dt = time.perf_counter() - t0
                        key = f"{cls_name}.{name}"
                        _PW_TOTALS[key] = _PW_TOTALS.get(key, 0.0) + dt
                        if PW_SLOW and dt >= PW_SLOW:
                            _log(f"pwslow {dt:5.2f}s {key}  {_callsite(3)}")

                wrapper._pw_prof = True
                return wrapper

            setattr(cls, name, make())


def pytest_collection_finish(session):
    _patch_playwright()
    for name, mod in list(sys.modules.items()):
        if (name == "conftest" or name.startswith("test_")) and hasattr(mod, "wait_until"):
            _wrap_wait(mod)
        if name.startswith("test_") and getattr(mod, "time", None) is time:
            shim = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time)})
            shim.sleep = _sleep_probe
            mod.time = shim


def pytest_runtest_logreport(report):
    if report.when == "call" and _PW_TOTALS:
        top = sorted(_PW_TOTALS.items(), key=lambda kv: -kv[1])[:6]
        _log(
            f"pw-totals {report.nodeid.split('::')[-1]}: "
            + ", ".join(f"{k}={v:.2f}s" for k, v in top)
        )
