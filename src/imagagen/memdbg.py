"""Memory-leak debug instrumentation (imagagen.ini ``memory-debug``).

Two halves, both no-ops unless :func:`configure` enabled them:

* **event stamps** -- one short ``MEMDBG evt=...`` line at every point that
  is known to move multi-GiB model data or hold references to it (model
  load / park / restore, prompt prefetch, the parking timer, job lifecycle).
  Each line carries the current process RSS/VmSwap so a leak shows up as a
  monotonic drift *attributed to an event kind*. Reading ``/proc/self/status``
  is a few microseconds, so this is safe to leave on during a real session.

* **censuses** -- an expensive (sub-second to seconds) walk of
  ``gc.get_objects()`` that splits all live memory into "model weights
  owned by a backend instance", "other CPU tensors", "CUDA tensors",
  "numpy arrays" and prints the biggest shape groups with a referrer label
  for one representative each. That answers WHERE the RAM went and (via the
  per-backend footprint + a ``weakref`` created/freed log) whether a whole
  model copy is being retained instead of freed.

The sampler thread prints a cheap line every ``interval`` seconds and a full
census when RSS jumped since the last one (or every ``census_every`` samples).
``GET /api/memdbg`` runs one on demand -- use it at the moment the system
monitor shows RAM nearly full.

When enabled, everything also goes to ``user/memdbg.log`` so the report can
be handed over without scraping a terminal.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
import weakref
from collections import Counter
from pathlib import Path

#: every line starts with this so the log can be grepped/filtered trivially
PREFIX = "MEMDBG"

_state = {
    "enabled": False,
    "interval": 30.0,
    "census_every": 6,
    "census_trigger_gib": 4.0,
    "log_path": None,
    "fh": None,
}
_fh_lock = threading.Lock()
_counts: Counter[str] = Counter()
_lock = threading.Lock()  # one census at a time
_sampler_stop: threading.Event | None = None
_last: dict = {}  # sampler bookkeeping (last rss, census rss)
_refs: dict[str, weakref.ref] = {}
_ref_seq = 0
_T0 = time.monotonic()
logger = None  # set by configure()


# --------------------------------------------------------------- control ----


def configure(
    enabled: bool,
    interval: float = 30.0,
    census_every: int = 6,
    census_trigger_gib: float = 4.0,
    log_path: Path | str | None = None,
) -> None:
    """Enable/disable the instrumentation (imagagen.ini memory-debug).

    ``log_path`` additionally mirrors every line into a file (the terminal
    keeps them too, the app's own logging is untouched otherwise).
    """
    global _sampler_stop, logger
    logger = logging.getLogger("imagagen.memdbg")
    if logger.level is logging.NOTSET or logger.level > logging.INFO:
        # independent of how the app configures logging: the diagnostics
        # must reach the file even in scripts/tests without log setup
        logger.setLevel(logging.INFO)
    _state["enabled"] = bool(enabled)
    _state["interval"] = max(2.0, float(interval))
    _state["census_every"] = max(1, int(census_every))
    _state["census_trigger_gib"] = float(census_trigger_gib)
    _state["log_path"] = Path(log_path) if log_path else None
    if not _state["enabled"]:
        _stop_sampler()
        _close_fh()
        return
    _close_fh()
    if log_path is not None:
        path = _state["log_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered plain file, deliberately NOT a logging.FileHandler:
        # uvicorn.run applies logging.config.dictConfig (non-incremental),
        # whose _clearExistingHandlers() closes every handler created so
        # far -- a logging handler opened in create_app() is dead
        # (stream = None) by the time the first generation logs anything.
        _state["fh"] = open(path, "w", encoding="utf-8", buffering=1)
    _log(
        f"enabled interval={_state['interval']}s "
        f"census_every={_state['census_every']} "
        f"trigger={_state['census_trigger_gib']}GiB log={_state['log_path'] or '-'}"
    )


def _close_fh() -> None:
    fh = _state.get("fh")
    if fh is not None:
        _state["fh"] = None
        try:
            fh.close()
        except Exception:
            pass


def enabled() -> bool:
    return bool(_state["enabled"])


# ------------------------------------------------------------- primitives ----


def _log(msg: str) -> None:
    if logger is not None:
        logger.info("%s %s", PREFIX, msg)
    fh = _state.get("fh")
    if fh is not None:
        try:
            with _fh_lock:
                fh.write(f"{time.strftime('%H:%M:%S')} {PREFIX} {msg}\n")
        except Exception:
            pass  # diagnostics must never break the app


def proc_kib() -> dict[str, float]:
    """Selected /proc/self/status counters in kiB (missing keys omitted)."""
    out: dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if not line.startswith(("VmRSS:", "VmHWM:", "VmSwap:", "RssAnon:", "VmSize:")):
                    continue
                key, _, rest = line.partition(":")
                out[key] = float(rest.strip().split()[0])
    except OSError:
        pass
    return out


def _host_kib() -> dict[str, float]:
    """MemAvailable / SwapUsed / MemTotal in kiB."""
    want = {
        "MemTotal": "mem_total",
        "MemAvailable": "mem_avail",
        "SwapTotal": "swap_total",
        "SwapFree": "swap_free",
    }
    vals: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in want:
                    vals[want[key]] = float(line.split(":", 1)[1].strip().split()[0])
    except OSError:
        pass
    out = dict(vals)
    if "swap_total" in out and "swap_free" in out:
        out["swap_used"] = out["swap_total"] - out["swap_free"]
    return out


def gib(kib: float) -> str:
    return "-" if kib < 0 else f"{kib / 2**20:.2f}"


def cuda_kib() -> dict[int, tuple[float, float, float, int]]:
    """{device: (allocated, reserved, max_allocated, alloc_retries)} in kiB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        out = {}
        for i in range(torch.cuda.device_count()):
            stats = torch.cuda.memory_stats(i)
            out[i] = (
                torch.cuda.memory_allocated(i) / 2**10,
                torch.cuda.memory_reserved(i) / 2**10,
                torch.cuda.max_memory_allocated(i) / 2**10,
                int(stats.get("num_alloc_retries", 0)),
            )
        return out
    except Exception:
        return {}


# ------------------------------------------------------------------ events ----


def stamp(event: str, **fields) -> None:
    """One cheap ``MEMDBG evt=...`` line with an RSS/swap stamp.

    ``event`` is the operation (``park``, ``load``, ``prefetch``, ...);
    ``fields`` are rendered as ``key=value``. Never raises.
    """
    if not _state["enabled"]:
        return
    try:
        _counts[event] += 1
        st = proc_kib()
        parts = [
            f"evt={event}",
            f"n={_counts[event]}",
            f"t={time.monotonic() - _T0:.0f}s",
            f"rss={gib(st.get('VmRSS', -1))}GiB",
            f"swap={gib(st.get('VmSwap', -1))}GiB",
            f"hwm={gib(st.get('VmHWM', -1))}GiB",
        ]
        for k, v in fields.items():
            if v is not None:
                parts.append(f"{k}={v}")
        cuda = cuda_kib()
        for dev, (alloc, res, _mx, retries) in sorted(cuda.items()):
            parts.append(f"cuda{dev}={alloc / 2**20:.2f}/{res / 2**20:.2f}GiB")
            if retries:
                parts.append(f"cuda{dev}retries={retries}")
        _log(" ".join(parts))
    except Exception:  # pragma: no cover - diagnostics must never break a run
        pass


def track(obj: object, label: str) -> None:
    """Log creation and (hopefully) collection of a big object.

    A whole model copy is ~7 GiB of CPU tensors; if a dropped copy is
    actually retained, its ``freed`` line never arrives and the alive count
    keeps climbing -- that is the difference between "parked copy dropped"
    and "parked copy leaked".
    """
    if not _state["enabled"]:
        return
    global _ref_seq
    try:
        _ref_seq += 1
        key = f"{label}#{_ref_seq}"
        alive = sum(1 for r in _refs.values() if r() is not None) + 1

        def _gone(_dead_ref, k=key, lbl=label) -> None:
            # weakref callbacks are invoked WITH the dead weakref as
            # positional arg -- it must be accepted or the callback
            # raises TypeError before running
            try:
                _refs.pop(k, None)
                still = sum(1 for r in _refs.values() if r() is not None) + 1
                stamp("freed", obj=lbl, key=k, alive=still)
            except Exception:  # interpreter shutdown races here
                pass

        _refs[key] = weakref.ref(obj, _gone)
        stamp("created", obj=label, key=key, alive=alive)
    except Exception as e:  # pragma: no cover
        # e.g. an instance without __weakref__ support: surface it, a
        # silently missing created/freed pair would fake a leak signal
        try:
            stamp("track-failed", obj=label, err=type(e).__name__)
        except Exception:
            pass


# ----------------------------------------------------------------- census ----


def _backend_instances(objects) -> list:
    from .backends import ModelBackend

    return [o for o in objects if isinstance(o, ModelBackend)]


def _storage_of(tensor) -> tuple[int, int] | None:
    """(storage ptr, storage bytes) of a tensor, or None on any failure."""
    try:
        st = tensor.untyped_storage()
        return (st.data_ptr(), st.nbytes())
    except Exception:
        try:
            return (tensor.data_ptr(), tensor.nbytes())
        except Exception:
            return None


def _module_cpu_bytes(obj) -> tuple[float, float, float, int]:
    """Walk ``obj`` (a backend or pipeline) and sum the storages of every
    ``nn.Module`` parameter/buffer it owns.

    Returns ``(cpu_kib, cuda_kib, pinned_kib, n_storages)`` plus -- via the
    mutable ``seen`` sets in ``_CENSUS`` -- the storage pointers that belong
    to a model copy, so the generic tensor walk can call the rest
    "unattributed".
    """
    import torch

    cpu = cuda = pinned = 0.0
    n = 0
    stack = [obj]
    visited: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in visited:
            continue
        visited.add(id(cur))
        if isinstance(cur, torch.nn.Module):
            for holder in (cur._parameters, cur._buffers):
                for t in holder.values():
                    if t is None:
                        continue
                    info = _storage_of(t)
                    if info is None or info[0] in _seen_storages:
                        continue
                    _seen_storages.add(info[0])
                    n += 1
                    if getattr(t, "is_pinned", lambda: False)():
                        pinned += info[1] / 2**10
                    elif t.device.type == "cpu":
                        cpu += info[1] / 2**10
                    else:
                        cuda += info[1] / 2**10
            stack.extend(cur._modules.values())
            continue
        # not a Module: follow the containers a backend/pipeline keeps its
        # submodules in (diffusers pipelines: .components; anything else:
        # the instance dict, including module-valued dicts like .blocks)
        comps = getattr(cur, "components", None)
        if isinstance(comps, dict):
            stack.extend(comps.values())
        for v in list(getattr(cur, "__dict__", {}).values()):
            if isinstance(v, torch.nn.Module):
                stack.append(v)
            elif isinstance(v, dict):
                stack.extend(x for x in v.values() if isinstance(x, torch.nn.Module))
            elif hasattr(v, "components") or hasattr(v, "_parameters"):
                stack.append(v)
    return cpu, cuda, pinned, n


_seen_storages: set[int] = set()


def _owner_label(obj, skip_ids: frozenset[int] = frozenset()) -> str:
    """Best-effort "who holds this" label, one referrer level deep.

    ``skip_ids`` keeps the census's own bookkeeping lists out of the result.
    """
    try:
        labels: list[str] = []
        for r in gc.get_referrers(obj):
            if id(r) in skip_ids:
                continue
            if type(r).__name__ in ("frame",):
                if getattr(r, "f_code", None) is not None and r.f_code.co_filename.endswith(
                    "memdbg.py"
                ):
                    continue  # our own walk
                labels.append("frame")
            elif isinstance(r, dict):
                keys = [k for k, v in r.items() if v is obj][:3]
                labels.append(f"dict[{','.join(map(str, keys))}]")
            elif isinstance(r, (list, tuple)):
                labels.append(f"{type(r).__name__}x{len(r)}")
            else:
                labels.append(type(r).__name__)
            if len(labels) >= 4:
                break
        if not labels:
            return "unreachable"
        return "<-" + ",".join(labels[:4])
    except Exception:
        return "?"


def census(reason: str = "manual") -> str:
    """Full object-graph census: where are all the bytes, and who owns them?

    Logs a multi-line ``MEMDBG census`` report and returns it as text.
    """
    if not _state["enabled"]:
        return "memory-debug is disabled (imagagen.ini: memory-debug = true)"
    with _lock:
        return _census_locked(reason)


def _census_locked(reason: str) -> str:
    lines: list[str] = []
    started = time.perf_counter()
    st = proc_kib()
    host = _host_kib()
    lines.append(
        f"census[{reason}] rss={gib(st.get('VmRSS', -1))}GiB "
        f"anon={gib(st.get('RssAnon', -1))}GiB "
        f"swap={gib(st.get('VmSwap', -1))}GiB hwm={gib(st.get('VmHWM', -1))}GiB "
        f"| host avail={gib(host.get('mem_avail', -1))}GiB "
        f"swap_used={gib(host.get('swap_used', -1))}GiB "
        f"| events={dict(_counts)}"
    )
    if logger is not None:
        hs = [
            f"{type(h).__name__}:{getattr(h, 'baseFilename', getattr(h, 'stream', '?'))}"
            for h in logger.handlers
        ]
        fh = _state.get("fh")
        lines.append(
            f"census memdbg-logger level={logging.getLevelName(logger.level)} "
            f"propagate={logger.propagate} handlers={hs} "
            f"mirror={'open' if fh is not None and not fh.closed else 'none'}"
        )

    objects = gc.get_objects()
    try:
        import torch
    except Exception:
        torch = None
        lines.append("census: torch not importable")

    # 1) model copies: every live backend, and which device its weights are on
    global _seen_storages
    _seen_storages = set()
    backends = _backend_instances(objects) if torch else []
    tot_cpu = tot_cuda = 0.0
    lines.append(
        f"census backends={len(backends)} "
        f"(registry slots below; a copy is ~6.9GiB SDXL fp16) "
        f"tracked_alive={sum(1 for r in _refs.values() if r() is not None)}"
    )
    for b in backends:
        cpu, cuda, pinned, n = _module_cpu_bytes(b)
        tot_cpu += cpu
        tot_cuda += cuda
        name = getattr(b, "name", "?")
        dev = getattr(b, "device", "?")
        lines.append(
            f"census   backend {id(b):#x} {type(b).__name__} model={name} "
            f"dev={dev} weights: cpu={cpu / 2**20:.2f}GiB "
            f"cuda={cuda / 2**20:.2f}GiB pinned={pinned / 2**20:.2f}GiB "
            f"storages={n}"
        )

    # 2) all tensors, split into model weights vs. everything else
    cpu_kib = cpu_n = 0.0
    cuda_tensor_kib = 0.0
    graph_kib = 0.0
    pinned_kib = 0.0
    np_kib = 0.0
    np_n = 0
    np_groups: dict[tuple, list] = {}
    groups: dict[tuple, list] = {}
    unattributed: dict[tuple, list] = {}
    unattr_kib = 0.0
    if torch:
        import warnings as _warnings

        import numpy as np

        seen_t: set[int] = set()
        # isinstance() probes torch lazily (torch.distributed deprecation
        # FutureWarnings on first touch); they would flood the log.
        # Indentation kept: the loop body accumulates the locals below.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for obj in objects:
                if isinstance(obj, torch.Tensor):
                    if obj.grad_fn is not None:
                        graph_kib += obj.nbytes() / 2**10
                    info = _storage_of(obj)
                    if info is None or info[0] in seen_t:
                        continue
                    seen_t.add(info[0])
                    k = info[1] / 2**10
                    if obj.device.type == "cpu":
                        if getattr(obj, "is_pinned", lambda: False)():
                            pinned_kib += k
                        cpu_kib += k
                        cpu_n += 1
                        if info[0] in _seen_storages:
                            continue  # belongs to a live backend
                        key = (str(obj.dtype), tuple(obj.shape))
                        unattr_kib += k
                        ent = unattributed.setdefault(key, [0, 0.0, obj])
                        ent[0] += 1
                        ent[1] += k
                    else:
                        cuda_tensor_kib += k
                        key = (str(obj.dtype), tuple(obj.shape))
                        ent = groups.setdefault(key, [0, 0.0, obj])
                        ent[0] += 1
                        ent[1] += k
                elif isinstance(obj, np.ndarray):
                    np_n += 1
                    np_kib += obj.nbytes / 2**10
                    key = (str(obj.dtype), tuple(obj.shape))
                    ent = np_groups.setdefault(key, [0, 0.0, obj])
                    ent[0] += 1
                    ent[1] += obj.nbytes / 2**10
        lines.append(
            f"census host-cpu-tensors={cpu_n:.0f} {cpu_kib / 2**20:.2f}GiB "
            f"(of which live-backend weights {tot_cpu / 2**20:.2f}GiB, "
            f"UNATTRIBUTED {unattr_kib / 2**20:.2f}GiB) | "
            f"pinned={pinned_kib / 2**20:.2f}GiB | "
            f"cuda-tensors={cuda_tensor_kib / 2**20:.2f}GiB | "
            f"grad-fn-held={graph_kib / 2**20:.2f}GiB | "
            f"numpy arrays={np_n} {np_kib / 2**20:.2f}GiB"
        )
        top = sorted(unattributed.items(), key=lambda kv: -kv[1][1])[:12]
        lines.append(
            "census top UNATTRIBUTED cpu tensor groups (dtype, shape) -> count, GiB, holder:"
        )
        if not top:
            lines.append("census   (none)")
        # our own bookkeeping lists hold the representative tensors, so they
        # must not show up as "owners"
        skip = frozenset(id(v) for v in unattributed.values()) | {id(unattributed)}
        for (dtype, shape), (cnt, k, rep) in top:
            kind = "Parameter" if isinstance(rep, torch.nn.Parameter) else type(rep).__name__
            lines.append(
                f"census   {dtype} {shape} x{cnt:.0f} {k / 2**20:.2f}GiB "
                f"({kind}) {_owner_label(rep, skip)}"
            )
        top = sorted(groups.items(), key=lambda kv: -kv[1][1])[:6]
        if top:
            lines.append("census top cuda tensor groups:")
            for (dtype, shape), (cnt, k, _rep) in top:
                lines.append(f"census   {dtype} {shape} x{cnt:.0f} {k / 2**20:.2f}GiB")
    # numpy (canvas entities, InpaintPlans, PIL buffers, VAE arrays):
    # canvas/img2img leaks live here, so group by shape too
    if np_groups:
        skip = frozenset(id(v) for v in np_groups.values()) | {id(np_groups)}
        lines.append(f"census numpy arrays={np_n} total={np_kib / 2**20:.2f}GiB, top groups:")
        for (dtype, shape), (cnt, k, rep) in sorted(np_groups.items(), key=lambda kv: -kv[1][1])[
            :12
        ]:
            lines.append(
                f"census   {dtype} {shape} x{cnt} {k / 2**20:.2f}GiB "
                f"{type(rep).__name__} {_owner_label(rep, skip)}"
            )

    # 3) live CUDA allocator view
    for dev, (alloc, res, mx, retries) in sorted(cuda_kib().items()):
        lines.append(
            f"census   cuda{dev} allocated={alloc / 2**20:.2f}GiB "
            f"reserved={res / 2**20:.2f}GiB peak={mx / 2**20:.2f}GiB "
            f"alloc_retries={retries}"
        )

    # 4) python-level object counts (growth of tracebacks/frames/arrays is a
    #    classic way to retain multi-GiB tensors)
    counts = Counter(type(o).__name__ for o in objects)
    delta = ""
    prev = _last.get("types")
    if prev:
        ups = sorted(
            ((c - prev.get(name, 0), name)
             for name, c in counts.items()
             if c - prev.get(name, 0) > 200),
            reverse=True,
        )[:8]
        delta = " delta=" + (str(ups) if ups else "(none >200)")
    _last["types"] = dict(counts)
    lines.append(f"census gc objects={len(objects)} top={counts.most_common(12)}{delta}")

    # 5) app-level holders
    registry = _last.get("registry")
    if registry is not None:
        try:
            states = getattr(registry, "_gpu_states", None)
            if not states:
                lines.append("census   registry has no _gpu_states (mock?)")
            slots = 0
            slot_objs: set[int] = set()
            for gpu, state in sorted(getattr(registry, "_gpu_states", {}).items()):
                resident = state.model_id if state.backend is not None else None
                parked = state.parked_id if state.parked is not None else None
                slots += (state.backend is not None) + (state.parked is not None)
                slot_objs |= {id(state.backend), id(state.parked)}
                lines.append(
                    f"census   gpu{gpu}: resident={resident} "
                    f"resident_id={id(state.backend):#x} "
                    f"parked={parked} parked_id={id(state.parked):#x} "
                    f"lock_held={state.lock.locked() if hasattr(state.lock, 'locked') else '?'}"
                )
            extra = [b for b in backends if id(b) not in slot_objs]
            if extra and states:
                lines.append(
                    f"census   !! {len(extra)} backend(s) alive OUTSIDE the "
                    f"registry slots (retained model copy, multi-GiB each); "
                    f"referrers of up to 4 of them:"
                )
                skip = frozenset({id(backends), id(objects), id(extra)})
                for b in extra[:4]:
                    lines.append(
                        f"census   !! {type(b).__name__} "
                        f"model={getattr(b, 'name', '?')} id={id(b):#x} "
                        f"{_owner_label(b, skip)}"
                    )
        except Exception as e:  # pragma: no cover
            lines.append(f"census   registry snapshot failed: {e!r}")
    queue = _last.get("queue")
    if queue is not None:
        try:
            jobs = getattr(queue, "_jobs", {})
            blob = 0
            for j in jobs.values():
                if j.source_png:
                    blob += len(j.source_png)
                for v in (j.results or {}).values():
                    if isinstance(v, (bytes, bytearray)):
                        blob += len(v)
            lines.append(f"census   jobs retained={len(jobs)} png-blobs={blob / 2**20:.1f}MiB")
        except Exception as e:  # pragma: no cover
            lines.append(f"census   queue snapshot failed: {e!r}")
    try:
        from .backends import _shared_enc_caches

        for key, cache in list(_shared_enc_caches.items()):
            ent_bytes = 0
            for entry in cache.values():
                for t in entry:
                    ent_bytes += getattr(t, "nbytes", 0)
            lines.append(
                f"census   enc-cache {key}: {len(cache)}/"
                f"{getattr(cache, 'max', '?')} entries "
                f"{ent_bytes / 2**20:.1f}MiB"
            )
    except Exception as e:  # pragma: no cover
        lines.append(f"census   enc-cache snapshot failed: {e!r}")

    lines.append(f"census done in {time.perf_counter() - started:.1f}s")
    for line in lines:
        _log(line)
    _last["rss_kib"] = st.get("VmRSS", -1)
    return "\n".join(lines)


# ---------------------------------------------------------------- sampler ----


def _sampler_loop(stop: threading.Event) -> None:
    ticks = 0
    while not stop.wait(_state["interval"]):
        try:
            st = proc_kib()
            host = _host_kib()
            rss = st.get("VmRSS", -1)
            parts = [
                "evt=sampler",
                f"rss={gib(rss)}GiB",
                f"anon={gib(st.get('RssAnon', -1))}GiB",
                f"swap={gib(st.get('VmSwap', -1))}GiB",
                f"host_avail={gib(host.get('mem_avail', -1))}GiB",
                f"host_swap_used={gib(host.get('swap_used', -1))}GiB",
            ]
            for dev, (alloc, res, _mx, retries) in sorted(cuda_kib().items()):
                parts.append(f"cuda{dev}={alloc / 2**20:.2f}/{res / 2**20:.2f}GiB")
                if retries:
                    parts.append(f"cuda{dev}retries={retries}")
            _log(" ".join(parts))
            ticks += 1
            last_census_rss = _last.get("census_rss_kib")
            grew = (
                last_census_rss is None
                or (rss - last_census_rss) >= _state["census_trigger_gib"] * 2**20
            )
            if ticks >= _state["census_every"] or grew:
                census(
                    "interval"
                    if not grew
                    else f"rss +{(rss - (last_census_rss or 0)) / 2**20:.1f}GiB since last census"
                )
                ticks = 0
                _last["census_rss_kib"] = rss
        except Exception:  # pragma: no cover
            import traceback

            _log("census sampler tick failed:\n" + traceback.format_exc())


def start(registry=None, queue=None) -> None:
    """Start the sampler thread (idempotent) and remember the holders."""
    global _sampler_stop
    if not _state["enabled"]:
        return
    _last["registry"] = registry
    _last["queue"] = queue
    if _sampler_stop is not None:
        return
    _sampler_stop = threading.Event()
    t = threading.Thread(
        target=_sampler_loop, args=(_sampler_stop,), name="memdbg-sampler", daemon=True
    )
    t.start()
    _log("sampler started")
    census("startup")


def _stop_sampler() -> None:
    global _sampler_stop
    if _sampler_stop is not None:
        _sampler_stop.set()
        _sampler_stop = None
