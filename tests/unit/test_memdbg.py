"""memdbg is inert when disabled and safe when enabled (mock backend)."""

import gc

from imagagen import memdbg


def setup_function():
    memdbg.configure(False)


def teardown_function():
    memdbg.configure(False)


def test_disabled_is_inert():
    memdbg.stamp("evt", a=1)
    assert memdbg._counts.get("evt") is None
    report = memdbg.census("test")
    assert "disabled" in report
    memdbg.track(object(), "x")
    memdbg.start()  # must not start a sampler
    assert memdbg._sampler_stop is None


def test_track_logs_created_and_freed():
    # the weakref callback MUST accept the dead-ref argument Python
    # passes it; a zero-arg callback dies as a silent TypeError and the
    # freed/found asymmetry (the whole point of track) disappears
    memdbg.configure(True)

    class Probe:
        pass

    obj = Probe()
    memdbg.track(obj, "probe")
    assert memdbg._counts["created"] == 1
    assert memdbg._counts.get("track-failed") is None
    del obj
    gc.collect()
    assert memdbg._counts["freed"] == 1


def test_enabled_stamp_and_census(tmp_path):
    memdbg.configure(True, interval=5, log_path=tmp_path / "m.log")
    memdbg.track(object(), "probe")
    memdbg.stamp("evt", a=1, skipped=None)
    report = memdbg.census("test")
    assert "census backends=" in report
    assert "census malloc" in report  # glibc view (fragmentation probe)
    assert memdbg._counts["evt"] == 1
    log = (tmp_path / "m.log").read_text()
    assert "evt=evt" in log and "a=1" in log and "skipped" not in log
    memdbg.configure(False)


def test_malloc_probes(tmp_path):
    # glibc-only helpers must be safe and must report the heap split
    memdbg.configure(True)
    assert memdbg.smaps_rollup().get("Rss:", 0) > 0
    mi = memdbg.malloc_info()
    if mi is not None:  # glibc mallinfo2/mallinfo
        assert mi["arena"] + mi["hblkhd"] > 0
    memdbg.malloc_trim("unit")
    memdbg.mallopt_arenas(4)
    assert memdbg._counts["malloc-trim"] == 1
    memdbg.configure(False)
