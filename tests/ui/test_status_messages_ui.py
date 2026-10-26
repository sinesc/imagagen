"""Browser tests for the generation status messages.

The status line reports the generation phases: "Encoding prompt…" while the
text encoder runs, "<Generating|Inpainting|Outpainting> @ <W>x<H>…" while
the transformer samples, and the batch summary ("Completed <n> images in
<t>s") when the batch ends. A phase text starts with a capital letter and
ends on an ellipsis while its stage is still running; the progress label
stays reserved for the step counter.

The mock backend generates in milliseconds, so the status line's text is
unobservable: the tests hook setStatus (a global function of a classic
script, so reassigning window.setStatus intercepts every call) and watch
#progress-label with a MutationObserver.
"""
import re

from conftest import dot, mask_arr, wait_until
from test_canvas_ui import _open, _show

ELLIPSIS = "\u2026"
PHASES = ("Encoding prompt" + ELLIPSIS, "Generating @ 1024x1024" + ELLIPSIS,
          "Inpainting @ 1024x1024" + ELLIPSIS,
          "Outpainting @ 1024x1024" + ELLIPSIS)


def _record(page):
    """Record every setStatus() call and every #progress-label text."""
    page.wait_for_function("ws && ws.readyState === 1")
    page.evaluate("""() => {
        window.__status = [];
        window.__label = [];
        const origSet = setStatus;
        setStatus = (text, isError) => {
            window.__status.push(text || ''); return origSet(text, isError);
        };
        const label = document.getElementById('progress-label');
        new MutationObserver(() => window.__label.push(label.textContent))
            .observe(label, {childList: true, subtree: true, characterData: true,
                             characterDataOldValue: true});
    }""")


def _wait_summary(page):
    page.wait_for_function(
        "window.__status.some(t => t.startsWith('Completed'))", timeout=20000)
    return page.evaluate("window.__status")


def test_generate_phases_and_summary_go_to_the_status_line(page, server):
    base, _ = server
    _open(page, base)
    _record(page)
    page.click("#btn-generate")
    status = _wait_summary(page)
    # encoding first, then sampling at the size the model generates at,
    # then the batch summary -- all of them on the status line
    assert "Encoding prompt" + ELLIPSIS in status
    assert "Generating @ 1024x1024" + ELLIPSIS in status
    summary = [t for t in status if t.startswith("Completed")]
    assert len(summary) == 1
    assert re.fullmatch(r"Completed 1 images in \d+\.\ds", summary[0])
    # the summary is what the line ends up showing
    assert wait_until(lambda: page.text_content("#status-line") == summary[0])
    # the progress label only ever carries the step counter
    label = page.evaluate("window.__label")
    assert any(t and t.startswith("step") for t in label), label
    assert not any(t in PHASES for t in label), label


def test_inpaint_and_outpaint_phase_verbs(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)
    page.select_option("#mode", "inpaint_masked")
    page.click("#tool-mask")
    dot(page, 100, 75)
    assert wait_until(lambda: mask_arr(base) is not None and mask_arr(base).any())
    _record(page)

    page.click("#btn-generate")
    status = _wait_summary(page)
    assert "Inpainting @ 1024x1024" + ELLIPSIS in status
    assert "Outpainting @ 1024x1024" + ELLIPSIS not in status
    assert not any(t in PHASES for t in page.evaluate("window.__label"))

    page.evaluate("window.__status = []; window.__label = []")
    page.select_option("#mode", "inpaint_not_masked")
    page.click("#btn-generate")
    status = _wait_summary(page)
    assert "Outpainting @ 1024x1024" + ELLIPSIS in status
    assert not any(t in PHASES for t in page.evaluate("window.__label"))
