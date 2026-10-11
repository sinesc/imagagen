"""Browser tests for the generation status messages.

The progress label shows "Encoding prompt" while the text encoder runs and
"<Generating|Inpainting|Outpainting> @ <W>x<H>" while the transformer
samples; the status line ends with the batch summary
("Completed <n> images in <t>s").

The mock backend generates in milliseconds, so the label text itself is
unobservable: the tests record the WebSocket messages the client renders
(a "phase" event carries the label text, "done" the summary).
"""
import re

from conftest import dot, mask_arr, wait_until
from test_canvas_ui import _open, _show


def _record(page):
    """Collect every phase / done message the server sends for this job."""
    page.wait_for_function("ws && ws.readyState === 1")
    page.evaluate("""() => {
        window.__msgs = [];
        ws.addEventListener('message', (ev) => {
            const m = JSON.parse(ev.data);
            if (m.type === 'phase' || m.type === 'done') window.__msgs.push(m.message);
        });
    }""")


def _wait_summary(page):
    page.wait_for_function(
        "window.__msgs.some(m => m && m.startsWith('Completed'))", timeout=20000)
    return page.evaluate("window.__msgs")


def test_generate_phase_messages_and_summary(page, server):
    base, _ = server
    _open(page, base)
    _record(page)
    page.click("#btn-generate")
    msgs = _wait_summary(page)
    # encoding first, then sampling at the size the model generates at
    assert msgs[0] == "Encoding prompt"
    assert "Generating @ 1024x1024" in msgs
    assert re.fullmatch(r"Completed 1 images in \d+\.\ds", msgs[-1])
    # the summary is what the status line ends up showing
    assert wait_until(lambda: page.text_content("#status-line") == msgs[-1])


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
    msgs = _wait_summary(page)
    assert "Inpainting @ 1024x1024" in msgs
    assert "Outpainting @ 1024x1024" not in msgs

    page.evaluate("window.__msgs = []")
    page.select_option("#mode", "inpaint_not_masked")
    page.click("#btn-generate")
    msgs = _wait_summary(page)
    assert "Outpainting @ 1024x1024" in msgs
