"""UI checks for the Oversample field and the width/height row alignment."""


def test_oversample_field(page, server):
    base, _root = server
    page.goto(base)
    page.wait_for_function("document.getElementById('loading-overlay').hidden")

    # oversample disabled in generate mode, enabled in the inpaint modes
    assert page.is_disabled("#oversample")
    page.select_option("#mode", "inpaint_masked")
    assert page.is_enabled("#oversample")
    page.select_option("#mode", "inpaint_not_masked")
    assert page.is_enabled("#oversample")
    page.select_option("#mode", "generate")
    assert page.is_disabled("#oversample")

    # row alignment: width+swap+height = left half (right edge of height
    # aligns with the mode select's right edge), rotate/oversample/batch
    # equal width filling the right half
    b = page.evaluate("""() => {
      const r = (s) => document.querySelector(s).getBoundingClientRect();
      return {row: r('.dims-row'), h: r('#height'), m: r('#mode'),
              o: r('#oversample'), rt: r('#rotation'), bt: r('#batch')};
    }""")
    assert abs(b["h"]["right"] - b["m"]["right"]) < 1.5, (b["h"], b["m"])
    assert abs(b["o"]["width"] - b["rt"]["width"]) < 1.5
    assert abs(b["o"]["width"] - b["bt"]["width"]) < 1.5
    # right half starts at the row midpoint
    mid = b["row"]["left"] + b["row"]["width"] / 2
    assert abs(b["o"]["left"] - mid) < 1.5, (b["o"], mid)
