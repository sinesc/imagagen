# Plan: source rotation before generate/inpaint/outpaint

## Purpose and semantics

Some models generate upside-down faces or sideways text poorly. The workaround:
the user tells the system that the *final canvas content* is rotated relative to
what the model should see. The server then

1. **before** generation: rotates the source (canvas image + mask) by the user's
   angle `θ` — the model works in this rotated "working frame" and sees upright
   content;
2. **after** generation: counter-rotates the result by `-θ` and center-crops it
   to the final size.

Angle convention: **positive = clockwise** (as viewed on screen), values
`-180, -135, -90, -45, 0, 45, 90, 135, 180` (multiples of 45). `0` is the default
and changes nothing (no plan, current behaviour).

Working-frame size (bounding box of the rotated rectangle, both dims `ceil`):

- `0`/`180`: unchanged `(w, h)`; `±90`: swapped `(h, w)`;
- `±45`/`±135`: both dimensions `(w+h)/√2` — the generation dimensions are
  extended so the rotated image fits entirely into the generated result.

Per mode (final canvas is always the *requested* `W×H` unless stated otherwise):

- **generate**: no source to rotate; the model generates at the extended size,
  the result is counter-rotated to `W×H` (the user can e.g. prompt a normal face
  with `180°` to get an upside-down one).
- **inpaint_masked**: canvas image + mask are rotated first; the inpaint box is
  computed **on the rotated mask** (aspect-grown + padding as today, so the box
  is minimal w.r.t. the rotated mask); box → crop → generate at `W×H` (no size
  extension — the box already covers the rotated content). The generation is
  blended into the *rotated* canvas, which is then counter-rotated back to the
  original canvas size.
- **inpaint_not_masked (outpaint)**: the target `W×H` canvas is built exactly as
  today (source AR-fit, centered, edge-replicated border); the *whole target*
  is then rotated to the working size and generated. Result is counter-rotated
  to `W×H`.

The result entity's mask is inherited from the source image as today
(`Canvas._inherit_mask_locked`, aspect-fit to the result size). Both the final
image and the inherited mask are in the original (counter-rotated) orientation,
so they stay consistent. The source thumbnail (green border) stays the
**unrotated** canvas snapshot.

## 1. `src/imagagen/generation.py`

Add to imports: `import math`, `from PIL import Image` (already imports numpy).

```python
#: the UI's rotation dropdown values (user degrees, positive = clockwise)
ROTATIONS = (-180, -135, -90, -45, 0, 45, 90, 135, 180)


def rotation_size(w: int, h: int, angle: int) -> tuple[int, int]:
    """Bounding-box (width, height) of a w×h rectangle rotated by `angle`
    (user degrees, positive = clockwise). 0/180: (w, h); ±90: (h, w);
    ±45/±135: both dimensions (w+h)/√2 (rounded up)."""
    a = abs(angle) % 180
    c, s = abs(math.cos(math.radians(a))), abs(math.sin(math.radians(a)))
    return max(1, math.ceil(w * c + h * s)), max(1, math.ceil(w * s + h * c))


def rotate_canvas(arr: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an HxW[x3] uint8 array by `angle` (user degrees, positive =
    clockwise), expanding to the rotated rectangle's bounding box.

    0/±90/180 are exact pixel permutations; ±45/±135 are bicubic-resampled and
    the uncovered corners (which only exist for ±45/±135) are filled by
    replicating the nearest edge pixel of `arr` (images) or zeroed (masks), so
    image and mask share the same pixel grid while the mask never invents
    coverage outside the original canvas."""
    angle = angle % 360
    if angle == 0:
        return arr.copy()
    if angle == 180:
        return np.flip(arr, axis=(0, 1))
    if angle in (90, 270):
        return np.rot90(arr, k=-1 if angle == 90 else 1)
    h, w = arr.shape[:2]
    cw, ch = rotation_size(w, h, angle)
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    # The corners of the rotated rectangle reach past the original edges by at
    # most max(w, h)·|sin 2θ|/2 (= max(w,h)/2 at 45°). Rotate an edge-padded
    # oversize copy so every output pixel samples (edge-replicated) source.
    pad = int(math.ceil(max(w, h) * abs(math.sin(2 * th)) / 2)) + 2
    base = np.pad(
        arr, ((pad, pad), (pad, pad)) + ((0, 0),) * (arr.ndim - 2),
        mode="constant" if arr.ndim == 2 else "edge")
    bh, bw = base.shape[:2]
    cx, cy = (cw - 1) / 2.0, (ch - 1) / 2.0
    # Output pixel (x, y) -> source pixel: the working frame is the source
    # rotated by `angle` clockwise, so the inverse mapping rotates back by
    # -angle around the shared center.
    m = (
        c, s, -c * cx - s * cy + (bw - 1) / 2.0,
        -s, c, s * cx - c * cy + (bh - 1) / 2.0,
    )
    return np.asarray(Image.fromarray(base).transform(
        (cw, ch), Image.AFFINE, m, resample=Image.BICUBIC, fillcolor=0))


def unrotate(img: np.ndarray, angle: int, w: int, h: int) -> np.ndarray:
    """Inverse of :func:`rotate_canvas` for a known final size: counter-rotate
    by -`angle` and center-crop to (w, h). (A ±1 px centring error is possible
    for the ±45/±135 family — the crop is centred on the rotation centre.)"""
    if angle % 360 == 0:
        if img.shape[:2] != (h, w):
            return resize_image(img, w, h)
        return img
    out = rotate_canvas(img, -angle)
    oh, ow = out.shape[:2]
    x0, y0 = (ow - w) // 2, (oh - h) // 2
    return out[y0 : y0 + h, x0 : x0 + w].copy()
```

(Verified: `rotate`/`unrotate` round-trip is exact for 0/±90/180 and ≤ 2/255
interior for ±45/±135; 90°-rotating a 20×10 image moves a top-left marker to
(row 0, col 9), i.e. clockwise.)

### `InpaintPlan` — reworked fields

`base_image`/`base_mask` are dropped (the source thumbnail now comes straight
from `canvas.image` in the queue — see §2); `base_mask` is never needed by
postprocess (the result mask is inherited by `Canvas.add_result`).

```python
@dataclass
class InpaintPlan:
    """Captured at job submit time; maps the backend's width x height output
    back to canvas state. For a non-zero rotation all geometry (bbox, masks)
    is in the *rotated working frame*: the source canvas is rotated by
    `rotation` before generation and the output is counter-rotated back to
    `requested_size`. (The rest is unchanged: the canvas stays editable
    during generation — the generation overwrites in-flight edits, restorable
    via undo.)"""

    mode: str  # "generate" | "inpaint_masked" | "inpaint_not_masked"
    requested_size: tuple[int, int]  # the user's W×H (final canvas size)
    rotation: int = 0  # user angle, degrees, positive = clockwise
    # inpaint_masked, working frame
    bbox: tuple[int, int, int, int] | None = None
    work_image: np.ndarray | None = None  # rotated canvas (the blend target)
    # inpaint_masked: rotated mask (blend alpha); inpaint_not_masked:
    # rotated target mask (resize reference)
    work_mask: np.ndarray | None = None

    def postprocess(self, gen: np.ndarray) -> np.ndarray:
        if self.mode == "inpaint_masked":
            assert self.bbox is not None and self.work_image is not None
            x0, y0, w, h = self.bbox
            resized = resize_image(gen, w, h)
            out = self.work_image.copy()
            H, W = out.shape[:2]
            a = self.work_mask.astype(np.float32) / 255.0
            dx0, dy0 = max(x0, 0), max(y0, 0)
            dx1, dy1 = min(x0 + w, W), min(y0 + h, H)
            if dx1 > dx0 and dy1 > dy0:
                patch = resized[dy0 - y0 : dy1 - y0, dx0 - x0 : dx1 - x0].astype(
                    np.float32)
                aa = a[dy0:dy1, dx0:dx1][..., None]
                reg = out[dy0:dy1, dx0:dx1].astype(np.float32)
                out[dy0:dy1, dx0:dx1] = np.clip(
                    reg * (1 - aa) + patch * aa, 0, 255).astype(np.uint8)
        elif self.mode == "inpaint_not_masked":
            assert self.work_mask is not None
            if gen.shape[:2] != self.work_mask.shape[:2]:
                # the backend may round the generation size (SDXL: multiples
                # of 8); the canvas must come out exactly at the requested size
                gen = resize_image(gen, self.work_mask.shape[1],
                                   self.work_mask.shape[0])
            out = gen
        else:  # "generate": the raw output is the (possibly rotated) result
            out = gen
        if self.rotation:
            fw, fh = self.requested_size
            out = unrotate(out, self.rotation, fw, fh)
        return out
```

For rotation 0 every branch behaves exactly as the current code does
(`bbox` blend against the unrotated canvas; `work_mask` == old `new_mask`).

### `with_inpaint`

New keyword `rotation: int = 0` (validate `rotation in ROTATIONS`, else
`GenerationError`). Returns `None` only for `"generate"` with rotation 0:

```python
def with_inpaint(request, mode, canvas, denoise, rotation=0,
                 padding=INPAINT_PADDING) -> InpaintPlan | None:
    if rotation not in ROTATIONS:
        raise GenerationError(f"invalid rotation {rotation!r}")
    if mode == "generate":
        if rotation == 0:
            return None
        fw, fh = request.width, request.height
        request.width, request.height = rotation_size(fw, fh, rotation)
        return InpaintPlan("generate", requested_size=(fw, fh),
                           rotation=rotation)
    if mode not in ("inpaint_masked", "inpaint_not_masked"):
        raise GenerationError(f"unknown mode {mode!r}")
    request.denoise = denoise
    image = canvas.image
    if image is None:
        raise GenerationError("canvas is empty")
    mask = canvas.mask
    work_image = rotate_canvas(image, rotation) if rotation else image
    work_mask = rotate_canvas(mask, rotation) if rotation else mask

    if mode == "inpaint_masked":
        if not (work_mask > 0).any():
            raise GenerationError("mask is empty")
        # context ring, capped at the rotated mask's largest extent
        rows = np.where((work_mask > 0).any(axis=1))[0]
        cols = np.where((work_mask > 0).any(axis=0))[0]
        extent = int(max(cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1))
        x0, y0, w, h = masked_bbox(work_mask, request.width, request.height,
                                   padding=min(padding, extent))
        request.init_image = resize_image(
            crop_padded(work_image, x0, y0, w, h), request.width,
            request.height)
        request.init_mask = resize_image(
            crop_mask(work_mask, x0, y0, w, h), request.width, request.height)
        return InpaintPlan("inpaint_masked",
                           requested_size=(request.width, request.height),
                           rotation=rotation, bbox=(x0, y0, w, h),
                           work_image=work_image, work_mask=work_mask)

    # inpaint_not_masked
    ch, cw = image.shape[:2]
    sw, sh, ox, oy = fit_ar(cw, ch, request.width, request.height)
    target = paste_centered(
        resize_image(image, sw, sh), request.width, request.height)
    new_mask = (
        resize_shift_mask(mask, sw, sh, ox, oy, request.width, request.height)
        if mask.any()
        else np.zeros((request.height, request.width), dtype=np.uint8)
    )
    fw, fh = request.width, request.height
    if rotation:
        request.width, request.height = rotation_size(fw, fh, rotation)
        request.init_image = rotate_canvas(target, rotation)
        work_mask = rotate_canvas(new_mask, rotation)
    else:
        request.init_image = target
        work_mask = new_mask
    request.init_mask = np.uint8(255) - work_mask  # protect the mask region
    return InpaintPlan("inpaint_not_masked", requested_size=(fw, fh),
                       rotation=rotation, work_mask=work_mask)
```

Notes:

- "inpaint_masked" keeps the request at `W×H` (no extension): the box on the
  rotated mask covers the rotated content, and the crop is resized to `W×H`
  as today.
- "inpaint_not_masked" rotates the *whole target* (source + borders), so the
  source content is upright in the working frame while the outpaint borders
  and the discarded 45° corners stay edge-replicated.
- Update the module docstring: a "generate" job may carry a rotation plan,
  and all plan geometry for rotated jobs is in the working frame.

## 2. `src/imagagen/queue.py`

- `submit()`: the source thumbnail is always the unrotated canvas:

  ```python
  if job.source_image_id is not None:
      job.source_png = _plain_png(self.canvas.image)
  ```

  (replaces the `plan.base_image` / `canvas.image` split — `plan.base_image`
  was a copy of `canvas.image` at submit time).

- `_result_to_png()`: metadata records the **user's requested size** and the
  rotation (the generation size differs from it only for rotated
  generate/outpaint jobs). With `p = job.plan if job is not None else None`:

  ```python
  width=p.requested_size[0] if p is not None else req.width,
  height=p.requested_size[1] if p is not None else req.height,
  # A1111/Forge write the strength only for inpaint/img2img jobs
  denoise=req.denoise if (p is not None and p.mode != "generate") else None,
  rotation=p.rotation or None if p is not None else None,
  ```

  (update the neighbouring comment: width/height is the *user's requested*
  size — the generation size except for rotated generate/outpaint, where it is
  the final canvas size; `rotation` rides along so an import regenerates the
  same extended generation).
- Update the `Job.plan` comment ("inpaint modes" → "maps output to canvas
  state; present for inpaint modes and for rotated generate jobs").

## 3. `src/imagagen/metadata.py`

`build_parameters(..., rotation: int | None = None)` — new keyword, written
only when non-zero (like `denoise`):

```python
if rotation:
    doc["rotation"] = rotation
```

No parser changes: the JSON path returns the document verbatim, so an import
sees `meta["rotation"]` (int); legacy text images simply lack the key.

## 4. `src/imagagen/app.py`

- `GenerateBody`:

  ```python
  # rotate the source by this many degrees before generating and
  # counter-rotate the result (positive = clockwise); 0 = no rotation.
  # ±45/±135 extend the generate/outpaint generation size
  rotation: int = Field(default=0, ge=-180, le=180)

  @field_validator("rotation")
  @classmethod
  def _rotation_multiple_of_45(cls, v: int) -> int:
      if v % 45:
          raise ValueError("rotation must be a multiple of 45")
      return v
  ```

  (`field_validator` imported from pydantic alongside the existing imports.)
- `generate()`: `plan = with_inpaint(req, body.mode, canvas, body.denoise,
  rotation=body.rotation, padding=body.mask_padding)`.

## 5. `src/imagagen/uistate.py`

```python
# rotate the source before generating (UI dropdown; 0 = default)
rotation: int = Field(default=0, ge=-180, le=180)
```

plus the same multiple-of-45 `field_validator` (import from pydantic). Old
state files without the key load with the default 0.

## 6. `src/imagagen/presets.py`

- `PRESET_FIELDS`: insert `"rotation"` after `"height"` (UI order).
- `Preset`: `rotation: int | None = Field(default=None, ge=-180, le=180)` plus
  the same multiple-of-45 `field_validator` (`field_validator` is already
  imported there).

## 7. Frontend

### `static/index.html`

In the Width/Height row, between the Height label and the Batch label:

```html
<label class="rotate" title="rotate the source before generating and counter-rotate the result: the model always sees upright content (+ = clockwise); ±45°/±135° extend the generation size so the rotated image fits">Rotate
  <select id="rotation">
    <option value="-180">-180°</option>
    <option value="-135">-135°</option>
    <option value="-90">-90°</option>
    <option value="-45">-45°</option>
    <option value="0" selected>0°</option>
    <option value="45">+45°</option>
    <option value="90">+90°</option>
    <option value="135">+135°</option>
    <option value="180">+180°</option>
  </select>
</label>
```

### `static/style.css`

```css
/* width/height/rotate row: rotate is a fixed slot next to the width/height
   slots (same width); batch (flex: 1) absorbs the remainder, so the new
   field takes its space from the batch slot */
.row > label.rotate { flex: 0 0 72px; }
```

(Batch stays `flex: 1` — no rule needed for it.)

### `static/app.js`

- `generate()`: add `rotation: parseInt($("rotation").value, 10),` to the body
  (next to `width`/`height`).
- `readUiState()`: add `rotation: parseInt($("rotation").value, 10),`.
- `applyUiState()`: add `["rotation", "rotation"]` to the `fields` list.
- `applyPngMeta()`: after the numeric-field loop:

  ```js
  // the source rotation round-trips through the PNG (absent = leave as-is)
  if (meta.rotation !== undefined && meta.rotation !== null)
    $("rotation").value = String(meta.rotation);
  ```

- Wiring (next to the sampler/scheduler `onchange` block):

  ```js
  $("rotation").onchange = () => { onUserFieldChange("rotation");
                                   scheduleSaveUiState(0); };
  ```

- Presets:
  - `PRESET_KEYS`: insert `"rotation"` after `"height"`.
  - `openPresetDialog()`: in `defs`, add `["rotation", "rotation", "select", p.rotation]`
    after the height row; in the option-populating loop add
    `if (key === "rotation") [-180, -135, -90, -45, 0, 45, 90, 135, 180].forEach((d) => input.append(opt(d, d + "°")));`
    (the "—" empty option and `input.value = value || ""` come from the
    existing select-kind handling, exactly like the mode row).
  - Dialog save: before the generic select branch add
    `else if (key === "rotation") body[key] = input.value === "" ? null : parseInt(input.value, 10);`
  - `applyPreset()`: no change needed — `$("rotation").value = p.rotation`
    works via the existing `key.replace(/_/g, "-")` path, and
    `onUserFieldChange("rotation")` maps id→preset key 1:1.
- `syncMode()`: no change — rotation is valid in all three modes.

## 8. Tests

### `tests/unit/test_generation.py`

- `rotation_size`: `(100,100,0)==(100,100)`, `(100,100,180)==(100,100)`,
  `(100,60,90)==(60,100)`, `(100,60,-90)==(60,100)`, `(100,60,±45)==(80,80)`,
  `(100,60,±135)==(80,80)` (`ceil(112/√2)==80`).
- `rotate_canvas`/`unrotate`:
  - 20×10 image, unique marker at (row 0, col 0): `rotate 90` → shape
    (20,10), marker at (0,9) (clockwise); `-90` → (19,0); `180` → (9,19).
    `unrotate(rotate(a, θ), θ, w, h)` is **exact** for 90/-90/180.
  - 45°: uniform 200 image → rotated is uniformly 200 (edge-replicated
    corners); binary mask → rotated corners are 0; roundtrip interior
    max diff ≤ 3.
- `with_inpaint` rotation:
  - `"generate"` 64×48, rotation 90 → request becomes 48×64,
    `plan.requested_size == (64,48)`; `postprocess` of a (64,48,3) generation
    → (48,64,3). Rotation 0 → still `None` (existing test keeps passing).
  - `"generate"` 100×100, rotation 45 → request 142×142; postprocess of a
    (142,142,3) generation → (100,100,3).
  - `"inpaint_masked"` 64×64 canvas (fill 100), 24×24 mask blob at rows 12:36,
    cols 20:44, request 64×64 padding 0, rotation 90: the rotated blob is
    rows 20:44, cols 28:52 in the working canvas, so
    `plan.bbox == (28, 20, 24, 24)` (the box on the *rotated* mask — on the
    unrotated mask it would be (20, 12, 24, 24)); `req.init_image/init_mask`
    are (64,64); `postprocess` of a uniform 250 (64,64,3) generation →
    (64,64,3) with the value 250 exactly on rows 12:36, cols 20:44 (the
    counter-rotated blend box) and fill 100 elsewhere.
  - `"inpaint_not_masked"` 64×48 canvas, request 64×48, rotation 90 → request
    48×64, `init_image` shape (64,48,3); request 100×100 rotation 45 →
    request 142×142, `init_image` shape (142,142,3), `postprocess` → (100,100,3).
- Invalid rotation: `with_inpaint(req, "generate", canvas, 1.0, rotation=30)`
  raises `GenerationError`.

### `tests/unit/test_queue.py` and existing `InpaintPlan` constructions

- `test_queue.py::test_inpaint_retains_canvas_until_thumb_selected` →
  `InpaintPlan("inpaint_masked", requested_size=(8, 8), bbox=(0, 0, 8, 8),
  work_image=base, work_mask=mask)`.
- `test_generation.py::test_inpaint_not_masked_postprocess_resizes_rounded_output`
  → `InpaintPlan("inpaint_not_masked", requested_size=(100, 100),
  work_mask=np.zeros((100, 100), dtype=np.uint8))`.

### `tests/integration/test_generation_modes.py`

New section (mock paints R = horizontal ramp, G = vertical ramp over the
**request** size, B = seed; a column/row slice of the result reveals the
ramp's orientation). The blend box in the *final* canvas lands back on the
original mask box (rotation and counter-rotation cancel), so the inpaint
tests assert the replaced region **plus** the counter-rotated content
orientation:

- `test_generate_rotation_90_swaps_ramp_axes`: canvas 64×48 (flat), generate
  `width=64, height=48, rotation=90` → after selecting the result the canvas
  is (48,64,3) and the R ramp (which runs along the generation's width 48)
  is counter-rotated to run along the **rows**: a column slice
  `img[:, 32, 0]` has spread > 150, a row slice `img[30, :, 0]` is flat
  (`max-min ≤ 60`, mock noise aside).
- `test_generate_rotation_45_fits_and_extends`: request 64×48, rotation 45 →
  result canvas exactly (48,64,3) (generation was 80×80 = `ceil(112/√2)`).
- `test_generate_rotation_180_flips`: request 64×48, rotation 180 → after
  selecting, the R ramp is **mirrored** vs the rotation-0 result:
  `img[24, 5, 0] > img[24, 58, 0]` (the 0° job's inequality is reversed).
- `test_inpaint_masked_rotation_90_replaces_upright`: 64×64 canvas (fill 99),
  `_mask_dot(client, 32, 24, size=24)` (rows 12:36, cols 20:44),
  `mode=inpaint_masked, width=64, height=64, rotation=90, mask_padding=0`
  → after selecting the result: canvas stays (64,64,3); inside the mask box
  the R ramp runs **along the rows** (the mock's horizontal ramp was
  counter-rotated to vertical): a column slice over rows 13:35 at a middle
  column has spread > 40, a row slice over cols 21:43 at a middle row is flat
  (`max-min ≤ 60`); everything outside rows 8:40, cols 16:48 is still exactly
  99.
- `test_inpaint_masked_rotation_180_flips_content`: same setup with
  `rotation=180`: inside the mask box the R ramp is **mirrored** —
  `img[24, 22, 0] > img[24, 42, 0]` (increasing-x ramp for rotation 0, so the
  inequality is reversed; ≈ 81 signal vs ≈ 17 per-pixel noise) — and
  everything outside the margin box is 99.
- `test_inpaint_not_masked_rotation_90_outpaints_upright`: 32×24 canvas (flat),
  `mode=inpaint_not_masked, width=64, height=48, rotation=90` → result canvas
  (48,64,3), whole canvas freshly generated with the R ramp along the rows
  (`_r_spread` over a column slice > 150), per the same ramp-axis argument as
  the generate test.
- `test_rotation_rejected`: `rotation: 30` → 422; `rotation: 185` → 422.
- `test_rotation_metadata_round_trips`: generate 64×48 rotation 90 → read the
  result PNG's metadata: `width == 64, height == 48, rotation == 90` (the
  user's requested size, not the 48×64 generation size); rotation 0 job →
  `"rotation" not in meta`.

### `tests/unit/test_presets.py` / `tests/integration/test_ui_state.py`

- UiState: `{"rotation": 90}` persists and is served back; `{"rotation": 30}`
  → 422; `{"rotation": 0}` persists as 0.
- Preset: a preset with `rotation: 90` in `controls` round-trips through
  save/load (`to_dict` includes `controls["rotation"]`); `rotation: 45` in a
  preset file validates; `rotation: 30` → 422.

### `tests/ui/test_rotation_ui.py` (new, Playwright)

- The `#rotation` select sits between `#height` and `#batch` in the dims row,
  options `-180° … +180°` (9 entries, `0°` selected by default).
- Selecting `+90°` persists: `page.select_option("#rotation", "90")`,
  `saveUiState()`, reload → select shows `90`.
- A generate with rotation actually rotates: upload a 64×48 flat canvas via
  the API, `page.select_option("#rotation", "180")`, click Generate, wait for
  the `image` event, select the result via `POST /api/canvas/thumb` (httpx),
  and assert the canvas PNG's R ramp is mirrored (the same two-pixel check as
  the integration test).

## 9. Verification

1. `uv run pytest tests/unit tests/integration -q` (full non-UI suite).
2. `uv run pytest tests/ui/test_rotation_ui.py -q` (Playwright).
3. Smoke on the live server (GPU1 free per AGENTS.local.md), e.g.
   `dev/smoke`-style script: 1024×1024 generate with rotation 180 and 45
   (check the 45° job generates at 1449×1449 = `ceil(1024·√2)` — log the
   batch-queued line — and the result canvas is 1024×1024), one inpaint
   with rotation 90.
