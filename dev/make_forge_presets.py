"""One-off: create imagagen presets from the old Forge presets in
config-img2img.json / config-txt2img.json. Idempotent (overwrites in place)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from imagagen.presets import Preset, PresetStore  # noqa: E402

MAX_DIM = 4096  # form/preset cap


def sanitize(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:120].rstrip("-")


def write(store: PresetStore, group=None, **fields) -> str:
    name = fields.pop("name")
    controls = {k: True for k in fields if fields[k] is not None}
    p = Preset(name=name, group=group, **fields, controls=controls)
    store.save(p)
    return p.name


store = PresetStore(ROOT / "user" / "presets")

# ---------------- img2img: detail touchup (mask_mode 0 -> inpaint_masked) --
for w, h, denoise in [
    (1024, 1024, 0.55), (1024, 1280, 0.55), (1024, 1440, 0.55),
    (1280, 1440, 0.55), (1280, 1600, 0.55), (1280, 1920, 0.45),
    (1440, 1600, 0.55), (1440, 1920, 0.55),
]:
    write(store,
          group="Inpaint",
          name=f"Detail touchup {w}x{h}",
          width=w, height=h, steps=12, cfg=1.5,
          sampler="Euler a", scheduler="Polyexponential",
          mode="inpaint_masked", denoise=denoise, mask_padding=8)

# ---------------- img2img: outpaint (-> inpaint_not_masked) ----------------
def outpaint(name, w, h, denoise):
    if w > MAX_DIM or h > MAX_DIM:
        # the form caps at 4096 -- drop the unsupported dimensions
        w = h = None
    write(store,
          group="Outpaint 1440p",
          name=name,
          width=w, height=h, steps=12, cfg=1.5,
          sampler="Euler a", scheduler="Polyexponential",
          mode="inpaint_not_masked", denoise=denoise)

for i, w in enumerate([1280, 1440, 1728, 1920, 2244, 2544, 2944], start=1):
    outpaint(f"Outpaint Step {i}", w, 1440, 0.8)
for denoise in [0.40, 0.80, 0.55, 0.45, 0.3]:
    # the 0.40 Forge preset ran ADetailer afterwards; no equivalent here
    outpaint(f"Outpaint Background {denoise:.2f}", 3440, 1440, denoise)

# ---------------- txt2img (-> generate) -------------------------------------
for name, w, h, steps, cfg, sampler in [
    ("SDXL 1024x1280 Standard", 1024, 1280, 24, 7.5, "DPM++ 2M SDE"),
    ("SDXL 1728x736 Standard", 1728, 736, 24, 7.5, "DPM++ 2M SDE"),
    ("SDXL 1024x1280 DMD2", 1024, 1280, 12, 1.5, "Euler a"),
    ("SDXL 1728x736 DMD2", 1728, 736, 12, 1.5, "Euler a"),
]:
    write(store,
          group="Generate",
          name=name,
          width=w, height=h, steps=steps, cfg=cfg,
          sampler=sampler, scheduler="Polyexponential",
          mode="generate")
# "SDXL ... LCM" presets skipped: no LCM sampler.

# ---------------- sanity check ----------------------------------------------
for name in store.names():
    p = store.get(name)
    assert p is not None, name
print("presets in store:")
for p in store.all():
    print(f"  [{p.group or '-'}] {p.name}: {p.controlled()}")
