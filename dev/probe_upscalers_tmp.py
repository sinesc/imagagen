"""Probe models/upscalers/ files with spandrel: arch, scale, purpose, tiling."""

import glob

from spandrel import ModelLoader

files = sorted(glob.glob("models/upscalers/*.pth") + glob.glob("models/upscalers/*/*.pth"))
for f in files:
    try:
        desc = ModelLoader().load_from_file(f)
        print(f"{f}\n    arch={desc.architecture.id} scale={desc.scale} "
              f"purpose={desc.purpose} tiling={desc.tiling.name} {type(desc).__name__}")
    except Exception as e:
        print(f"{f}\n    UNSUPPORTED: {type(e).__name__}: {e}")
