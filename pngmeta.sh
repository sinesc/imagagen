#!/bin/sh
# Print an image file's generation metadata as imagagen-format JSON
# (A1111/Forge metadata is converted to our format). See
# src/imagagen/pngmeta.py for options (--help).
cd "${0%/*}"
# Same relocatable-venv repair as launch.sh (no-op when not needed).
if [ -d .venv ]; then
  for pybin in .python313/cpython-*/bin; do
    if [ -x "$pybin/python3.13" ]; then
      ln -sfn "../../$pybin/python3.13" .venv/bin/python
      sed -i "s|^home = .*|home = ../../$pybin|" .venv/pyvenv.cfg
      break
    fi
  done
  for pkg in .venv/lib/python*/site-packages; do
    if [ -f "$pkg/_editable_impl_imagagen.pth" ]; then
      printf '../../../../src\n' > "$pkg/_editable_impl_imagagen.pth"
    fi
  done
fi

.venv/bin/python -m imagagen.pngmeta "$@"
