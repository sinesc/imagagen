#!/bin/sh
cd "${0%/*}"
mkdir -p user
[ -f "user/imagagen.ini" ] || cp res/imagagen.ini.default user/imagagen.ini
[ -f "user/styles.csv" ] || cp res/styles.csv.default user/styles.csv
# A uv sync may rewrite the venv's launcher files with absolute paths (and
# does on any machine that is not the one where they were first written);
# restore the relocatable relative versions. No-op when .venv or a
# project-local CPython (.python313/cpython-*/) does not exist.
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

.venv/bin/python -m imagagen $@
