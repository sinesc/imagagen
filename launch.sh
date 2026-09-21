#!/bin/sh
cd "${0%/*}"
mkdir -p user
[ -f "user/imagagen.ini" ] || cp res/imagagen.ini.default user/imagagen.ini
[ -f "user/styles.csv" ] || cp res/styles.csv.default user/styles.csv
# uv sync (in the container) rewrites these three venv files with absolute
# container paths, which break host usage; restore the relative versions
# (also documented in README.md, "Running on the host system")
ln -sfn ../../.python313/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13 .venv/bin/python
sed -i 's|^home = .*|home = ../../.python313/cpython-3.13.15-linux-x86_64-gnu/bin|' .venv/pyvenv.cfg
printf '../../../../src\n' > .venv/lib/python3.13/site-packages/_editable_impl_imagagen.pth

.venv/bin/python -m imagagen $@
