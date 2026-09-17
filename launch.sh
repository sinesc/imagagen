#!/bin/sh
cd "${0%/*}"
[ -f "imagagen.ini" ] || cp imagagen.ini.default imagagen.ini
[ -f "styles.csv" ] || cp styles.csv.default styles.csv
.venv/bin/python -m imagagen $@
