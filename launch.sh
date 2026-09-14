#!/bin/sh
[ -f "imagagen.ini" ] || cp imagagen.ini.default imagagen.ini
.venv3.13/bin/python -m imagagen $@
