# AGENTS.md

This file provides guidance to agents when working with code in this repository. Environment-specific instructions live in `.pi/APPEND_SYSTEM.md` (git-excluded, auto-injected into the agent system prompt by pi); other agents should read that file if it exists.

## What this is

This is `imagagen`, an AI image generation tool with a Python backend and Javascript/HTML frontend.

- source in `src/imagagen/`, frontend in `src/imagagen/static/`.
- dev tools, plans and reference images in `dev/` (some environments may also contain git-ignored reference implementations in `dev/implementations/`)
- relocatable python in `.python313` (exact version number might differ)
- venv in `.venv`, modified by `launch.sh` to be relocatable (fixes absolute paths used in venv so it can be used from both a host machine and a dev-container mounting the host folder)
- ui test (playwright), unit/integration tests in `tests/`

# Commit policy

Before committing your changes make sure the relevant tests still pass. If you encounter pre-existing test failures fix and commit these as well (separate commit). 
Use past tense in the commit message, e.g. "Fixed glorp widget calibration", "Added support for ACME widgets".
