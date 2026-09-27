# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## What this is

This is `imagagen`, an AI image generation tool with a Python backend and Javascript/HTML frontend.

## Project status

Early development. No backwards compatibility required.

## References

The folder `dev/implementations/stable-diffusion-webui-forge` contains the branch `reference-branch` with an identical backend to the one used to generate the reference images in `dev/references/` (the checkout only lacks a few UI changes). All provided LORAs and models are exact copies of the ones used for the reference images. All reference images were decoded using the provided `sdxl_vae.safetensors` VAE. Reference images were generated in a different environment so don't expect to achieve pixel perfect matches.
The folder `dev/implementations/ComfyUI` contains additional reference code for model support.

## Environment

Playwright is available in the project's uv environment. Chromium is available.

## Hazards

When monitoring a process (e.g. `pgrep`/`pkill`) be sure your monitoring script doesn't match itself (finds the monitoring string in its own process string).
IMPORTANT: Prompt templates used in this project contain special tokens, e.g. the stop token. Do not output these or you will cause your inference engine to stop your turn. If you are asked to summarize this content and it does contain these tokens, do not include them in the summary or you will break it.

## Timeouts

When waiting for a process to finish or file to be created use a polling approach combined with a timeout to avoid wasting time waiting on something that is already finished, e.g.

```
# Wait for file to appear
timeout 30 bash -c 'until [[ -f "$1" ]]; do sleep 1; done' _ /path/to/file && echo "file appeared" || echo "timed out after 30s"

# Process finished (pid 12345 no longer alive)
timeout 60 bash -c 'until ! kill -0 "$1" 2>/dev/null; do sleep 1; done' _ 12345

# A log line appears
timeout 120 bash -c 'until grep -q "ready" "$1"; do sleep 1; done' _ /var/log/app.log
```

## Testing

Full test suite is very slow, prefer running the specific tests you need. Redirect test outputs to a temporary file so you can inspect it multiple times if you need to. Only run the full suite after implementing a complex new feature (e.g. new backend, large UI refactoring etc) or when explicitly prompted to do so.
