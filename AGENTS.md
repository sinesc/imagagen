# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## What this is

This is `imagagen`, an AI image generation tool with a Python backend and Javascript/HTML frontend.

## Project status

Early development. No backwards compatibility required.

## References

The parallel folder `stable-diffusion-webui-forge` contains the branch `reference-branch` with an identical backend to the one used to generate the reference images in `devtools/references/` (the checkout only lacks a few UI changes). All provided LORAs and models are exact copies of the ones used for the reference images. All reference images were decoded using the provided `sdxl_vae.safetensors` VAE. Reference images were generated in a different environment so don't expect to achieve pixel perfect matches.

## Environment

This environment has access to two GPUs. One of them should have enough free VRAM to run Stable Diffusion.
Playwright is available in the project's uv environment. Chromium is available.

## Process monitoring

When monitoring a process (e.g. `pgrep`/`pkill`) be sure your monitoring script doesn't match itself (finds the monitoring string in its own process string).

## Testing

Full test suite is very slow, prefer running the specific tests you need. Redirect test outputs to a temporary file so you can inspect it multiple times if you need to.
