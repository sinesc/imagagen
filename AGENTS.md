# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## What this is

This is `imagagen`, an AI image generation tool with a Python backend and Javascript/HTML frontend.

## Project status

Early development.

## References

The parallel folder `sd-webui-forge-classic` contains a StableDiffusion WebUI implementation that may be useful to look at for reference implementations (e.g. sampler code).

The imagagen subdirectory `devtools/references/` contains reference images generated with Forge. Note: these were generated in a different environment and not necessarily the exact same version of the provided StableDiffusion WebUI reference implementation so don't expect to achieve pixel perfect matches.

## Environment

This environment has access to two GPUs, one of which should have enough VRAM to spare for running stable diffusion.
