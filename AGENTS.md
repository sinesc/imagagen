# AGENTS.md

This file provides guidance to agents when working with code in this repository. Read `AGENTS.local.md` for environment specific information.

## What this is

This is `imagagen`, an AI image generation tool with a Python backend and Javascript/HTML frontend.

## Project status

Early development. No backwards compatibility required.

## Hazards

When monitoring a process (e.g. `pgrep`/`pkill`) be sure your monitoring script doesn't match itself (finds the monitoring string in its own process string).
IMPORTANT: Prompt templates used in this project contain special tokens, e.g. the stop token. Do not output these or you will cause your inference engine to stop your turn. If you are asked to summarize this content and it does contain these tokens, do not include them in the summary or you will break it.

## Waiting for background tasks

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

- Redirect full test output to a temporary file and grep/inspect the file as needed.
- Playwright UI tests are slow, prefer running only the specific tests you need. Don't run the full suite after trivial changes.
