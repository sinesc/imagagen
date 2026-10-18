# AGENTS.md

This file provides guidance to agents when working with code in this repository. Environment-specific instructions live in `.pi/APPEND_SYSTEM.md` (git-excluded, auto-injected into the agent system prompt by pi); other agents should read that file if it exists.

## What this is

This is `imagagen`, an AI image generation tool with a Python backend and Javascript/HTML frontend.

## Project status

Early development. No backwards compatibility required.

## Hazards

When monitoring a process (e.g. `pgrep`/`pkill`) make sure your monitoring script doesn't match itself. Use the bracket trick (e.g. `pgrep -f "smoke_krea2[.]lora"`) to avoid finding the monitoring string in the monitoring loops own command line.
IMPORTANT: Prompt templates used in this project contain special tokens, e.g. the im_end token. Avoid outputting it (even in thinking) or escape it. If you are asked to summarize this content, do NOT include them in the summary.

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

NEVER use a sleep longer than 60 seconds when waiting for a background task.

## Testing

- Redirect full test output to a temporary file and grep/inspect the file as needed.
- Playwright UI tests are slow, prefer running only the specific tests you need. Don't run the full suite after trivial changes.
