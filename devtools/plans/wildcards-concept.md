# Prompt wildcard processing

- wildcard textfiles (*.txt) located in wildcards directory
- can be included in prompt via __<wildcard-name>__, e.g. `__weather__` (relative path within the wildcards folder without file extension, wrapped in double underscores)
- wildcards can reference other wildcards (same syntax as above) in the same folder or subfolders (but not parent folders), e.g. `__sub-dir/clouds__`
- on prompt processing, wildcards are replaced with a pseudo-random line (based on seed and wildcard processing order) from the respective text file
- error on infinite recursion (when processing sees an already processed wildcard again)
- wildcards that couldn't be found should trigger an error (message to terminal and ui status line)
- image meta data should store the fully resolved wildcard replacement

I have added some example wildcards to the wildcards folder (note: since these are untested they may contain errors).