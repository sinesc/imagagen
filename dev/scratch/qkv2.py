import collections
import re

from safetensors import safe_open

with safe_open("infra/qwen_3vl_4b.safetensors", framework="pt") as f:
    ks = list(f.keys())
    info = []
    for k in ks:
        s = f.get_slice(k)
        info.append((k, s.get_dtype(), s.get_shape()))

pat = collections.defaultdict(list)
for k, dt, shp in info:
    p = re.sub(r"\.\d+\.", "." , k)
    pat[p].append((k, dt, shp))

for p, items in pat.items():
    dt = collections.Counter(i[1] for i in items)
    shp = items[0][2]
    print(f"{len(items):3d}  {p}   {shp}  {dict(dt)}")
