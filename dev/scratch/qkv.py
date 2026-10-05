from safetensors import safe_open
import collections

def keys_of(path, n=0):
    with safe_open(path, framework="pt") as f:
        ks = list(f.keys())
        tot = 0
        dt = collections.Counter()
        for k in ks:
            t = f.get_slice(k)
            dt[t.get_dtype()] += 1
            tot += t.get_shape() and int(__import__("math").prod(t.get_shape()))
        print("==", path)
        print("   n keys", len(ks), "dtypes", dict(dt))
        pref = collections.Counter(k.split(".")[0] for k in ks)
        print("   top prefixes", dict(pref))
        for k in ks[:12]:
            print("   ", k, f.get_slice(k).get_shape())
        if n:
            for k in ks[-n:]:
                print("   tail", k, f.get_slice(k).get_shape())
    return ks

ks = keys_of("infra/qwen_3vl_4b.safetensors", n=6)

# layer count for the text model
layers = sorted({k.split(".")[2] for k in ks if k.startswith("model.layers.") or k.startswith("language_model.layers.")}, key=lambda x: int(x) if x.isdigit() else 0)
print("   text layers:", layers[:5], "..." if len(layers) > 8 else "", f"count={len(layers)}")
