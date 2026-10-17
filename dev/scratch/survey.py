import collections

from safetensors import safe_open


def survey(path):
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        print("==", path, "n keys:", len(keys))
        dt = collections.Counter()
        for k in keys:
            dt[f.get_slice(k).get_dtype()] += 1
        print("  dtypes:", dict(dt))
        fams = collections.Counter()
        for k in keys:
            fams[k.split(".")[0]] += 1
        print("  top:", dict(fams))
        for k in keys[:16]:
            print("   ", k, f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())
        mid = [k for k in keys if ".5." in k][:10]
        for k in mid:
            print("   mid", k, f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())

survey("infra/qwen_3vl_4b.safetensors")
survey("infra/ae.safetensors")
survey("loras/krea2/rustic-ambience.safetensors")
