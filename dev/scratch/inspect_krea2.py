"""One-off inspection of the Krea2 Int8Extended checkpoint layout."""
from collections import Counter
from safetensors import safe_open

path = "models/krea2/krea2_Int8Extended.safetensors"
with safe_open(path, framework="pt") as f:
    keys = f.keys()
    print("total keys:", len(keys))
    dtypes = Counter()
    for k in keys:
        dtypes[f.get_slice(k).get_dtype()] += 1
    print("dtypes:", dict(dtypes))

    # show a sample of keys per family
    fams = Counter()
    for k in keys:
        top = k.split(".")[0]
        fams[top] += 1
    print("top-level:", dict(fams))

    # find keys NOT of dtype INT8 (U8 is the marker string tensor)
    non_int8 = [k for k in keys if f.get_slice(k).get_dtype() not in ("INT8", "U8")]
    print("\nnon-int8 keys ({}):".format(len(non_int8)))
    for k in non_int8:
        print(" ", k, f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())

    # the U8 marker: print its text
    for k in keys:
        if f.get_slice(k).get_dtype() == "U8":
            t = f.get_tensor(k)
            print("\nU8 tensor:", k, "len", len(t))
            try:
                print(t.tobytes().decode("utf-8")[:400])
            except Exception:
                print(t[:100])
            break

    # shapes of main block keys
    for k in ("model.0.attn.wq.weight", "model.0.mlp.up.weight", "model.first.weight"):
        print(k, f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())
