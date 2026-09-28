# RAM master-copy plan — one shared RAM copy per parked model

## Context

`ModelRegistry` parks at most one model per GPU, but each parked model is a
**separate fully-materialized CPU backend**. Parking the same model on both
GPUs (the normal end-state after every 2-GPU batch) keeps **two** full copies
in RAM (measured: Flux Q4 25.4 GB, SDXL 13.8 GB, FP8 Flux 34.2 GB, Z-Image
~12.6 GB). Weights are immutable after load (inference only; LoRAs are
additive deltas re-applied per generate), so a single shared copy per model
is always valid.

Goal: keep **one RAM master per parked model**, shared by every GPU that
parked it. Parking an extra copy of an already-parked model then frees the
GPU copy instead of creating a second CPU copy. Idle RAM for a same-model
batch drops by one parked-model size (Flux Q4: 25.4 → 12.7 GB).

No change to the park policy itself (still ≤ 1 parked model per GPU; a new
model's load drops the GPU's previous park).

## Design

### Objects and invariants

- `ModelRegistry._masters: dict[str, _Master]` — one entry per model that has
  a RAM master. Guarded by `self._meta_lock` (all dict mutations), owner GPU
  operations run while the owner's `state.lock` is held (callers of the
  park/load functions already do).
- `_Master(backend: ModelBackend, owners: set[int])` — `backend` is
  **CPU-resident and LoRA-free**; `owners` = GPU ids whose park slot points
  at this master. A master exists iff ≥ 1 owner.
- Per-GPU state keeps `state.parked_id` (both mechanisms). The legacy
  `state.parked` (backend object) is kept **only** for the non-cloneable
  fallback (see below).
- Lock ordering: `state.lock` → `_meta_lock`, never the reverse (existing
  code never holds `_meta_lock` while acquiring a `state.lock`; `unload_all`
  and `backend_for`/`run` release it first — unchanged).
- **Ownership races are safe by construction:** every "decide + mutate
  owners/dict" step is one `_meta_lock` critical section. Cloning a master
  whose owner count just hit zero is safe because the local `master`
  reference keeps the backend alive even after another thread removes it
  from the dict mid-clone. A concurrent park that finds no master simply
  creates a new one from its own copy (first-park branch).

### Master lifecycle

| Event | Action |
|---|---|
| GPU parks model M, no master M | strip LoRAs → `unload()` (GPU→CPU) → create master, `owners={gpu}` |
| GPU parks model M, master M exists | strip LoRAs → **drop the GPU copy** (no CPU copy) → add gpu to `owners` |
| GPU loads M, its park slot is M, **sole** owner | pop master from dict → `master.backend.load()` back to this GPU (old behaviour, zero extra copy) |
| GPU loads M, its park slot is M, **shared** | discard gpu from `owners` → `master.backend.clone(device)` to this GPU; master stays in RAM |
| GPU loads M, master M exists (no park slot) | `master.backend.clone(device)`; no ownership change |
| GPU loads M, no master M | fresh load from disk (current behaviour) |
| GPU's park slot released (replaced by another park, or its model loaded) | discard gpu from `owners`; when the last owner goes, `del master` (RAM returned) |

### Non-cloneable fallback (`supports_clone = False`)

NF4-quantized SDXL UNets contain `bitsandbytes` `Linear4bit` modules whose
quant state does not `deepcopy` safely. Such backends keep today's exact
behaviour: `state.parked` holds the backend object, sole-owner move on
restore, drop on replace, **no master is ever created**. One
class-level flag, checked at the two branch points (park, load fast path).
Flux/Z-Image/FP8/FP16-SDXL are cloneable (plain tensors / `nn.Parameter`
GGUF bytes / fp8 tensors — all deepcopy + `.to()` cleanly).

### Clone semantics

`ModelBackend.clone(device) -> ModelBackend` — returns an independent copy
with weights on `device`, starting **LoRA-free, empty caches**, fresh
locks/cancel state. The caller (registry) has already LoRA-stripped the
master, so per-generate LoRA application on the clone is unchanged.
Tokenizers are shared (stateless in inference). Text-encoder placement
mirrors each backend's `load()` co-residency decision.

## Changes

### 1. `src/imagagen/backends.py`

- Add to `ModelBackend`:
  ```python
  supports_clone: bool = True

  def clone(self, device) -> "ModelBackend":
      """Independent copy of this backend with its weights on ``device``.

      Source must be a parked master (CPU-resident, no LoRAs). The clone
      starts LoRA-free with empty caches; callers re-apply LoRAs per
      generate as usual.
      """
      raise NotImplementedError  # MockBackend never parks; real backends override
  ```

### 2. `src/imagagen/sdxl_backend.py`

In `__init__`, after the NF4 branch:
`self.supports_clone = config.unet_quantization != "nf4" if config else True`

```python
def clone(self, device) -> "SdxlBackend":
    import copy
    import torch
    b = SdxlBackend.__new__(SdxlBackend)
    b.model_id = self.model_id
    b.name = self.name
    b.model_hash = self.model_hash
    b.device = device
    b.config = self.config
    b._lock = threading.Lock()
    b._cancel = False
    b._step = 0
    b._steps_total = 1
    b._active_loras = None
    b._enc_cache = OrderedDict()
    b.vae_dtype = self.vae_dtype
    with _PARAM_INIT_LOCK:
        b.pipe = copy.deepcopy(self.pipe)  # CPU->CPU copy of all weights
        b.pipe.to(device, silence_dtype_warnings=True)
        b.text_encoder_device = (
            "cpu" if (self.config and self.config.text_encoder_device == "cpu")
            else device
        )
        b.pipe.text_encoder.to(b.text_encoder_device)
        if getattr(b.pipe, "text_encoder_2", None) is not None:
            b.pipe.text_encoder_2.to(b.text_encoder_device)
    torch.cuda.empty_cache()
    return b
```
Deepcopy preserves everything the live pipeline already configured: VAE
dtype cast, `enable_slicing()`, `AttnProcessor2_0`, (no LoRA adapters —
master is stripped).

### 3. `src/imagagen/flux_backend.py`

```python
def clone(self, device) -> "FluxBackend":
    import copy
    import torch
    b = FluxBackend.__new__(FluxBackend)
    b.model_id = self.model_id
    b.name = self.name
    b.model_hash = self.model_hash
    b.device = device
    b.config = self.config
    b.model_path = self.model_path
    b._lock = threading.Lock()
    b._cancel = False
    b._step = 0
    b._steps_total = 1
    b._active_loras = None
    b._lora_cache = {}
    b._enc_cache = OrderedDict()
    b.supports_distilled = self.supports_distilled
    b.te_device = self.te_device
    b._encoder_bytes = self._encoder_bytes
    with _PARAM_INIT_LOCK:
        b.transformer = copy.deepcopy(self.transformer).to(device)
        b.vae = copy.deepcopy(self.vae).to(device)
        b.clip = copy.deepcopy(self.clip)
        b.t5 = copy.deepcopy(self.t5)
        if (self.te_device != "cpu"
                and self._encoder_bytes + (1 << 30)
                <= torch.cuda.mem_get_info(device)[0]):
            b._text_to(device)  # same co-residency decision as load()
    b.clip_tok = self.clip_tok  # stateless tokenizers shared
    b.t5_tok = self.t5_tok
    torch.cuda.empty_cache()
    return b
```
Deepcopy preserves `GgufParam` bake flags / `Fp8Linear` scale params /
QuantLinear structure; no `dummy` params exist post-load (deleted at init).

### 4. `src/imagagen/zimage_backend.py`

Same pattern:
```python
def clone(self, device) -> "ZImageBackend":
    import copy
    import torch
    b = ZImageBackend.__new__(ZImageBackend)
    b.model_id = self.model_id
    b.name = self.name
    b.model_hash = self.model_hash
    b.device = device
    b.config = self.config
    b.model_path = self.model_path
    b._lock = threading.Lock()
    b._cancel = False
    b._step = 0
    b._steps_total = 1
    b._active_loras = None
    b._lora_cache = {}
    b._enc_cache = OrderedDict()
    b.te_device = self.te_device
    b._encoder_bytes = self._encoder_bytes
    with _PARAM_INIT_LOCK:
        b.transformer = copy.deepcopy(self.transformer).to(device)
        b.vae = copy.deepcopy(self.vae).to(device)
        b.te = copy.deepcopy(self.te)
        if (self.te_device != "cpu"
                and self._encoder_bytes + (1 << 30)
                <= torch.cuda.mem_get_info(device)[0]):
            b._text_to(device)
    b.te_tok = self.te_tok
    torch.cuda.empty_cache()
    return b
```

### 5. `src/imagagen/models.py`

Add:
```python
@dataclass
class _Master:
    """Single RAM-resident copy of a model shared by every GPU that parked
    it. ``backend`` is CPU-resident and LoRA-free; ``owners`` are the GPU
    ids whose park slot points at this master (>=1 while in _masters)."""
    backend: ModelBackend
    owners: set[int]
```

In `ModelRegistry.__init__`: `self._masters: dict[str, _Master] = {}`
(dict mutations guarded by `_meta_lock`).

Helpers:
```python
def _gc_and_empty(self, gpu: int) -> None:
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
    except Exception:
        pass

def _strip_loras(self, backend: ModelBackend) -> None:
    """LoRA-strip a backend before it becomes a shared master (the master
    always holds base weights; clones re-apply LoRAs per generate)."""
    with backend._lock, _PARAM_INIT_LOCK:
        backend._unload_loras()
        backend._active_loras = None

def _release_owner(self, model_id: str, gpu: int) -> None:
    """Drop ``gpu`` from the master's owners; free the master when it is
    the last owner. Caller holds the owner GPU's state.lock."""
    with self._meta_lock:
        master = self._masters.get(model_id)
        if master is None:
            return
        master.owners.discard(gpu)
        if master.owners:
            return
        del self._masters[model_id]
    log.info("model master: %s freed from RAM (last owner gone)", model_id)
    del master
```
(import `_PARAM_INIT_LOCK` from `.backends`.)

Rewrite `_park_state_locked` (caller holds `state.lock`):
```python
def _park_state_locked(self, state: _GpuState) -> None:
    # Release whatever this GPU previously parked (either mechanism).
    if state.parked is not None:
        log.info("model park: dropping %s parked on GPU %d",
                 state.parked_id, state.gpu)
        state.parked = None
        state.parked_id = None
    elif state.parked_id is not None:
        log.info("model park: releasing %s parked on GPU %d",
                 state.parked_id, state.gpu)
        self._release_owner(state.parked_id, state.gpu)
        state.parked_id = None
    if state.backend is None:
        self._gc_and_empty(state.gpu)
        return
    if not state.backend.can_park:
        return  # leave resident
    model_id = state.model_id
    backend = state.backend
    state.backend = None
    state.model_id = None
    if not backend.supports_clone:
        # Legacy: this copy is parked for this GPU only.
        backend.unload()
        state.parked = backend
        state.parked_id = model_id
        log.info("model park: %s on GPU %d -> RAM", model_id, state.gpu)
        self._gc_and_empty(state.gpu)
        return
    with self._meta_lock:
        master = self._masters.get(model_id)
    if master is None:
        # First GPU to park this model: this copy becomes the shared master.
        self._strip_loras(backend)
        backend.unload()  # GPU -> CPU
        with self._meta_lock:
            self._masters[model_id] = _Master(backend=backend, owners={state.gpu})
        log.info("model park: %s on GPU %d -> RAM (new master)",
                 model_id, state.gpu)
    else:
        # Master exists: free this GPU copy, the shared one stays.
        self._strip_loras(backend)
        del backend
        with self._meta_lock:
            master.owners.add(state.gpu)
        log.info("model park: %s on GPU %d -> RAM (shared master, GPU copy freed)",
                 model_id, state.gpu)
    state.parked_id = model_id
    self._gc_and_empty(state.gpu)
```

Rewrite `_load_backend_locked` (caller holds `state.lock`); keep the
existing start/done logging + `emit` around each path:
```python
def _load_backend_locked(self, state: _GpuState, model_id: str,
                         loras: list[str] | None = None) -> ModelBackend:
    if state.backend is not None and state.model_id == model_id:
        return state.backend
    info = self.get(model_id)
    if info is None:
        raise ValueError(f"unknown model: {model_id}")
    device = torch.device(f"cuda:{state.gpu}")
    self.emit({"type": "status",
               "message": f"loading {info.name} on GPU {state.gpu}"})
    started = time.monotonic()

    # Fast path 1: legacy per-GPU parked copy (non-cloneable backend).
    if state.parked is not None and state.parked_id == model_id:
        restoring = state.parked
        state.parked = None
        state.parked_id = None
        self._park_state_locked(state)  # park the current resident, if any
        state.backend = restoring
        state.model_id = model_id
        how = "parked restore"
        restoring.load()
    else:
        # Fast path 2: shared master owned by this GPU.
        if state.parked_id is not None and state.parked_id == model_id:
            with self._meta_lock:
                master = self._masters.get(model_id)
                if master is not None and master.owners <= {state.gpu}:
                    self._masters.pop(model_id, None)  # sole owner: take it
                    sole = True
                else:
                    if master is not None:
                        master.owners.discard(state.gpu)
                    sole = False
            state.parked_id = None
            self._park_state_locked(state)
            if sole:
                state.backend = master.backend
                state.model_id = model_id
                how = "parked restore"
                state.backend.load()
            else:
                # Shared: local ref keeps the master alive even if a
                # concurrent _release_owner drops the dict entry mid-clone.
                state.backend = master.backend.clone(device)
                state.model_id = model_id
                how = "clone from shared master"
        else:
            # Slow path: drop old park, park resident, then clone or fresh.
            self._park_state_locked(state)
            with self._meta_lock:
                master = self._masters.get(model_id)
            if master is not None and master.backend.supports_clone:
                state.backend = master.backend.clone(device)
                how = "clone from shared master"
            else:
                state.backend = self._new_backend(info, device)
                how = "fresh"
            state.model_id = model_id

    log.info("model load done: %s on GPU %d (%.1fs, %s)",
             info.name, state.gpu, time.monotonic() - started, how)
    self.emit({"type": "status",
               "message": f"loaded {info.name} on GPU {state.gpu}"})
    with self._meta_lock:
        self._current_id = model_id
    return state.backend
```

Note on fast path 2 when `master is None` (a concurrent sole-owner restore
took it between our slot check and the critical section): we fall into the
clone branch with `master is None` — that must be handled: treat
`master is None` as "no master" → **fresh load** instead. Implement the
critical section as:
```python
            with self._meta_lock:
                master = self._masters.get(model_id)
                sole = master is not None and master.owners <= {state.gpu}
                if sole:
                    self._masters.pop(model_id, None)
                elif master is not None:
                    master.owners.discard(state.gpu)
            ...
            if sole:
                ...
            elif master is not None:
                clone branch
            else:
                state.backend = self._new_backend(info, device)  # fresh
                how = "fresh"
```
(`master.owners <= {state.gpu}` with non-empty `owners` ⇒ sole owner is us;
the invariant owners ≥ 1 while in the dict holds.)

Update the module docstring of `models.py` (parking semantics) and the
`_GpuState` docstring (`parked` = legacy per-GPU copy; `parked_id` = master
ownership marker).

`unload_all` is unchanged (it calls `_park_state_locked`).

## Tests

New file `tests/unit/test_registry_masters.py`. Fake backend (no CUDA —
`clone`/`load`/`unload` only record the target device, weights stay
CPU tensors):

```python
class TinyBackend(ModelBackend):
    can_park = True

    def __init__(self, model_id, device, cloneable=True):
        self.model_id = model_id
        self.name = model_id
        self.model_hash = 0
        self.device = device
        self.weight = torch.nn.Parameter(torch.zeros(16))
        self._lock = threading.Lock()
        self._cancel = False
        self._step = 0
        self._steps_total = 1
        self.supports_clone = cloneable
        self._active_loras = None

    def generate(self, request, progress=None): ...  # unused
    def cancel(self): self._cancel = True
    def unload(self): self.weight.data = self.weight.data.to("cpu")
    def load(self): pass  # device already recorded

    def clone(self, device):
        b = TinyBackend(self.model_id, device, cloneable=self.supports_clone)
        b.weight = torch.nn.Parameter(self.weight.data.clone())
        return b

    def _unload_loras(self): self._active_loras = None
```

Construct `ModelRegistry(Config(root=tmp_path), gpus=[0, 1])`, monkeypatch
`reg._new_backend` to return fresh `TinyBackend`s (unique `weight`
storages). `_park_state_locked`'s `torch.cuda` calls are guarded by
`is_available()` — on a CPU-only test env `empty_cache` is skipped.

Scenarios (assert on `reg._masters`, live objects, `state.backend/parked_id`):

1. **Same model both GPUs** — load A on GPU 0, load A on GPU 1 (via
   `reg.backend_for("A", gpu=n)`), `reg.unload_all()` ⇒ exactly one master,
   `owners == {0, 1}`, both `state.backend is None`; the master's weight
   storage is the original GPU-0 one; a live-object scan finds exactly one
   `TinyBackend` for A.
2. **Clone from shared master** — from state of test 1, `backend_for("A",
   gpu=1)` ⇒ returned object is *not* the master, `weight.data_ptr()`
   differs, master still present with `owners == {0}` (GPU 1 had no park
   slot, so no ownership was added; the clone is owned by no one). Re-park
   (`unload_all`) ⇒ owners `{0, 1}` again, single master.
3. **Sole-owner restore moves, no clone** — from state of test 1,
   `backend_for("A", gpu=0)` ⇒ returned object *is* the master's backend
   (same `data_ptr()`), `_masters` empty.
4. **Load while shared master exists, no park slot** — GPU 0 has A
   resident (cloned), master A parked on GPU 1; `backend_for("A", gpu=0)`
   is the early-return (resident) case — instead test: GPU 0 resident B,
   parked A (master, owners {0,1}), request A on GPU 0 ⇒ resident B parks
   (new master B, owners {0}), A is cloned for GPU 0, master A owners {1}.
5. **Last-owner release frees the master** — from state of test 4, load C on
   GPU 0 (park slot A, master A owners {1} → released, stays) then unload
   B/replace on GPU 1 until A's owners are empty ⇒ `A` key gone, master
   object collectable (gc + id-scan or `del` + reference count check).
6. **unload_all same model both GPUs** ⇒ one master, owners {0,1}.
7. **can_park=False** ⇒ never parked, no master (regression).
8. **supports_clone=False** ⇒ legacy path: `state.parked` holds the object,
   no `_masters` entry; two GPUs parking the same model ⇒ two independent
   parked copies; restore moves the same object back.
9. **LoRA strip on master creation** — set `backend._active_loras = ["x"]`
   before parking ⇒ master's `_active_loras is None`.

Run: `.venv/bin/pytest tests/unit/test_registry_masters.py
tests/unit/test_queue.py tests/unit/test_backends.py
tests/unit/test_model_discovery.py`. Ruff on touched files.

## Live verification (2 GPUs)

New probe `dev/probe_master_copy.py` (modeled on `dev/probe_park_ram.py`,
reusing its `rss_gb()` / `cpu_torch_gb()` helpers):

```
reg = ModelRegistry(cfg, gpus=[0, 1])
1. backend_for("agFluxQ4KS_v21", 0); backend_for("agFluxQ4KS_v21", 1)
   reg.unload_all()
   -> expect log "shared master, GPU copy freed", cpuTorch ≈ 12.7 GB
      (one parked Flux), NOT 25.4 GB
2. backend_for("agFluxQ4KS_v21", 1)  -> log "clone from shared master"
      (GPU 0 sole-owner case is step: also try backend_for(..., 0)
      right after a solo park -> "parked restore" on the same object)
3. backend_for("epicrealismXL11.0", 0); backend_for("epicrealismXL11.0", 1)
   reg.unload_all()
   -> expect masters: Flux still owned by GPU 1, SDXL new master (GPU 0
      copy becomes master, GPU 1 copy freed); cpuTorch ≈ 12.7 + 6.9 GB
4. backend_for("agFluxQ4KS_v21", 1)  -> sole-owner move (same object)
5. backend_for("intorealism_zitV40", 1) -> Flux master freed (last owner),
      cpuTorch drops by ~12.7 GB
```
Print RSS + `cpu_torch_gb()` + `len(reg._masters)` after each step.
(Existing ramdebug instrumentation is stashed and intentionally not
re-applied — it would conflict with the rewritten `models.py`; the probe
carries its own heavy scan.)

## Out of scope (follow-ups, not part of this change)

- RAM budget: dropping instead of parking a model when total parked bytes
  exceed a threshold.
- Not parking when the next queued job is a different model.
- NF4 clone support (needs bnb-safe deep copy).
- Avoiding the transient CPU copy inside `clone()` (module-by-module
  GPU move) — current deepcopy cost is one model-size for a short window.
