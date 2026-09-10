Diagnosis complete. Here's what I found:

**Current state of the container**
- Host NVIDIA **kernel module is loaded and visible** (driver `580.173.02`, via `/proc/driver/nvidia/version`)
- But **no `/dev/nvidia*` devices** are passed into the container
- **No NVIDIA userspace driver libraries** (no `libnvidia-ml.so`, no `nvidia-smi`)
- I confirmed the container's **device cgroup blocks opening GPU devices** — I could `mknod /dev/nvidia0 c 195 0` (we have `CAP_MKNOD` via sudo), but `open()` returns `EPERM`. So this can't be fixed from inside; the container was started without GPU access.

**What you need to do (host side)**

1. **Install the NVIDIA Container Toolkit** (if not already installed):
   ```
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
   (The GPG key URL changes with toolkit version — the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) has the current one.)

2. **Recreate the dev container with GPU access.** Docker needs to be told at `docker run` time, so this means editing whatever defines the container:
   - If it's a devcontainer: add `"runArgs": ["--gpus=all"]` to `.devcontainer.json`
   - If compose: add
     ```yaml
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: all
               capabilities: [gpu]
     ```
   - Or plain `docker run`: `--gpus all`

   If you want **only GPU 0** inside the container, use `--gpus "device=0"` instead of `--gpus all`.

**A note on "GPU 0 hidden from the X windows server"**

That's a host-side display concern (presumably X runs on GPU 1, or the host sets `CUDA_VISIBLE_DEVICES` for AI apps). The toolkit passes *both* GPUs into the container by default, so inside we'll just export `CUDA_VISIBLE_DEVICES=0` — which is also the mechanism concept.md anticipated. No special host trickery needed as long as `nvidia-smi` shows both GPUs once the container is up.

**Driver 580 compatibility**

580.x supports CUDA 13.x; the toolkit mounts the host driver into the container, so we should **not** install any driver inside the image — just the CUDA runtime bits (or nothing, since modern PyTorch pip wheels bundle their CUDA runtime). I'll use pip-installed torch (cu12x wheels) + `diffusers`/`k_diffusion`.

**After you've done that**, I'll verify with:
```
nvidia-smi
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Let me know once the container is recreated with `--gpus` and I'll re-check.