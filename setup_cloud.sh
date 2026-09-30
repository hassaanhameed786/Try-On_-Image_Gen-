#!/usr/bin/env bash
# Bootstrap ComfyUI on any rented GPU box (RunPod, Vast.ai, Lambda, etc.).
# Target: Ubuntu + NVIDIA GPU with >= 16 GB VRAM (24 GB recommended; needed later for video).
set -euo pipefail
WORKDIR="${WORKDIR:-/workspace}"
cd "$WORKDIR"
[ -d ComfyUI ] || git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
mkdir -p models/checkpoints
CKPT=models/checkpoints/sd_xl_base_1.0.safetensors
if [ ! -f "$CKPT" ]; then
  wget -O "$CKPT" \
    https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
fi
echo "Starting ComfyUI on :8188 (expose this port in your provider's UI)"
python main.py --listen 0.0.0.0 --port 8188
