# ComfyUI Fashion Pipeline (cloud-first POC)

A small, working proof of concept for a **men's apparel brand** that wants consistent AI fashion imagery
generated **entirely in the cloud** and operated without touching a local GPU.

It covers the three jobs an apparel team actually needs first:

| # | Workflow | Use case | Key knob |
|---|----------|----------|----------|
| 1 | `01_txt2img_sdxl_api.json` | New campaign / lookbook shots from a prompt | prompt, seed |
| 2 | `02_img2img_reference_sdxl_api.json` | Re-style a reference photo (new colourway, setting) keeping pose/composition | `denoise` (0.4 subtle → 0.7 big change) |
| 3 | `03_inpaint_garment_sdxl_api.json` | Swap **only the garment**; model, pose and background stay untouched | mask + `denoise` |

Plus the parts that make it usable day to day:

- `scripts/comfy_client.py` — tiny client that patches any workflow by node title, queues it on a remote ComfyUI, and downloads results.
- `scripts/generate_batch.py` — SKU sheet (`prompts/products.csv`) × prompt template → N variants per SKU.
- `scripts/garment_qa.py` — **garment colour-accuracy check** (CIELAB distance vs. the brand hex) so off-colour generations are flagged before anyone sees them.
- `cloud/setup_cloud.sh` — one-shot bootstrap for any rented GPU (RunPod, Vast.ai, Lambda…).

Workflows use **only ComfyUI core nodes**, so nothing breaks when custom-node packs update.

## Quick start

```bash
# 1. On a cloud GPU box (>=16 GB VRAM; 24 GB recommended):
bash cloud/setup_cloud.sh          # installs ComfyUI, downloads SDXL base, serves on :8188

# 2. On your laptop:
pip install -r requirements.txt
export COMFY_URL="https://<your-pod>-8188.proxy.runpod.net"   # or http://<ip>:8188

# 3. Generate + QA a batch
python scripts/generate_batch.py --template studio_front --variants 3
# -> outputs/<sku>/*.png and outputs/report.csv (dE colour score + PASS/REVIEW)
```

Workflows are stored in ComfyUI **API format** (what the scripts submit). To edit one visually, rebuild/modify it in the
ComfyUI editor, then export with *Dev mode → Save (API format)* and drop the file into `workflows/`.
Nodes are addressed by `_meta.title` (`POSITIVE_PROMPT`, `SAMPLER`, `INPUT_IMAGE`, ...), so keep those titles.

### Workflow 3 (garment swap) input
Erase the garment in a PNG (transparent area = region to regenerate), upload it, then run with
`prompt="navy blue crew neck t-shirt, ..."`. Mask edges are grown by 8 px to hide seams.

## Garment accuracy: how it is handled
1. Prompt templates keep garment wording consistent per SKU.
2. Inpainting (workflow 3) changes only the garment region instead of re-rolling the whole image.
3. `garment_qa.py` scores colour drift against the official hex; anything above the threshold is marked `REVIEW`.
   Pass `--mask` for precise scoring; without it a torso-region heuristic is used.

## Roadmap (not implemented in this POC)
- **Consistent model identity:** IPAdapter / InstantID face + body reference, then a small LoRA trained on the brand's chosen model.
- **Exact product fidelity:** garment reference conditioning / try-on nodes, LoRA per hero product, logo-preservation pass.
- **Video:** image-to-video (e.g. Wan 2.x / LTX) for 5–10 s social clips and try-on motion, same client + batch structure.
- **Hand-off UI:** a simple Streamlit/Gradio front-end so non-technical staff can run batches.

## Sample outputs
<!-- Add 3-4 images from examples/ here after your first cloud run -->

## Notes
- Default checkpoint is `sd_xl_base_1.0.safetensors`; swap it via `ComfyClient.patch(..., checkpoint="your_model.safetensors")`.
- Respect model and image licences when using outputs commercially.
