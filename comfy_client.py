"""Minimal ComfyUI client: patch a workflow by node title, queue it, wait, download results.

Works against any remote ComfyUI (RunPod proxy URL, Vast.ai, your own VM).
    export COMFY_URL="https://<pod-id>-8188.proxy.runpod.net"
    export COMFY_API_TOKEN="..."   # optional, sent as Bearer header if your proxy needs it
"""
from __future__ import annotations
import copy, json, os, time, uuid
from pathlib import Path
import requests


class ComfyClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: int = 600):
        self.base = (base_url or os.environ.get("COMFY_URL", "http://127.0.0.1:8188")).rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self.s = requests.Session()
        token = token or os.environ.get("COMFY_API_TOKEN")
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"

    # ---------- workflow helpers ----------
    @staticmethod
    def load_workflow(path: str | Path) -> dict:
        return json.loads(Path(path).read_text())

    @staticmethod
    def find(workflow: dict, title: str) -> str:
        for nid, node in workflow.items():
            if node.get("_meta", {}).get("title") == title:
                return nid
        raise KeyError(f"No node titled {title!r} in workflow")

    def patch(self, workflow: dict, *, prompt=None, negative=None, seed=None, steps=None, cfg=None,
              denoise=None, width=None, height=None, image=None, checkpoint=None, prefix=None) -> dict:
        wf = copy.deepcopy(workflow)
        def setv(title, key, value):
            if value is not None:
                wf[self.find(wf, title)]["inputs"][key] = value
        setv("POSITIVE_PROMPT", "text", prompt)
        setv("NEGATIVE_PROMPT", "text", negative)
        setv("CHECKPOINT", "ckpt_name", checkpoint)
        setv("SAMPLER", "seed", seed); setv("SAMPLER", "steps", steps)
        setv("SAMPLER", "cfg", cfg);   setv("SAMPLER", "denoise", denoise)
        setv("SAVE", "filename_prefix", prefix)
        setv("INPUT_IMAGE", "image", image)
        if width or height:
            try:
                n = wf[self.find(wf, "LATENT_SIZE")]["inputs"]
                n["width"] = width or n["width"]; n["height"] = height or n["height"]
            except KeyError:
                pass  # img2img/inpaint resize nodes are fixed at 832x1216 in the JSON
        return wf

    # ---------- server I/O ----------
    def upload_image(self, path: str | Path) -> str:
        p = Path(path)
        with p.open("rb") as f:
            r = self.s.post(f"{self.base}/upload/image", files={"image": (p.name, f)},
                            data={"overwrite": "true"}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["name"]

    def queue(self, workflow: dict) -> str:
        r = self.s.post(f"{self.base}/prompt", json={"prompt": workflow, "client_id": self.client_id},
                        timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {r.text}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 1.5) -> dict:
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            r = self.s.get(f"{self.base}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            hist = r.json()
            if prompt_id in hist:
                return hist[prompt_id]
            time.sleep(poll)
        raise TimeoutError(f"Prompt {prompt_id} did not finish in {self.timeout}s")

    def download_outputs(self, history: dict, out_dir: str | Path) -> list[Path]:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for node_out in history.get("outputs", {}).values():
            for img in node_out.get("images", []):
                r = self.s.get(f"{self.base}/view", params={
                    "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output")}, timeout=self.timeout)
                r.raise_for_status()
                dest = out_dir / img["filename"]
                dest.write_bytes(r.content); saved.append(dest)
        return saved

    def run(self, workflow: dict, out_dir: str | Path) -> list[Path]:
        pid = self.queue(workflow)
        return self.download_outputs(self.wait(pid), out_dir)
