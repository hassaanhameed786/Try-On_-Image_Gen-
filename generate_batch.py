"""SKU batch generator: products.csv x template -> ComfyUI -> outputs/<sku>/ + QA report.

    python scripts/generate_batch.py --template studio_front --variants 3
Writes outputs/report.csv with a dE colour score per image so you can reject off-colour garments fast.
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
from comfy_client import ComfyClient
from garment_qa import check

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="studio_front")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--workflow", default=str(ROOT / "workflows/01_txt2img_sdxl_api.json"))
    ap.add_argument("--threshold", type=float, default=15.0)
    a = ap.parse_args()

    template = json.loads((ROOT / "prompts/templates.json").read_text())[a.template]
    wf = ComfyClient.load_workflow(a.workflow)
    client = ComfyClient()
    rows = []
    with open(ROOT / "prompts/products.csv", newline="") as f:
        for p in csv.DictReader(f):
            prompt = template.format(**p)
            for v in range(a.variants):
                seed = random.randint(0, 2**31 - 1)
                job = client.patch(wf, prompt=prompt, seed=seed, prefix=f"fashion/{p['sku']}")
                for img in client.run(job, ROOT / "outputs" / p["sku"]):
                    res = check(str(img), p["color_hex"], threshold=a.threshold)
                    rows.append({"sku": p["sku"], "seed": seed, **res})
                    print(rows[-1])
    if rows:
        with open(ROOT / "outputs/report.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
