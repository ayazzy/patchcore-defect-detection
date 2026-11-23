---
title: PatchCore Defect Detection
emoji: 🔩
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# PatchCore visual defect detection

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yazimov/patchcore-defect-detection)

**[Try the live demo →](https://huggingface.co/spaces/yazimov/patchcore-defect-detection)** Upload a metal-nut image or click an example to get a verdict, an anomaly score and a heatmap.

Detect defects on manufactured parts **without ever labelling a defect**. The model learns what *good* parts look like, then flags and localises anything that deviates, with a heatmap of where.

Built to understand how industrial vision inspection can be deployed quickly on a real production line, where a labelled catalogue of every possible defect never exists but good parts come off the line all day.

## How it works

1. **Learn normality.** A Wide ResNet-50 backbone (ImageNet-pretrained, frozen) extracts mid-level patch features from defect-free images only. PatchCore keeps a coreset-subsampled **memory bank** of those patch embeddings, a compact description of "normal".
2. **Score by distance.** At inference, each patch of a new image is compared with its nearest neighbour in the memory bank. The largest patch distance gives the image-level anomaly score, and the per-patch distances, upsampled, give the anomaly heatmap.
3. **Decide.** The score is thresholded into Normal / Anomalous.

No defect labels are used in training, which means a new part type can go from zero to working detection with a few hundred good images and no annotation effort.

## What this demo does

Upload a metal-nut image (or click an example) and the app returns three things:

- **Verdict** — Normal / Anomalous
- **Anomaly score** — a single number (higher = more abnormal)
- **Heatmap overlay** — red = where the part deviates most from normal

## Scope (deliberately tight)

- **One category** — MVTec AD `metal_nut`. No multi-category support.
- **CPU inference** — fine for a single-image demo; not tuned for throughput.
- **Not benchmarked** — the test step prints sanity metrics, but the goal is to
  demonstrate the approach, not chase state-of-the-art AUROC.

## Tech

- [`anomalib`](https://github.com/openvinotoolkit/anomalib) `2.5.0` (Intel) — PatchCore model
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset, `metal_nut` category
- Gradio UI, deployable to Hugging Face Spaces

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train on good parts only, then export a Torch model (auto-downloads the dataset).
python train.py --category metal_nut

# train.py exports to model/weights/torch/model.pt — copy it where the app expects:
cp model/weights/torch/model.pt model/model.pt

# Launch the demo.
python app.py
```

The model file `model/model.pt` is committed so the Hugging Face Space can run
inference without retraining.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Load MVTec AD, fit PatchCore on good parts, test, export a Torch model. |
| `app.py` | Gradio app: verdict + anomaly score + heatmap overlay. |
| `model/model.pt` | The exported model the Space loads. |
| `examples/` | A few good and defective sample images for the UI. |

## What I built

The end-to-end pipeline (training script, model export, inference app and heatmap overlay) on top of Intel's open-source [anomalib](https://github.com/openvinotoolkit/anomalib) PatchCore implementation, assembled over a weekend with AI coding tools. The value here is the deployment-oriented framing and a working, inspectable demo, not novel model research.

## Credit

- Model & training library: [anomalib](https://github.com/openvinotoolkit/anomalib) (Apache-2.0)
- PatchCore: Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022)
- Dataset: MVTec AD (research/non-commercial licence, see MVTec's terms)
