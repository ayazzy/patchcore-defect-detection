---
title: Matta-style Defect Detection POC
emoji: 🔩
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Matta "Sentry"-style Defect Detection — Weekend POC

A small, honest proof-of-concept of the approach behind [Matta's](https://www.matta.ai)
**Sentry** product: **vision-based defect detection that learns from defect-free
parts only**, instead of labelling defects.

> **Honest framing.** This is a weekend learning exercise I built to understand
> Matta's problem, not a claim of original engineering. I assembled it with the help
> of AI tools and I understand how every part works — it wires together Intel's
> open-source [`anomalib`](https://github.com/openvinotoolkit/anomalib) library
> (PatchCore model) and the public MVTec AD dataset into a tiny Gradio demo. The
> interesting part here is the *approach*, not benchmark numbers or novel code.

## The idea: learn normality, don't label defects

Conventional supervised defect detection needs a labelled library of every defect
type — expensive to collect and always incomplete, because the next defect is the
one you haven't seen.

The anomaly-detection approach flips this:

1. **Train on good parts only.** PatchCore extracts mid-level features from
   defect-free images and stores a compressed "memory bank" of what normal looks like.
   No defect labels are used.
2. **Score by deviation.** At inference, any region of a new part that doesn't match
   the memory bank scores as anomalous — and you get a heatmap of *where*.

**Why this is what enables fast, low-data factory deployment:** you rarely have a
labelled catalogue of every possible defect, but you always have good parts coming
off the line. Learning normality means you can stand up detection for a new part
with very little data and zero labelling effort — which is exactly what makes it
deployable on a real production line.

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

## Credit

- Model & training: [anomalib](https://github.com/openvinotoolkit/anomalib) (Apache-2.0)
- Dataset: MVTec AD (research/non-commercial license — see MVTec's terms)
- Inspiration: [Matta](https://www.matta.ai) and their Sentry product
