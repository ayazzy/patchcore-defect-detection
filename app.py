"""Gradio app: inspect a part image and get a defect verdict + heatmap.

The model was trained only on defect-free metal nuts (see train.py). Here we load
the exported Torch model and, for any uploaded image, return:
  - a verdict (Normal / Anomalous)
  - an anomaly score (higher = more abnormal)
  - a heatmap overlay showing *where* the model thinks the part is abnormal.
"""

import os
from pathlib import Path

# anomalib's TorchInferencer refuses to unpickle a model unless this is set,
# since loading a .pt executes arbitrary code. We trust this file — we trained and
# exported it ourselves in train.py — so we opt in here rather than per-environment.
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import gradio as gr
import matplotlib.cm as cm
import numpy as np
from PIL import Image

from anomalib.deploy import TorchInferencer

MODEL_PATH = "model/model.pt"
EXAMPLES_DIR = Path("examples")

# Load the model once at startup. CPU is plenty for a single-image demo.
inferencer = TorchInferencer(path=MODEL_PATH, device="cpu")


def make_overlay(original: Image.Image, anomaly_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a jet-colored anomaly heatmap over the original image.

    The raw anomaly_map is in arbitrary score units, so we min-max normalize it to
    0-1 *for visualization only*. After the jet colormap: red = most abnormal
    region, blue = most normal. We resize the heatmap back to the original image
    size and alpha-blend.
    """
    amap = np.array(anomaly_map).squeeze().astype(np.float32)

    # Min-max normalize to 0-1. Guard against a flat map (all-equal) which would
    # divide by zero and produce a uniform image.
    amin, amax = float(amap.min()), float(amap.max())
    if amax - amin < 1e-8:
        norm = np.zeros_like(amap)
    else:
        norm = (amap - amin) / (amax - amin)

    # Apply jet colormap -> RGB (drop alpha channel from RGBA), to 0-255 uint8.
    heat_rgb = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb).resize(original.size, Image.BILINEAR)

    base = original.convert("RGB")
    return Image.blend(base, heat_img, alpha)


def inspect(image: Image.Image):
    """Run inference on one image and return (overlay, result_text)."""
    if image is None:
        return None, "Upload an image first."

    original = image.convert("RGB")
    result = inferencer.predict(original)

    # squeeze() pulls scalars out of the single-image batch tensors.
    score = float(np.array(result.pred_score).squeeze())
    is_anomalous = bool(np.array(result.pred_label).squeeze())
    overlay = make_overlay(original, result.anomaly_map)

    verdict = "ANOMALOUS" if is_anomalous else "NORMAL"
    text = (
        f"Verdict: {verdict}\n"
        f"Anomaly score: {score:.4f}  (higher = more abnormal)\n\n"
        f"Red regions in the heatmap are where the part deviates most from the "
        f"defect-free examples the model learned from."
    )
    return overlay, text


DESCRIPTION = """
# Matta "Sentry"-style Defect Detection — Weekend POC

A small, honest proof-of-concept of **vision-based anomaly detection that learns
from defect-free parts only** — the core idea behind Matta's *Sentry* product.

**How it works:** the model (PatchCore, via Intel's `anomalib`) was trained on
images of *good* metal nuts only — **no defect labels**. It builds a memory bank of
what "normal" looks like. Anything that doesn't match scores as anomalous, and the
heatmap shows *where*.

**Why this matters on a factory floor:** you rarely have a labelled library of every
possible defect, but you always have good parts coming off the line. Learning
normality instead of enumerating defects means a new part can be deployed fast, with
very little data and no labelling effort.

Upload a metal-nut image (or click an example) and press **Inspect**.
""".strip()


def build_examples():
    if not EXAMPLES_DIR.exists():
        return []
    return [[str(p)] for p in sorted(EXAMPLES_DIR.glob("*.png"))]


with gr.Blocks(title="Matta-style Defect Detection POC") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Part image")
            inspect_btn = gr.Button("Inspect", variant="primary")
            gr.Examples(examples=build_examples(), inputs=input_image, label="Examples")
        with gr.Column():
            output_overlay = gr.Image(type="pil", label="Anomaly heatmap (red = abnormal)")
            output_text = gr.Textbox(label="Result", lines=6)

    inspect_btn.click(fn=inspect, inputs=input_image, outputs=[output_overlay, output_text])


if __name__ == "__main__":
    demo.launch()
