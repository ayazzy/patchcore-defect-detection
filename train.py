"""Train a PatchCore anomaly-detection model on one MVTec AD category.

This is the "learn from good parts only" step: PatchCore builds a memory bank of
features extracted from defect-free training images. At inference time, anything
that does not look like those good parts scores as anomalous. No defect labels
are used during training.

Usage:
    python train.py --category metal_nut
"""

import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PatchCore on an MVTec AD category.")
    parser.add_argument("--category", default="metal_nut", help="MVTec AD category to train on.")
    parser.add_argument("--root", default="./datasets/MVTecAD", help="Where to download/read the dataset.")
    parser.add_argument("--export_dir", default="./model", help="Where to export the trained Torch model.")
    args = parser.parse_args()

    # 1. Data: the datamodule auto-downloads MVTec AD on first run. Training only
    #    ever sees the "good" images in train/; defective images live in test/.
    datamodule = MVTecAD(
        root=args.root,
        category=args.category,
        train_batch_size=8,
        eval_batch_size=8,
    )

    # 2. Model: PatchCore with a Wide ResNet-50 backbone. We pull mid-level
    #    features (layer2/layer3) and keep a 10% coreset of them as the memory bank.
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    # 3. Fit: build the memory bank from the good training images.
    engine = Engine()
    print(f"\n[train] Fitting PatchCore on category '{args.category}' (good parts only)...")
    engine.fit(model=model, datamodule=datamodule)

    # 4. Test: sanity-check metrics against the labelled test set (image/pixel AUROC etc.).
    #    These numbers are just a smoke test — the point of the demo is the approach.
    print("\n[train] Running test for sanity metrics...")
    engine.test(model=model, datamodule=datamodule)

    # 5. Export: write a self-contained Torch model for CPU inference in the app.
    print("\n[train] Exporting Torch model...")
    engine.export(model=model, export_type=ExportType.TORCH, export_root=args.export_dir)

    exported = Path(args.export_dir) / "weights" / "torch" / "model.pt"
    target = Path("model") / "model.pt"
    print("\n" + "=" * 70)
    print(f"[train] Done. Model exported to: {exported}")
    print(f"[train] REMINDER: copy it to where app.py expects it:")
    print(f"        cp {exported} {target}")
    print("=" * 70)


if __name__ == "__main__":
    main()
