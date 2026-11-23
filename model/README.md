# Trained model

`model.pt` is the exported PatchCore model (Wide ResNet-50 backbone, trained on defect-free MVTec AD `metal_nut` images only), stored with Git LFS so the demo runs without retraining.

Regenerate it with:

```bash
python train.py --category metal_nut
cp model/weights/torch/model.pt model/model.pt
```
