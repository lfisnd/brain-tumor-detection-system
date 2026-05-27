# Model Weights

Model weights are intentionally excluded from Git because they are large and may contain private training artifacts.

## Detection GUI weights

Place trained weights here:

```text
weights/classification/weights/best.pt
weights/segmentation/weights/best.pt
weights/classification+segmentation/train/weights/best.pt
```

The main GUI checks these paths when it starts.

## Training GUI pretrained weights

Place YOLO pretrained weights here when you want local offline training:

```text
models/yolo11n.pt
models/yolo11s.pt
models/yolo11m.pt
models/yolo11l.pt
models/yolo11x.pt
models/yolo11n-seg.pt
models/yolo11s-seg.pt
models/yolo11m-seg.pt
models/yolo11l-seg.pt
models/yolo11x-seg.pt
```

`train_worker.py` first checks `models/`, then `weights/pretrained/`. If no local file exists, Ultralytics may try to download the requested model.

## Recommended release flow

For public sharing, upload large weights to GitHub Releases, cloud storage, or a model registry, then link them from the README instead of committing them to Git.
