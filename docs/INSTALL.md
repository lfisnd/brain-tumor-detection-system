# Installation

## 1. Create a Python environment

```bash
conda create -n brain-tumor-detection python=3.10 -y
conda activate brain-tumor-detection
```

## 2. Install PyTorch

Choose one option from the official PyTorch installation matrix for your GPU/CPU environment. Example for CUDA 12.1:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

CPU-only example:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

## 3. Install project dependencies

```bash
pip install -r requirements.txt
```

## 4. Verify installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
python -c "from PyQt5.QtWidgets import QApplication; print('pyqt ok')"
```

## 5. Run the GUI

```bash
python tumor_detection_app.py
```

Training GUI:

```bash
python train_gui.py
```

## Notes

- Put local model weights under `weights/` or `models/` as described in `docs/MODELS.md`.
- Put datasets under `datasets/` as described in `docs/DATASETS.md`.
- AI assistant features are optional. Configure `DEEPSEEK_API_KEY` only on your local machine.
