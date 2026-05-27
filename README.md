# Brain Tumor Detection System

基于 YOLO11 的脑肿瘤检测、分类和实例分割桌面系统，提供检测 GUI、训练 GUI、NII/MRI 可视化、批量检测和可选 AI 辅助解读功能。

> 重要提示：本项目仅用于学习、研究和演示，不能替代医生诊断或临床决策。

## 功能

- 脑肿瘤分类检测：Glioma、Meningioma、No Tumor、Pituitary
- 脑肿瘤实例分割：输出肿瘤区域掩膜
- 双模型协同：分类模型与分割模型联合展示
- 支持常见图片格式和 NII 医学影像文件
- PyQt5 图形界面：检测、批量处理、训练参数配置和结果查看
- 可选 AI 辅助解读：通过环境变量配置 OpenAI-compatible API

## 项目结构

```text
.
├── ablation_experiment/        # 消融实验脚本和示例结果
├── datasets/                   # 数据集配置和目录占位，不提交真实医学数据
├── docs/                       # 安装、数据集、模型权重说明
├── models/                     # 预训练模型占位，不提交 .pt 权重
├── utils/                      # 数据集和可视化工具
├── weights/                    # 训练权重占位，不提交 .pt 权重
├── train_gui.py                # 训练可视化界面
├── train_worker.py             # 训练子进程入口
├── tumor_detection_app.py      # 主检测 GUI
├── requirements.txt            # pip 依赖
├── pyproject.toml              # Python 项目元数据
├── .env.example                # AI API 环境变量示例
└── .gitignore                  # 公开仓库忽略规则
```

## 快速开始

1. 创建环境并安装依赖：

```bash
conda create -n brain-tumor-detection python=3.10 -y
conda activate brain-tumor-detection
pip install -r requirements.txt
```

2. 放置模型权重：

```text
weights/classification/weights/best.pt
weights/segmentation/weights/best.pt
weights/classification+segmentation/train/weights/best.pt
```

也可以把 YOLO 预训练权重放到：

```text
models/yolo11n.pt
models/yolo11m-seg.pt
```

3. 放置数据集：

```text
datasets/classification/Train/
datasets/classification/Val/
datasets/segmentation/images/train/
datasets/segmentation/images/val/
datasets/segmentation/labels/train/
datasets/segmentation/labels/val/
```

4. 启动程序：

```bash
python tumor_detection_app.py
```

训练界面：

```bash
python train_gui.py
```

## AI API 配置

仓库不包含任何私有 API Key。需要使用 AI 辅助解读时，在本机设置环境变量：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

Windows PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
python tumor_detection_app.py
```

更多安装、模型和数据集说明见 [docs/INSTALL.md](docs/INSTALL.md)、[docs/MODELS.md](docs/MODELS.md)、[docs/DATASETS.md](docs/DATASETS.md)。

## 公开仓库说明

为保护隐私和控制仓库大小，以下内容默认不会提交到 GitHub：

- 医学影像数据、标签文件和缓存
- 训练得到的 `.pt`、`.onnx`、`.engine` 等模型文件
- 训练输出目录 `runs/`、`results/`、`outputs/`
- `.env` 等本地密钥配置
- 本地 vendored 的 `ultralytics/` 副本

克隆仓库后按文档把数据集和权重放回对应目录即可保持项目可用。
