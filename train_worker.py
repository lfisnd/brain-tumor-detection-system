"""
训练工作进程 - 在独立进程中运行训练，支持被主进程终止
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ultralytics import YOLO


def train(config_json):
    """执行训练"""
    config = json.loads(config_json)
    
    task = config.get("task", "classification")
    model_name = config.get("model", "yolo11n" if task == "classification" else "yolo11m-seg")
    epochs = config.get("epochs", 100)
    batch = config.get("batch", 4)
    imgsz = config.get("imgsz", 640)
    lr0 = config.get("lr0", 0.01)
    lrf = config.get("lrf", 0.01)
    momentum = config.get("momentum", 0.937)
    weight_decay = config.get("weight_decay", 0.0005)
    warmup_epochs = config.get("warmup_epochs", 3.0)
    patience = config.get("patience", 50)
    save_period = config.get("save_period", 10)
    device = config.get("device", "0")
    workers = config.get("workers", 8)
    optimizer = config.get("optimizer", "SGD")
    seed = config.get("seed", 0)
    cos_lr = config.get("cos_lr", False)
    freeze = config.get("freeze", 0)
    
    project_root = Path(__file__).parent.resolve()
    
    # 优先从 models 文件夹加载模型，如果不存在则从 weights/pretrained 加载
    models_dir = project_root / "models"
    weights_dir = project_root / "weights" / "pretrained"
    
    if task == "classification":
        # 分类模型
        model_filename = f"{model_name}.pt"
        model_path = models_dir / model_filename
        if not model_path.exists():
            model_path = weights_dir / model_filename
        data_path = project_root / "datasets" / "classification" / "icon.yaml"
        project_name = "classification"
    else:
        # 分割模型
        if not model_name.endswith("-seg"):
            model_name = f"{model_name}-seg"
        model_filename = f"{model_name}.pt"
        model_path = models_dir / model_filename
        if not model_path.exists():
            model_path = weights_dir / model_filename
        data_path = project_root / "datasets" / "segmentation" / "data.yaml"
        project_name = "segmentation"
    
    print(f"[INFO] 正在加载模型: {model_path}", flush=True)
    print(f"[INFO] 数据集路径: {data_path}", flush=True)
    print(f"[INFO] 任务类型: {task}", flush=True)
    
    model = YOLO(str(model_path))
    
    results_dir = project_root / "results" / project_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] 开始训练...", flush=True)
    
    args = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "save_period": save_period,
        "device": device,
        "workers": workers,
        "optimizer": optimizer,
        "seed": seed,
        "cos_lr": cos_lr,
        "freeze": freeze,
        "project": str(results_dir),
        "name": "train",
        "exist_ok": True,
        "verbose": True,
    }
    
    if task == "segmentation":
        args["overlap_mask"] = True
        args["mask_ratio"] = 4
    
    try:
        results = model.train(**args)
        print("[INFO] 训练完成!", flush=True)
        print(f"[RESULT] {results_dir / 'train'}", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="JSON格式的训练配置")
    args = parser.parse_args()
    
    success = train(args.config)
    sys.exit(0 if success else 1)
