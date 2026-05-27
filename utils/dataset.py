"""
数据集处理工具模块
"""
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import yaml


def split_dataset(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        data_dir: 数据目录，包含 images 和 labels 文件夹
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        (train_files, val_files, test_files) 文件列表
    """
    random.seed(seed)
    
    image_dir = Path(data_dir) / "images"
    image_files = list(image_dir.glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    random.shuffle(image_files)
    
    total = len(image_files)
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)
    
    train_files = image_files[:train_num]
    val_files = image_files[train_num:train_num + val_num]
    test_files = image_files[train_num + val_num:]
    
    return train_files, val_files, test_files


def organize_dataset(
    src_dir: str,
    dst_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> None:
    """
    组织数据集为YOLO格式
    
    Args:
        src_dir: 源数据目录
        dst_dir: 目标数据目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    dst_path = Path(dst_dir)
    splits = ['train', 'val', 'test']
    
    # 创建目录结构
    for split in splits:
        (dst_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 分割数据集
    train_files, val_files, test_files = split_dataset(src_dir, train_ratio, val_ratio)
    
    # 复制文件
    file_groups = [
        (train_files, 'train'),
        (val_files, 'val'),
        (test_files, 'test')
    ]
    
    for files, split in file_groups:
        for img_path in files:
            # 复制图片
            dst_img = dst_path / 'images' / split / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # 复制对应的标签文件
            label_path = Path(src_dir) / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = dst_path / 'labels' / split / f"{img_path.stem}.txt"
                shutil.copy2(label_path, dst_label)
    
    print(f"数据集组织完成:")
    print(f"  训练集: {len(train_files)} 张")
    print(f"  验证集: {len(val_files)} 张")
    print(f"  测试集: {len(test_files)} 张")


def create_data_yaml(
    data_dir: str,
    class_names: List[str],
    output_path: str = "data.yaml"
) -> None:
    """
    创建数据集配置文件
    
    Args:
        data_dir: 数据目录
        class_names: 类别名称列表
        output_path: 输出文件路径
    """
    data = {
        'path': os.path.abspath(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    
    print(f"数据集配置文件已创建: {output_path}")
