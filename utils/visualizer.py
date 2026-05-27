"""
可视化工具模块
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


# 默认颜色映射
COLORS = [
    (255, 0, 0),    # 红
    (0, 255, 0),    # 绿
    (0, 0, 255),    # 蓝
    (255, 255, 0),  # 黄
    (255, 0, 255),  # 紫
    (0, 255, 255),  # 青
    (128, 0, 0),    # 深红
    (0, 128, 0),    # 深绿
    (0, 0, 128),    # 深蓝
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str],
    color_map: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像 (H, W, 3)
        boxes: 边界框坐标 (N, 4) [x1, y1, x2, y2]
        scores: 置信度分数 (N,)
        class_ids: 类别ID (N,)
        class_names: 类别名称列表
        color_map: 颜色映射
        thickness: 边框线宽
        
    Returns:
        绘制后的图像
    """
    if color_map is None:
        color_map = COLORS
    
    result = image.copy()
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = color_map[class_id % len(color_map)]
        
        # 绘制边框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签
        label = f"{class_names[class_id]}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        # 标签背景
        cv2.rectangle(
            result,
            (x1, y1 - text_h - 10),
            (x1 + text_w, y1),
            color,
            -1
        )
        
        # 标签文字
        cv2.putText(
            result,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    return result


def plot_training_results(results_csv: str, save_path: Optional[str] = None) -> None:
    """
    绘制训练结果曲线
    
    Args:
        results_csv: 训练结果CSV文件路径
        save_path: 保存路径
    """
    import pandas as pd
    
    df = pd.read_csv(results_csv)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Results', fontsize=16)
    
    # 损失曲线
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='box')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='cls')
    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='dfl')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 验证损失
    axes[0, 1].plot(df['epoch'], df['val/box_loss'], label='box')
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='cls')
    axes[0, 1].plot(df['epoch'], df['val/dfl_loss'], label='dfl')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Val Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # mAP
    axes[0, 2].plot(df['epoch'], df['metrics/mAP50'], label='mAP50')
    axes[0, 2].plot(df['epoch'], df['metrics/mAP50-95'], label='mAP50-95')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('mAP')
    axes[0, 2].set_title('mAP')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 精确率
    axes[1, 0].plot(df['epoch'], df['metrics/precision'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].grid(True)
    
    # 召回率
    axes[1, 1].plot(df['epoch'], df['metrics/recall'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].grid(True)
    
    # 学习率
    axes[1, 2].plot(df['epoch'], df['lr/pg0'], label='lr0')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练结果图已保存: {save_path}")
    else:
        plt.show()


def visualize_batch(
    images: List[np.ndarray],
    batch_size: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    可视化批量图像
    
    Args:
        images: 图像列表
        batch_size: 每行显示数量
        save_path: 保存路径
    """
    n = len(images)
    rows = (n + batch_size - 1) // batch_size
    
    fig, axes = plt.subplots(rows, batch_size, figsize=(batch_size * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img in enumerate(images):
        row = i // batch_size
        col = i % batch_size
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(n, rows * batch_size):
        row = i // batch_size
        col = i % batch_size
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
