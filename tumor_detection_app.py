import sys
import os
from pathlib import Path
from datetime import datetime
import json
import tempfile
import shutil

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QGroupBox, QSplitter, QMessageBox, QFrame, QTabWidget,
    QListWidget, QListWidgetItem, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QLineEdit, QStackedWidget, QStackedLayout, QScrollArea,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy,
    QMenu, QAction, QToolButton, QStatusBar, QGridLayout, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRunnable, QThreadPool, QSize, QTimer, QPointF, QObject
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon, QCursor, QTransform, QPainter

import cv2
import numpy as np
try:
    from scipy import ndimage
except ImportError:
    ndimage = None
from PIL import Image

# DeepSeek API配置
AI_API_URL_ENV = "DEEPSEEK_API_URL"
AI_API_KEY_ENV = "DEEPSEEK_API_KEY"
AI_MODEL_ENV = "DEEPSEEK_MODEL"


def get_ai_api_url():
    return os.getenv(AI_API_URL_ENV, "https://api.deepseek.com/v1")


def get_ai_api_key():
    return os.getenv(AI_API_KEY_ENV) or os.getenv("OPENAI_API_KEY") or ""


def get_ai_model_name():
    return os.getenv(AI_MODEL_ENV, "deepseek-chat")


def get_ai_config_error_message():
    return f"AI API 未配置：请先设置环境变量 {AI_API_KEY_ENV}。"

# 添加 ultralytics 到路径
sys.path.insert(0, str(Path(__file__).parent))
from ultralytics import YOLO

# 尝试导入nibabel处理NII文件
try:
    import nibabel as nib
    NII_AVAILABLE = True
except ImportError:
    NII_AVAILABLE = False

# 尝试导入pyvistaqt进行3D可视化
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PV_AVAILABLE = True
except ImportError:
    PV_AVAILABLE = False


# 主题颜色配置
class Theme:
    # 主色调
    PRIMARY = "#165DFF"  # 医疗深蓝
    PRIMARY_HOVER = "#1147CC"
    PRIMARY_LIGHT = "#E8F0FF"

    # 辅助色
    SUCCESS = "#00B42A"  # 成功绿
    SUCCESS_LIGHT = "#E8FFEA"
    WARNING = "#F53F3F"  # 警告红
    WARNING_LIGHT = "#FFE8E8"
    NEUTRAL = "#86909C"  # 中性灰

    # 浅色模式
    LIGHT_BG = "#F7F8FA"
    LIGHT_CARD = "#FFFFFF"
    LIGHT_TEXT = "#1D2129"
    LIGHT_TEXT_SECONDARY = "#4E5969"
    LIGHT_BORDER = "#E5E6EB"
    LIGHT_HOVER = "#F2F3F5"

    # 深色模式
    DARK_BG = "#1A1D21"
    DARK_CARD = "#2E3440"
    DARK_TEXT = "#E8E9EA"
    DARK_TEXT_SECONDARY = "#A6A9AD"
    DARK_BORDER = "#3D4149"
    DARK_HOVER = "#3A3F4A"

    # 阴影
    SHADOW = "0 2px 12px rgba(0,0,0,0.08)"
    SHADOW_HOVER = "0 4px 20px rgba(0,0,0,0.12)"


class NII3DViewer(QDialog):
    """NII文件3D查看器 - 使用PyVista，支持MPR、分割叠加、体积计算"""
    
    def __init__(self, nii_img, seg_img=None, detection_boxes=None, parent=None):
        """
        初始化3D查看器
        
        Args:
            nii_img: NII图像数据
            seg_img: 分割掩码图像（可选）
            detection_boxes: 检测框列表 [{'x1', 'y1', 'x2', 'y2', 'cls_name', 'conf'}]（可选）
            parent: 父窗口
        """
        super().__init__(parent)
        self.setWindowTitle("3D MRI查看器 - PyVista")
        self.setMinimumSize(1200, 800)
        self.nii_img = nii_img
        self.seg_img = seg_img
        self.detection_boxes = detection_boxes or []
        
        # 获取图像数据
        self.volume_data = nii_img.get_fdata()
        self.shape = self.volume_data.shape
        
        # 计算体素大小（用于体积计算）
        self.voxel_size = self._get_voxel_size()
        
        # 创建布局
        self.setup_ui()
        
        # 加载数据
        self.load_volume()
        
        # 如果有分割数据，加载分割
        if seg_img is not None:
            self.load_segmentation()
        
        # 显示统计信息
        self.update_statistics()
    
    def _get_voxel_size(self):
        """获取体素大小（mm³）- 使用行列式计算，更严谨"""
        try:
            affine = self.nii_img.affine
            # 计算体素体积：使用仿射矩阵左上角3x3子矩阵的行列式绝对值
            # 这种方法兼容所有情况（包括存在旋转的影像）
            voxel_size = abs(np.linalg.det(affine[:3, :3]))
            return voxel_size
        except:
            return 1.0  # 默认1mm³
    
    def setup_ui(self):
        """设置UI界面 - 包含3D视图、MPR视图和统计信息"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：3D视图
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # 创建PyVista交互器
        self.plotter = QtInteractor(left_widget)
        left_layout.addWidget(self.plotter, 1)
        
        main_splitter.addWidget(left_widget)
        
        # 右侧：MPR视图和统计信息
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        
        # MPR视图组
        mpr_group = QFrame()
        mpr_group.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #E5E6EB;
                border-radius: 8px;
            }
        """)
        mpr_layout = QVBoxLayout(mpr_group)
        mpr_layout.setContentsMargins(12, 12, 12, 12)
        
        mpr_title = QLabel("多平面重建 (MPR)")
        mpr_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #1D2129;")
        mpr_layout.addWidget(mpr_title)
        
        # 三个MPR视图
        mpr_grid = QGridLayout()
        mpr_grid.setSpacing(4)
        
        # 横断面 (Axial)
        self.mpr_axial = QLabel("横断面")
        self.mpr_axial.setStyleSheet("""
            QLabel {
                background-color: #F7F8FA;
                border: 1px solid #E5E6EB;
                min-height: 150px;
            }
        """)
        self.mpr_axial.setAlignment(Qt.AlignCenter)
        self.mpr_axial.setMinimumSize(150, 150)
        mpr_grid.addWidget(self.mpr_axial, 0, 0)
        
        # 冠状面 (Coronal)
        self.mpr_coronal = QLabel("冠状面")
        self.mpr_coronal.setStyleSheet("""
            QLabel {
                background-color: #F7F8FA;
                border: 1px solid #E5E6EB;
                min-height: 150px;
            }
        """)
        self.mpr_coronal.setAlignment(Qt.AlignCenter)
        self.mpr_coronal.setMinimumSize(150, 150)
        mpr_grid.addWidget(self.mpr_coronal, 0, 1)
        
        # 矢状面 (Sagittal)
        self.mpr_sagittal = QLabel("矢状面")
        self.mpr_sagittal.setStyleSheet("""
            QLabel {
                background-color: #F7F8FA;
                border: 1px solid #E5E6EB;
                min-height: 150px;
            }
        """)
        self.mpr_sagittal.setAlignment(Qt.AlignCenter)
        self.mpr_sagittal.setMinimumSize(150, 150)
        mpr_grid.addWidget(self.mpr_sagittal, 1, 0)
        
        # 切片滑块
        slider_widget = QWidget()
        slider_layout = QVBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0, 4, 0, 0)
        
        self.slice_slider = QSpinBox()
        self.slice_slider.setRange(0, self.shape[2] - 1)
        self.slice_slider.setValue(self.shape[2] // 2)
        self.slice_slider.setPrefix("切片: ")
        self.slice_slider.valueChanged.connect(self.update_mpr_slices)
        slider_layout.addWidget(self.slice_slider)
        
        mpr_grid.addWidget(slider_widget, 1, 1)
        
        mpr_layout.addLayout(mpr_grid)
        right_layout.addWidget(mpr_group)
        
        # 统计信息组 - 使用滚动区域
        stats_group = QFrame()
        stats_group.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #E5E6EB;
                border-radius: 8px;
            }
        """)
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setSpacing(8)
        
        stats_title = QLabel("统计信息")
        stats_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #1D2129;")
        stats_layout.addWidget(stats_title)
        
        # 创建滚动区域
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stats_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        stats_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #F2F3F5;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #C9CDD4;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #86909C;
            }
        """)
        
        # 创建统计信息标签容器
        stats_content = QWidget()
        stats_content_layout = QVBoxLayout(stats_content)
        stats_content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_stats = QLabel("加载中...")
        self.lbl_stats.setStyleSheet("font-size: 14px; color: #4E5969; line-height: 1.6;")
        self.lbl_stats.setWordWrap(True)
        stats_content_layout.addWidget(self.lbl_stats)
        
        stats_scroll.setWidget(stats_content)
        stats_layout.addWidget(stats_scroll)
        
        right_layout.addWidget(stats_group, 1)  # 添加拉伸因子，让统计区域可以扩展
        
        # 检测框信息
        if self.detection_boxes:
            boxes_group = QFrame()
            boxes_group.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border: 1px solid #E5E6EB;
                    border-radius: 8px;
                }
            """)
            boxes_layout = QVBoxLayout(boxes_group)
            boxes_layout.setContentsMargins(12, 12, 12, 12)
            
            boxes_title = QLabel(f"检测框信息 ({len(self.detection_boxes)}个)")
            boxes_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #1D2129;")
            boxes_layout.addWidget(boxes_title)
            
            boxes_text = ""
            for i, box in enumerate(self.detection_boxes[:5], 1):  # 最多显示5个
                cls_name = box.get('cls_name', 'Unknown')
                conf = box.get('conf', 0)
                boxes_text += f"{i}. {cls_name} (置信度: {conf:.2%})\n"
            
            lbl_boxes = QLabel(boxes_text)
            lbl_boxes.setStyleSheet("font-size: 12px; color: #4E5969;")
            boxes_layout.addWidget(lbl_boxes)
            
            right_layout.addWidget(boxes_group)
        
        right_layout.addStretch()
        main_splitter.addWidget(right_widget)
        
        # 设置分割比例
        main_splitter.setSizes([800, 400])
        layout.addWidget(main_splitter, 1)
        
        # 创建控制面板
        control_widget = QWidget()
        control_widget.setStyleSheet("""
            QWidget {
                background-color: #F7F8FA;
                border-top: 1px solid #E5E6EB;
            }
        """)
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(16, 12, 16, 12)
        control_layout.setSpacing(12)
        
        # 标题
        title = QLabel("3D 体绘制控制")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #1D2129;")
        control_layout.addWidget(title)
        
        control_layout.addSpacing(20)
        
        # 显示模式切换
        self.btn_toggle_mpr = QPushButton("显示/隐藏 MPR")
        self.btn_toggle_mpr.setCheckable(True)
        self.btn_toggle_mpr.setChecked(True)
        self.btn_toggle_mpr.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1147CC;
            }
            QPushButton:checked {
                background-color: #00B42A;
            }
        """)
        self.btn_toggle_mpr.clicked.connect(self.toggle_mpr)
        control_layout.addWidget(self.btn_toggle_mpr)
        
        # 不透明度滑块
        opacity_label = QLabel("不透明度:")
        opacity_label.setStyleSheet("font-size: 13px; color: #4E5969;")
        control_layout.addWidget(opacity_label)
        
        self.opacity_slider = QSpinBox()
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setSuffix("%")
        self.opacity_slider.setStyleSheet("""
            QSpinBox {
                border: 1px solid #E5E6EB;
                border-radius: 4px;
                padding: 4px 8px;
                background: white;
            }
        """)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        control_layout.addWidget(self.opacity_slider)
        
        control_layout.addSpacing(20)
        
        # 按钮组
        btn_reset = QPushButton("重置视角")
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1147CC;
            }
        """)
        btn_reset.clicked.connect(self.plotter.reset_camera)
        control_layout.addWidget(btn_reset)
        
        btn_screenshot = QPushButton("保存截图")
        btn_screenshot.setStyleSheet("""
            QPushButton {
                background-color: #00B42A;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #009926;
            }
        """)
        btn_screenshot.clicked.connect(self.save_screenshot)
        control_layout.addWidget(btn_screenshot)
        
        control_layout.addStretch()
        
        layout.addWidget(control_widget)
        
        # 延迟初始化MPR视图，确保UI已完全创建
        QTimer.singleShot(100, lambda: self.update_mpr_slices(self.shape[2] // 2))
    
    def toggle_mpr(self, checked):
        """切换MPR视图显示/隐藏"""
        # 找到右侧部件（主分割器的第二个子部件）
        splitter = self.findChild(QSplitter)
        if splitter:
            right_widget = splitter.widget(1)
            if right_widget:
                right_widget.setVisible(checked)
    
    def update_mpr_slices(self, slice_idx):
        """更新MPR三个正交切面的显示 - 支持融合视图和同步更新"""
        try:
            print(f"[MPR] 更新切片: {slice_idx}, 数据形状: {self.shape}")
            
            # 限制切片范围
            axial_idx = max(0, min(slice_idx, self.shape[2] - 1))
            # 根据Z轴切片位置，同步计算Y和X轴位置（保持三个平面交于同一点）
            coronal_idx = max(0, min(slice_idx, self.shape[1] - 1))
            sagittal_idx = max(0, min(slice_idx, self.shape[0] - 1))
            
            # 保存当前切片索引用于其他操作
            self.current_mpr_slice_idx = axial_idx
            
            # 获取 QLabel 的当前大小
            axial_size = self.mpr_axial.size()
            coronal_size = self.mpr_coronal.size()
            sagittal_size = self.mpr_sagittal.size()
            
            print(f"[MPR] 切片索引: 横断面(Z)={axial_idx}, 冠状面(Y)={coronal_idx}, 矢状面(X)={sagittal_idx}")
            
            # 获取分割数据（如果有）
            seg_data = None
            if self.seg_img is not None:
                seg_data = self.seg_img.get_fdata()
            
            # 横断面 (Axial) - Z轴切片
            axial_slice = self.volume_data[:, :, axial_idx]
            if seg_data is not None:
                seg_slice = seg_data[:, :, axial_idx]
                axial_slice = self._create_fusion_slice(axial_slice, seg_slice)
            print(f"[MPR] 横断面数据形状: {axial_slice.shape}")
            axial_qimg = self._numpy_to_qimage(axial_slice)
            if axial_qimg:
                pixmap = QPixmap.fromImage(axial_qimg)
                target_size = axial_size if axial_size.width() > 10 and axial_size.height() > 10 else QSize(150, 150)
                scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.mpr_axial.setPixmap(scaled_pixmap)
                print(f"[MPR] 横断面更新成功, 图像大小: {pixmap.width()}x{pixmap.height()}")
            else:
                print(f"[MPR] 横断面转换失败")
                self.mpr_axial.setText("横断面\n(转换失败)")
            
            # 冠状面 (Coronal) - Y轴切片（现在随滑块同步更新）
            coronal_slice = self.volume_data[:, coronal_idx, :]
            if seg_data is not None:
                seg_coronal = seg_data[:, coronal_idx, :]
                coronal_slice = self._create_fusion_slice(coronal_slice, seg_coronal)
            print(f"[MPR] 冠状面数据形状: {coronal_slice.shape}")
            coronal_qimg = self._numpy_to_qimage(coronal_slice)
            if coronal_qimg:
                pixmap = QPixmap.fromImage(coronal_qimg)
                target_size = coronal_size if coronal_size.width() > 10 and coronal_size.height() > 10 else QSize(150, 150)
                scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.mpr_coronal.setPixmap(scaled_pixmap)
                print(f"[MPR] 冠状面更新成功")
            else:
                print(f"[MPR] 冠状面转换失败")
                self.mpr_coronal.setText("冠状面\n(转换失败)")
            
            # 矢状面 (Sagittal) - X轴切片（现在随滑块同步更新）
            sagittal_slice = self.volume_data[sagittal_idx, :, :]
            if seg_data is not None:
                seg_sagittal = seg_data[sagittal_idx, :, :]
                sagittal_slice = self._create_fusion_slice(sagittal_slice, seg_sagittal)
            print(f"[MPR] 矢状面数据形状: {sagittal_slice.shape}")
            sagittal_qimg = self._numpy_to_qimage(sagittal_slice)
            if sagittal_qimg:
                pixmap = QPixmap.fromImage(sagittal_qimg)
                target_size = sagittal_size if sagittal_size.width() > 10 and sagittal_size.height() > 10 else QSize(150, 150)
                scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.mpr_sagittal.setPixmap(scaled_pixmap)
                print(f"[MPR] 矢状面更新成功")
            else:
                print(f"[MPR] 矢状面转换失败")
                self.mpr_sagittal.setText("矢状面\n(转换失败)")
            
        except Exception as e:
            print(f"[MPR] 更新切片失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_fusion_slice(self, mri_slice, seg_slice, alpha=0.5):
        """创建融合切片（MRI + 分割标签）"""
        try:
            print(f"[MPR] 创建融合切片: MRI形状={mri_slice.shape}, SEG形状={seg_slice.shape}")
            print(f"[MPR] 分割标签值: {np.unique(seg_slice)}")
            
            # 归一化MRI到0-255
            mri_min, mri_max = mri_slice.min(), mri_slice.max()
            if mri_max > mri_min:
                mri_norm = (mri_slice - mri_min) / (mri_max - mri_min) * 255
            else:
                mri_norm = np.zeros_like(mri_slice)
            
            # 创建RGB图像
            h, w = mri_slice.shape
            fusion = np.stack([mri_norm] * 3, axis=-1).astype(np.float32)
            
            # 定义分割颜色 - 使用更鲜艳的颜色
            colors = {
                1: [255, 0, 0],     # 坏死 - 纯红色
                2: [0, 255, 0],     # 水肿 - 纯绿色
                4: [0, 0, 255]      # 增强肿瘤 - 纯蓝色
            }
            
            # 融合分割标签
            has_color = False
            for label_id, color in colors.items():
                mask = (seg_slice == label_id)
                mask_count = mask.sum()
                if mask_count > 0:
                    print(f"[MPR] 标签 {label_id}: {mask_count} 像素")
                    has_color = True
                    for c in range(3):
                        fusion[:, :, c] = np.where(
                            mask,
                            (1 - alpha) * fusion[:, :, c] + alpha * color[c],
                            fusion[:, :, c]
                        )
            
            if has_color:
                print(f"[MPR] 融合切片创建成功，包含彩色分割标注")
            else:
                print(f"[MPR] 警告: 当前切片没有分割标签")
            
            return fusion.astype(np.uint8)
            
        except Exception as e:
            print(f"[MPR] 创建融合切片失败: {e}")
            import traceback
            traceback.print_exc()
            return mri_slice
    
    def _numpy_to_qimage(self, arr):
        """将numpy数组转换为QImage - 支持RGB彩色图像"""
        try:
            # 检查是否是RGB图像 (3通道)
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                # 已经是RGB图像，直接使用
                h, w = arr.shape[:2]
                rgb = arr.astype(np.uint8)
                
                # 创建QImage - 使用Format_RGB888
                bytes_per_line = 3 * w
                qimg = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
                return qimg.copy()
            
            # 处理2D灰度图像
            elif len(arr.shape) == 2:
                # 归一化到0-255
                arr_min = arr.min()
                arr_max = arr.max()
                if arr_max > arr_min:
                    arr = (arr - arr_min) / (arr_max - arr_min) * 255
                else:
                    arr = np.zeros_like(arr)
                
                arr = arr.astype(np.uint8)
                
                # 获取尺寸
                h, w = arr.shape
                
                # 转换为RGB格式 (灰度图复制3通道)
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgb[:, :, 0] = arr
                rgb[:, :, 1] = arr
                rgb[:, :, 2] = arr
                
                # 创建QImage - 使用Format_RGB888
                bytes_per_line = 3 * w
                qimg = QImage(rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
                
                # 复制数据确保内存安全
                return qimg.copy()
            
            else:
                print(f"[MPR] 不支持的数组形状: {arr.shape}")
                return None
                
        except Exception as e:
            print(f"[MPR] 转换图像失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_volume(self):
        """加载3D体数据"""
        try:
            # 归一化到0-1
            data_min = self.volume_data.min()
            data_max = self.volume_data.max()
            if data_max > data_min:
                normalized_data = (self.volume_data - data_min) / (data_max - data_min)
            else:
                normalized_data = self.volume_data
            
            # 创建体绘制 - 使用灰度颜色映射
            self.volume = self.plotter.add_volume(
                normalized_data,
                cmap="gray",
                opacity="linear",
                show_scalar_bar=False,
                name="MRI Volume"
            )
            
            # 设置背景色
            self.plotter.set_background("white")
            
            # 添加坐标轴指示器
            self.plotter.add_axes()
            
            # 添加三个正交切片平面（作为参考）
            self._add_slice_planes()
            
            # 重置相机
            self.plotter.reset_camera()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载3D数据失败: {str(e)}")
    
    def _add_slice_planes(self):
        """在3D视图中添加三个正交切片平面作为参考"""
        try:
            # 添加三个正交切片
            x_slice = self.shape[0] // 2
            y_slice = self.shape[1] // 2
            z_slice = self.shape[2] // 2
            
            # 创建切片网格
            # X平面（矢状面）
            self.plotter.add_mesh_slice_orthogonal(
                self.volume,
                normal='x',
                origin=(x_slice, 0, 0)
            )
            
        except Exception as e:
            print(f"添加切片平面失败: {e}")
    
    def load_segmentation(self):
        """加载分割掩码并在3D视图中叠加显示 - 融合视图风格"""
        try:
            if self.seg_img is None:
                print("[3D分割] 未提供分割图像")
                return
            
            seg_data = self.seg_img.get_fdata()
            
            print(f"[3D分割] 加载分割数据，形状: {seg_data.shape}")
            print(f"[3D分割] 分割标签值: {np.unique(seg_data)}")
            
            # 方法1: 创建融合风格的彩色体数据 (MRI + 分割标签)
            fusion_volume = self._create_fusion_volume(seg_data)
            
            if fusion_volume is not None:
                # 使用RGB颜色映射显示融合体数据
                self._add_rgb_volume(fusion_volume)
                print("[3D分割] 融合视图加载成功")
            
            # 方法2: 同时添加独立的分割标签体（更明显的颜色）
            self._add_separate_segmentation_volumes(seg_data)
            
            # 计算分割体积
            self.calculate_segmentation_volume(seg_data)
            
        except Exception as e:
            print(f"[3D分割] 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_rgb_volume(self, rgb_volume):
        """添加RGB彩色体数据到3D视图"""
        try:
            # PyVista处理RGB体数据需要使用特定方法
            # 方法：分别添加三个颜色通道作为独立的体绘制
            channel_configs = [
                (rgb_volume[:,:,:,0], "Red_Fusion", "hot"),       # 红色通道 - 使用hot映射更明显
                (rgb_volume[:,:,:,1], "Green_Fusion", "Greens"),  # 绿色通道
                (rgb_volume[:,:,:,2], "Blue_Fusion", "Blues")     # 蓝色通道
            ]
            
            for channel_data, name, cmap_name in channel_configs:
                # 只显示非零区域（有颜色的部分）
                if channel_data.max() > 0.1:
                    self.plotter.add_volume(
                        channel_data,
                        cmap=cmap_name,  # 使用matplotlib颜色映射名称
                        opacity="sigmoid",
                        show_scalar_bar=False,
                        name=name
                    )
                    
        except Exception as e:
            print(f"[3D分割] 添加RGB体数据失败: {e}")
    
    def _create_fusion_volume(self, seg_data):
        """创建融合风格的3D体数据（MRI + 彩色分割标签）"""
        try:
            # 归一化MRI数据到0-1
            mri_min = self.volume_data.min()
            mri_max = self.volume_data.max()
            if mri_max > mri_min:
                mri_normalized = (self.volume_data - mri_min) / (mri_max - mri_min)
            else:
                mri_normalized = np.zeros_like(self.volume_data)
            
            # 创建RGB体数据
            shape = self.volume_data.shape
            fusion = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.float32)
            
            # 设置基础MRI灰度
            fusion[:, :, :, 0] = mri_normalized  # R
            fusion[:, :, :, 1] = mri_normalized  # G
            fusion[:, :, :, 2] = mri_normalized  # B
            
            # 定义分割标签颜色（与2D融合视图一致）
            colors = {
                1: [1.0, 0.2, 0.2],  # 坏死 - 红色（稍柔和）
                2: [0.2, 1.0, 0.2],  # 水肿 - 绿色
                4: [0.2, 0.4, 1.0]   # 增强肿瘤 - 蓝色
            }
            
            # 叠加分割标签（半透明融合）
            alpha = 0.6  # 融合透明度
            for label_id, color in colors.items():
                mask = (seg_data == label_id)
                if mask.any():
                    print(f"[3D分割] 标签 {label_id}: {mask.sum()} 体素")
                    for c in range(3):
                        fusion[:, :, :, c] = np.where(
                            mask,
                            (1 - alpha) * fusion[:, :, :, c] + alpha * color[c],
                            fusion[:, :, :, c]
                        )
            
            return fusion
            
        except Exception as e:
            print(f"[3D分割] 创建融合体数据失败: {e}")
            return None
    
    def _rgb_to_color_name(self, rgb):
        """将RGB颜色列表转换为matplotlib颜色映射名称"""
        # PyVista/matplotlib支持的颜色映射
        # 根据RGB值返回最接近的颜色映射
        r, g, b = rgb
        
        # 判断主要颜色 - 使用更明显的颜色映射
        if r > 0.5 and g < 0.5 and b < 0.5:
            return "hot"  # 热图，红色更明显
        elif r < 0.5 and g > 0.5 and b < 0.5:
            return "Greens"  # 绿色渐变
        elif r < 0.5 and g < 0.5 and b > 0.5:
            return "Blues"  # 蓝色渐变
        elif r > 0.5 and g > 0.5 and b < 0.5:
            return "Oranges"  # 黄色/橙色
        else:
            return "viridis"  # 默认
    
    def _add_separate_segmentation_volumes(self, seg_data):
        """添加独立的分割标签体绘制（用于高亮显示）"""
        try:
            colors = {
                1: [1.0, 0.0, 0.0],  # 红色 - 坏死
                2: [0.0, 1.0, 0.0],  # 绿色 - 水肿
                4: [0.0, 0.0, 1.0]   # 蓝色 - 增强肿瘤
            }
            
            label_names = {
                1: "坏死/非增强肿瘤",
                2: "水肿",
                4: "增强肿瘤"
            }
            
            for label_id, color in colors.items():
                mask = (seg_data == label_id).astype(np.float32)
                voxel_count = mask.sum()
                
                print(f"[3D分割] 标签 {label_id} ({label_names.get(label_id, '未知')}): {voxel_count} 体素")
                
                if voxel_count > 10:  # 进一步降低阈值，确保小区域也能显示
                    print(f"[3D分割] 添加 {label_names.get(label_id, '未知')} 体绘制")
                    
                    # 将RGB颜色转换为颜色名称
                    color_name = self._rgb_to_color_name(color)
                    print(f"[3D分割] 使用颜色映射: {color_name}")
                    
                    # 使用颜色名称而不是RGB列表
                    self.plotter.add_volume(
                        mask,
                        cmap=color_name,  # 使用颜色名称字符串
                        opacity="sigmoid",  # 使用sigmoid透明度
                        show_scalar_bar=False,
                        name=f"Seg_{label_id}"
                    )
                    
                    # 同时添加表面网格（轮廓）以增强可见性
                    try:
                        from skimage import measure
                        import pyvista as pv
                        # 使用 marching cubes 提取表面
                        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
                        if len(faces) > 0:
                            # 创建面数组
                            faces_pv = np.hstack([[3] + f.tolist() for f in faces])
                            mesh_pd = pv.PolyData(verts, faces_pv)
                            self.plotter.add_mesh(
                                mesh_pd,
                                color=color,
                                opacity=0.4,
                                name=f"SegMesh_{label_id}"
                            )
                    except Exception as mesh_e:
                        print(f"[3D分割] 添加表面网格失败: {mesh_e}")
                        
        except Exception as e:
            print(f"[3D分割] 添加独立分割体失败: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_segmentation_volume(self, seg_data):
        """计算分割区域的体积，包含体积校验逻辑
        
        医学定义（严格遵守）：
        - 肿瘤实体总体积 = Label 1(坏死/非增强) + Label 4(增强肿瘤) 【不含水肿】
        - 瘤周水肿体积 = Label 2 【独立评估指标】
        - 全病灶体积 = Label 1 + 2 + 4 【仅用于科研，需特别注明】
        """
        try:
            self.seg_volumes = {}
            self.volume_warnings = []  # 存储体积异常警告
            
            labels = {
                1: '坏死/非增强肿瘤',
                2: '瘤周水肿',
                4: '增强肿瘤'
            }
            
            # 体积阈值设置 (cm³)
            VOLUME_THRESHOLDS = {
                1: 30,   # 坏死/非增强肿瘤: 通常较小
                2: 60,   # 瘤周水肿: 临床中通常不超过60cm³
                4: 50    # 增强肿瘤: 恶性程度高但也有限度
            }
            
            # 分别计算各区域体积
            tumor_core_volume = 0  # Label 1 + Label 4 (肿瘤实体)
            edema_volume = 0       # Label 2 (水肿，独立统计)
            
            for label_id, label_name in labels.items():
                voxel_count = np.sum(seg_data == label_id)
                volume_mm3 = voxel_count * self.voxel_size
                volume_cm3 = volume_mm3 / 1000.0
                
                # 体积校验
                threshold = VOLUME_THRESHOLDS.get(label_id, 60)
                is_abnormal = volume_cm3 > threshold
                
                self.seg_volumes[label_id] = {
                    'name': label_name,
                    'voxels': int(voxel_count),
                    'volume_mm3': volume_mm3,
                    'volume_cm3': volume_cm3,
                    'threshold': threshold,
                    'is_abnormal': is_abnormal
                }
                
                # 区分肿瘤实体和水肿
                if label_id in [1, 4]:  # 肿瘤实体部分
                    tumor_core_volume += volume_cm3
                elif label_id == 2:     # 水肿部分（独立统计）
                    edema_volume = volume_cm3
                
                # 如果体积异常，添加警告
                if is_abnormal:
                    warning_msg = f"{label_name} 体积({volume_cm3:.1f}cm³)超过阈值({threshold}cm³)，可能存在过度分割"
                    self.volume_warnings.append({
                        'label_id': label_id,
                        'label_name': label_name,
                        'volume': volume_cm3,
                        'threshold': threshold,
                        'message': warning_msg
                    })
                    print(f"[体积警告] {warning_msg}")
            
            # 【关键】肿瘤实体总体积（不含水肿）- 临床最关注的指标
            self.tumor_core_volume = tumor_core_volume
            # 全病灶体积（仅用于科研参考）
            self.total_lesion_volume = tumor_core_volume + edema_volume
            # 水肿体积（独立评估指标）
            self.edema_volume = edema_volume
            
            # 向后兼容：total_tumor_volume 指向肿瘤实体体积
            self.total_tumor_volume = tumor_core_volume
            
            # 肿瘤实体体积校验（不含水肿）
            if tumor_core_volume > 80:
                warning_msg = f"肿瘤实体体积({tumor_core_volume:.1f}cm³)异常偏大，请检查分割准确性"
                self.volume_warnings.append({
                    'label_id': 'tumor_core',
                    'label_name': '肿瘤实体体积',
                    'volume': tumor_core_volume,
                    'threshold': 80,
                    'message': warning_msg
                })
                print(f"[体积警告] {warning_msg}")
            
        except Exception as e:
            print(f"计算体积失败: {e}")
    
    def update_statistics(self):
        """更新统计信息显示"""
        try:
            # 计算脑组织体积（整个图像体积）
            total_voxels = self.shape[0] * self.shape[1] * self.shape[2]
            brain_volume_cm3 = total_voxels * self.voxel_size / 1000.0
            
            stats_text = f"""
<b>📊 图像信息:</b><br>
• 尺寸: {self.shape[0]} × {self.shape[1]} × {self.shape[2]} 体素<br>
• 总切片数: {self.shape[2]} 张<br>
• 体素大小: {self.voxel_size:.4f} mm³<br>
• 数据范围: [{self.volume_data.min():.2f}, {self.volume_data.max():.2f}]<br>
• 估算脑体积: ~{brain_volume_cm3:.1f} cm³<br>
<br>
<b>🔬 体积统计:</b><br>"""
            
            if hasattr(self, 'seg_volumes') and self.seg_volumes:
                # 有分割数据，显示详细统计
                # 1. 显示肿瘤实体部分（Label 1 + Label 4）
                stats_text += """<span style='color: #165DFF; font-weight: bold;'>【肿瘤实体】</span><br>"""
                for label_id in [1, 4]:  # 只显示肿瘤实体部分
                    if label_id in self.seg_volumes:
                        info = self.seg_volumes[label_id]
                        if info['voxels'] > 0:
                            # 如果体积异常，标记为红色
                            if info.get('is_abnormal', False):
                                stats_text += f"• <span style='color: #F53F3F; font-weight: bold;'>{info['name']}: {info['volume_cm3']:.2f} cm³ (⚠️ 超限)</span><br>"
                            else:
                                stats_text += f"• {info['name']}: {info['volume_cm3']:.2f} cm³ ({info['voxels']} 体素)<br>"
                
                # 显示肿瘤实体总体积（临床最关注的指标）
                if hasattr(self, 'tumor_core_volume') and self.tumor_core_volume > 0:
                    stats_text += f"<br>• <b>肿瘤实体总体积: {self.tumor_core_volume:.2f} cm³</b> <span style='color: #86909C; font-size: 11px;'>(Label 1+4, 不含水肿)</span>"
                
                # 2. 显示水肿部分（独立评估指标）
                if 2 in self.seg_volumes and self.seg_volumes[2]['voxels'] > 0:
                    edema_info = self.seg_volumes[2]
                    stats_text += """<br><br><span style='color: #00B42A; font-weight: bold;'>【瘤周水肿】</span><br>"""
                    if edema_info.get('is_abnormal', False):
                        stats_text += f"• <span style='color: #F53F3F; font-weight: bold;'>瘤周水肿: {edema_info['volume_cm3']:.2f} cm³ (⚠️ 超限)</span><br>"
                    else:
                        stats_text += f"• 瘤周水肿: {edema_info['volume_cm3']:.2f} cm³ ({edema_info['voxels']} 体素)<br>"
                    stats_text += f"<span style='color: #86909C; font-size: 11px;'><i>水肿为独立评估指标，不计入肿瘤体积</i></span>"
                
                # 3. 显示全病灶体积（仅科研参考）
                if hasattr(self, 'total_lesion_volume') and self.total_lesion_volume > 0:
                    stats_text += f"""<br><br><span style='color: #86909C;'>【全病灶体积(科研参考)】</span><br>
• 全病灶: {self.total_lesion_volume:.2f} cm³ <span style='font-size: 11px;'>(含水肿，仅用于科研)</span>"""
                
                # 计算肿瘤占比（基于肿瘤实体体积）
                if hasattr(self, 'tumor_core_volume') and self.tumor_core_volume > 0:
                    tumor_ratio = (self.tumor_core_volume / brain_volume_cm3) * 100
                    stats_text += f"<br><br>• <b>肿瘤占比: {tumor_ratio:.2f}%</b> <span style='color: #86909C; font-size: 11px;'>(肿瘤实体/脑体积)</span>"
                
                # 添加体积异常警告
                if hasattr(self, 'volume_warnings') and self.volume_warnings:
                    stats_text += f"""<br><br>
<b style='color: #F53F3F;'>⚠️ 体积异常警告:</b><br>"""
                    for warning in self.volume_warnings:
                        stats_text += f"<span style='color: #F53F3F; font-size: 12px;'>• {warning['message']}</span><br>"
                    stats_text += "<span style='color: #86909C; font-size: 11px;'><i>提示: 可能是分割错误（脑脊液/正常组织被误识别）</i></span>"
            else:
                # 无分割数据，显示提示
                stats_text += """• <i>未找到分割标签文件</i><br>
• 分割文件命名格式: <code>XXX_seg.nii</code> 或 <code>XXX_seg.nii.gz</code><br>
• 包含分割数据后可计算肿瘤体积"""
            
            # 添加检测框信息
            if self.detection_boxes:
                stats_text += f"""<br><br>
<b>🎯 AI检测结果:</b><br>
• 检测到 {len(self.detection_boxes)} 个病灶<br>"""
                for i, box in enumerate(self.detection_boxes[:3], 1):  # 最多显示3个
                    cls_name = box.get('cls_name', 'Unknown')
                    conf = box.get('conf', 0)
                    stats_text += f"• 病灶{i}: {cls_name} ({conf:.1%})<br>"
            
            self.lbl_stats.setText(stats_text)
            
        except Exception as e:
            print(f"更新统计信息失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_opacity(self, value):
        """更新不透明度 - 使用PyVista的scalar opacity方法"""
        if hasattr(self, 'volume'):
            try:
                opacity = value / 100.0
                # 获取volume属性并设置不透明度
                # 使用property的SetColor方法间接控制
                if hasattr(self.volume, 'GetProperty'):
                    prop = self.volume.GetProperty()
                    if prop:
                        # 设置整体不透明度
                        prop.SetScalarOpacityUnitDistance(opacity * 10)
                # 重新渲染
                self.plotter.render()
            except Exception as e:
                print(f"更新不透明度失败: {e}")
    
    def save_screenshot(self):
        """保存截图"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存截图", 
            "mri_3d_view.png",
            "PNG图片 (*.png);;JPEG图片 (*.jpg);;所有文件 (*)"
        )
        if file_path:
            try:
                self.plotter.screenshot(file_path)
                QMessageBox.information(self, "成功", f"截图已保存:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存截图失败: {str(e)}")
    
    def closeEvent(self, event):
        """关闭事件"""
        if hasattr(self, 'plotter'):
            self.plotter.close()
        event.accept()


class BatchWorker(QThread):
    """批量处理工作线程"""
    progress_signal = pyqtSignal(int, int, str)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(list)

    def __init__(self, file_list, model_cls_path, model_seg_path, output_dir, is_nii=False):
        super().__init__()
        self.file_list = file_list
        self.model_cls_path = model_cls_path
        self.model_seg_path = model_seg_path
        self.output_dir = output_dir
        self.is_nii = is_nii
        self.results = []
        self.is_running = True

    def run(self):
        try:
            self.progress_signal.emit(0, len(self.file_list), "正在加载模型...")
            model_cls = YOLO(self.model_cls_path)
            model_seg = YOLO(self.model_seg_path)

            for idx, file_path in enumerate(self.file_list):
                if not self.is_running:
                    break

                self.progress_signal.emit(idx + 1, len(self.file_list), os.path.basename(file_path))

                try:
                    if self.is_nii:
                        result = self.process_nii_file(file_path, model_cls, model_seg)
                    else:
                        result = self.process_image_file(file_path, model_cls, model_seg)
                    self.results.append(result)
                    self.result_signal.emit(result)
                except Exception as e:
                    error_result = {
                        'file': file_path,
                        'status': 'error',
                        'error': str(e)
                    }
                    self.results.append(error_result)
                    self.result_signal.emit(error_result)

            self.finished_signal.emit(self.results)
        except Exception as e:
            self.finished_signal.emit([{'status': 'error', 'error': str(e)}])

    def process_image_file(self, file_path, model_cls, model_seg):
        """处理单个图像文件"""
        start_time = datetime.now()

        # 分类检测 (best5)
        cls_results = model_cls(file_path)
        cls_data_raw = []

        for result in cls_results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id]
                    
                    # 获取检测框坐标 (x1, y1, x2, y2)
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    cls_data_raw.append({
                        'cls_id': cls_id,
                        'cls_name': cls_name,
                        'conf': conf,
                        'box': xyxy
                    })
        
        # 对best5的结果进行NMS处理：去除重叠的框（IOU > 0.5），保留置信度高的
        cls_data = self.nms_boxes(cls_data_raw, iou_threshold=0.5)
        cls_boxes = [item['box'] for item in cls_data]
        
        # 判断是否为肿瘤
        has_tumor = any(item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常'] 
                       for item in cls_data)

        seg_save_path = None
        matched_boxes = []  # 两个模型共同识别到的框
        
        # 统计best5识别到的肿瘤数量（用于显示）
        tumor_count = sum(1 for item in cls_data 
                         if item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常'])
        
        # 只有best5检测到肿瘤时才进行分割
        if has_tumor:
            # 读取原图
            original_img = cv2.imread(file_path)
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w = original_img_rgb.shape[:2]
            result_img = original_img_rgb.copy()
            
            # 进行全图分割
            seg_results = model_seg(file_path)
            
            # 获取best6的分割结果（框和掩码）
            seg_boxes = []
            seg_masks = []
            for result in seg_results:
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        xyxy = box.xyxy[0].cpu().numpy()
                        seg_boxes.append(xyxy)
                        # 保存对应的掩码
                        if result.masks is not None and i < len(result.masks):
                            seg_masks.append(result.masks[i])
                        else:
                            seg_masks.append(None)
            
            # 获取best5的检测框
            cls_boxes = [item['box'] for item in cls_data if 'box' in item]
            
            matched_boxes = []  # 记录有匹配分割的框
            
            # 匹配best5和best6的框：只保留与best5框位置相同的best6分割
            matched_seg_indices = set()
            
            for cls_idx, cls_box in enumerate(cls_boxes):
                best_iou = 0
                best_seg_idx = -1
                
                # 找到与best5框最匹配的best6分割框
                for seg_idx, seg_box in enumerate(seg_boxes):
                    if seg_idx in matched_seg_indices:
                        continue
                    iou = self.calculate_iou(cls_box, seg_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_seg_idx = seg_idx
                
                # IOU > 0.3 认为是同一个位置，保留该分割
                if best_iou > 0.3 and best_seg_idx >= 0:
                    matched_seg_indices.add(best_seg_idx)
                    matched_boxes.append(cls_data[cls_idx])
                    seg_mask = seg_masks[best_seg_idx]
                    
                    if seg_mask is not None:
                        # 获取分割掩码
                        mask_data = seg_mask.data.cpu().numpy()[0]
                        
                        # 将掩码缩放到原图尺寸
                        mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # 创建彩色掩码 (橙色)
                        color = np.array([0, 165, 255], dtype=np.uint8)
                        colored_mask = np.zeros_like(original_img_rgb)
                        colored_mask[mask_resized > 0.5] = color
                        
                        # 叠加掩码
                        alpha = 0.5
                        result_img = cv2.addWeighted(result_img, 1, colored_mask, alpha, 0)
            
            # 绘制best5的检测框
            for cls_item in cls_data:
                if 'box' not in cls_item:
                    continue
                    
                box = cls_item['box']
                x1, y1, x2, y2 = map(int, box)
                
                # 检查该框是否有匹配的分割
                has_matching_seg = cls_item in matched_boxes
                
                # 绘制检测框 - 识别到肿瘤用红色
                is_tumor = cls_item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']
                color = (0, 0, 255) if is_tumor else (0, 165, 255)  # 红色或橙色
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                
                # 添加类别标签
                label = cls_item.get('cls_name', 'Unknown')
                conf = cls_item.get('conf', 0)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(result_img, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 保存结果图
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            seg_save_path = os.path.join(self.output_dir, f"{base_name}_result.jpg")
            Image.fromarray(result_img).save(seg_save_path)

        process_time = (datetime.now() - start_time).total_seconds()

        return {
            'file': file_path,
            'filename': os.path.basename(file_path),
            'status': 'success',
            'cls_count': len(cls_data),
            'seg_count': tumor_count,  # 使用best5识别的肿瘤数量
            'seg_path': seg_save_path,
            'cls_data': cls_data,
            'has_tumor': has_tumor,
            'matched_boxes': matched_boxes,
            'process_time': process_time
        }
    
    def calculate_iou(self, box1, box2):
        """计算两个框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        # 计算IOU
        if union_area == 0:
            return 0
        return inter_area / union_area

    def nms_boxes(self, boxes_data, iou_threshold=0.5):
        """对检测框进行NMS处理，去除重叠的框，保留置信度高的
        
        Args:
            boxes_data: 包含框信息的列表，每个元素是 {'cls_id', 'cls_name', 'conf', 'box'}
            iou_threshold: IOU阈值，超过此值认为是重叠
        
        Returns:
            过滤后的框列表
        """
        if not boxes_data:
            return []
        
        # 按置信度降序排序
        sorted_boxes = sorted(boxes_data, key=lambda x: x['conf'], reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, box_i in enumerate(sorted_boxes):
            if i in suppressed:
                continue
            
            keep.append(box_i)
            
            # 检查后续所有框
            for j in range(i + 1, len(sorted_boxes)):
                if j in suppressed:
                    continue
                
                box_j = sorted_boxes[j]
                iou = self.calculate_iou(box_i['box'], box_j['box'])
                
                # 如果IOU超过阈值，抑制（去除）后面的框
                if iou > iou_threshold:
                    suppressed.add(j)
        
        return keep

    def process_nii_file(self, file_path, model_cls, model_seg):
        """处理NII文件"""
        if not NII_AVAILABLE:
            raise ImportError("未安装nibabel库，无法处理NII文件。请运行: pip install nibabel")

        start_time = datetime.now()

        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()

        middle_slice_idx = data.shape[2] // 2
        slice_data = data[:, :, middle_slice_idx]
        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        slice_rgb = np.stack([slice_data] * 3, axis=-1)

        # 使用tempfile创建临时文件，自动清理
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_img_path = tmp.name
        try:
            Image.fromarray(slice_rgb).save(temp_img_path)
            result = self.process_image_file(temp_img_path, model_cls, model_seg)
        finally:
            # 确保临时文件被删除
            try:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            except Exception:
                pass

        result['file'] = file_path
        result['filename'] = os.path.basename(file_path)
        result['nii_slice'] = middle_slice_idx
        result['nii_shape'] = data.shape

        process_time = (datetime.now() - start_time).total_seconds()
        result['process_time'] = process_time

        return result

    def stop(self):
        self.is_running = False


class MultiModalMRIProcessor:
    """多模态MRI处理器 - 处理患者的5个NII文件"""
    
    # 分割标签定义
    SEG_LABELS = {
        0: {'name': '背景', 'en_name': 'Background', 'desc': '正常脑组织'},
        1: {'name': '坏死/非增强肿瘤', 'en_name': 'Necrotic/Non-enhancing Tumor', 'desc': 'NCR/NET'},
        2: {'name': '瘤周水肿', 'en_name': 'Peritumoral Edema', 'desc': 'ED'},
        4: {'name': '增强肿瘤', 'en_name': 'Enhancing Tumor', 'desc': 'ET'}
    }
    
    def __init__(self, patient_folder):
        self.patient_folder = patient_folder
        self.files = {}
        self.data = {}
        self.shape = None
        self.affine = None
        
    def scan_patient_folder(self):
        """扫描患者文件夹，查找5个NII文件"""
        if not os.path.isdir(self.patient_folder):
            raise ValueError(f"无效的文件夹路径: {self.patient_folder}")
        
        # 查找所有.nii和.nii.gz文件
        nii_files = []
        for f in os.listdir(self.patient_folder):
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nii_files.append(f)
        
        # 识别5种模态文件
        for f in nii_files:
            lower_f = f.lower()
            if '_flair' in lower_f:
                self.files['flair'] = os.path.join(self.patient_folder, f)
            elif '_t1ce' in lower_f:
                self.files['t1ce'] = os.path.join(self.patient_folder, f)
            elif '_t1' in lower_f and '_t1ce' not in lower_f:
                self.files['t1'] = os.path.join(self.patient_folder, f)
            elif '_t2' in lower_f:
                self.files['t2'] = os.path.join(self.patient_folder, f)
            elif '_seg' in lower_f:
                self.files['seg'] = os.path.join(self.patient_folder, f)
        
        return self.files
    
    def load_all_modalities(self):
        """加载所有模态数据"""
        if not self.files:
            self.scan_patient_folder()
        
        for modality, filepath in self.files.items():
            if os.path.exists(filepath):
                nii_img = nib.load(filepath)
                self.data[modality] = nii_img.get_fdata()
                if self.shape is None:
                    self.shape = self.data[modality].shape
                    self.affine = nii_img.affine
            else:
                raise FileNotFoundError(f"找不到文件: {filepath}")
        
        return self.data
    
    def get_slice(self, modality, slice_idx, normalize=True):
        """获取指定模态的切片"""
        if modality not in self.data:
            raise ValueError(f"未加载模态: {modality}")
        
        slice_data = self.data[modality][:, :, slice_idx].copy()
        
        if normalize and modality != 'seg':
            # 归一化到0-255
            min_val = slice_data.min()
            max_val = slice_data.max()
            if max_val > min_val:
                slice_data = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                slice_data = np.zeros_like(slice_data, dtype=np.uint8)
        
        return slice_data
    
    def create_fusion_image(self, slice_idx, alpha=0.5):
        """创建融合图像（t1ce + seg标签叠加）"""
        # 获取t1ce切片（主图像）
        t1ce_slice = self.get_slice('t1ce', slice_idx)
        
        # 转换为RGB
        fusion = np.stack([t1ce_slice] * 3, axis=-1)
        
        # 如果有分割标签，叠加显示
        if 'seg' in self.data:
            seg_slice = self.get_slice('seg', slice_idx, normalize=False)
            
            # 定义颜色映射
            colors = {
                1: [255, 0, 0],    # 坏死 - 红色
                2: [0, 255, 0],    # 水肿 - 绿色
                4: [0, 0, 255]     # 增强肿瘤 - 蓝色
            }
            
            # 创建彩色标签叠加
            for label_value, color in colors.items():
                mask = (seg_slice == label_value)
                for c in range(3):
                    fusion[:, :, c] = np.where(mask, 
                        (1 - alpha) * fusion[:, :, c] + alpha * color[c],
                        fusion[:, :, c]
                    )
        
        return fusion.astype(np.uint8)
    
    def analyze_tumor_info(self, slice_idx):
        """分析指定切片的肿瘤信息"""
        info = {
            'slice_idx': slice_idx,
            'has_tumor': False,
            'regions': {},
            'total_tumor_pixels': 0,
            'tumor_area_mm2': 0
        }
        
        if 'seg' not in self.data:
            return info
        
        seg_slice = self.get_slice('seg', slice_idx, normalize=False)
        
        # 计算体素大小（用于计算实际面积）
        voxel_size = 1.0
        if self.affine is not None:
            voxel_size = abs(self.affine[0, 0] * self.affine[1, 1])  # mm²
        
        # 分析每种肿瘤区域
        for label_value, label_info in self.SEG_LABELS.items():
            if label_value == 0:
                continue
            
            pixel_count = np.sum(seg_slice == label_value)
            
            if pixel_count > 0:
                info['has_tumor'] = True
                info['regions'][label_value] = {
                    'name': label_info['name'],
                    'en_name': label_info['en_name'],
                    'pixel_count': int(pixel_count),
                    'area_mm2': float(pixel_count * voxel_size)
                }
                info['total_tumor_pixels'] += int(pixel_count)
        
        info['tumor_area_mm2'] = float(info['total_tumor_pixels'] * voxel_size)
        
        return info
    
    def get_patient_summary(self):
        """获取患者整体肿瘤总结"""
        summary = {
            'folder': self.patient_folder,
            'files_found': list(self.files.keys()),
            'shape': self.shape,
            'slices_with_tumor': [],
            'tumor_statistics': {},
            'total_tumor_volume_mm3': 0
        }
        
        if 'seg' not in self.data:
            return summary
        
        # 遍历所有切片
        for slice_idx in range(self.shape[2]):
            slice_info = self.analyze_tumor_info(slice_idx)
            
            if slice_info['has_tumor']:
                summary['slices_with_tumor'].append(slice_idx)
                
                # 累加各区域统计
                for label_value, region_info in slice_info['regions'].items():
                    if label_value not in summary['tumor_statistics']:
                        summary['tumor_statistics'][label_value] = {
                            'name': region_info['name'],
                            'en_name': region_info['en_name'],
                            'total_pixels': 0,
                            'slice_count': 0
                        }
                    summary['tumor_statistics'][label_value]['total_pixels'] += region_info['pixel_count']
                    summary['tumor_statistics'][label_value]['slice_count'] += 1
        
        # 计算体积 - 使用行列式计算体素体积（更严谨，兼容旋转情况）
        voxel_volume = 1.0
        if self.affine is not None:
            voxel_volume = abs(np.linalg.det(self.affine[:3, :3]))  # mm³
        
        summary['total_tumor_volume_mm3'] = sum(
            stats['total_pixels'] for stats in summary['tumor_statistics'].values()
        ) * voxel_volume
        
        return summary
    
    def get_t1ce_path_for_model(self):
        """获取t1ce文件路径供模型处理"""
        return self.files.get('t1ce')
    
    def create_mixed_view(self, slice_idx):
        """创建混合视图 - 用于AI分析
        
        将T1CE、FLAIR、T2三个模态合成为一张RGB图像
        R通道: T1CE (显示增强肿瘤)
        G通道: FLAIR (显示水肿)
        B通道: T2 (显示解剖结构)
        如果有分割标签，叠加彩色轮廓
        """
        try:
            # 获取各模态切片
            available_modalities = []
            
            # T1CE - 红色通道
            if 't1ce' in self.data:
                t1ce_slice = self.get_slice('t1ce', slice_idx)
                available_modalities.append('t1ce')
            else:
                t1ce_slice = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            
            # FLAIR - 绿色通道
            if 'flair' in self.data:
                flair_slice = self.get_slice('flair', slice_idx)
                available_modalities.append('flair')
            else:
                flair_slice = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            
            # T2 - 蓝色通道
            if 't2' in self.data:
                t2_slice = self.get_slice('t2', slice_idx)
                available_modalities.append('t2')
            else:
                t2_slice = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
            
            # 创建RGB图像
            mixed_view = np.stack([t1ce_slice, flair_slice, t2_slice], axis=-1)
            
            # 如果有分割标签，添加彩色轮廓
            if 'seg' in self.data:
                seg_slice = self.get_slice('seg', slice_idx, normalize=False)
                
                # 定义颜色 (RGB)
                colors = {
                    1: [255, 100, 100],    # 坏死 - 浅红色
                    2: [100, 255, 100],    # 水肿 - 浅绿色
                    4: [100, 100, 255]     # 增强肿瘤 - 浅蓝色
                }
                
                # 创建轮廓叠加
                for label_value, color in colors.items():
                    mask = (seg_slice == label_value)
                    if np.any(mask):
                        # 找到轮廓边缘
                        if ndimage is not None:
                            eroded = ndimage.binary_erosion(mask, iterations=1)
                            contour = mask & ~eroded
                        else:
                            # 如果没有scipy，直接填充整个区域
                            contour = mask
                        
                        # 叠加轮廓
                        for c in range(3):
                            mixed_view[:, :, c] = np.where(contour, color[c], mixed_view[:, :, c])
            
            return mixed_view.astype(np.uint8)
            
        except Exception as e:
            print(f"[DEBUG] 创建混合视图失败: {e}")
            return None


class ModelWorker(QThread):
    """模型推理工作线程 - 使用信号安全地与GUI通信"""
    finished_signal = pyqtSignal(str, object, str)  # task_type, results, error

    def __init__(self, model_path, image_path, task_type):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.task_type = task_type
        self.model = None

    def run(self):
        try:
            self.model = YOLO(self.model_path)
            results = self.model(self.image_path)
            self.finished_signal.emit(self.task_type, results, "")
        except Exception as e:
            self.finished_signal.emit(self.task_type, None, str(e))


class SegmentationWorker(QThread):
    """分割处理工作线程"""
    finished_signal = pyqtSignal(object, list)  # 返回处理后的图像和结果
    error_signal = pyqtSignal(str)

    def __init__(self, image_path, model_seg_path, classification_data):
        super().__init__()
        self.image_path = image_path
        self.model_seg_path = model_seg_path
        self.classification_data = classification_data

    def run(self):
        try:
            import cv2
            import numpy as np
            from PIL import Image
            from ultralytics import YOLO

            # 读取原图
            original_img = cv2.imread(self.image_path)
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w = original_img.shape[:2]

            # 加载分割模型，进行全图分割
            model_seg = YOLO(self.model_seg_path)
            seg_results = model_seg(self.image_path)

            # 获取best6的分割结果（框和掩码）
            seg_boxes = []
            seg_masks = []
            for result in seg_results:
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        xyxy = box.xyxy[0].cpu().numpy()
                        seg_boxes.append(xyxy)
                        # 保存对应的掩码
                        if result.masks is not None and i < len(result.masks):
                            seg_masks.append(result.masks[i])
                        else:
                            seg_masks.append(None)

            # 处理结果图像
            result_img = original_img_rgb.copy()
            all_seg_results = []

            # 获取best5的检测框
            cls_boxes = []
            for item in self.classification_data:
                if 'box' in item:
                    cls_boxes.append(item['box'])
            
            # 匹配best5和best6的框：只保留与best5框位置相同的best6分割
            matched_seg_indices = set()
            
            for cls_idx, cls_box in enumerate(cls_boxes):
                best_iou = 0
                best_seg_idx = -1
                
                # 找到与best5框最匹配的best6分割框
                for seg_idx, seg_box in enumerate(seg_boxes):
                    if seg_idx in matched_seg_indices:
                        continue
                    iou = self.calculate_iou(cls_box, seg_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_seg_idx = seg_idx
                
                # IOU > 0.3 认为是同一个位置，保留该分割
                if best_iou > 0.3 and best_seg_idx >= 0:
                    matched_seg_indices.add(best_seg_idx)
                    seg_mask = seg_masks[best_seg_idx]
                    
                    if seg_mask is not None:
                        # 获取分割掩码
                        mask_data = seg_mask.data.cpu().numpy()[0]
                        
                        # 将掩码缩放到原图尺寸
                        mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # 创建彩色掩码 (橙色)
                        color = np.array([0, 165, 255], dtype=np.uint8)
                        colored_mask = np.zeros_like(original_img_rgb)
                        colored_mask[mask_resized > 0.5] = color
                        
                        # 叠加掩码
                        alpha = 0.5
                        result_img = cv2.addWeighted(result_img, 1, colored_mask, alpha, 0)
                        
                        all_seg_results.append(seg_results[0])
            
            # 绘制best5的检测框
            for cls_item in self.classification_data:
                if 'box' not in cls_item:
                    continue
                    
                box = cls_item['box']
                x1, y1, x2, y2 = map(int, box)
                
                # 检查该框是否有匹配的分割
                has_matching_seg = False
                for seg_idx in matched_seg_indices:
                    if seg_idx < len(seg_boxes):
                        iou = self.calculate_iou(box, seg_boxes[seg_idx])
                        if iou > 0.3:
                            has_matching_seg = True
                            break
                
                # 绘制检测框 - 识别到肿瘤用红色
                is_tumor = cls_item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']
                color = (0, 0, 255) if is_tumor else (0, 165, 255)  # 红色或橙色
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                
                # 添加类别标签
                label = cls_item.get('cls_name', 'Unknown')
                conf = cls_item.get('conf', 0)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(result_img, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.finished_signal.emit(result_img, all_seg_results)

        except Exception as e:
            self.error_signal.emit(str(e))

    def calculate_iou(self, box1, box2):
        """计算两个框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0
        return inter_area / union_area


class CombinedModelWorker(QThread):
    """单模型（分类+分割合并）工作线程"""
    finished_signal = pyqtSignal(object, list, str)  # result_img, results, error

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path

    def run(self):
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO

            # 读取原图
            original_img = cv2.imread(self.image_path)
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w = original_img.shape[:2]

            # 加载单模型
            model = YOLO(self.model_path)
            results = model(self.image_path)

            # 处理结果图像
            result_img = original_img_rgb.copy()
            all_results = []

            for result in results:
                all_results.append(result)

                # 绘制检测框和标签
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = result.names[cls_id]
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)

                        # 判断是否为肿瘤
                        is_tumor = cls_name.lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']
                        color = (0, 0, 255) if is_tumor else (0, 165, 255)

                        # 绘制检测框
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

                        # 添加类别标签
                        label_text = f"{cls_name} {conf:.2f}"
                        cv2.putText(result_img, label_text, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 绘制分割掩码
                if result.masks is not None:
                    for mask in result.masks:
                        mask_data = mask.data.cpu().numpy()[0]
                        mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)

                        # 创建彩色掩码 (绿色)
                        color = np.array([0, 255, 0], dtype=np.uint8)
                        colored_mask = np.zeros_like(original_img_rgb)
                        colored_mask[mask_resized > 0.5] = color

                        # 叠加掩码
                        alpha = 0.5
                        result_img = cv2.addWeighted(result_img, 1, colored_mask, alpha, 0)

            self.finished_signal.emit(result_img, all_results, "")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(None, [], str(e))


class ImageViewer(QGraphicsView):
    """支持缩放和平移的图像查看器"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.zoom_factor = 1.0

    def set_image(self, pixmap):
        """设置并显示图像"""
        if pixmap is None or pixmap.isNull():
            print("[WARNING] ImageViewer.set_image: 无效的pixmap")
            return
        
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        # 立即调整视图
        self._fit_image()
        self.zoom_factor = 1.0

    def _fit_image(self):
        """调整图像适应视图"""
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.2
            self.scale(1.2, 1.2)
        else:
            self.zoom_factor *= 0.8
            self.scale(0.8, 0.8)

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def reset_view(self):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0

    def show_loading(self):
        """显示加载状态"""
        from PyQt5.QtGui import QPainter, QFont, QColor
        
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(240, 240, 240))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont("Microsoft YaHei", 14))
        
        rect = pixmap.rect()
        painter.drawText(rect, Qt.AlignCenter, "正在检测...")
        painter.end()
        
        self.set_image(pixmap)


class CameraWorker(QThread):
    """摄像头实时检测工作线程"""
    frame_signal = pyqtSignal(np.ndarray)  # 发送视频帧
    result_signal = pyqtSignal(dict)  # 发送检测结果
    status_signal = pyqtSignal(str)  # 发送状态信息
    error_signal = pyqtSignal(str)  # 发送错误信息

    def __init__(self, camera_id, model_cls_path, model_seg_path, detection_mode="综合检测"):
        super().__init__()
        self.camera_id = camera_id
        self.model_cls_path = model_cls_path
        self.model_seg_path = model_seg_path
        self.detection_mode = detection_mode
        self.is_running = False
        self.cap = None
        self.model_cls = None
        self.model_seg = None
        self.last_detection_result = None  # 保存最后一次检测结果
        self.last_result_frame = None  # 保存最后一次带检测框的帧

    def run(self):
        """运行摄像头检测"""
        try:
            self.status_signal.emit("正在加载模型...")
            # 加载模型
            self.model_cls = YOLO(self.model_cls_path)
            self.model_seg = YOLO(self.model_seg_path)
            
            self.status_signal.emit("正在打开摄像头...")
            # 打开摄像头
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.error_signal.emit(f"无法打开摄像头 {self.camera_id}")
                return
            
            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            self.status_signal.emit("检测运行中...")
            
            frame_count = 0
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # 每3帧进行一次检测（降低CPU占用）
                if frame_count % 3 == 0:
                    result_frame, detection_result = self.process_frame(frame)
                    self.last_result_frame = result_frame  # 保存带检测框的帧
                    if detection_result:
                        self.last_detection_result = detection_result
                        self.result_signal.emit(detection_result)
                    self.frame_signal.emit(result_frame)
                else:
                    # 使用上一帧带检测框的结果，避免闪烁
                    if self.last_result_frame is not None:
                        self.frame_signal.emit(self.last_result_frame)
                    else:
                        self.frame_signal.emit(frame)
                
                frame_count += 1
                # 短暂休眠，避免占用过多CPU
                self.msleep(30)
            
        except Exception as e:
            self.error_signal.emit(f"摄像头检测错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.status_signal.emit("检测已停止")

    def process_frame(self, frame):
        """处理单帧图像"""
        result_frame = frame.copy()
        detection_result = None
        temp_path = None
        
        try:
            # 使用tempfile创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
            cv2.imwrite(temp_path, frame)
            
            # 分类检测
            if self.detection_mode in ["分类检测", "综合检测"]:
                cls_results = self.model_cls(temp_path)
                cls_data = []
                
                for result in cls_results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        # 调试信息
                        print(f"[CameraWorker] 检测到 {len(boxes)} 个框")
                        print(f"[CameraWorker] result.names: {result.names if hasattr(result, 'names') else 'N/A'}")
                        
                        # 获取所有检测框，按置信度排序
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # 安全获取类别名称
                            if hasattr(result, 'names') and cls_id in result.names:
                                cls_name = result.names[cls_id]
                            else:
                                cls_name = f"类别_{cls_id}"
                            
                            print(f"[CameraWorker] 框 {i}: cls_id={cls_id}, cls_name={cls_name}, conf={conf:.2f}")
                            
                            cls_data.append({
                                'cls_name': cls_name,
                                'conf': conf,
                                'rank': i + 1,
                                'box': box.xyxy[0].cpu().numpy()
                            })
                        
                        # 按置信度降序排序
                        cls_data.sort(key=lambda x: x['conf'], reverse=True)
                        
                        # 重新设置排名
                        for i, item in enumerate(cls_data):
                            item['rank'] = i + 1
                
                # 判断是否检测到肿瘤
                has_tumor = any(item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常'] 
                               for item in cls_data[:3]) if cls_data else False
                
                # 获取主要检测结果
                if cls_data:
                    top_result = cls_data[0]
                    detection_result = {
                        'has_tumor': has_tumor,
                        'tumor_type': top_result['cls_name'],
                        'confidence': top_result['conf'],
                        'all_results': cls_data[:3] if len(cls_data) >= 3 else cls_data
                    }
                    
                    # 在画面上显示结果 - 显示所有检测到的框
                    colors = {
                        'Meningioma': (0, 0, 255),    # 红色 - 脑膜瘤
                        'Glioma': (0, 165, 255),       # 橙色 - 胶质瘤
                        'Pituitary': (255, 0, 255),    # 紫色 - 垂体瘤
                        'No Tumor': (0, 255, 0),       # 绿色 - 无肿瘤
                        'Notumor': (0, 255, 0),
                        'Healthy': (0, 255, 0),
                        'Normal': (0, 255, 0)
                    }
                    
                    # 显示所有检测到的框（最多显示3个）
                    for idx, item in enumerate(cls_data[:3]):
                        cls_name = item['cls_name']
                        conf = item['conf']
                        box = item['box']
                        
                        # 获取颜色
                        color = colors.get(cls_name, (128, 128, 128))  # 默认灰色
                        
                        # 绘制检测框
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # 显示标签 - 使用PIL绘制中文
                        tumor_type_cn = self.get_tumor_name_cn(cls_name)
                        label = f"{idx+1}.{tumor_type_cn} {conf:.1%}"
                        result_frame = self._draw_chinese_text(result_frame, label, (x1, y1 - 35), color)
            
            # 实例分割
            if self.detection_mode in ["实例分割", "综合检测"]:
                seg_results = self.model_seg(temp_path)
                
                for result in seg_results:
                    if result.masks is not None:
                        for mask in result.masks:
                            mask_data = mask.data.cpu().numpy()[0]
                            mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]), 
                                                     interpolation=cv2.INTER_LINEAR)
                            
                            # 创建彩色掩码
                            color_mask = np.zeros_like(result_frame)
                            color_mask[mask_resized > 0.5] = [0, 165, 255]  # 橙色
                            
                            # 叠加掩码
                            result_frame = cv2.addWeighted(result_frame, 1, color_mask, 0.5, 0)
            
        except Exception as e:
            print(f"处理帧错误: {e}")
        finally:
            # 确保临时文件被删除
            if temp_path:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
        
        return result_frame, detection_result

    def get_tumor_name_cn(self, tumor_type_en):
        """肿瘤类型英文转中文"""
        if tumor_type_en is None:
            return "未知"
        
        tumor_name_map = {
            'Meningioma': '脑膜瘤',
            'Glioma': '胶质瘤',
            'Pituitary': '垂体瘤',
            'No Tumor': '无肿瘤',
            'Notumor': '无肿瘤',
            'Healthy': '健康',
            'Normal': '正常',
            '无肿瘤': '无肿瘤',
            '正常': '正常'
        }
        return tumor_name_map.get(tumor_type_en, tumor_type_en)
    
    def _draw_chinese_text(self, img, text, position, color):
        """使用PIL绘制中文文本"""
        from PIL import Image, ImageDraw, ImageFont
        
        # 转换颜色格式 (BGR -> RGB)
        color_rgb = (color[2], color[1], color[0])
        
        # 将OpenCV图像转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载中文字体
        font = None
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 20)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # 绘制文本背景
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x, y = position
        
        # 绘制背景矩形
        draw.rectangle([x, y, x + text_width + 10, y + text_height + 6], 
                       fill=(255, 255, 255, 200))
        
        # 绘制文本
        draw.text((x + 5, y + 3), text, font=font, fill=color_rgb)
        
        # 转换回OpenCV图像
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def stop(self):
        """停止检测"""
        self.is_running = False
        self.wait(1000)  # 等待1秒


# -------------------------- 仿微信AI聊天组件 --------------------------
class ChatBubble(QFrame):
    """自定义消息气泡（完全仿微信）"""
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QFrame { background: transparent; }")

        # 布局：用户右对齐，AI左对齐
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        # 消息文本标签
        msg_label = QLabel()
        msg_label.setWordWrap(True)
        msg_label.setFont(QFont("微软雅黑", 10))
        msg_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        msg_label.setMaximumWidth(500)  # 限制最大宽度，仿微信
        msg_label.setTextFormat(Qt.RichText)  # 支持HTML富文本
        msg_label.setText(text)  # 设置文本

        # 样式：用户绿色气泡，AI白色气泡
        if is_user:
            msg_label.setStyleSheet("""
                QLabel {
                    background-color: #95ec69;
                    color: #000000;
                    padding: 10px 14px;
                    border-radius: 12px;
                    border-top-right-radius: 2px;
                }
            """)
            layout.addStretch(1)  # 占位，让气泡靠右
            layout.addWidget(msg_label)
        else:
            msg_label.setStyleSheet("""
                QLabel {
                    background-color: #ffffff;
                    color: #333333;
                    padding: 10px 14px;
                    border-radius: 12px;
                    border-top-left-radius: 2px;
                    border: 1px solid #eeeeee;
                }
            """)
            layout.addWidget(msg_label)
            layout.addStretch(1)  # 占位，让气泡靠左


class VoicePlayThread(QThread):
    """语音播放线程（使用Edge TTS自然人声，支持暂停/继续）"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    started = pyqtSignal()  # 开始播放信号

    def __init__(self, text, voice="zh-CN-XiaoxiaoNeural"):
        super().__init__()
        self.text = text
        self.voice = voice
        self.is_paused = False
        self.is_stopped = False
        self.tmp_path = None

    def run(self):
        try:
            import edge_tts
            import asyncio
            import tempfile
            import os
            import re
            import pygame

            # 清理文本（去除HTML标签）
            clean_text = re.sub(r'<[^>]+>', '', self.text)
            clean_text = re.sub(r'[\n\r]', ' ', clean_text)
            clean_text = clean_text[:800]  # 限制长度，避免过长

            # 临时文件存放音频
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                self.tmp_path = tmp_file.name

            # 异步生成语音
            async def do_tts():
                communicate = edge_tts.Communicate(clean_text, self.voice)
                await communicate.save(self.tmp_path)

            asyncio.run(do_tts())

            # 使用pygame播放（支持暂停）
            pygame.mixer.init()
            pygame.mixer.music.load(self.tmp_path)
            pygame.mixer.music.play()
            
            self.started.emit()
            
            # 等待播放完成，同时检查暂停状态
            while pygame.mixer.music.get_busy() or self.is_paused:
                if self.is_stopped:
                    pygame.mixer.music.stop()
                    break
                
                if self.is_paused:
                    pygame.mixer.music.pause()
                    # 暂停时持续等待
                    while self.is_paused and not self.is_stopped:
                        self.msleep(100)
                    
                    if not self.is_stopped:
                        pygame.mixer.music.unpause()
                
                self.msleep(100)
            
            # 清理
            pygame.mixer.quit()
            if self.tmp_path and os.path.exists(self.tmp_path):
                os.unlink(self.tmp_path)
            
            if not self.is_stopped:
                self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def pause(self):
        """暂停播放"""
        self.is_paused = True

    def resume(self):
        """继续播放"""
        self.is_paused = False

    def stop(self):
        """停止播放"""
        self.is_stopped = True
        self.is_paused = False


class AIRequestThread(QThread):
    """API请求线程（防止界面卡顿）"""
    reply_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, user_msg, api_url, api_key, model="deepseek-chat"):
        super().__init__()
        self.user_msg = user_msg
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def run(self):
        if not self.api_key:
            self.error_occurred.emit(get_ai_config_error_message())
            return

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                from openai import OpenAI

                client = OpenAI(
                    base_url=self.api_url,
                    api_key=self.api_key,
                    timeout=120.0  # 增加超时时间到120秒
                )

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self.user_msg}],
                    temperature=0.3,  # 降低temperature以获得更快更确定的回复
                    max_tokens=500  # 减少token数以加快生成速度
                )

                ai_reply = response.choices[0].message.content
                self.reply_finished.emit(ai_reply)
                return  # 成功则直接返回
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if retry_count < max_retries:
                    # 等待后重试
                    import time
                    time.sleep(2 * retry_count)  # 递增等待时间
                    continue
                else:
                    # 所有重试都失败
                    if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                        self.error_occurred.emit("请求超时：服务器响应时间过长，请检查网络连接后重试")
                    elif "connection" in error_msg.lower():
                        self.error_occurred.emit("网络连接失败：请检查网络连接")
                    elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                        self.error_occurred.emit("API密钥错误：请检查API配置")
                    else:
                        self.error_occurred.emit(f"请求失败：{error_msg}")


class WeChatChatWidget(QWidget):
    """仿微信AI聊天组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # DeepSeek API配置
        self.API_URL = get_ai_api_url()
        self.API_KEY = get_ai_api_key()
        self.MODEL_NAME = get_ai_model_name()

        self.init_ui()

    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 使用QSplitter分割聊天区和输入区
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
            }
            QSplitter::handle:hover {
                background-color: #07c160;
            }
        """)

        # 1. 聊天滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f5f5;
            }
            QScrollBar:vertical {
                width: 8px;
                background: #f5f5f5;
            }
            QScrollBar::handle:vertical {
                background: #cccccc;
                border-radius: 4px;
            }
        """)

        # 消息容器
        self.msg_container = QWidget()
        self.msg_layout = QVBoxLayout(self.msg_container)
        self.msg_layout.setAlignment(Qt.AlignTop)
        self.msg_layout.setContentsMargins(0, 10, 0, 10)
        self.msg_layout.setSpacing(8)
        self.scroll_area.setWidget(self.msg_container)
        splitter.addWidget(self.scroll_area)

        # 2. 底部区域（包含发送检测结果按钮、音色选择和输入栏）
        bottom_widget = QWidget()
        bottom_widget.setFixedHeight(140)  # 增加高度
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        
        # 初始化语音相关变量
        self.last_ai_reply = ""  # 存储最后一条AI回复
        self.current_voice = "zh-CN-XiaoxiaoNeural"  # 默认音色
        self.voice_map = {
            "晓晓（女声）": "zh-CN-XiaoxiaoNeural",
            "小艺（女声）": "zh-CN-XiaoyiNeural",
            "云希（男声）": "zh-CN-YunxiNeural",
            "云霞（童声）": "zh-CN-YunxiaNeural",
            "云扬（男声）": "zh-CN-YunyangNeural",
            "云健（男声）": "zh-CN-YunjianNeural",
            "晓北（东北话）": "zh-CN-liaoning-XiaobeiNeural",
            "小妮（陕西话）": "zh-CN-shaanxi-XiaoniNeural"
        }
  
        # 2.1 发送检测结果按钮栏
        btn_widget = QWidget()
        btn_widget.setFixedHeight(32)
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(8, 3, 8, 3)
        btn_layout.setSpacing(8)
        btn_widget.setStyleSheet("background-color: #f7f7f7;")

        # 发送检测结果按钮
        self.send_result_btn = QPushButton("📊 发送检测结果")
        self.send_result_btn.setFixedHeight(26)
        self.send_result_btn.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 5px 15px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1147D9;
            }
            QPushButton:pressed {
                background-color: #0E3CB5;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.send_result_btn.clicked.connect(self.send_detection_result)
        btn_layout.addWidget(self.send_result_btn)
        btn_layout.addStretch()
        bottom_layout.addWidget(btn_widget)

        # 2.2 音色选择栏
        voice_bar = QWidget()
        voice_bar.setFixedHeight(28)
        voice_layout = QHBoxLayout(voice_bar)
        voice_layout.setContentsMargins(8, 2, 8, 2)
        voice_layout.setSpacing(6)
        voice_bar.setStyleSheet("background-color: #f0f0f0;")

        voice_label = QLabel("音色:")
        voice_label.setFont(QFont("微软雅黑", 8))
        self.voice_combo = QComboBox()
        self.voice_combo.setFont(QFont("微软雅黑", 8))
        self.voice_combo.addItems(list(self.voice_map.keys()))
        self.voice_combo.currentTextChanged.connect(self.on_voice_change)
        self.voice_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 1px 6px;
                background-color: white;
                min-width: 90px;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #cccccc;
                selection-background-color: #165DFF;
            }
        """)

        # 播放按钮
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.setFont(QFont("微软雅黑", 8))
        self.play_btn.setFixedHeight(22)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #07c160;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 2px 12px;
            }
            QPushButton:hover {
                background-color: #06ae56;
            }
            QPushButton:pressed {
                background-color: #059c4c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.play_btn.clicked.connect(self.play_voice)

        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_combo)
        voice_layout.addStretch()
        voice_layout.addWidget(self.play_btn)
        bottom_layout.addWidget(voice_bar)

        # 2.3 输入栏
        input_widget = QWidget()
        input_widget.setFixedHeight(70)
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)
        input_widget.setStyleSheet("background-color: #f7f7f7;")

        # 输入框
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("输入问题，按Enter发送...")
        self.input_edit.setFont(QFont("微软雅黑", 10))
        self.input_edit.setFixedHeight(54)
        self.input_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dddddd;
                border-radius: 8px;
                padding: 8px;
                background-color: #ffffff;
            }
        """)
        self.input_edit.installEventFilter(self)
        input_layout.addWidget(self.input_edit)

        # 发送按钮
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedSize(70, 54)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #07c160;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #06ae56;
            }
            QPushButton:pressed {
                background-color: #059c4c;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        bottom_layout.addWidget(input_widget)
        splitter.addWidget(bottom_widget)
        # 设置默认比例：聊天区占大部分，输入区固定高度
        splitter.setSizes([300, 140])
        splitter.setStretchFactor(0, 1)  # 聊天区可拉伸
        splitter.setStretchFactor(1, 0)  # 输入区固定

        main_layout.addWidget(splitter)

    def eventFilter(self, obj, event):
        if obj == self.input_edit and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and not event.modifiers():
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    def send_message(self):
        user_text = self.input_edit.toPlainText().strip()
        if not user_text:
            return

        # 添加用户消息气泡
        self.add_message(user_text, is_user=True)
        # 清空输入框
        self.input_edit.clear()
        # 禁用按钮，显示思考中
        self.send_btn.setEnabled(False)
        self.send_btn.setText("思考中...")

        # 启动API线程
        self.ai_thread = AIRequestThread(user_text, self.API_URL, self.API_KEY, self.MODEL_NAME)
        self.ai_thread.reply_finished.connect(self.on_ai_reply)
        self.ai_thread.error_occurred.connect(self.on_ai_error)
        self.ai_thread.start()

    def add_message(self, text, is_user):
        # 对AI回复进行Markdown格式转换
        if not is_user:
            text = self._markdown_to_html(text)
        bubble = ChatBubble(text, is_user)
        self.msg_layout.addWidget(bubble)
        # 自动滚动到底部
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
    
    def _markdown_to_html(self, text):
        """将Markdown格式转换为HTML"""
        import re
        
        # 处理标题 (# 标题) - 先处理，避免与粗体冲突
        text = re.sub(r'^###\s+(.+)$', r'<h3 style="margin: 8px 0; font-size: 16px; color: #1D2129;">\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'<h2 style="margin: 10px 0; font-size: 18px; color: #1D2129;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.+)$', r'<h1 style="margin: 12px 0; font-size: 20px; color: #165DFF;">\1</h1>', text, flags=re.MULTILINE)
        
        # 处理粗体 (**文本**)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #165DFF;">\1</strong>', text)
        
        # 处理斜体 (*文本*)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        
        # 处理无序列表 (- 项目)
        lines = text.split('\n')
        result_lines = []
        in_list = False
        
        for line in lines:
            list_match = re.match(r'^-\s+(.+)$', line)
            if list_match:
                if not in_list:
                    result_lines.append('<ul style="margin: 8px 0; padding-left: 20px;">')
                    in_list = True
                result_lines.append(f'<li style="margin: 4px 0;">{list_match.group(1)}</li>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                result_lines.append(line)
        
        if in_list:
            result_lines.append('</ul>')
        
        text = '\n'.join(result_lines)
        
        # 处理有序列表 (1. 项目)
        lines = text.split('\n')
        result_lines = []
        in_olist = False
        
        for line in lines:
            olist_match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if olist_match and not line.startswith('<li>'):  # 避免重复处理
                if not in_olist:
                    result_lines.append('<ol style="margin: 8px 0; padding-left: 20px;">')
                    in_olist = True
                result_lines.append(f'<li style="margin: 4px 0;">{olist_match.group(2)}</li>')
            else:
                if in_olist and not line.startswith('<li>'):
                    result_lines.append('</ol>')
                    in_olist = False
                result_lines.append(line)
        
        if in_olist:
            result_lines.append('</ol>')
        
        text = '\n'.join(result_lines)
        
        # 处理换行（跳过已包含HTML标签的行）
        lines = text.split('\n')
        result_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('<'):
                result_lines.append(line + '<br>')
            else:
                result_lines.append(line)
        text = ''.join(result_lines)
        
        return text

    def on_ai_reply(self, reply_text):
        self.add_message(reply_text, is_user=False)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.send_result_btn.setEnabled(True)
        # 保存最后一条AI回复用于语音播报（清理Markdown格式）
        self.last_ai_reply = self._clean_text_for_voice(reply_text)
        # 自动语音播报（如果是AI诊断建议）
        if len(reply_text) > 20:  # 只有较长的回复才自动播报
            self.play_voice()
    
    def _clean_text_for_voice(self, text):
        """清理文本中的Markdown和HTML标记，用于语音播报"""
        import re
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除Markdown标记
        text = re.sub(r'\*\*', '', text)  # 粗体
        text = re.sub(r'\*', '', text)     # 斜体
        text = re.sub(r'#+\s*', '', text)  # 标题
        text = re.sub(r'-\s*', '', text)   # 无序列表
        text = re.sub(r'\d+\.\s*', '', text)  # 有序列表
        
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def on_ai_error(self, error_text):
        self.add_message(error_text, is_user=False)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.send_result_btn.setEnabled(True)

    def clear_chat(self):
        """清空聊天历史"""
        while self.msg_layout.count():
            item = self.msg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.last_ai_reply = ""  # 清空语音记录

    def on_voice_change(self, voice_name):
        """切换语音音色"""
        if voice_name in self.voice_map:
            self.current_voice = self.voice_map[voice_name]

    def play_voice(self):
        """播放/暂停语音播报（使用Edge TTS自然人声）"""
        if not self.last_ai_reply:
            self.add_message("⚠️ 没有可播报的内容", is_user=False)
            return
        
        # 如果正在播放，则暂停
        if hasattr(self, 'voice_thread') and self.voice_thread and self.voice_thread.isRunning():
            if hasattr(self.voice_thread, 'is_paused') and not self.voice_thread.is_paused:
                # 暂停
                self.voice_thread.pause()
                self.play_btn.setText("▶ 继续")
                return
            else:
                # 继续播放
                self.voice_thread.resume()
                self.play_btn.setText("⏸ 暂停")
                return
        
        # 开始新播放
        self.voice_thread = VoicePlayThread(self.last_ai_reply, self.current_voice)
        self.voice_thread.finished.connect(self.on_voice_finished)
        self.voice_thread.error.connect(self.on_voice_error)
        self.voice_thread.started.connect(self.on_voice_started)
        self.voice_thread.start()

    def on_voice_started(self):
        """语音开始播放"""
        self.play_btn.setText("⏸ 暂停")

    def on_voice_finished(self):
        """语音播放完成"""
        self.play_btn.setText("▶ 播放")

    def on_voice_error(self, error_msg):
        """语音播放错误"""
        self.play_btn.setText("▶ 播放")
        self.add_message(f"⚠️ 语音播报失败: {error_msg}", is_user=False)

    def _get_3d_statistics_info(self, main_window):
        """获取3D查看器的统计信息用于AI分析"""
        info = {
            'shape': '未知',
            'total_slices': '未知',
            'volume_stats': '未加载分割数据',
            'detection_info': '未获取检测框信息'
        }
        
        try:
            # 获取图像维度信息
            if hasattr(main_window, 'current_nii_img') and main_window.current_nii_img is not None:
                nii_img = main_window.current_nii_img
                shape = nii_img.shape
                info['shape'] = f"{shape[0]}×{shape[1]}×{shape[2]}"
                info['total_slices'] = str(shape[2])
            
            # 尝试从mri_processor获取分割统计
            if hasattr(main_window, 'mri_processor') and main_window.mri_processor is not None:
                processor = main_window.mri_processor
                
                # 获取体积统计
                volume_stats = []
                if hasattr(processor, 'data') and 'seg' in processor.data:
                    seg_data = processor.data['seg']
                    labels = {
                        1: '坏死/非增强肿瘤',
                        2: '瘤周水肿',
                        4: '增强肿瘤'
                    }
                    
                    total_volume = 0
                    for label_id, label_name in labels.items():
                        voxel_count = np.sum(seg_data == label_id)
                        if voxel_count > 0:
                            # 估算体积（假设体素大小为1mm³）
                            volume_cm3 = voxel_count / 1000.0
                            volume_stats.append(f"• {label_name}: {volume_cm3:.2f} cm³ ({int(voxel_count)} 体素)")
                            total_volume += volume_cm3
                    
                    if total_volume > 0:
                        volume_stats.append(f"• 肿瘤总体积: {total_volume:.2f} cm³")
                        # 计算肿瘤占比
                        total_voxels = seg_data.size
                        tumor_ratio = (np.sum(seg_data > 0) / total_voxels) * 100
                        volume_stats.append(f"• 肿瘤占比: {tumor_ratio:.2f}%")
                
                if volume_stats:
                    info['volume_stats'] = "\n".join(volume_stats)
            
            # 获取检测框信息
            if hasattr(main_window, 'last_nii_detection_result') and main_window.last_nii_detection_result:
                nii_result = main_window.last_nii_detection_result
                detection_info = []
                
                if nii_result.get('has_tumor', False):
                    cls_data = nii_result.get('cls_data', [])
                    detection_info.append(f"• 检测到的病灶数量: {len(cls_data)} 个")
                    
                    for i, cls_item in enumerate(cls_data[:3], 1):  # 最多显示3个
                        cls_name = cls_item.get('cls_name', 'Unknown')
                        conf = cls_item.get('conf', 0)
                        detection_info.append(f"• 病灶{i}: {cls_name} (置信度: {conf:.1%})")
                
                if detection_info:
                    info['detection_info'] = "\n".join(detection_info)
            
        except Exception as e:
            print(f"[AI统计] 获取3D统计信息失败: {e}")
        
        return info
    
    def send_detection_result(self):
        """发送检测结果到AI进行分析 - 支持NII 3D统计信息"""
        # 获取主窗口的检测结果
        main_window = self.window()
        if not hasattr(main_window, 'last_detection_result') or not main_window.last_detection_result:
            self.add_message("⚠️ 请先进行肿瘤检测", is_user=False)
            return

        result = main_window.last_detection_result
        has_tumor = result.get('has_tumor', False)

        # 基础提示词
        base_prompt = """你是专业神经影像科AI辅助诊断专家，沟通对象为临床神经外科、影像科执业医师。请基于脑肿瘤MRI影像识别检测结果，以专业医学视角输出结构化诊断建议：
1. 根据肿瘤类型（脑膜瘤/胶质瘤/垂体瘤）判断典型好发部位，给出病灶定位分析；
2. 给出疑似肿瘤病理分型倾向及分级参考；
3. 提供鉴别诊断方向，排除相似影像学病变；
4. 提出针对性临床诊疗建议、进一步检查方案（如增强MRI、病理活检等）；
5. 分析病灶对周围脑组织的潜在影响与风险提示；
全程使用医学专业术语，简洁严谨，无冗余表述。

【肿瘤类型与典型位置参考】
- 脑膜瘤：好发于大脑凸面、矢状窦旁、蝶骨嵴、鞍结节等部位，多为脑外肿瘤
- 胶质瘤：好发于大脑半球白质区（额叶、颞叶常见），为脑内肿瘤，浸润性生长
- 垂体瘤：位于鞍区、垂体窝，可向上压迫视交叉，向两侧侵犯海绵窦

【重要格式要求】
- 对诊断结论、关键指标、风险提示等重点内容，使用HTML标签进行加粗和变色突出显示，例如：<b style="color: #FF0000;">重点内容</b>
- 可使用颜色：红色(#FF0000)表示警告/风险，蓝色(#165DFF)表示关键指标，绿色(#07c160)表示正常/良性，橙色(#FF9500)表示注意事项
- 输出内容中严禁使用星号(*)和井号()符号，禁止使用Markdown格式
- 使用纯文本和HTML标签进行排版
- 段落之间必须使用<br><br>进行分隔，确保内容清晰分段，不要挤在一起
- 每个小节之间留出空行，提高可读性

【回复速度要求】
- 请快速生成回复，不需要深度思考
- 优先保证回复速度，内容简洁明了即可
- 控制在300字以内，快速给出核心诊断意见

免责声明：本结果为AI辅助诊断参考，不可替代执业医师的临床诊断与诊疗决策。"""

        # 检查是否在NII模式且有3D统计信息
        is_nii_mode = hasattr(main_window, 'current_nii_img') and main_window.current_nii_img is not None
        
        if is_nii_mode and has_tumor:
            # NII模式：使用3D查看器的统计信息
            tumor_type = result.get('tumor_type', '未知')
            confidence = result.get('confidence', 0)
            
            # 获取3D统计信息
            stats_info = self._get_3d_statistics_info(main_window)
            
            prompt = f"""{base_prompt}

【3D MRI多模态综合分析结果】
- 肿瘤类别：{tumor_type}
- AI检测置信度：{confidence:.1%}
- 数据维度：{stats_info.get('shape', '未知')}
- 总切片数：{stats_info.get('total_slices', '未知')}张

【体积统计分析】
{stats_info.get('volume_stats', '未加载分割数据')}

【检测病灶信息】
{stats_info.get('detection_info', '未获取检测框信息')}

请基于上述3D多模态MRI检测结果和体积统计分析，提供专业的结构化诊断建议。重点分析肿瘤体积、空间分布特征及临床意义。"""
        elif has_tumor:
            # 普通图像模式
            tumor_type = result.get('tumor_type', '未知')
            confidence = result.get('confidence', 0)

            prompt = f"""{base_prompt}

【影像识别检测结果】
- 肿瘤类别：{tumor_type}
- AI检测置信度：{confidence:.1%}

请基于上述检测结果，结合肿瘤类型的典型好发部位，提供专业的结构化诊断建议。请自行分析该类型肿瘤最可能的解剖位置。"""
        else:
            prompt = f"""{base_prompt}

【影像识别检测结果】
未检测到肿瘤病灶，影像表现正常。

请基于上述检测结果，提供专业的影像解读。"""

        # 添加用户消息气泡（显示提示词）
        self.add_message("📊 发送检测结果进行分析...", is_user=True)

        # 禁用按钮
        self.send_result_btn.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("思考中...")

        # 启动API线程
        self.ai_thread = AIRequestThread(prompt, self.API_URL, self.API_KEY, self.MODEL_NAME)
        self.ai_thread.reply_finished.connect(self.on_ai_reply)
        self.ai_thread.error_occurred.connect(self.on_ai_error)
        self.ai_thread.start()


class TumorDetectionApp(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO11 多模态脑肿瘤智能识别系统")
        self.setGeometry(100, 100, 1000, 700)

        # 先计算字体缩放因子（基于屏幕DPI）- 必须在其他初始化之前
        self.font_scale = self.calculate_font_scale()

        # 设置窗口最小尺寸（使用缩放后的尺寸）
        self.setMinimumSize(self.scaled_px(1000), self.scaled_px(650))

        # 模型路径（使用项目相对路径）
        project_root = Path(__file__).parent
        self.model_classification_path = str(project_root / "weights" / "classification" / "weights" / "best.pt")
        self.model_segmentation_path = str(project_root / "weights" / "segmentation" / "weights" / "best.pt")
        # 单模型（分类+分割合并）路径
        self.model_combined_path = str(project_root / "weights" / "classification+segmentation" / "train" / "weights" / "best.pt")
        # 训练GUI路径
        self.train_gui_path = str(project_root / "train_gui.py")

        # 模型模式: "dual"=双模型协同, "combined"=单模型分类+分割
        self.model_mode = "dual"

        # 状态变量
        self.current_image_path = None
        self.current_nii_path = None
        self.batch_file_list = []
        self.batch_nii_list = []
        self.is_dark_mode = False
        self.batch_results = []

        # 统计
        self.total_processed = 0
        self.success_count = 0
        self.error_count = 0

        # 线程池
        self.thread_pool = QThreadPool()
        self.batch_worker = None

        # 批量预览当前索引
        self.current_preview_index = -1

        # 摄像头相关
        self.camera_worker = None
        self.is_camera_running = False

        # 初始化UI
        self.init_ui()

        # 应用主题和检查模型
        self.apply_theme()
        self.check_models()
        self.check_nii_support()

    def calculate_font_scale(self):
        """计算字体缩放因子，适配不同DPI屏幕"""
        from PyQt5.QtWidgets import QDesktopWidget
        from PyQt5.QtGui import QFontDatabase

        # 获取主屏幕
        desktop = QDesktopWidget()
        screen = desktop.screenGeometry()
        dpi = desktop.logicalDpiX()

        # 基准DPI为96（标准100%缩放）
        base_dpi = 96
        scale = dpi / base_dpi

        # 根据屏幕分辨率进一步调整
        screen_width = screen.width()
        if screen_width >= 3840:  # 4K屏幕
            scale *= 1.3
        elif screen_width >= 2560:  # 2K屏幕
            scale *= 1.15
        elif screen_width >= 1920:  # 1080p
            scale *= 1.0
        else:  # 低分辨率
            scale *= 0.9

        return max(0.8, min(scale, 2.0))  # 限制在0.8-2.0之间

    def scaled_font_size(self, base_size):
        """根据缩放因子计算字体大小"""
        return int(base_size * self.font_scale)

    def scaled_px(self, base_px):
        """根据缩放因子计算像素值"""
        return int(base_px * self.font_scale)

    def init_ui(self):
        """初始化用户界面 - 三栏布局（使用QSplitter支持拖动）"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 顶部导航栏
        self.create_top_navbar()
        main_layout.addWidget(self.top_navbar)

        # 内容区域 - 使用QSplitter实现可拖动调整
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(6, 6, 6, 6)

        # 创建水平分割器
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
            }
            QSplitter::handle:hover {
                background-color: #165DFF;
            }
        """)

        # 左侧：功能操作栏
        self.create_left_sidebar()
        self.left_sidebar.setMinimumWidth(250)
        self.left_sidebar.setMaximumWidth(400)
        self.main_splitter.addWidget(self.left_sidebar)

        # 中间：图片展示区
        self.create_center_content()
        self.center_content.setMinimumWidth(300)
        self.main_splitter.addWidget(self.center_content)

        # 右侧：AI聊天框
        self.create_right_chat()
        self.right_chat.setMinimumWidth(350)
        self.right_chat.setMaximumWidth(600)
        self.main_splitter.addWidget(self.right_chat)

        # 设置初始宽度比例
        self.main_splitter.setSizes([300, 700, 400])

        content_layout.addWidget(self.main_splitter)
        main_layout.addWidget(content_widget, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def create_top_navbar(self):
        """创建顶部导航栏"""
        self.top_navbar = QFrame()
        self.top_navbar.setFixedHeight(self.scaled_px(64))
        self.top_navbar.setObjectName("top_navbar")

        layout = QHBoxLayout(self.top_navbar)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 0, 24, 0)

        # 左侧 Logo 和名称
        left_layout = QHBoxLayout()
        left_layout.setSpacing(12)

        # Logo图标
        logo_label = QLabel("🔬")
        logo_label.setStyleSheet(f"font-size: {self.scaled_font_size(28)}px;")
        left_layout.addWidget(logo_label)

        # 系统名称
        title_label = QLabel("YOLO11 多模态脑肿瘤智能识别系统")
        title_label.setObjectName("nav_title")
        title_label.setStyleSheet(f"font-size: {self.scaled_font_size(20)}px; font-weight: 600;")
        left_layout.addWidget(title_label)

        # 英文副标题
        subtitle_label = QLabel("Tumor Detection System")
        subtitle_label.setObjectName("nav_subtitle")
        subtitle_label.setStyleSheet(f"font-size: {self.scaled_font_size(12)}px; opacity: 0.7;")
        left_layout.addWidget(subtitle_label)

        # 开发者信息
        author_label = QLabel("Brain Tumor Detection System")
        author_label.setObjectName("nav_author")
        author_label.setStyleSheet(f"font-size: {self.scaled_font_size(14)}px; opacity: 0.8; color: #165DFF; font-weight: 500;")
        left_layout.addWidget(author_label)

        left_layout.addStretch()
        layout.addLayout(left_layout)

        # 右侧控制区
        right_layout = QHBoxLayout()
        right_layout.setSpacing(16)

        # 版本号
        version_label = QLabel("v1.0")
        version_label.setObjectName("nav_version")
        right_layout.addWidget(version_label)

        # 作者信息
        author_info_label = QLabel("Public Release")
        author_info_label.setObjectName("nav_author_info")
        author_info_label.setStyleSheet(f"font-size: {self.scaled_font_size(11)}px; color: #86909C;")
        right_layout.addWidget(author_info_label)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setObjectName("nav_separator")
        separator.setFixedSize(1, self.scaled_px(24))
        right_layout.addWidget(separator)

        # 模型训练按钮
        train_btn = QPushButton("🎯 模型训练")
        train_btn.setObjectName("nav_train_btn")
        train_btn.setFixedHeight(self.scaled_px(36))
        train_btn.setCursor(QCursor(Qt.PointingHandCursor))
        train_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
                font-size: {self.scaled_font_size(12)}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #1147D9;
            }}
        """)
        train_btn.clicked.connect(self.open_train_gui)
        right_layout.addWidget(train_btn)

        # 深浅模式切换
        self.theme_btn = QPushButton("🌙")
        self.theme_btn.setObjectName("theme_btn")
        self.theme_btn.setFixedSize(self.scaled_px(36), self.scaled_px(36))
        self.theme_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.theme_btn.clicked.connect(self.toggle_theme)
        right_layout.addWidget(self.theme_btn)

        # 设置按钮
        settings_btn = QPushButton("⚙️")
        settings_btn.setObjectName("icon_btn")
        settings_btn.setFixedSize(self.scaled_px(36), self.scaled_px(36))
        settings_btn.setCursor(QCursor(Qt.PointingHandCursor))
        right_layout.addWidget(settings_btn)

        layout.addLayout(right_layout)

    def create_left_sidebar(self):
        """创建左侧边栏 - 固定宽度300px"""
        self.left_sidebar = QFrame()
        self.left_sidebar.setObjectName("left_sidebar")

        layout = QVBoxLayout(self.left_sidebar)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 模型状态卡片
        model_card = self.create_model_status_card()
        layout.addWidget(model_card)

        # 功能切换
        function_group = self.create_function_switch()
        layout.addWidget(function_group)

        # 操作区（动态内容）
        self.operation_stack = QStackedWidget()
        self.operation_stack.addWidget(self.create_single_operation_widget())
        self.operation_stack.addWidget(self.create_batch_operation_widget())
        self.operation_stack.addWidget(self.create_nii_operation_widget())
        self.operation_stack.addWidget(self.create_camera_operation_widget())
        layout.addWidget(self.operation_stack, 1)

        # 统计卡片（默认隐藏，只在批量检测时显示）
        self.stats_card = self.create_stats_card()
        self.stats_card.setVisible(False)
        layout.addWidget(self.stats_card)

    def create_model_status_card(self):
        """创建模型状态卡片 - 支持模型模式切换"""
        card = QFrame()
        card.setObjectName("model_card")
        card.setFixedHeight(self.scaled_px(180))

        layout = QVBoxLayout(card)
        layout.setSpacing(6)
        layout.setContentsMargins(16, 10, 16, 10)

        # 标题 - 12pt 加粗
        title = QLabel("模型状态")
        title.setObjectName("card_title")
        title.setStyleSheet(f"font-size: {self.scaled_font_size(12)}px; font-weight: 600; color: #1D2129;")
        layout.addWidget(title)

        # 模型模式切换
        mode_layout = QHBoxLayout()
        mode_label = QLabel("模型模式:")
        mode_label.setStyleSheet(f"font-size: {self.scaled_font_size(12)}px; color: #4E5969;")
        mode_layout.addWidget(mode_label)

        self.model_mode_combo = QComboBox()
        self.model_mode_combo.addItem("双模型协同", "dual")
        self.model_mode_combo.addItem("单模型分类+分割", "combined")
        self.model_mode_combo.setStyleSheet(f"""
            QComboBox {{
                font-size: {self.scaled_font_size(12)}px;
                padding: 4px 8px;
                border: 1px solid #E5E6EB;
                border-radius: 4px;
                background: white;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """)
        self.model_mode_combo.currentIndexChanged.connect(self.on_model_mode_changed)
        mode_layout.addWidget(self.model_mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # 模型状态列表 - 16pt 常规（字体再增大）
        self.model_cls_status = QLabel("❌ 分类模型: 未加载")
        self.model_cls_status.setObjectName("model_status_item")
        self.model_cls_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: #00B42A;")
        layout.addWidget(self.model_cls_status)

        self.model_seg_status = QLabel("❌ 分割模型: 未加载")
        self.model_seg_status.setObjectName("model_status_item")
        self.model_seg_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: #00B42A;")
        layout.addWidget(self.model_seg_status)

        self.model_nii_status = QLabel("❌ NII支持: 未启用")
        self.model_nii_status.setObjectName("model_status_item")
        self.model_nii_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: #00B42A;")
        layout.addWidget(self.model_nii_status)

        return card

    def create_function_switch(self):
        """创建功能切换"""
        group = QFrame()
        group.setObjectName("function_group")

        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题 - 12pt 加粗
        title = QLabel("功能选择")
        title.setObjectName("section_title")
        title.setStyleSheet(f"font-size: {self.scaled_font_size(12)}px; font-weight: 600; color: #1D2129;")
        layout.addWidget(title)

        # 功能按钮组 - 统一尺寸
        self.function_btns = []

        # 单张图像 - 高度48px，11pt常规
        self.btn_single = QPushButton("📷 单张图像检测")
        self.btn_single.setObjectName("function_btn_active")
        self.btn_single.setCheckable(True)
        self.btn_single.setChecked(True)
        self.btn_single.setFixedHeight(self.scaled_px(48))
        self.btn_single.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_single.clicked.connect(lambda: self.switch_function(0))
        layout.addWidget(self.btn_single)
        self.function_btns.append(self.btn_single)

        # 批量图像
        self.btn_batch = QPushButton("📂 批量图像检测")
        self.btn_batch.setObjectName("function_btn")
        self.btn_batch.setCheckable(True)
        self.btn_batch.setFixedHeight(self.scaled_px(48))
        self.btn_batch.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_batch.clicked.connect(lambda: self.switch_function(1))
        layout.addWidget(self.btn_batch)
        self.function_btns.append(self.btn_batch)

        # NII文件
        self.btn_nii = QPushButton("📄 NII文件检测")
        self.btn_nii.setObjectName("function_btn")
        self.btn_nii.setCheckable(True)
        self.btn_nii.setFixedHeight(self.scaled_px(48))
        self.btn_nii.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_nii.clicked.connect(lambda: self.switch_function(2))
        layout.addWidget(self.btn_nii)
        self.function_btns.append(self.btn_nii)

        # 摄像头检测
        self.btn_camera = QPushButton("📹 实时摄像头检测")
        self.btn_camera.setObjectName("function_btn")
        self.btn_camera.setCheckable(True)
        self.btn_camera.setFixedHeight(self.scaled_px(48))
        self.btn_camera.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_camera.clicked.connect(lambda: self.switch_function(3))
        layout.addWidget(self.btn_camera)
        self.function_btns.append(self.btn_camera)

        return group

    def create_single_operation_widget(self):
        """创建单张图像操作区"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 选择图像按钮 - 主按钮 44px高
        self.btn_select_image = QPushButton("选择图像")
        self.btn_select_image.setObjectName("primary_btn")
        self.btn_select_image.setFixedHeight(self.scaled_px(44))
        self.btn_select_image.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_select_image.clicked.connect(self.select_image)
        layout.addWidget(self.btn_select_image)

        # 文件信息 - 9pt 浅灰
        self.lbl_image_info = QLabel("未选择图像")
        self.lbl_image_info.setObjectName("file_info")
        self.lbl_image_info.setAlignment(Qt.AlignCenter)
        self.lbl_image_info.setStyleSheet(f"font-size: {self.scaled_font_size(9)}px; color: #86909C;")
        layout.addWidget(self.lbl_image_info)

        # 操作按钮组 - 次按钮 48px高
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.btn_classify = QPushButton("分类检测")
        self.btn_classify.setObjectName("secondary_btn")
        self.btn_classify.setFixedHeight(self.scaled_px(48))
        self.btn_classify.setEnabled(False)
        self.btn_classify.clicked.connect(self.run_classification)
        btn_layout.addWidget(self.btn_classify)

        self.btn_segment = QPushButton("实例分割")
        self.btn_segment.setObjectName("secondary_btn")
        self.btn_segment.setFixedHeight(self.scaled_px(48))
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self.run_segmentation)
        btn_layout.addWidget(self.btn_segment)

        layout.addLayout(btn_layout)

        # 一键检测按钮 - 主按钮 44px高
        self.btn_run_all = QPushButton("一键检测")
        self.btn_run_all.setObjectName("primary_btn")
        self.btn_run_all.setFixedHeight(self.scaled_px(44))
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.clicked.connect(self.run_all_detection)
        layout.addWidget(self.btn_run_all)

        layout.addStretch()
        return widget

    def create_batch_operation_widget(self):
        """创建批量操作区"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 选择文件夹按钮 - 主按钮 44px高
        self.btn_select_folder = QPushButton("选择图像文件夹")
        self.btn_select_folder.setObjectName("primary_btn")
        self.btn_select_folder.setFixedHeight(self.scaled_px(44))
        self.btn_select_folder.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_select_folder.clicked.connect(self.select_image_folder)
        layout.addWidget(self.btn_select_folder)

        # 文件列表
        list_frame = QFrame()
        list_frame.setObjectName("list_frame")
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(8, 8, 8, 8)

        self.list_image_files = QListWidget()
        self.list_image_files.setObjectName("file_list")
        self.list_image_files.setMaximumHeight(self.scaled_px(160))
        list_layout.addWidget(self.list_image_files)

        layout.addWidget(list_frame)

        # 文件夹信息 - 9pt 浅灰
        self.lbl_folder_info = QLabel("未选择文件夹")
        self.lbl_folder_info.setObjectName("file_info")
        self.lbl_folder_info.setAlignment(Qt.AlignCenter)
        self.lbl_folder_info.setStyleSheet(f"font-size: {self.scaled_font_size(9)}px; color: #86909C;")
        layout.addWidget(self.lbl_folder_info)

        # 操作按钮 - 次按钮 48px高
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.btn_batch_run = QPushButton("开始检测")
        self.btn_batch_run.setObjectName("primary_btn")
        self.btn_batch_run.setFixedHeight(self.scaled_px(48))
        self.btn_batch_run.setEnabled(False)
        self.btn_batch_run.clicked.connect(self.run_batch_image_detection)
        btn_layout.addWidget(self.btn_batch_run)

        self.btn_batch_stop = QPushButton("停止")
        self.btn_batch_stop.setObjectName("secondary_btn")
        self.btn_batch_stop.setFixedHeight(self.scaled_px(48))
        self.btn_batch_stop.setEnabled(False)
        self.btn_batch_stop.clicked.connect(self.stop_batch_processing)
        btn_layout.addWidget(self.btn_batch_stop)

        layout.addLayout(btn_layout)
        layout.addStretch()
        return widget

    def create_nii_operation_widget(self):
        """创建NII操作区 - 支持多模态MRI患者文件夹"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 选择患者文件夹按钮 - 主按钮 44px高
        self.btn_select_patient = QPushButton("选择患者文件夹")
        self.btn_select_patient.setObjectName("primary_btn")
        self.btn_select_patient.setFixedHeight(self.scaled_px(44))
        self.btn_select_patient.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_select_patient.clicked.connect(self.select_patient_folder)
        layout.addWidget(self.btn_select_patient)

        # 模态选择
        modality_layout = QHBoxLayout()
        modality_label = QLabel("显示模态:")
        modality_label.setStyleSheet(f"font-size: {self.scaled_font_size(10)}px; color: #1D2129;")
        modality_layout.addWidget(modality_label)
        self.combo_modality = QComboBox()
        self.combo_modality.addItems(["T1CE (增强T1)", "FLAIR", "T1", "T2", "融合视图", "专家标注"])
        self.combo_modality.setEnabled(False)
        self.combo_modality.currentIndexChanged.connect(self.on_modality_changed)
        modality_layout.addWidget(self.combo_modality)
        layout.addLayout(modality_layout)

        # 切片选择
        slice_layout = QHBoxLayout()
        slice_label = QLabel("切片:")
        slice_label.setStyleSheet(f"font-size: {self.scaled_font_size(10)}px; color: #1D2129;")
        slice_layout.addWidget(slice_label)
        self.spin_nii_slice = QSpinBox()
        self.spin_nii_slice.setEnabled(False)
        self.spin_nii_slice.valueChanged.connect(self.on_nii_slice_changed)
        slice_layout.addWidget(self.spin_nii_slice)
        layout.addLayout(slice_layout)

        # 患者信息 - 9pt 浅灰
        self.lbl_nii_info = QLabel("未选择患者文件夹")
        self.lbl_nii_info.setObjectName("file_info")
        self.lbl_nii_info.setAlignment(Qt.AlignCenter)
        self.lbl_nii_info.setWordWrap(True)
        self.lbl_nii_info.setStyleSheet(f"font-size: {self.scaled_font_size(9)}px; color: #86909C;")
        layout.addWidget(self.lbl_nii_info)

        # 文件状态
        self.lbl_modality_status = QLabel("")
        self.lbl_modality_status.setObjectName("file_info")
        self.lbl_modality_status.setAlignment(Qt.AlignLeft)
        self.lbl_modality_status.setWordWrap(True)
        self.lbl_modality_status.setStyleSheet(f"font-size: {self.scaled_font_size(9)}px; color: #86909C;")
        layout.addWidget(self.lbl_modality_status)

        # 检测按钮 - 主按钮 44px高
        self.btn_nii_run = QPushButton("检测当前切片 (T1CE)")
        self.btn_nii_run.setObjectName("primary_btn")
        self.btn_nii_run.setFixedHeight(self.scaled_px(44))
        self.btn_nii_run.setEnabled(False)
        self.btn_nii_run.clicked.connect(self.run_nii_detection)
        layout.addWidget(self.btn_nii_run)

        layout.addStretch()
        return widget

    def create_camera_operation_widget(self):
        """创建摄像头操作区"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 摄像头选择
        camera_layout = QHBoxLayout()
        camera_label = QLabel("摄像头:")
        camera_label.setStyleSheet(f"font-size: {self.scaled_font_size(10)}px; color: #1D2129;")
        camera_layout.addWidget(camera_label)
        self.combo_camera = QComboBox()
        self.combo_camera.addItem("摄像头 0", 0)
        self.combo_camera.addItem("摄像头 1", 1)
        self.combo_camera.addItem("摄像头 2", 2)
        camera_layout.addWidget(self.combo_camera)
        layout.addLayout(camera_layout)

        # 开始/停止按钮 - 次按钮 48px高
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.btn_camera_start = QPushButton("▶️ 开始检测")
        self.btn_camera_start.setObjectName("secondary_btn")
        self.btn_camera_start.setFixedHeight(self.scaled_px(48))
        self.btn_camera_start.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_camera_start.clicked.connect(self.start_camera_detection)
        btn_layout.addWidget(self.btn_camera_start)

        self.btn_camera_stop = QPushButton("⏹️ 停止检测")
        self.btn_camera_stop.setObjectName("secondary_btn")
        self.btn_camera_stop.setFixedHeight(self.scaled_px(48))
        self.btn_camera_stop.setEnabled(False)
        self.btn_camera_stop.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_camera_stop.clicked.connect(self.stop_camera_detection)
        btn_layout.addWidget(self.btn_camera_stop)

        layout.addLayout(btn_layout)

        # 刷新摄像头按钮 - 次按钮 48px高
        self.btn_refresh_camera = QPushButton("🔄 刷新摄像头列表")
        self.btn_refresh_camera.setObjectName("secondary_btn")
        self.btn_refresh_camera.setFixedHeight(self.scaled_px(48))
        self.btn_refresh_camera.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_refresh_camera.clicked.connect(self.refresh_camera_list)
        layout.addWidget(self.btn_refresh_camera)

        # 摄像头状态 - 9pt 浅灰
        self.lbl_camera_status = QLabel("摄像头状态: 未启动")
        self.lbl_camera_status.setObjectName("file_info")
        self.lbl_camera_status.setAlignment(Qt.AlignCenter)
        self.lbl_camera_status.setStyleSheet(f"font-size: {self.scaled_font_size(9)}px; color: #86909C;")
        layout.addWidget(self.lbl_camera_status)

        # 实时检测结果
        result_frame = QFrame()
        result_frame.setObjectName("result_frame")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(12, 12, 12, 12)

        result_title = QLabel("实时检测结果")
        result_title.setObjectName("section_title")
        result_title.setStyleSheet(f"font-size: {self.scaled_font_size(12)}px; font-weight: 600; color: #1D2129;")
        result_layout.addWidget(result_title)

        self.lbl_camera_result = QLabel("等待检测...")
        self.lbl_camera_result.setObjectName("camera_result")
        self.lbl_camera_result.setAlignment(Qt.AlignCenter)
        self.lbl_camera_result.setStyleSheet(f"font-size: {self.scaled_font_size(14)}px; color: #165DFF;")
        result_layout.addWidget(self.lbl_camera_result)

        self.lbl_camera_confidence = QLabel("")
        self.lbl_camera_confidence.setObjectName("camera_confidence")
        self.lbl_camera_confidence.setAlignment(Qt.AlignCenter)
        self.lbl_camera_confidence.setStyleSheet(f"font-size: {self.scaled_font_size(10)}px; color: #86909C;")
        result_layout.addWidget(self.lbl_camera_confidence)

        # 发送给AI分析按钮
        self.btn_camera_send_ai = QPushButton("🤖 发送给AI分析")
        self.btn_camera_send_ai.setObjectName("primary_btn")
        self.btn_camera_send_ai.setFixedHeight(self.scaled_px(44))
        self.btn_camera_send_ai.setEnabled(False)
        self.btn_camera_send_ai.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_camera_send_ai.clicked.connect(self.send_camera_result_to_ai)
        result_layout.addWidget(self.btn_camera_send_ai)

        layout.addWidget(result_frame)

        # 提示信息 - 9pt 浅灰
        tip_label = QLabel("💡 提示: 将脑部影像对准摄像头进行检测")
        tip_label.setObjectName("tip_label")
        tip_label.setStyleSheet(f"font-size: {self.scaled_font_size(9)}px; color: #86909C; padding: {self.scaled_px(8)}px;")
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)

        layout.addStretch()
        return widget

    def create_stats_card(self):
        """创建统计卡片"""
        card = QFrame()
        card.setObjectName("stats_card")
        card.setFixedHeight(120)

        layout = QGridLayout(card)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题
        title = QLabel("检测统计")
        title.setObjectName("card_title")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title, 0, 0, 1, 2)

        # 统计数据
        self.lbl_total = QLabel("总处理: 0")
        self.lbl_total.setObjectName("stat_item")
        layout.addWidget(self.lbl_total, 1, 0)

        self.lbl_success = QLabel("成功: 0")
        self.lbl_success.setObjectName("stat_success")
        layout.addWidget(self.lbl_success, 1, 1)

        self.lbl_error = QLabel("失败: 0")
        self.lbl_error.setObjectName("stat_error")
        layout.addWidget(self.lbl_error, 2, 0)

        self.lbl_processing = QLabel("检测中: 0")
        self.lbl_processing.setObjectName("stat_item")
        layout.addWidget(self.lbl_processing, 2, 1)

        return card

    def create_center_content(self):
        """创建中间内容区 - 图片展示区（自适应宽度）"""
        self.center_content = QFrame()
        self.center_content.setObjectName("center_content")

        layout = QVBoxLayout(self.center_content)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 创建堆叠布局，用于切换不同视图
        self.content_stack = QStackedLayout()

        # 页面1: 单张检测视图
        self.single_view = QFrame()
        single_layout = QVBoxLayout(self.single_view)
        single_layout.setSpacing(8)
        single_layout.setContentsMargins(0, 0, 0, 0)

        # 原始图像
        card1 = self.create_image_card("原始图像", "original")
        single_layout.addWidget(card1, 1)

        # 最终识别结果（分割结果）
        card3 = self.create_segmentation_card()
        single_layout.addWidget(card3, 1)

        self.content_stack.addWidget(self.single_view)

        # 页面2: 批量检测视图
        self.batch_view = self.create_batch_view()
        self.content_stack.addWidget(self.batch_view)

        # 页面3: NII检测视图
        self.nii_view = self.create_nii_view()
        self.content_stack.addWidget(self.nii_view)

        # 页面4: 摄像头检测视图
        self.camera_view = self.create_camera_view()
        self.content_stack.addWidget(self.camera_view)

        layout.addLayout(self.content_stack, 1)

    def create_right_chat(self):
        """创建右侧AI聊天框 - 固定宽度400px"""
        self.right_chat = QFrame()
        self.right_chat.setObjectName("right_chat")

        layout = QVBoxLayout(self.right_chat)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建仿微信AI聊天组件
        self.chat_widget = WeChatChatWidget()
        layout.addWidget(self.chat_widget)

    def create_image_card(self, title, card_type):
        """创建图像显示卡片"""
        card = QFrame()
        card.setObjectName("content_card")
        card.setMinimumHeight(350)

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题栏
        header = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setObjectName("card_title")
        title_label.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; font-weight: 600;")
        header.addWidget(title_label)

        header.addStretch()

        # 功能按钮
        if card_type == "original":
            fullscreen_btn = QPushButton("⛶")
            fullscreen_btn.setObjectName("icon_btn_small")
            fullscreen_btn.setFixedSize(28, 28)
            fullscreen_btn.setCursor(QCursor(Qt.PointingHandCursor))
            header.addWidget(fullscreen_btn)

        layout.addLayout(header)

        # 图像显示区
        if card_type == "original":
            self.image_viewer = ImageViewer()
            self.image_viewer.setObjectName("image_viewer")
            self.image_viewer.setMinimumHeight(250)
            layout.addWidget(self.image_viewer, 1)

            # 底部信息
            self.lbl_image_meta = QLabel("未加载图像")
            self.lbl_image_meta.setObjectName("image_meta")
            layout.addWidget(self.lbl_image_meta)
        elif card_type == "batch_preview":
            # 批量预览模式
            self.batch_image_viewer = ImageViewer()
            self.batch_image_viewer.setObjectName("batch_image_viewer")
            self.batch_image_viewer.setMinimumHeight(200)
            layout.addWidget(self.batch_image_viewer, 1)
        else:
            placeholder = QLabel("等待检测...")
            placeholder.setObjectName("placeholder")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setMinimumHeight(250)
            layout.addWidget(placeholder, 1)

        return card

    def create_classification_card(self):
        """创建肿瘤类别卡片 - 简化版"""
        card = QFrame()
        card.setObjectName("content_card")
        card.setMinimumHeight(350)

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题栏
        header = QHBoxLayout()

        title = QLabel("肿瘤类别")
        title.setObjectName("card_title")
        title.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; font-weight: 600;")
        header.addWidget(title)

        header.addStretch()

        layout.addLayout(header)

        # 结果内容 - 只显示肿瘤类别
        self.cls_result_widget = QFrame()
        cls_layout = QVBoxLayout(self.cls_result_widget)
        cls_layout.setSpacing(20)
        cls_layout.setAlignment(Qt.AlignCenter)

        # 状态指示 - 是否检测到肿瘤
        self.cls_status = QLabel("未检测")
        self.cls_status.setObjectName("cls_status")
        self.cls_status.setAlignment(Qt.AlignCenter)
        self.cls_status.setStyleSheet(f"""
            font-size: {self.scaled_font_size(24)}px;
            font-weight: 600;
            padding: {self.scaled_px(30)}px;
            border-radius: 12px;
            background-color: #F2F3F5;
            color: #86909C;
        """)
        cls_layout.addWidget(self.cls_status)

        # 肿瘤类别 - 大字体突出显示
        self.cls_category = QLabel("-")
        self.cls_category.setObjectName("category_label")
        self.cls_category.setAlignment(Qt.AlignCenter)
        self.cls_category.setStyleSheet(f"""
            font-size: {self.scaled_font_size(32)}px;
            font-weight: 700;
            color: #165DFF;
            padding: {self.scaled_px(20)}px;
        """)
        cls_layout.addWidget(self.cls_category)

        # 置信度 - 简化显示
        self.cls_confidence_label = QLabel("置信度: -")
        self.cls_confidence_label.setObjectName("confidence_label")
        self.cls_confidence_label.setAlignment(Qt.AlignCenter)
        self.cls_confidence_label.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: #86909C;")
        cls_layout.addWidget(self.cls_confidence_label)

        cls_layout.addStretch()
        layout.addWidget(self.cls_result_widget)

        return card

    def create_segmentation_card(self):
        """创建最终识别结果卡片 - 简化版，只显示分割图像"""
        card = QFrame()
        card.setObjectName("content_card")
        card.setMinimumHeight(350)

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题栏
        header = QHBoxLayout()

        title = QLabel("最终识别结果")
        title.setObjectName("card_title")
        title.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; font-weight: 600;")
        header.addWidget(title)

        header.addStretch()

        # 全屏查看按钮
        fullscreen_btn = QPushButton("⛶")
        fullscreen_btn.setObjectName("icon_btn_small")
        fullscreen_btn.setFixedSize(self.scaled_px(28), self.scaled_px(28))
        fullscreen_btn.setCursor(QCursor(Qt.PointingHandCursor))
        fullscreen_btn.clicked.connect(self.show_segmentation_fullscreen)
        header.addWidget(fullscreen_btn)

        layout.addLayout(header)

        # 分割图像显示 - 使用支持缩放的ImageViewer
        self.seg_image_viewer = ImageViewer()
        self.seg_image_viewer.setObjectName("seg_image_viewer")
        self.seg_image_viewer.setMinimumHeight(self.scaled_px(250))
        layout.addWidget(self.seg_image_viewer, 1)

        return card

    def create_batch_segmentation_card(self):
        """创建批量检测专用的最终识别结果卡片"""
        card = QFrame()
        card.setObjectName("content_card")
        card.setMinimumHeight(self.scaled_px(350))

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题栏
        header = QHBoxLayout()

        title = QLabel("最终识别结果")
        title.setObjectName("card_title")
        title.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; font-weight: 600;")
        header.addWidget(title)

        header.addStretch()

        # 全屏查看按钮
        fullscreen_btn = QPushButton("⛶")
        fullscreen_btn.setObjectName("icon_btn_small")
        fullscreen_btn.setFixedSize(self.scaled_px(28), self.scaled_px(28))
        fullscreen_btn.setCursor(QCursor(Qt.PointingHandCursor))
        fullscreen_btn.clicked.connect(self.show_batch_segmentation_fullscreen)
        header.addWidget(fullscreen_btn)

        layout.addLayout(header)

        # 分割图像显示 - 批量检测专用
        self.batch_seg_image_viewer = ImageViewer()
        self.batch_seg_image_viewer.setObjectName("batch_seg_image_viewer")
        self.batch_seg_image_viewer.setMinimumHeight(self.scaled_px(250))
        layout.addWidget(self.batch_seg_image_viewer, 1)

        return card

    def create_diagnosis_card(self):
        """创建AI对话卡片（单张检测时显示在右下角）"""
        card = QFrame()
        card.setObjectName("content_card")
        card.setMinimumHeight(self.scaled_px(350))

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题栏
        header = QHBoxLayout()

        title = QLabel("🤖 AI实时对话")
        title.setObjectName("card_title")
        title.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; font-weight: 600;")
        header.addWidget(title)

        header.addStretch()

        # 清空对话按钮
        clear_btn = QPushButton("🗑️ 清空")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: #86909C;
                border: 1px solid #E5E6EB;
                border-radius: 4px;
                padding: {self.scaled_px(4)}px {self.scaled_px(8)}px;
                font-size: {self.scaled_font_size(11)}px;
            }}
            QPushButton:hover {{
                background-color: #F2F3F5;
                color: #4E5969;
            }}
        """)
        clear_btn.clicked.connect(self.clear_chat_history)
        header.addWidget(clear_btn)

        layout.addLayout(header)

        # 对话历史区域
        self.diagnosis_text = QTextEdit()
        self.diagnosis_text.setObjectName("diagnosis_text")
        self.diagnosis_text.setReadOnly(True)
        self.diagnosis_text.setPlaceholderText("请先进行肿瘤检测，AI将为您提供诊断建议...\n\n检测完成后，您可以直接在这里输入问题与AI对话。")
        self.diagnosis_text.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid #E5E6EB;
                border-radius: 8px;
                padding: {self.scaled_px(12)}px;
                background-color: #F7F8FA;
                font-size: {self.scaled_font_size(14)}px;
                line-height: 1.6;
            }}
        """)
        layout.addWidget(self.diagnosis_text, 1)

        # 快捷功能按钮区域
        quick_actions_layout = QHBoxLayout()
        quick_actions_layout.setSpacing(8)
        
        # 识别检测结果照片按钮
        btn_recognize_image = QPushButton("🔍 识别检测照片")
        btn_recognize_image.setStyleSheet(f"""
            QPushButton {{
                background-color: #E8F0FF;
                color: #165DFF;
                border: 1px solid #165DFF;
                border-radius: 6px;
                padding: {self.scaled_px(6)}px {self.scaled_px(12)}px;
                font-size: {self.scaled_font_size(12)}px;
            }}
            QPushButton:hover {{
                background-color: #165DFF;
                color: white;
            }}
        """)
        btn_recognize_image.clicked.connect(self.quick_recognize_image)
        quick_actions_layout.addWidget(btn_recognize_image)
        
        # 语音播报按钮
        btn_voice_broadcast = QPushButton("🔊 语音播报")
        btn_voice_broadcast.setStyleSheet(f"""
            QPushButton {{
                background-color: #E8FFEA;
                color: #00B42A;
                border: 1px solid #00B42A;
                border-radius: 6px;
                padding: {self.scaled_px(6)}px {self.scaled_px(12)}px;
                font-size: {self.scaled_font_size(12)}px;
            }}
            QPushButton:hover {{
                background-color: #00B42A;
                color: white;
            }}
        """)
        btn_voice_broadcast.clicked.connect(self.quick_voice_broadcast)
        quick_actions_layout.addWidget(btn_voice_broadcast)
        
        # 查看统计信息按钮
        btn_view_stats = QPushButton("📊 查看统计")
        btn_view_stats.setStyleSheet(f"""
            QPushButton {{
                background-color: #FFF7E6;
                color: #FF7D00;
                border: 1px solid #FF7D00;
                border-radius: 6px;
                padding: {self.scaled_px(6)}px {self.scaled_px(12)}px;
                font-size: {self.scaled_font_size(12)}px;
            }}
            QPushButton:hover {{
                background-color: #FF7D00;
                color: white;
            }}
        """)
        btn_view_stats.clicked.connect(self.quick_view_stats)
        quick_actions_layout.addWidget(btn_view_stats)
        
        quick_actions_layout.addStretch()
        layout.addLayout(quick_actions_layout)

        # 用户输入区域
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        self.diagnosis_input = QLineEdit()
        self.diagnosis_input.setObjectName("diagnosis_input")
        self.diagnosis_input.setPlaceholderText("输入问题，按Enter发送...")
        self.diagnosis_input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid #E5E6EB;
                border-radius: 6px;
                padding: {self.scaled_px(8)}px {self.scaled_px(12)}px;
                background-color: white;
                font-size: {self.scaled_font_size(14)}px;
            }}
            QLineEdit:focus {{
                border: 1px solid #165DFF;
            }}
        """)
        self.diagnosis_input.returnPressed.connect(self.send_diagnosis_message)
        input_layout.addWidget(self.diagnosis_input, 1)

        send_btn = QPushButton("发送")
        send_btn.setObjectName("send_btn")
        send_btn.setFixedWidth(self.scaled_px(60))
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: {self.scaled_px(8)}px;
                font-size: {self.scaled_font_size(13)}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #1147D9;
            }}
            QPushButton:pressed {{
                background-color: #0E3CB5;
            }}
        """)
        send_btn.clicked.connect(self.send_diagnosis_message)
        input_layout.addWidget(send_btn)

        layout.addLayout(input_layout)

        # 免责声明
        disclaimer = QLabel("⚠️ 仅供参考，请以专业医生诊断为准")
        disclaimer.setObjectName("disclaimer")
        disclaimer.setStyleSheet(f"color: #86909C; font-size: {self.scaled_font_size(11)}px;")
        disclaimer.setAlignment(Qt.AlignCenter)
        layout.addWidget(disclaimer)

        return card

    def create_placeholder_card(self):
        """创建占位卡片（批量检测时显示）"""
        card = QFrame()
        card.setObjectName("content_card")
        card.setMinimumHeight(self.scaled_px(350))

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setAlignment(Qt.AlignCenter)

        # 提示图标
        icon_label = QLabel("📊")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # 提示文字
        hint = QLabel("切换到「批量图像检测」以查看批量处理进度")
        hint.setObjectName("placeholder_hint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #86909C; font-size: 14px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch()
        return card

    def create_batch_view(self):
        """创建批量检测视图 - 支持逐个切换预览"""
        widget = QFrame()
        layout = QGridLayout(widget)
        layout.setSpacing(16)

        # 左侧：批量检测进度
        progress_card = self.create_batch_progress_card()
        layout.addWidget(progress_card, 0, 0, 2, 1)

        # 右侧：预览区域（包含切换按钮）
        preview_widget = QFrame()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setSpacing(8)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # 预览标题和导航
        nav_layout = QHBoxLayout()
        
        self.preview_title = QLabel("图像预览")
        self.preview_title.setObjectName("card_title")
        self.preview_title.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; font-weight: 600;")
        nav_layout.addWidget(self.preview_title)

        nav_layout.addStretch()

        # 上一个按钮
        self.btn_prev = QPushButton("◀ 上一个")
        self.btn_prev.setObjectName("nav_btn")
        self.btn_prev.setFixedHeight(self.scaled_px(32))
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.show_prev_image)
        nav_layout.addWidget(self.btn_prev)

        # 当前索引显示
        self.lbl_preview_index = QLabel("0 / 0")
        self.lbl_preview_index.setObjectName("preview_index")
        self.lbl_preview_index.setStyleSheet(f"font-size: {self.scaled_font_size(14)}px; color: #86909C;")
        nav_layout.addWidget(self.lbl_preview_index)

        # 下一个按钮
        self.btn_next = QPushButton("下一个 ▶")
        self.btn_next.setObjectName("nav_btn")
        self.btn_next.setFixedHeight(self.scaled_px(32))
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.btn_next)
        
        preview_layout.addLayout(nav_layout)

        # 原始图像预览
        image_card = self.create_image_card("原始图像", "batch_preview")
        preview_layout.addWidget(image_card, 1)

        # 检测结果预览 - 批量检测专用
        result_card = self.create_batch_segmentation_card()
        preview_layout.addWidget(result_card, 1)

        layout.addWidget(preview_widget, 0, 1, 2, 1)

        return widget

    def create_batch_progress_card(self):
        """创建批量进度卡片 - 包含AI统计信息"""
        card = QFrame()
        card.setObjectName("content_card")
        card.setMinimumHeight(350)

        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题栏
        header = QHBoxLayout()

        title = QLabel("批量检测进度")
        title.setObjectName("card_title")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(title)

        header.addStretch()

        layout.addLayout(header)

        # 进度信息
        self.batch_progress_label = QLabel("0/0 已完成")
        self.batch_progress_label.setObjectName("progress_label")
        layout.addWidget(self.batch_progress_label)

        # 进度条
        self.batch_progress = QProgressBar()
        self.batch_progress.setObjectName("batch_progress")
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress.setFixedHeight(8)
        layout.addWidget(self.batch_progress)

        # AI统计信息区域
        self.batch_stats_widget = QFrame()
        self.batch_stats_widget.setObjectName("batch_stats")
        self.batch_stats_widget.setStyleSheet("""
            QFrame#batch_stats {
                background-color: #F7F8FA;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        stats_layout = QVBoxLayout(self.batch_stats_widget)
        stats_layout.setSpacing(8)
        stats_layout.setContentsMargins(12, 12, 12, 12)

        # 统计标题
        stats_title = QLabel("🤖 AI统计分析")
        stats_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #165DFF;")
        stats_layout.addWidget(stats_title)

        # 统计内容
        self.batch_stats_text = QTextEdit()
        self.batch_stats_text.setObjectName("batch_stats_text")
        self.batch_stats_text.setReadOnly(True)
        self.batch_stats_text.setMaximumHeight(120)
        self.batch_stats_text.setPlaceholderText("检测完成后显示统计信息...")
        self.batch_stats_text.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: transparent;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        stats_layout.addWidget(self.batch_stats_text)

        # 默认隐藏统计区域
        self.batch_stats_widget.setVisible(False)
        layout.addWidget(self.batch_stats_widget)

        # 结果表格
        self.batch_table = QTableWidget()
        self.batch_table.setObjectName("batch_table")
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels(["文件名", "分类结果", "肿瘤数", "耗时", "状态"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.batch_table.itemClicked.connect(self.on_batch_item_clicked)
        layout.addWidget(self.batch_table, 1)

        # 保存按钮
        self.btn_save_batch = QPushButton("保存所有结果")
        self.btn_save_batch.setObjectName("primary_btn")
        self.btn_save_batch.setFixedHeight(40)
        self.btn_save_batch.setEnabled(False)
        self.btn_save_batch.clicked.connect(self.save_batch_results)
        layout.addWidget(self.btn_save_batch)

        return card

    def create_nii_view(self):
        """创建NII多模态检测视图 - 添加3D查看功能"""
        widget = QFrame()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # MRI图像显示卡片
        image_card = QFrame()
        image_card.setObjectName("content_card")
        image_layout = QVBoxLayout(image_card)
        image_layout.setSpacing(8)
        image_layout.setContentsMargins(12, 12, 12, 12)

        # 标题和3D按钮
        header = QHBoxLayout()
        title = QLabel("MRI图像显示")
        title.setObjectName("card_title")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(title)
        
        # 添加3D查看按钮
        if PV_AVAILABLE:
            btn_3d = QPushButton("3D查看")
            btn_3d.setObjectName("btn_3d_view")
            btn_3d.setStyleSheet("""
                QPushButton {
                    background-color: #165DFF;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 16px;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #1147CC;
                }
                QPushButton:disabled {
                    background-color: #C9CDD4;
                    color: #86909C;
                }
            """)
            btn_3d.setEnabled(False)  # 初始禁用，加载NII后启用
            btn_3d.clicked.connect(self.open_3d_viewer)
            self.btn_3d_viewer = btn_3d  # 保存引用以便启用/禁用
            header.addWidget(btn_3d)
        
        header.addStretch()
        image_layout.addLayout(header)

        # MRI图像查看器
        self.nii_image_viewer = ImageViewer()
        self.nii_image_viewer.setObjectName("nii_image_viewer")
        self.nii_image_viewer.setMinimumHeight(200)
        image_layout.addWidget(self.nii_image_viewer, 1)

        # 模态信息
        self.lbl_nii_modality = QLabel("当前模态: 未选择")
        self.lbl_nii_modality.setObjectName("image_meta")
        image_layout.addWidget(self.lbl_nii_modality)

        layout.addWidget(image_card, 1)

        # 分割结果显示卡片
        seg_card = QFrame()
        seg_card.setObjectName("content_card")
        seg_layout = QVBoxLayout(seg_card)
        seg_layout.setSpacing(8)
        seg_layout.setContentsMargins(12, 12, 12, 12)

        seg_header = QHBoxLayout()
        seg_title = QLabel("分割结果")
        seg_title.setObjectName("card_title")
        seg_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        seg_header.addWidget(seg_title)
        seg_header.addStretch()
        seg_layout.addLayout(seg_header)

        # 分割图像查看器（NII专用）
        self.nii_seg_viewer = ImageViewer()
        self.nii_seg_viewer.setObjectName("nii_seg_viewer")
        self.nii_seg_viewer.setMinimumHeight(180)
        seg_layout.addWidget(self.nii_seg_viewer, 1)

        layout.addWidget(seg_card, 1)

        return widget

    def create_camera_view(self):
        """创建摄像头实时检测视图"""
        widget = QFrame()
        layout = QGridLayout(widget)
        layout.setSpacing(16)

        # 左侧：实时视频显示
        video_card = QFrame()
        video_card.setObjectName("content_card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setSpacing(12)
        video_layout.setContentsMargins(16, 16, 16, 16)

        # 标题
        header = QHBoxLayout()
        title = QLabel("实时视频检测")
        title.setObjectName("card_title")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header.addWidget(title)

        # 检测模式显示
        self.lbl_camera_mode_display = QLabel("模式: 未启动")
        self.lbl_camera_mode_display.setObjectName("camera_mode_display")
        self.lbl_camera_mode_display.setStyleSheet("font-size: 12px; color: #86909C;")
        header.addWidget(self.lbl_camera_mode_display)
        header.addStretch()
        video_layout.addLayout(header)

        # 视频显示区域
        self.camera_video_label = QLabel()
        self.camera_video_label.setObjectName("camera_video_label")
        self.camera_video_label.setAlignment(Qt.AlignCenter)
        self.camera_video_label.setMinimumHeight(400)
        self.camera_video_label.setStyleSheet("""
            background-color: #1A1D21;
            border-radius: 8px;
            color: #86909C;
            font-size: 16px;
        """)
        self.camera_video_label.setText("摄像头未启动\n点击「开始检测」按钮启动")
        video_layout.addWidget(self.camera_video_label, 1)

        # FPS显示
        self.lbl_camera_fps = QLabel("FPS: 0")
        self.lbl_camera_fps.setObjectName("camera_fps")
        self.lbl_camera_fps.setStyleSheet("font-size: 12px; color: #86909C;")
        video_layout.addWidget(self.lbl_camera_fps)

        layout.addWidget(video_card, 0, 0, 2, 1)

        # 右侧：实时检测结果
        # 检测结果卡片
        result_card = QFrame()
        result_card.setObjectName("content_card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setSpacing(12)
        result_layout.setContentsMargins(16, 16, 16, 16)

        result_title = QLabel("实时检测结果")
        result_title.setObjectName("card_title")
        result_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        result_layout.addWidget(result_title)

        # 检测状态
        self.camera_cls_status = QLabel("未检测")
        self.camera_cls_status.setObjectName("cls_status")
        self.camera_cls_status.setAlignment(Qt.AlignCenter)
        self.camera_cls_status.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            padding: 40px;
            border-radius: 12px;
            background-color: #F7F8FA;
            color: #86909C;
        """)
        result_layout.addWidget(self.camera_cls_status)

        # 置信度
        self.camera_confidence_label = QLabel("")
        self.camera_confidence_label.setObjectName("confidence_label")
        self.camera_confidence_label.setAlignment(Qt.AlignCenter)
        self.camera_confidence_label.setStyleSheet("font-size: 16px; color: #4E5969;")
        result_layout.addWidget(self.camera_confidence_label)

        # 详细结果列表
        self.camera_result_list = QTextEdit()
        self.camera_result_list.setObjectName("camera_result_list")
        self.camera_result_list.setReadOnly(True)
        self.camera_result_list.setMaximumHeight(150)
        self.camera_result_list.setPlaceholderText("检测结果将显示在这里...")
        self.camera_result_list.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E5E6EB;
                border-radius: 8px;
                padding: 8px;
                background-color: #F7F8FA;
                font-size: 13px;
            }
        """)
        result_layout.addWidget(self.camera_result_list)

        result_layout.addStretch()
        layout.addWidget(result_card, 0, 1)

        # AI诊断建议卡片（摄像头检测专用）
        camera_diagnosis_card = QFrame()
        camera_diagnosis_card.setObjectName("content_card")
        camera_diagnosis_layout = QVBoxLayout(camera_diagnosis_card)
        camera_diagnosis_layout.setSpacing(12)
        camera_diagnosis_layout.setContentsMargins(16, 16, 16, 16)

        diagnosis_title = QLabel("AI诊断建议")
        diagnosis_title.setObjectName("card_title")
        diagnosis_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        camera_diagnosis_layout.addWidget(diagnosis_title)

        self.camera_diagnosis_text = QTextEdit()
        self.camera_diagnosis_text.setObjectName("camera_diagnosis_text")
        self.camera_diagnosis_text.setReadOnly(True)
        self.camera_diagnosis_text.setPlaceholderText("启动检测后将显示AI诊断建议...")
        self.camera_diagnosis_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E5E6EB;
                border-radius: 8px;
                padding: 12px;
                background-color: #F7F8FA;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        camera_diagnosis_layout.addWidget(self.camera_diagnosis_text, 1)

        disclaimer = QLabel("⚠️ 仅供参考，请以专业医生诊断为准")
        disclaimer.setObjectName("disclaimer")
        disclaimer.setStyleSheet("color: #86909C; font-size: 11px;")
        disclaimer.setAlignment(Qt.AlignCenter)
        camera_diagnosis_layout.addWidget(disclaimer)

        layout.addWidget(camera_diagnosis_card, 1, 1)

        return widget

    def apply_theme(self):
        """应用主题样式"""
        if self.is_dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    def apply_light_theme(self):
        """应用浅色主题"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Theme.LIGHT_BG};
            }}
            QWidget {{
                font-family: "Microsoft YaHei", "Source Han Sans", sans-serif;
                font-size: 13px;
                color: {Theme.LIGHT_TEXT};
            }}
            #top_navbar {{
                background-color: {Theme.LIGHT_CARD};
                border-bottom: 1px solid {Theme.LIGHT_BORDER};
            }}
            #nav_title {{
                color: {Theme.LIGHT_TEXT};
            }}
            #nav_subtitle {{
                color: {Theme.LIGHT_TEXT_SECONDARY};
            }}
            #left_sidebar {{
                background-color: {Theme.LIGHT_CARD};
                border-right: 1px solid {Theme.LIGHT_BORDER};
            }}
            #main_content {{
                background-color: {Theme.LIGHT_BG};
            }}
            #model_card, #stats_card {{
                background-color: #F5F7FA;
                border-radius: 12px;
                border: none;
            }}
            #content_card {{
                background-color: {Theme.LIGHT_CARD};
                border-radius: 12px;
                border: 1px solid {Theme.LIGHT_BORDER};
            }}
            #primary_btn {{
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
                font-size: {self.scaled_font_size(10)}px;
                font-weight: bold;
            }}
            #primary_btn:hover {{
                background-color: #1249CC;
            }}
            #primary_btn:disabled {{
                background-color: #C9CDD4;
                color: white;
            }}
            #secondary_btn {{
                background-color: #F2F3F5;
                color: #1D2129;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
                font-size: {self.scaled_font_size(10)}px;
                font-weight: normal;
            }}
            #secondary_btn:hover {{
                background-color: #E5E6EB;
            }}
            #secondary_btn:disabled {{
                background-color: #C9CDD4;
                color: white;
            }}
            #function_btn {{
                background-color: #F5F7FA;
                color: #1D2129;
                border: none;
                border-radius: 8px;
                padding: 0 16px;
                text-align: left;
                font-size: {self.scaled_font_size(11)}px;
                font-weight: normal;
            }}
            #function_btn:hover {{
                background-color: #E5E6EB;
            }}
            #function_btn_active {{
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 16px;
                text-align: left;
                font-size: {self.scaled_font_size(11)}px;
                font-weight: normal;
            }}
            #function_btn_active:hover {{
                background-color: #1249CC;
            }}
            #file_list {{
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
            }}
            #file_list::item {{
                padding: 8px;
                border-radius: 4px;
            }}
            #file_list::item:selected {{
                background-color: {Theme.PRIMARY_LIGHT};
                color: {Theme.PRIMARY};
            }}
            QTableWidget {{
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                gridline-color: {Theme.LIGHT_BORDER};
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
            QTableWidget::item:selected {{
                background-color: {Theme.PRIMARY_LIGHT};
                color: {Theme.PRIMARY};
            }}
            QHeaderView::section {{
                background-color: {Theme.LIGHT_HOVER};
                padding: 10px;
                border: none;
                font-weight: 600;
            }}
            QProgressBar {{
                background-color: {Theme.LIGHT_HOVER};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {Theme.PRIMARY};
                border-radius: 4px;
            }}
            #confidence_bar::chunk {{
                background-color: {Theme.SUCCESS};
            }}
            #placeholder {{
                background-color: {Theme.LIGHT_HOVER};
                color: {Theme.NEUTRAL};
                border-radius: 8px;
            }}
            #image_viewer {{
                background-color: {Theme.LIGHT_HOVER};
                border-radius: 8px;
            }}
        """)

    def apply_dark_theme(self):
        """应用深色主题"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Theme.DARK_BG};
            }}
            QWidget {{
                font-family: "Microsoft YaHei", "Source Han Sans", sans-serif;
                font-size: 13px;
                color: {Theme.DARK_TEXT};
            }}
            #top_navbar {{
                background-color: {Theme.DARK_CARD};
                border-bottom: 1px solid {Theme.DARK_BORDER};
            }}
            #nav_title {{
                color: {Theme.DARK_TEXT};
            }}
            #nav_subtitle {{
                color: {Theme.DARK_TEXT_SECONDARY};
            }}
            #left_sidebar {{
                background-color: {Theme.DARK_CARD};
                border-right: 1px solid {Theme.DARK_BORDER};
            }}
            #main_content {{
                background-color: {Theme.DARK_BG};
            }}
            #model_card, #stats_card {{
                background-color: {Theme.DARK_HOVER};
                border-radius: 12px;
                border: 1px solid {Theme.DARK_BORDER};
            }}
            #content_card {{
                background-color: {Theme.DARK_CARD};
                border-radius: 12px;
                border: 1px solid {Theme.DARK_BORDER};
            }}
            #primary_btn {{
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
                font-size: {self.scaled_font_size(10)}px;
                font-weight: bold;
            }}
            #primary_btn:hover {{
                background-color: #1249CC;
            }}
            #primary_btn:disabled {{
                background-color: #4A4A4A;
                color: #888888;
            }}
            #secondary_btn {{
                background-color: #3A3A3C;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
                font-size: {self.scaled_font_size(10)}px;
                font-weight: normal;
            }}
            #secondary_btn:hover {{
                background-color: #4A4A4C;
            }}
            #secondary_btn:disabled {{
                background-color: #4A4A4A;
                color: #888888;
            }}
            #function_btn {{
                background-color: #2C2C2E;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 0 16px;
                text-align: left;
                font-size: {self.scaled_font_size(11)}px;
                font-weight: normal;
            }}
            #function_btn:hover {{
                background-color: #3A3A3C;
            }}
            #function_btn_active {{
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 16px;
                text-align: left;
                font-size: {self.scaled_font_size(11)}px;
                font-weight: normal;
            }}
            #function_btn_active:hover {{
                background-color: #1249CC;
            }}
            #file_list {{
                background-color: {Theme.DARK_CARD};
                border: 1px solid {Theme.DARK_BORDER};
                border-radius: 8px;
                color: {Theme.DARK_TEXT};
            }}
            #file_list::item {{
                padding: 8px;
                border-radius: 4px;
            }}
            #file_list::item:selected {{
                background-color: {Theme.PRIMARY};
                color: white;
            }}
            QTableWidget {{
                background-color: {Theme.DARK_CARD};
                border: 1px solid {Theme.DARK_BORDER};
                border-radius: 8px;
                gridline-color: {Theme.DARK_BORDER};
                color: {Theme.DARK_TEXT};
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
            QTableWidget::item:selected {{
                background-color: {Theme.PRIMARY};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {Theme.DARK_HOVER};
                padding: 10px;
                border: none;
                font-weight: 600;
                color: {Theme.DARK_TEXT};
            }}
            QProgressBar {{
                background-color: {Theme.DARK_HOVER};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {Theme.PRIMARY};
                border-radius: 4px;
            }}
            #confidence_bar::chunk {{
                background-color: {Theme.SUCCESS};
            }}
            #placeholder {{
                background-color: {Theme.DARK_HOVER};
                color: {Theme.DARK_TEXT_SECONDARY};
                border-radius: 8px;
            }}
            #image_viewer {{
                background-color: {Theme.DARK_HOVER};
                border-radius: 8px;
            }}
        """)

    def open_train_gui(self):
        """打开模型训练GUI"""
        try:
            import subprocess
            import sys
            
            # 使用项目相对路径启动训练GUI
            train_gui_path = Path(__file__).parent / "train_gui.py"
            
            if not train_gui_path.exists():
                QMessageBox.warning(
                    self, 
                    "警告", 
                    f"训练界面文件未找到: {train_gui_path}\n请确保 train_gui.py 存在于项目根目录。"
                )
                return
            
            # 启动训练GUI
            subprocess.Popen([sys.executable, str(train_gui_path)])
            self.status_bar.showMessage("已启动模型训练界面")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动训练界面失败: {str(e)}")

    def toggle_theme(self):
        """切换深浅模式"""
        self.is_dark_mode = not self.is_dark_mode
        self.theme_btn.setText("☀️" if self.is_dark_mode else "🌙")
        self.apply_theme()

    def switch_function(self, index):
        """切换功能"""
        # 更新按钮状态
        for i, btn in enumerate(self.function_btns):
            if i == index:
                btn.setObjectName("function_btn_active")
                btn.setChecked(True)
            else:
                btn.setObjectName("function_btn")
                btn.setChecked(False)

        # 应用样式更新
        self.apply_theme()

        # 切换操作区
        self.operation_stack.setCurrentIndex(index)

        # 切换主内容区视图
        if index == 0:  # 单张图像检测
            self.content_stack.setCurrentIndex(0)
            if hasattr(self, 'stats_card'):
                self.stats_card.setVisible(False)
        elif index == 1:  # 批量图像检测
            self.content_stack.setCurrentIndex(1)
            if hasattr(self, 'stats_card'):
                self.stats_card.setVisible(True)
        elif index == 2:  # NII文件检测
            self.content_stack.setCurrentIndex(2)  # NII视图
            if hasattr(self, 'stats_card'):
                self.stats_card.setVisible(False)
        elif index == 3:  # 摄像头检测
            self.content_stack.setCurrentIndex(3)  # 摄像头视图
            if hasattr(self, 'stats_card'):
                self.stats_card.setVisible(False)
        
        # 切换模式时完全重置所有数据和状态
        self.reset_results(clear_chat=True)
        
        # 重置语音播报相关数据
        if hasattr(self, 'voice_text'):
            self.voice_text = ""
        if hasattr(self, 'voice_engine'):
            try:
                self.voice_engine.stop()
            except:
                pass
        
        # 重置检测结果
        self.last_detection_result = None
        self.current_image_path = None
        self.classification_results_data = []
        self.tumor_boxes = []
        
        # 重置批量检测数据
        self.batch_file_list = []
        self.batch_results = []
        self.total_processed = 0
        self.success_count = 0
        self.error_count = 0
        self.current_preview_index = -1
        
        # 重置批量统计显示
        if hasattr(self, 'batch_stats_text'):
            self.batch_stats_text.clear()
        if hasattr(self, 'batch_table'):
            self.batch_table.setRowCount(0)
        
        # 重置单张图像显示
        if hasattr(self, 'single_image_viewer'):
            self.single_image_viewer.scene.clear()
        
        # 重置NII相关数据
        if hasattr(self, 'mri_processor'):
            self.mri_processor = None
        self.current_nii_slice_idx = 0
        self.nii_files = {}
        
        # 停止摄像头（如果在运行）
        if hasattr(self, 'is_camera_running') and self.is_camera_running:
            self.stop_camera()
        
        # 更新状态栏
        self.status_bar.showMessage(f"已切换到{'单张图像检测' if index == 0 else '批量图像检测' if index == 1 else 'NII文件检测' if index == 2 else '摄像头检测'}模式")

    def on_model_mode_changed(self, index):
        """模型模式切换回调"""
        self.model_mode = self.model_mode_combo.currentData()
        print(f"[INFO] 切换到模型模式: {self.model_mode}")
        self.check_models()
        self.status_bar.showMessage(f"已切换到{'双模型协同' if self.model_mode == 'dual' else '单模型分类+分割'}模式")

    def check_models(self):
        """检查模型文件 - 根据当前模式显示不同状态"""
        if self.model_mode == "dual":
            # 双模型模式
            cls_exists = os.path.exists(self.model_classification_path)
            seg_exists = os.path.exists(self.model_segmentation_path)

            if cls_exists:
                self.model_cls_status.setText(f"✅ 分类模型: 已加载")
                self.model_cls_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.SUCCESS};")
            else:
                self.model_cls_status.setText(f"❌ 分类模型: 未加载")
                self.model_cls_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.WARNING};")

            if seg_exists:
                self.model_seg_status.setText(f"✅ 分割模型: 已加载")
                self.model_seg_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.SUCCESS};")
            else:
                self.model_seg_status.setText(f"❌ 分割模型: 未加载")
                self.model_seg_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.WARNING};")
        else:
            # 单模型模式
            combined_exists = os.path.exists(self.model_combined_path)

            if combined_exists:
                self.model_cls_status.setText(f"✅ 分类+分割模型: 已加载")
                self.model_cls_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.SUCCESS};")
            else:
                self.model_cls_status.setText(f"❌ 分类+分割模型: 未加载")
                self.model_cls_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.WARNING};")

            # 单模型模式下分割模型状态显示为"已合并"
            self.model_seg_status.setText(f"ℹ️ 分割功能: 已合并到单模型")
            self.model_seg_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.NEUTRAL};")

    def check_nii_support(self):
        """检查NII支持"""
        if NII_AVAILABLE:
            self.model_nii_status.setText("✅ NII支持: 已启用")
            self.model_nii_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.SUCCESS};")
        else:
            self.model_nii_status.setText("❌ NII支持: 未启用")
            self.model_nii_status.setStyleSheet(f"font-size: {self.scaled_font_size(16)}px; color: {Theme.WARNING};")

    def update_stats(self):
        """更新统计信息"""
        self.lbl_total.setText(f"总处理: {self.total_processed}")
        self.lbl_success.setText(f"成功: {self.success_count}")
        self.lbl_error.setText(f"失败: {self.error_count}")

    def show_segmentation_fullscreen(self):
        """全屏查看分割结果"""
        if hasattr(self, 'current_seg_pixmap') and self.current_seg_pixmap:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle("分割结果 - 全屏查看")
            dialog.setGeometry(100, 100, 1200, 800)
            
            layout = QVBoxLayout(dialog)
            
            # 创建全屏图像查看器
            fullscreen_viewer = ImageViewer()
            fullscreen_viewer.set_image(self.current_seg_pixmap)
            layout.addWidget(fullscreen_viewer)
            
            dialog.exec_()

    def show_batch_segmentation_fullscreen(self):
        """全屏查看批量检测结果"""
        if hasattr(self, 'current_batch_seg_pixmap') and self.current_batch_seg_pixmap:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle("批量检测结果 - 全屏查看")
            dialog.setGeometry(100, 100, 1200, 800)
            
            layout = QVBoxLayout(dialog)
            
            # 创建全屏图像查看器
            fullscreen_viewer = ImageViewer()
            fullscreen_viewer.set_image(self.current_batch_seg_pixmap)
            layout.addWidget(fullscreen_viewer)
            
            dialog.exec_()

    # ==================== 单张图像功能 ====================

    def select_image(self):
        """选择图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*.*)"
        )

        if file_path:
            self.current_image_path = file_path
            self.lbl_image_info.setText(f"已选择: {os.path.basename(file_path)}")

            # 显示原始图像
            pixmap = QPixmap(file_path)
            self.image_viewer.set_image(pixmap)

            # 更新图像信息
            img = Image.open(file_path)
            self.lbl_image_meta.setText(f"{os.path.basename(file_path)} | {img.size[0]}x{img.size[1]} | {img.format}")

            # 启用检测按钮
            self.btn_classify.setEnabled(True)
            self.btn_segment.setEnabled(True)
            self.btn_run_all.setEnabled(True)

            # 重置结果
            self.reset_results()

    def reset_results(self, clear_chat=False):
        """重置结果显示
        
        Args:
            clear_chat: 是否清空聊天记录，默认False（选择新图片时不清空，切换模式时清空）
        """
        # 重置分类结果（如果存在）
        if hasattr(self, 'cls_status'):
            self.cls_status.setText("未检测")
            self.cls_status.setStyleSheet("""
                font-size: 24px;
                font-weight: 600;
                padding: 30px;
                border-radius: 12px;
                background-color: #F2F3F5;
                color: #86909C;
            """)
        if hasattr(self, 'cls_confidence_label'):
            self.cls_confidence_label.setText("置信度: -")
        if hasattr(self, 'cls_category'):
            self.cls_category.setText("-")
        
        # 重置分割图像查看器
        if hasattr(self, 'seg_image_viewer') and self.seg_image_viewer:
            self.seg_image_viewer.scene.clear()
        self.current_seg_pixmap = None
        
        # 重置AI诊断建议（仅在切换模式时清空）
        if clear_chat:
            if hasattr(self, 'diagnosis_text'):
                self.diagnosis_text.clear()
            if hasattr(self, 'nii_diagnosis_text'):
                self.nii_diagnosis_text.clear()

    def run_classification(self):
        """运行分类检测 - 双结合模式"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图像")
            return

        self.status_bar.showMessage("正在运行分类检测...")
        self.cls_worker = ModelWorker(
            self.model_classification_path,
            self.current_image_path,
            "classification"
        )
        self.cls_worker.finished_signal.connect(self.on_single_detection_complete)
        self.cls_worker.start()

    def run_combined_detection(self):
        """运行单模型（分类+分割）检测"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图像")
            return

        # 检查单模型文件是否存在
        if not os.path.exists(self.model_combined_path):
            QMessageBox.warning(self, "警告", f"单模型文件未找到: {self.model_combined_path}")
            return

        self.status_bar.showMessage("正在运行单模型分类+分割检测...")
        self.combined_worker = CombinedModelWorker(
            self.model_combined_path,
            self.current_image_path
        )
        self.combined_worker.finished_signal.connect(self.on_combined_detection_complete)
        self.combined_worker.start()

    def on_combined_detection_complete(self, result_img, results, error):
        """单模型检测完成回调"""
        if error:
            QMessageBox.critical(self, "错误", f"检测失败: {error}")
            self.status_bar.showMessage("检测失败")
            return

        # 显示结果图像
        self.display_combined_image(result_img, results)

        # 处理检测结果数据
        has_tumor = False
        tumor_count = 0
        self.classification_results_data = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy()

                    self.classification_results_data.append({
                        'cls_id': cls_id,
                        'cls_name': cls_name,
                        'conf': conf,
                        'box': xyxy
                    })

                    if cls_name.lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']:
                        has_tumor = True
                        tumor_count += 1

        # 保存检测结果供快捷功能使用
        if has_tumor and self.classification_results_data:
            first_tumor = self.classification_results_data[0]
            tumor_name_en = first_tumor['cls_name']
            tumor_name_cn = self.get_tumor_name_cn(tumor_name_en)
            box = first_tumor['box']
            width = box[2] - box[0]
            height = box[3] - box[1]

            self.last_detection_result = {
                'tumor_type': tumor_name_cn,
                'tumor_type_en': tumor_name_en,
                'confidence': first_tumor['conf'],
                'size': f"{width:.0f} x {height:.0f} 像素",
                'position': self.get_brain_region(box),
                'volume': f"{width * height:.0f} 像素²",
                'has_tumor': True,
                'box': box
            }
        else:
            self.last_detection_result = {
                'tumor_type': '无肿瘤',
                'tumor_type_en': 'No Tumor',
                'confidence': 0,
                'has_tumor': False
            }

        # 更新状态栏
        if has_tumor:
            self.status_bar.showMessage(f"检测完成 - 识别到 {tumor_count} 个肿瘤区域（含分割掩码）")
        else:
            self.status_bar.showMessage("检测完成 - 健康大脑")

        # 显示AI诊断建议
        if self.classification_results_data:
            first_result = self.classification_results_data[0]
            self.generate_diagnosis_suggestion(
                first_result['cls_name'],
                first_result['conf'],
                has_tumor,
                False,
                None,
                self.last_detection_result
            )

    def display_combined_image(self, result_img, results):
        """显示单模型检测结果图像"""
        import cv2
        import numpy as np
        from PyQt5.QtGui import QImage, QPixmap

        print(f"[DEBUG] display_combined_image called")
        print(f"[DEBUG] result_img type: {type(result_img)}")
        
        # 转换图像格式
        if isinstance(result_img, np.ndarray):
            print(f"[DEBUG] result_img shape: {result_img.shape}, dtype: {result_img.dtype}")
            
            # 确保图像数据是uint8格式
            if result_img.dtype != np.uint8:
                # 归一化到0-255
                result_img = ((result_img - result_img.min()) / (result_img.max() - result_img.min() + 1e-8) * 255).astype(np.uint8)
                print(f"[DEBUG] 图像已归一化")
            
            height, width = result_img.shape[:2]
            channels = result_img.shape[2] if len(result_img.shape) == 3 else 1
            
            print(f"[DEBUG] 图像尺寸: {width}x{height}, 通道数: {channels}")
            
            bytes_per_line = channels * width
            
            if channels == 3:
                q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(q_image)
            print(f"[DEBUG] pixmap created: {pixmap.width()}x{pixmap.height()}")
        else:
            pixmap = result_img
            print(f"[DEBUG] 使用原始pixmap")

        # 检查pixmap是否有效
        if pixmap is None or pixmap.isNull():
            print("[ERROR] display_combined_image: 无效的pixmap")
            return
            
        self.current_seg_pixmap = pixmap

        # 判断是否为NII模式
        is_nii_mode = hasattr(self, 'mri_processor') and hasattr(self, 'current_nii_slice_idx')
        if is_nii_mode and hasattr(self, 'nii_seg_viewer'):
            self.nii_seg_viewer.set_image(pixmap)
        else:
            # 确保seg_image_viewer存在
            if hasattr(self, 'seg_image_viewer') and self.seg_image_viewer:
                self.seg_image_viewer.set_image(pixmap)
                print(f"[DEBUG] 图像已显示到seg_image_viewer")
            else:
                print("[ERROR] seg_image_viewer 不存在")

        # 更新分类状态显示
        if hasattr(self, 'cls_status'):
            has_tumor = any(
                item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']
                for item in self.classification_results_data
            ) if self.classification_results_data else False

            if has_tumor:
                self.cls_status.setText("检测到肿瘤")
                self.cls_status.setStyleSheet("""
                    font-size: 24px;
                    font-weight: 600;
                    padding: 30px;
                    border-radius: 12px;
                    background-color: #FFE8E8;
                    color: #F53F3F;
                """)
            else:
                self.cls_status.setText("健康大脑")
                self.cls_status.setStyleSheet("""
                    font-size: 24px;
                    font-weight: 600;
                    padding: 30px;
                    border-radius: 12px;
                    background-color: #E8FFEA;
                    color: #00B42A;
                """)

        if hasattr(self, 'cls_confidence_label') and self.classification_results_data:
            conf = self.classification_results_data[0]['conf']
            self.cls_confidence_label.setText(f"置信度: {conf:.2%}")

        if hasattr(self, 'cls_category') and self.classification_results_data:
            cls_name = self.classification_results_data[0]['cls_name']
            self.cls_category.setText(self.get_tumor_name_cn(cls_name))

    def run_segmentation(self):
        """运行实例分割（在best5识别的肿瘤框内）"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图像")
            return
        
        if not hasattr(self, 'tumor_boxes') or not self.tumor_boxes:
            QMessageBox.information(self, "提示", "没有检测到肿瘤区域，无需分割")
            return

        self.status_bar.showMessage("正在运行实例分割...")
        
        # 在肿瘤框内进行分割
        self.run_segmentation_in_boxes()

    def run_segmentation_in_boxes(self):
        """两个模型全图推理，只保留共同识别框 - 使用工作线程"""
        # 创建工作线程
        self.seg_worker = SegmentationWorker(
            self.current_image_path,
            self.model_segmentation_path,
            self.classification_results_data
        )
        self.seg_worker.finished_signal.connect(self.on_segmentation_finished)
        self.seg_worker.error_signal.connect(self.on_segmentation_error)
        self.seg_worker.start()

    def on_segmentation_finished(self, result_img, all_seg_results):
        """分割完成回调"""
        print(f"[DEBUG] on_segmentation_finished called")
        print(f"[DEBUG] result_img shape: {result_img.shape if hasattr(result_img, 'shape') else 'N/A'}")
        print(f"[DEBUG] all_seg_results length: {len(all_seg_results)}")
        
        # 检查是否有分割掩码（判断是否有共同识别区域）
        has_segmentation = any(isinstance(r, object) and hasattr(r, 'masks') and r.masks is not None 
                              for r in all_seg_results if not isinstance(r, dict))
        
        # 显示分割结果
        self.display_segmentation_image(result_img, all_seg_results)
        
        # 统计best5识别的肿瘤数量（从classification_results_data中获取）
        tumor_count = sum(1 for item in self.classification_results_data 
                         if item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常'])
        
        if has_segmentation:
            self.status_bar.showMessage(f"分割完成 - 识别到 {tumor_count} 个肿瘤区域（含分割掩码）")
        else:
            self.status_bar.showMessage(f"分割完成 - 识别到 {tumor_count} 个肿瘤区域（无分割掩码）")
        
        # 清理临时文件
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if hasattr(self, 'current_image_path') and self.current_image_path:
                # 检查是否是临时文件（在系统临时目录中）
                temp_dir = tempfile.gettempdir()
                if self.current_image_path.startswith(temp_dir):
                    if os.path.exists(self.current_image_path):
                        os.remove(self.current_image_path)
                        print(f"[DEBUG] 清理临时文件: {self.current_image_path}")
        except Exception as e:
            print(f"[DEBUG] 清理临时文件失败: {e}")
    
    def on_segmentation_error(self, error_msg):
        """分割错误回调"""
        # 清理临时文件
        self._cleanup_temp_files()
        
        if "未找到两个模型共同识别的区域" in error_msg:
            self.status_bar.showMessage(error_msg)
            # 获取检测到的肿瘤类型
            if hasattr(self, 'classification_results_data') and self.classification_results_data:
                tumor_items = [item for item in self.classification_results_data 
                              if item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']]
                if tumor_items:
                    tumor_names_en = list(dict.fromkeys([item['cls_name'] for item in tumor_items]))[:3]
                    tumor_names_cn = [self.get_tumor_name_cn(name) for name in tumor_names_en]
                    tumor_type_str = '、'.join(tumor_names_cn)
                    message = f"检测到: {tumor_type_str}\n但分割模型未能在相同位置识别到肿瘤区域"
                else:
                    message = "检测到肿瘤\n但分割模型未能在相同位置识别到肿瘤区域"
            else:
                message = "检测到肿瘤\n但分割模型未能在相同位置识别到肿瘤区域"
            # 显示提示图像 - 没有共同识别区域
            self.show_no_segmentation_result(message)
        else:
            QMessageBox.critical(self, "错误", f"分割失败: {error_msg}")
            self.status_bar.showMessage("分割失败")
            # 显示错误提示图像
            self.show_no_segmentation_result(f"分割失败: {error_msg}")

    def show_original_image(self):
        """显示原始图像（无肿瘤时）"""
        try:
            import cv2
            from PyQt5.QtGui import QImage, QPixmap
            
            # 读取原图
            original_img = cv2.imread(self.current_image_path)
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # 转换为QPixmap
            height, width, channels = original_img_rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(original_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # 保存当前图像用于全屏查看
            self.current_seg_pixmap = pixmap
            
            # 显示图像
            self.seg_image_viewer.set_image(pixmap)
            
        except Exception as e:
            print(f"[DEBUG] 显示原图失败: {e}")
    
    def show_no_segmentation_result(self, message):
        """显示无分割结果的提示"""
        # 创建一个空白图像显示提示信息
        from PyQt5.QtGui import QPainter, QFont, QColor
        
        pixmap = QPixmap(640, 480)
        pixmap.fill(QColor(240, 240, 240))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont("Microsoft YaHei", 16))
        
        # 绘制提示文字
        rect = pixmap.rect()
        painter.drawText(rect, Qt.AlignCenter, message)
        painter.end()
        
        # 保存并显示
        self.current_seg_pixmap = pixmap
        
        # 判断是否为NII模式，选择正确的图像查看器
        is_nii_mode = hasattr(self, 'mri_processor') and hasattr(self, 'current_nii_slice_idx')
        if is_nii_mode and hasattr(self, 'nii_seg_viewer'):
            self.nii_seg_viewer.set_image(pixmap)
        else:
            self.seg_image_viewer.set_image(pixmap)

    def display_segmentation_image(self, seg_img_rgb, seg_results):
        """显示最终识别结果图像"""
        print(f"[DEBUG] display_segmentation_image called")
        print(f"[DEBUG] seg_img_rgb shape: {seg_img_rgb.shape if hasattr(seg_img_rgb, 'shape') else 'N/A'}")
        print(f"[DEBUG] seg_img_rgb dtype: {seg_img_rgb.dtype}")
        
        # 检查输入图像是否有效
        if seg_img_rgb is None:
            print("[ERROR] display_segmentation_image: seg_img_rgb is None")
            return
        
        # 确保图像数据是uint8格式
        if isinstance(seg_img_rgb, np.ndarray):
            if seg_img_rgb.dtype != np.uint8:
                # 归一化到0-255
                seg_img_rgb = ((seg_img_rgb - seg_img_rgb.min()) / (seg_img_rgb.max() - seg_img_rgb.min() + 1e-8) * 255).astype(np.uint8)
                print(f"[DEBUG] 图像已归一化")
        
        # 显示分割图像 - 使用支持缩放的ImageViewer
        height, width = seg_img_rgb.shape[:2]
        channels = seg_img_rgb.shape[2] if len(seg_img_rgb.shape) == 3 else 1
        bytes_per_line = channels * width
        
        if channels == 3:
            q_image = QImage(seg_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            q_image = QImage(seg_img_rgb.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        print(f"[DEBUG] pixmap size: {pixmap.width()}x{pixmap.height()}")
        
        # 检查pixmap是否有效
        if pixmap is None or pixmap.isNull():
            print("[ERROR] display_segmentation_image: 无效的pixmap")
            return
        
        # 保存当前分割图像用于全屏查看
        self.current_seg_pixmap = pixmap
        
        # 判断是否为NII模式，选择正确的图像查看器
        is_nii_mode = hasattr(self, 'mri_processor') and hasattr(self, 'current_nii_slice_idx')
        if is_nii_mode and hasattr(self, 'nii_seg_viewer'):
            target_viewer = self.nii_seg_viewer
        else:
            target_viewer = self.seg_image_viewer
        
        # 确保目标查看器存在
        if not hasattr(self, 'seg_image_viewer') or not self.seg_image_viewer:
            print("[ERROR] display_segmentation_image: seg_image_viewer 不存在")
            return
        
        # 使用ImageViewer显示（支持滚轮缩放、拖拽平移、双击重置）
        print(f"[DEBUG] Calling target_viewer.set_image")
        target_viewer.set_image(pixmap)
        print(f"[DEBUG] set_image completed")

    def run_all_detection(self):
        """运行完整检测 - 根据当前模型模式选择检测方式"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图像")
            return

        self.reset_results()

        if self.model_mode == "dual":
            # 双模型协同模式：先运行分类，再运行分割
            self.status_bar.showMessage("开始完整检测流程 (双模型协同)...")
            self.run_classification()
        else:
            # 单模型分类+分割模式
            self.status_bar.showMessage("开始完整检测流程 (单模型分类+分割)...")
            self.run_combined_detection()

    def on_single_detection_complete(self, task_type, results, error):
        """单张检测完成回调 - 双结合模式"""
        if error:
            QMessageBox.critical(self, "错误", f"检测失败: {error}")
            self.status_bar.showMessage("检测失败")
            # 清理临时文件
            self._cleanup_temp_files()
            return

        if task_type == "classification":
            has_tumor = self.process_classification_results(results)
            # 只有检测到肿瘤时才运行分割
            if has_tumor:
                self.run_segmentation_in_boxes()
            else:
                self.status_bar.showMessage("检测完成 - 健康大脑，无需分割")
                # 显示原图（无肿瘤标记）
                self.show_original_image()
                # 清理临时文件
                self._cleanup_temp_files()

    def process_classification_results(self, results):
        """处理分类结果，返回是否检测到肿瘤"""
        raw_results = []
        has_tumor = False

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # 检查是否检测到肿瘤（类别名不是 "No Tumor" 或类似）
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id]
                    
                    # 获取检测框坐标
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # 存储分类信息
                    raw_results.append({
                        'cls_id': cls_id,
                        'cls_name': cls_name,
                        'conf': conf,
                        'box': xyxy
                    })
        
        # 对best5的结果进行NMS处理：去除重叠的框（IOU > 0.5），保留置信度高的
        self.classification_results_data = self.nms_boxes(raw_results, iou_threshold=0.5)
        
        # 判断是否为NII模式
        is_nii_mode = hasattr(self, 'mri_processor') and hasattr(self, 'current_nii_slice_idx')
        
        # 重新计算肿瘤框
        self.tumor_boxes = []
        for item in self.classification_results_data:
            if item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']:
                has_tumor = True
                self.tumor_boxes.append(item['box'])
        
        # 保存检测结果供快捷功能使用
        if has_tumor and self.classification_results_data:
            first_tumor = self.classification_results_data[0]
            tumor_name_en = first_tumor['cls_name']
            tumor_name_cn = self.get_tumor_name_cn(tumor_name_en)
            
            # 计算肿瘤大小和位置信息
            box = first_tumor['box']
            width = box[2] - box[0]
            height = box[3] - box[1]
            size_info = f"{width:.0f} x {height:.0f} 像素"
            # 获取大脑解剖位置
            brain_region = self.get_brain_region(box)
            position_info = brain_region
            volume_info = f"{width * height:.0f} 像素²"
            
            self.last_detection_result = {
                'tumor_type': tumor_name_cn,
                'tumor_type_en': tumor_name_en,
                'confidence': first_tumor['conf'],
                'size': size_info,
                'position': position_info,
                'volume': volume_info,
                'has_tumor': True,
                'box': box
            }
        else:
            self.last_detection_result = {
                'tumor_type': '无肿瘤',
                'tumor_type_en': 'No Tumor',
                'confidence': 0,
                'has_tumor': False
            }
        
        # 根据模式选择正确的诊断文本控件
        if is_nii_mode and hasattr(self, 'nii_diagnosis_text'):
            target_diagnosis_text = self.nii_diagnosis_text
        else:
            target_diagnosis_text = self.diagnosis_text if hasattr(self, 'diagnosis_text') else None
        
        # 显示检测结果 - 使用新的常驻头部显示方式
        if has_tumor and self.classification_results_data:
            # 显示第一个肿瘤检测结果
            first_tumor = self.classification_results_data[0]
            tumor_name_en = first_tumor['cls_name']
            tumor_name_cn = self.get_tumor_name_cn(tumor_name_en)
            
            # 在AI对话区域显示检测结果 - 使用新的update_diagnosis_display方法
            if target_diagnosis_text:
                loading_html = """
                <div style="color: #86909C; padding: 10px;">
                    <p>🤖 正在调用AI分析，请稍候...</p>
                </div>
                """
                self.update_diagnosis_display(target_diagnosis_text, loading_html)
            
            # 生成AI诊断建议 - 传递检测详细信息
            nii_slice_idx = self.current_nii_slice_idx if is_nii_mode else None
            self.generate_diagnosis_suggestion(first_tumor['cls_name'], first_tumor['conf'], True, is_nii_mode, nii_slice_idx, self.last_detection_result)
        else:
            # 所有检测都是健康/无肿瘤
            if target_diagnosis_text:
                loading_html = """
                <div style="color: #86909C; padding: 10px;">
                    <p>🤖 正在调用AI分析，请稍候...</p>
                </div>
                """
                self.update_diagnosis_display(target_diagnosis_text, loading_html)
            
            # 生成健康诊断建议
            nii_slice_idx = self.current_nii_slice_idx if is_nii_mode else None
            self.generate_diagnosis_suggestion("无肿瘤", 0, False, is_nii_mode, nii_slice_idx, self.last_detection_result)
        
        return has_tumor

    def get_tumor_name_cn(self, tumor_type_en):
        """将肿瘤类型英文名称转换为中文
        
        Args:
            tumor_type_en: 英文肿瘤类型名称
            
        Returns:
            中文肿瘤类型名称
        """
        tumor_name_map = {
            'Meningioma': '脑膜瘤',
            'Glioma': '胶质瘤',
            'Pituitary': '垂体瘤',
            'No Tumor': '无肿瘤',
            'Notumor': '无肿瘤',
            'Healthy': '健康',
            'Normal': '正常',
            '无肿瘤': '无肿瘤',
            '正常': '正常'
        }
        return tumor_name_map.get(tumor_type_en, tumor_type_en)

    def get_brain_region(self, box):
        """根据检测框位置判断大脑解剖位置
        
        Args:
            box: 检测框坐标 [x1, y1, x2, y2]
            
        Returns:
            大脑解剖位置描述
        """
        x1, y1, x2, y2 = box
        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 假设标准MRI图像尺寸约为 512x512 或 640x640
        # 将位置归一化到 0-1 范围（以中心点为原点）
        # 左脑在图像左侧 (x较小)，右脑在图像右侧 (x较大)
        # 上部是前额叶，下部是枕叶
        
        # 简化的脑部分区判断
        # 水平方向：左、中、右
        # 垂直方向：上（前额）、中（顶叶）、下（枕叶/小脑）
        
        regions = []
        
        # 左右判断
        if center_x < 0.45:
            regions.append("左侧")
        elif center_x > 0.55:
            regions.append("右侧")
        else:
            regions.append("中线/中央")
        
        # 前后判断（垂直方向）
        if center_y < 0.35:
            regions.append("前额叶")
        elif center_y < 0.5:
            regions.append("额叶")
        elif center_y < 0.65:
            regions.append("顶叶")
        elif center_y < 0.8:
            regions.append("颞叶")
        else:
            regions.append("枕叶")
        
        # 深度判断（根据框的大小粗略估计）
        width = x2 - x1
        height = y2 - y1
        area_ratio = (width * height) / (512 * 512)  # 假设标准尺寸
        
        depth_desc = ""
        if area_ratio > 0.15:
            depth_desc = "（较大占位）"
        elif area_ratio > 0.08:
            depth_desc = "（中等大小）"
        elif area_ratio > 0.03:
            depth_desc = "（较小病灶）"
        
        return "".join(regions) + depth_desc

    def generate_diagnosis_suggestion(self, tumor_type, confidence, has_tumor, is_nii=False, nii_slice_idx=None, detection_info=None):
        """生成AI诊断建议 - 已禁用"""
        # AI功能已禁用，不执行任何操作
        pass
    
    def _build_image_prompt(self, tumor_type_cn, confidence, has_tumor):
        """构建普通图像检测的提示词"""
        if not has_tumor:
            return """你是一位专业的脑肿瘤诊断AI助手。请根据以下检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：健康，未检测到肿瘤
- 检测方法：基于YOLO11模型的单张MRI图像分析

请提供：
1. 诊断结果说明
2. 健康建议
3. 后续注意事项

请以HTML格式输出，使用中文。"""
        else:
            return f"""你是一位专业的脑肿瘤诊断AI助手。请根据以下检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：检测到肿瘤
- 肿瘤类型：{tumor_type_cn}
- AI检测置信度：{confidence:.1%}
- 检测方法：基于YOLO11模型的单张MRI图像分析

请提供：
1. 该肿瘤类型的基本介绍
2. 典型影像特征
3. 治疗建议
4. 预后情况
5. 注意事项

请以HTML格式输出，使用中文。标题使用<h3>标签，段落使用<p>标签，列表使用<ul><li>标签。"""
    
    def _build_image_prompt_with_info(self, tumor_type_cn, confidence, has_tumor, detection_info):
        """构建带详细信息的单张图像检测提示词"""
        if not has_tumor:
            return """你是一位专业的脑肿瘤诊断AI助手。请根据以下检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：健康，未检测到肿瘤
- 检测方法：基于YOLO11模型的单张MRI图像分析

请提供：
1. 诊断结果说明
2. 健康建议
3. 后续注意事项

请以HTML格式输出，使用中文。250字左右。"""
        else:
            # 构建详细信息
            info_text = ""
            if detection_info:
                info_text += f"\n- 肿瘤大小: {detection_info.get('size', '未知')}"
                info_text += f"\n- 肿瘤位置: {detection_info.get('position', '未知')}"
                info_text += f"\n- 肿瘤体积: {detection_info.get('volume', '未知')}"
            
            return f"""你是一位专业的脑肿瘤诊断AI助手。请根据以下检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：检测到肿瘤
- 肿瘤类型：{tumor_type_cn}
- AI检测置信度：{confidence:.1%}
- 检测方法：基于YOLO11模型的单张MRI图像分析{info_text}

请提供：
1. 该肿瘤类型的基本介绍
2. 典型影像特征
3. 治疗建议
4. 预后情况
5. 注意事项

请以HTML格式输出，使用中文。250字左右。"""
    
    def _build_nii_prompt_with_info(self, tumor_type_cn, confidence, has_tumor, slice_idx, detection_info):
        """构建带详细信息的NII检测提示词"""
        try:
            tumor_info = self.mri_processor.analyze_tumor_info(slice_idx)
            available_modalities = list(self.mri_processor.files.keys())
            
            regions_info = ""
            if tumor_info['has_tumor']:
                for label_value, region in tumor_info['regions'].items():
                    label_name = MultiModalMRIProcessor.SEG_LABELS.get(label_value, {}).get('name', '未知')
                    regions_info += f"- {label_name}: {region['pixel_count']}像素, {region['area_mm2']:.2f}mm²\n"
            
            # 添加额外信息
            extra_info = ""
            if detection_info:
                extra_info += f"\n- 肿瘤总体积: {detection_info.get('total_volume', '未知')} mm³"
                extra_info += f"\n- 涉及切片数: {detection_info.get('slice_count', '未知')}"
            
            return f"""你是一位专业的脑肿瘤诊断AI助手。请根据以下多模态MRI检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：{"检测到肿瘤" if has_tumor else "未检测到肿瘤"}
- 肿瘤类型：{tumor_type_cn}
- AI检测置信度：{confidence:.1%}
- 当前切片：第{slice_idx}层
- 可用模态数据：{', '.join(available_modalities)}{extra_info}

肿瘤区域分析：
{regions_info if regions_info else "当前切片未显示明显肿瘤区域"}

请提供：
1. 综合诊断分析
2. 各模态数据解读
3. 治疗建议
4. 后续检查建议

请以HTML格式输出，使用中文。250字左右。"""
        except Exception as e:
            return f"""你是一位专业的脑肿瘤诊断AI助手。请根据以下检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：{"检测到肿瘤" if has_tumor else "未检测到肿瘤"}
- 肿瘤类型：{tumor_type_cn}
- AI检测置信度：{confidence:.1%}
- 当前切片：第{slice_idx}层

请提供诊断建议。250字左右。"""
    
    def _build_nii_prompt(self, tumor_type_cn, confidence, has_tumor, slice_idx):
        """构建NII多模态检测的提示词"""
        try:
            tumor_info = self.mri_processor.analyze_tumor_info(slice_idx)
            available_modalities = list(self.mri_processor.files.keys())
            
            regions_info = ""
            if tumor_info['has_tumor']:
                for label_value, region in tumor_info['regions'].items():
                    label_name = MultiModalMRIProcessor.SEG_LABELS.get(label_value, {}).get('name', '未知')
                    regions_info += f"- {label_name}: {region['pixel_count']}像素, {region['area_mm2']:.2f}mm²\n"
            
            return f"""你是一位专业的脑肿瘤诊断AI助手。请根据以下多模态MRI检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：{"检测到肿瘤" if has_tumor else "未检测到肿瘤"}
- 肿瘤类型：{tumor_type_cn}
- AI检测置信度：{confidence:.1%}
- 当前切片：第{slice_idx}层
- 可用模态数据：{', '.join(available_modalities)}

肿瘤区域分析：
{regions_info if regions_info else "当前切片未显示明显肿瘤区域"}

请提供：
1. 综合诊断分析
2. 各模态数据解读
3. 治疗建议
4. 后续检查建议

请以HTML格式输出，使用中文。标题使用<h3>标签，段落使用<p>标签，列表使用<ul><li>标签。"""
        except Exception as e:
            return f"""你是一位专业的脑肿瘤诊断AI助手。请根据以下检测结果，为患者提供详细的诊断建议。

检测结果：
- 状态：{"检测到肿瘤" if has_tumor else "未检测到肿瘤"}
- 肿瘤类型：{tumor_type_cn}
- AI检测置信度：{confidence:.1%}
- 当前切片：第{slice_idx}层

请提供详细的诊断建议，以HTML格式输出，使用中文。"""

    def get_detection_header_html(self):
        """获取检测结果头部HTML - 常驻显示"""
        if not hasattr(self, 'last_detection_result') or not self.last_detection_result:
            return ""
        
        result = self.last_detection_result
        has_tumor = result.get('has_tumor', False)
        
        if has_tumor:
            tumor_type = result.get('tumor_type', '未知')
            confidence = result.get('confidence', 0)
            
            return f"""
            <div style="background: linear-gradient(135deg, #E8F4FF 0%, #D6EBFF 100%); 
                        border: 2px solid #165DFF; padding: 16px; margin: 0 0 15px 0; 
                        border-radius: 12px; box-shadow: 0 4px 12px rgba(22, 93, 255, 0.15);">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 12px; color: #165DFF;">
                    🔍 YOLO检测结果
                </div>
                <table cellpadding="0" cellspacing="0" border="0">
                    <tr>
                        <td style="padding-right: 30px;">
                            <div style="color: #1D2129; font-size: 13px; margin-bottom: 4px;">肿瘤类型</div>
                            <div style="color: #F53F3F; font-size: 22px; font-weight: bold;">{tumor_type}</div>
                        </td>
                        <td style="border-left: 2px solid #165DFF; padding-left: 30px;">
                            <div style="color: #1D2129; font-size: 13px; margin-bottom: 4px;">置信度</div>
                            <div style="color: #165DFF; font-size: 22px; font-weight: bold;">{confidence:.1%}</div>
                        </td>
                    </tr>
                </table>
            </div>
            """
        else:
            return """
            <div style="background: linear-gradient(135deg, #E8FFEA 0%, #D6F5D9 100%); 
                        border: 2px solid #00B42A; padding: 16px; margin: 0 0 15px 0; 
                        border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 180, 42, 0.15);">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 8px; color: #00B42A;">
                    ✅ YOLO检测结果
                </div>
                <div style="color: #1D2129; font-size: 18px; font-weight: bold;">
                    未检测到肿瘤 - 健康大脑
                </div>
            </div>
            """
    
    def update_diagnosis_display(self, text_widget, content_html):
        """更新诊断显示区域 - 保留检测结果头部"""
        header_html = self.get_detection_header_html()
        
        # 组合头部和内容
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ margin: 0; padding: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; }}
            </style>
        </head>
        <body>
            {header_html}
            <div id="chat-content">
                {content_html}
            </div>
        </body>
        </html>
        """
        text_widget.setHtml(full_html)
    
    def on_ai_reply_received(self, reply, text_widget):
        """处理AI回复 - 保留检测结果头部"""
        # 移除加载提示
        import re
        
        # 获取当前内容（不包括头部）
        current_html = text_widget.toHtml()
        
        # 提取chat-content部分的内容
        chat_content = ""
        if '<div id="chat-content">' in current_html:
            start = current_html.find('<div id="chat-content">') + len('<div id="chat-content">')
            end = current_html.find('</div>', start)
            if end > start:
                chat_content = current_html[start:end].strip()
        
        # 移除包含"思考中"的表格
        chat_content = re.sub(r'<table[^>]*>.*?🤔 思考中．．．.*?</table>.*?<div[^>]*>.*?</div>', '', chat_content, flags=re.DOTALL)
        # 再试一次简单的字符串替换
        if '🤔 思考中' in chat_content:
            start_idx = chat_content.find('<table')
            while start_idx != -1:
                end_idx = chat_content.find('</table>', start_idx)
                if end_idx != -1 and '🤔 思考中' in chat_content[start_idx:end_idx]:
                    div_start = chat_content.find('<div', end_idx)
                    div_end = chat_content.find('</div>', div_start) if div_start != -1 else -1
                    if div_start != -1 and div_end != -1:
                        chat_content = chat_content[:start_idx] + chat_content[div_end + 6:]
                    break
                start_idx = chat_content.find('<table', start_idx + 1)
        
        # 将Markdown格式转换为HTML
        formatted_reply = self._markdown_to_html(reply)
        
        ai_html = f"""
        <table width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="left">
                    <div style="display: inline-block; background-color: #F2F3F5; color: #333; padding: 10px 16px; border-radius: 12px 12px 12px 2px; max-width: 80%; font-size: 15px;">
                        {formatted_reply}
                    </div>
                </td>
            </tr>
        </table>
        <div style="height: 10px;"></div>
        """
        
        # 使用update_diagnosis_display更新显示
        self.update_diagnosis_display(text_widget, chat_content + ai_html)
        
        # 滚动到底部
        scrollbar = text_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _markdown_to_html(self, text):
        """将Markdown格式转换为HTML"""
        import html
        import re
        
        # 先转义HTML特殊字符
        text = html.escape(text)
        
        # 处理标题 (# 标题)
        text = re.sub(r'^###\s+(.+)$', r'<h3 style="margin: 8px 0; font-size: 16px; color: #1D2129;">\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'<h2 style="margin: 10px 0; font-size: 18px; color: #1D2129;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.+)$', r'<h1 style="margin: 12px 0; font-size: 20px; color: #165DFF;">\1</h1>', text, flags=re.MULTILINE)
        
        # 处理粗体 (**文本**)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #165DFF;">\1</strong>', text)
        
        # 处理斜体 (*文本*)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        
        # 处理换行
        text = text.replace('\n', '<br>')
        
        return text

    def quick_recognize_image(self):
        """快捷功能：识别检测照片并发送给AI"""
        # 获取当前检测结果图片
        image_base64 = self._get_detection_result_image_base64()
        if not image_base64:
            QMessageBox.information(self, "提示", "请先进行肿瘤检测")
            return
        
        # 判断当前模式
        is_nii = hasattr(self, 'mri_processor') and hasattr(self, 'current_nii_slice_idx')
        if is_nii and hasattr(self, 'nii_diagnosis_text'):
            text_widget = self.nii_diagnosis_text
            input_widget = self.nii_diagnosis_input
        else:
            text_widget = self.diagnosis_text
            input_widget = self.diagnosis_input
        
        # 显示用户消息
        message = "请帮我分析这张检测照片"
        import html
        escaped_message = html.escape(message)
        user_html = f"""
        <table width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="right">
                    <div style="display: inline-block; background-color: #0078D4; color: white; padding: 10px 16px; border-radius: 8px; max-width: 80%; text-align: left; font-size: 15px;">
                        {escaped_message}
                    </div>
                </td>
            </tr>
        </table>
        <div style="height: 10px;"></div>
        """
        
        current_html = text_widget.toHtml()
        if "请先进行肿瘤检测" in current_html or "正在调用AI分析" in current_html:
            text_widget.setHtml(user_html)
        else:
            if "<hr>" in current_html:
                current_html = current_html.split("<hr>")[0]
            text_widget.setHtml(current_html + user_html)
        
        # 显示加载中
        loading_html = """
        <table width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="left">
                    <div style="display: inline-block; background-color: white; color: #333; padding: 10px 16px; border-radius: 8px; font-size: 15px;">
                        🤔 正在分析图片...
                    </div>
                </td>
            </tr>
        </table>
        <div style="height: 10px;"></div>
        """
        text_widget.setHtml(text_widget.toHtml() + loading_html)
        scrollbar = text_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 调用API分析图片
        self._call_api_with_image(message, image_base64, text_widget)

    def quick_voice_broadcast(self):
        """快捷功能：语音播报检测结果和AI分析内容，支持暂停/开始"""
        try:
            # 判断当前模式，获取对应的诊断文本控件
            is_nii_mode = hasattr(self, 'mri_processor') and hasattr(self, 'current_nii_slice_idx')
            if is_nii_mode and hasattr(self, 'nii_diagnosis_text'):
                target_text = self.nii_diagnosis_text
            else:
                target_text = self.diagnosis_text if hasattr(self, 'diagnosis_text') else None
            
            if not target_text:
                QMessageBox.information(self, "提示", "无法获取对话内容")
                return
            
            # 获取所有文本内容
            full_html = target_text.toHtml()
            
            # 提取纯文本（去除HTML标签和特殊符号）
            import re
            # 移除HTML标签
            text = re.sub(r'<[^>]+>', '', full_html)
            # 移除CSS样式代码
            text = re.sub(r'[a-zA-Z-]+:\s*[^;{}]+[;}]?', '', text)
            # 移除emoji和特殊符号（只保留中文、数字、常用标点）
            text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303F\uFF00-\uFFEF0-9\s，。！？、：；]', '', text)
            # 移除多余空白
            text = re.sub(r'\s+', ' ', text).strip()
            # 移除提示文本
            text = text.replace('请先进行肿瘤检测，AI将为您提供诊断建议... 检测完成后，您可以直接在这里输入问题与AI对话。', '')
            text = text.replace('正在调用AI分析，请稍候...', '')
            text = text.replace('正在分析图片...', '')
            
            # 如果没有内容，使用检测结果生成通顺的播报文本
            if not text:
                if hasattr(self, 'last_detection_result'):
                    result = self.last_detection_result
                    has_tumor = result.get('has_tumor', False)
                    if has_tumor:
                        tumor_type = result.get('tumor_type', '未知')
                        confidence = result.get('confidence', 0)
                        position = result.get('position', '未知')
                        size = result.get('size', '未知')
                        # 将像素替换为更自然的说法
                        size = size.replace('像素', '').replace('x', '乘')
                        text = f"检测发现{tumor_type}，置信度为百分之{int(confidence * 100)}，位于{position}，大小约为{size}。"
                    else:
                        text = "检测结果显示未检测到肿瘤，大脑健康状况良好。"
                else:
                    text = "暂无检测结果，请先进行肿瘤检测。"
            
            # 创建语音播报控制对话框
            self._show_voice_broadcast_dialog(text)
            
        except Exception as e:
            QMessageBox.information(self, "提示", f"语音播报功能暂不可用: {str(e)}")
    
    def _show_voice_broadcast_dialog(self, text):
        """显示语音播报控制对话框"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, QProgressBar
        
        dialog = QDialog(self)
        dialog.setWindowTitle("🔊 语音播报")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #F7F8FA;
            }
            QTextEdit {
                border: 1px solid #E5E6EB;
                border-radius: 8px;
                padding: 12px;
                background-color: white;
                font-size: 14px;
                line-height: 1.6;
            }
            QPushButton {
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton#play {
                background-color: #00B42A;
                color: white;
            }
            QPushButton#play:hover {
                background-color: #009429;
            }
            QPushButton#pause {
                background-color: #FF7D00;
                color: white;
            }
            QPushButton#pause:hover {
                background-color: #E66D00;
            }
            QPushButton#stop {
                background-color: #F53F3F;
                color: white;
            }
            QPushButton#stop:hover {
                background-color: #D92B2B;
            }
            QProgressBar {
                border: 1px solid #E5E6EB;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #165DFF;
                border-radius: 4px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = QLabel("🔊 语音播报内容")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #1D2129;")
        layout.addWidget(title)
        
        # 文本显示区域
        text_edit = QTextEdit()
        text_edit.setPlainText(text)
        text_edit.setReadOnly(True)
        text_edit.setMaximumHeight(200)
        layout.addWidget(text_edit)
        
        # 进度条
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        layout.addWidget(progress_bar)
        
        # 控制按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        # 播放/继续按钮
        btn_play = QPushButton("▶️ 播放")
        btn_play.setObjectName("play")
        btn_layout.addWidget(btn_play)
        
        # 暂停按钮
        btn_pause = QPushButton("⏸️ 暂停")
        btn_pause.setObjectName("pause")
        btn_layout.addWidget(btn_pause)
        
        # 停止按钮
        btn_stop = QPushButton("⏹️ 停止")
        btn_stop.setObjectName("stop")
        btn_layout.addWidget(btn_stop)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 状态标签
        status_label = QLabel("准备就绪")
        status_label.setStyleSheet("color: #86909C; font-size: 12px;")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        
        # 语音播报控制变量
        self.voice_engine = None
        self.voice_thread = None
        self.is_playing = False
        self.is_paused = False
        self.current_text = text
        self.current_position = 0
        
        def start_broadcast():
            """开始播报"""
            if self.is_paused:
                # 如果是暂停状态，继续播放
                self.is_paused = False
                self.is_playing = True
                status_label.setText("正在播报...")
                btn_play.setEnabled(False)
                btn_pause.setEnabled(True)
                return
            
            if self.is_playing:
                return
            
            self.is_playing = True
            self.is_paused = False
            status_label.setText("正在播报...")
            btn_play.setEnabled(False)
            btn_pause.setEnabled(True)
            
            def speak():
                try:
                    import pyttsx3
                    import time
                    
                    # 初始化语音引擎
                    self.voice_engine = pyttsx3.init()
                    self.voice_engine.setProperty('rate', 150)  # 语速
                    self.voice_engine.setProperty('volume', 0.9)  # 音量
                    
                    # 获取文本片段
                    text_to_speak = self.current_text[self.current_position:]
                    
                    # 分段播报
                    sentences = re.split(r'[。！？.!?]', text_to_speak)
                    total_sentences = len(sentences)
                    
                    for i, sentence in enumerate(sentences):
                        if not sentence.strip():
                            continue
                        
                        # 检查是否停止
                        if not self.is_playing:
                            break
                        
                        # 检查是否暂停
                        while self.is_paused and self.is_playing:
                            time.sleep(0.1)
                        
                        if not self.is_playing:
                            break
                        
                        # 播报当前句子
                        self.voice_engine.say(sentence)
                        self.voice_engine.runAndWait()
                        
                        # 更新位置
                        self.current_position += len(sentence) + 1
                        
                        # 更新进度条
                        progress = int((i + 1) / total_sentences * 100) if total_sentences > 0 else 0
                        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                        QMetaObject.invokeMethod(
                            progress_bar,
                            "setValue",
                            Qt.QueuedConnection,
                            Q_ARG(int, progress)
                        )
                    
                    # 播报完成
                    if self.is_playing:
                        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                        QMetaObject.invokeMethod(
                            status_label,
                            "setText",
                            Qt.QueuedConnection,
                            Q_ARG(str, "播报完成")
                        )
                        QMetaObject.invokeMethod(
                            btn_play,
                            "setEnabled",
                            Qt.QueuedConnection,
                            Q_ARG(bool, True)
                        )
                        QMetaObject.invokeMethod(
                            btn_pause,
                            "setEnabled",
                            Qt.QueuedConnection,
                            Q_ARG(bool, False)
                        )
                        self.is_playing = False
                        self.current_position = 0
                        
                except ImportError:
                    # 如果没有pyttsx3，使用Windows系统语音
                    import os
                    # 将文本分段，避免过长
                    max_len = 200
                    text_to_speak = self.current_text[self.current_position:]
                    for i in range(0, len(text_to_speak), max_len):
                        if not self.is_playing:
                            break
                        segment = text_to_speak[i:i+max_len]
                        os.system(f'powershell -c "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{segment}\')"')
                        self.current_position += len(segment)
                        progress = int((i + len(segment)) / len(text_to_speak) * 100) if text_to_speak else 0
                        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                        QMetaObject.invokeMethod(
                            progress_bar,
                            "setValue",
                            Qt.QueuedConnection,
                            Q_ARG(int, progress)
                        )
                except Exception as e:
                    print(f"[DEBUG] 语音播报失败: {e}")
                    from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(
                        status_label,
                        "setText",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"播报失败: {str(e)}")
                    )
            
            # 启动播报线程
            import threading
            self.voice_thread = threading.Thread(target=speak, daemon=True)
            self.voice_thread.start()
        
        def pause_broadcast():
            """暂停播报"""
            if self.is_playing and not self.is_paused:
                self.is_paused = True
                status_label.setText("已暂停")
                btn_play.setEnabled(True)
                btn_play.setText("▶️ 继续")
                btn_pause.setEnabled(False)
        
        def stop_broadcast():
            """停止播报"""
            self.is_playing = False
            self.is_paused = False
            self.current_position = 0
            if self.voice_engine:
                try:
                    self.voice_engine.stop()
                except:
                    pass
            status_label.setText("已停止")
            btn_play.setEnabled(True)
            btn_play.setText("▶️ 播放")
            btn_pause.setEnabled(False)
            progress_bar.setValue(0)
        
        # 连接按钮信号
        btn_play.clicked.connect(start_broadcast)
        btn_pause.clicked.connect(pause_broadcast)
        btn_stop.clicked.connect(stop_broadcast)
        
        # 初始状态
        btn_pause.setEnabled(False)
        
        # 对话框关闭时停止播报
        def on_dialog_close():
            stop_broadcast()
        
        dialog.finished.connect(on_dialog_close)
        
        dialog.exec_()

    def quick_view_stats(self):
        """快捷功能：查看详细检测统计信息"""
        if not hasattr(self, 'last_detection_result') or not self.last_detection_result:
            QMessageBox.information(self, "提示", "请先进行肿瘤检测")
            return
        
        result = self.last_detection_result
        
        # 创建详细信息对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("📊 检测统计详情")
        dialog.setMinimumSize(450, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #F7F8FA;
            }
            QLabel {
                font-size: 14px;
                color: #1D2129;
            }
            QLabel#title {
                font-size: 18px;
                font-weight: 600;
                color: #165DFF;
            }
            QLabel#section_title {
                font-size: 15px;
                font-weight: 600;
                color: #165DFF;
                margin-top: 10px;
            }
            QLabel#value {
                font-size: 14px;
                color: #4E5969;
                padding-left: 10px;
            }
            QFrame#section {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin: 5px 0;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = QLabel("🔍 肿瘤检测统计报告")
        title.setObjectName("title")
        layout.addWidget(title)
        
        # 基本信息区域
        basic_frame = QFrame()
        basic_frame.setObjectName("section")
        basic_layout = QVBoxLayout(basic_frame)
        
        basic_title = QLabel("📋 基本信息")
        basic_title.setObjectName("section_title")
        basic_layout.addWidget(basic_title)
        
        # 检测状态
        has_tumor = result.get('has_tumor', False)
        status_text = "⚠️ 检测到肿瘤" if has_tumor else "✅ 未检测到肿瘤"
        status_color = "#F53F3F" if has_tumor else "#00B42A"
        status_label = QLabel(f"<b>检测状态:</b> <span style='color: {status_color};'>{status_text}</span>")
        status_label.setObjectName("value")
        basic_layout.addWidget(status_label)
        
        # 肿瘤类型
        tumor_type = result.get('tumor_type', '未知')
        type_label = QLabel(f"<b>肿瘤类型:</b> {tumor_type}")
        type_label.setObjectName("value")
        basic_layout.addWidget(type_label)
        
        # 置信度
        confidence = result.get('confidence', 0)
        conf_label = QLabel(f"<b>AI置信度:</b> {confidence:.2%}")
        conf_label.setObjectName("value")
        basic_layout.addWidget(conf_label)
        
        layout.addWidget(basic_frame)
        
        # 详细测量信息（仅当检测到肿瘤时显示）
        if has_tumor:
            detail_frame = QFrame()
            detail_frame.setObjectName("section")
            detail_layout = QVBoxLayout(detail_frame)
            
            detail_title = QLabel("📏 测量信息")
            detail_title.setObjectName("section_title")
            detail_layout.addWidget(detail_title)
            
            # 肿瘤大小
            if 'size' in result:
                size_label = QLabel(f"<b>肿瘤大小:</b> {result['size']}")
                size_label.setObjectName("value")
                detail_layout.addWidget(size_label)
            
            # 肿瘤位置（大脑解剖位置）
            if 'position' in result:
                pos_label = QLabel(f"<b>解剖位置:</b> {result['position']}")
                pos_label.setObjectName("value")
                detail_layout.addWidget(pos_label)
            
            # 肿瘤体积
            if 'volume' in result:
                vol_label = QLabel(f"<b>估算面积:</b> {result['volume']}")
                vol_label.setObjectName("value")
                detail_layout.addWidget(vol_label)
            
            # 检测框坐标（原始像素位置）
            if 'box' in result:
                box = result['box']
                box_label = QLabel(f"<b>图像坐标:</b> ({box[0]:.0f}, {box[1]:.0f}) - ({box[2]:.0f}, {box[3]:.0f})")
                box_label.setObjectName("value")
                detail_layout.addWidget(box_label)
            
            layout.addWidget(detail_frame)
        
        # 检测信息区域
        info_frame = QFrame()
        info_frame.setObjectName("section")
        info_layout = QVBoxLayout(info_frame)
        
        info_title = QLabel("ℹ️ 检测信息")
        info_title.setObjectName("section_title")
        info_layout.addWidget(info_title)
        
        # 检测时间
        from datetime import datetime
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_label = QLabel(f"<b>检测时间:</b> {time_str}")
        time_label.setObjectName("value")
        info_layout.addWidget(time_label)
        
        # 检测模型
        model_label = QLabel(f"<b>检测模型:</b> YOLO11 脑肿瘤检测模型")
        model_label.setObjectName("value")
        info_layout.addWidget(model_label)
        
        layout.addWidget(info_frame)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 关闭按钮
        btn_close = QPushButton("关闭")
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 30px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1147CC;
            }
        """)
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignCenter)
        
        dialog.exec_()

    def _call_api_with_image(self, message, image_base64, text_widget):
        """调用API分析图片"""
        import threading
        
        def api_call():
            try:
                from openai import OpenAI
                
                client = OpenAI(
                    base_url=get_ai_api_url(),
                    api_key=get_ai_api_key(),
                    timeout=60.0
                )
                
                response = client.chat.completions.create(
                    model=get_ai_model_name(),
                    messages=[
                        {
                            "role": "system",
                            "content": "你是脑肿瘤诊断AI助手。分析MRI影像照片，专业简洁回答，250字左右。"
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": message},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                            ]
                        }
                    ],
                    temperature=0.2,
                    max_tokens=400
                )
                
                reply = response.choices[0].message.content
                
                # 在主线程更新UI
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self,
                    "_update_chat_with_reply",
                    Qt.QueuedConnection,
                    Q_ARG(str, reply),
                    Q_ARG(object, text_widget)
                )
                    
            except Exception as e:
                error_msg = f"图片分析失败: {str(e)}"
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self,
                    "_update_chat_with_reply",
                    Qt.QueuedConnection,
                    Q_ARG(str, error_msg),
                    Q_ARG(object, text_widget)
                )
        
        threading.Thread(target=api_call, daemon=True).start()

    def _update_chat_with_reply(self, reply, text_widget):
        """更新聊天界面 with回复"""
        current_html = text_widget.toHtml()
        # 移除加载提示
        import re
        current_html = re.sub(r'<table[^>]*>.*?🤔 正在分析图片．．．.*?</table>.*?<div[^>]*>.*?</div>', '', current_html, flags=re.DOTALL)
        
        # 添加AI回复
        import html
        escaped_reply = html.escape(reply)
        ai_html = f"""
        <table width="100%" cellspacing="0" cellpadding="0" border="0">
            <tr>
                <td align="left">
                    <div style="display: inline-block; background-color: white; color: black; padding: 10px 16px; border-radius: 8px; max-width: 80%; font-size: 15px;">
                        {escaped_reply}
                    </div>
                </td>
            </tr>
        </table>
        <div style="height: 10px;"></div>
        """
        text_widget.setHtml(current_html + ai_html)
        scrollbar = text_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_ai_suggestion_received(self, suggestion_html, text_widget):
        """处理AI诊断建议 - 追加到现有内容，保留检测结果头部"""
        # 移除"正在调用AI分析"的提示
        import re
        suggestion_html = re.sub(r'<p[^>]*>🤖 正在调用AI分析，请稍候\.\.\.</p>', '', suggestion_html)
        
        # 获取当前内容（不包括头部）
        current_html = text_widget.toHtml()
        
        # 提取chat-content部分的内容
        chat_content = ""
        if '<div id="chat-content">' in current_html:
            start = current_html.find('<div id="chat-content">') + len('<div id="chat-content">')
            end = current_html.find('</div>', start)
            if end > start:
                chat_content = current_html[start:end].strip()
        
        # 移除"正在调用AI分析"的提示
        chat_content = re.sub(r'<div[^>]*>\s*<p>🤖 正在调用AI分析.*?</div>', '', chat_content, flags=re.DOTALL)
        
        # 追加AI建议
        new_content = chat_content + suggestion_html
        
        # 使用update_diagnosis_display更新显示
        self.update_diagnosis_display(text_widget, new_content)
        
        # 滚动到底部
        scrollbar = text_widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _get_detection_result_image_base64(self):
        """获取单张检测结果图片的base64编码"""
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            # 从seg_image_viewer获取当前显示的图片
            if hasattr(self, 'current_seg_pixmap') and self.current_seg_pixmap:
                # 将QPixmap转换为PIL Image
                qimage = self.current_seg_pixmap.toImage()
                buffer = BytesIO()
                
                # 转换为RGB格式
                if qimage.format() != QImage.Format_RGB888:
                    qimage = qimage.convertToFormat(QImage.Format_RGB888)
                
                width = qimage.width()
                height = qimage.height()
                ptr = qimage.bits()
                ptr.setsize(height * width * 3)
                
                # 创建PIL Image
                pil_image = Image.frombytes('RGB', (width, height), ptr.asstring())
                
                # 保存为PNG并编码为base64
                buffer = BytesIO()
                pil_image.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return img_str
            return None
        except Exception as e:
            print(f"[DEBUG] 获取检测结果图片失败: {e}")
            return None
    
    def _get_nii_mixed_view_base64(self, slice_idx):
        """获取NII混合视图的base64编码"""
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            import numpy as np
            
            if hasattr(self, 'mri_processor'):
                # 生成混合视图
                mixed_view = self.mri_processor.create_mixed_view(slice_idx)
                if mixed_view is not None:
                    # 转换为PIL Image
                    if isinstance(mixed_view, np.ndarray):
                        if mixed_view.dtype != np.uint8:
                            mixed_view = ((mixed_view - mixed_view.min()) / 
                                        (mixed_view.max() - mixed_view.min()) * 255).astype(np.uint8)
                        pil_image = Image.fromarray(mixed_view)
                    else:
                        return None
                    
                    # 保存为PNG并编码为base64
                    buffer = BytesIO()
                    pil_image.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    return img_str
            return None
        except Exception as e:
            print(f"[DEBUG] 获取NII混合视图失败: {e}")
            return None
    
    def _call_deepseek_api(self, prompt, target_text_edit):
        """调用DeepSeek API获取诊断建议 - 使用deepseek-chat模型"""
        self._call_deepseek_api_for_diagnosis(prompt, target_text_edit)

    def _call_deepseek_api_with_image(self, prompt, image_base64, target_text_edit):
        """调用DeepSeek API获取诊断建议 - 使用deepseek-chat模型"""
        self._call_deepseek_api_for_diagnosis(prompt, target_text_edit)

    def _call_deepseek_api_for_diagnosis(self, prompt, target_text_edit):
        """调用DeepSeek API获取诊断建议 - 使用deepseek-chat模型"""
        import threading
        import time

        def api_call():
            try:
                from openai import OpenAI

                start_time = time.time()

                # 初始化OpenAI客户端
                client = OpenAI(
                    base_url=get_ai_api_url(),
                    api_key=get_ai_api_key(),
                    timeout=30.0
                )

                # 调用API - 使用deepseek-chat模型
                response = client.chat.completions.create(
                    model=get_ai_model_name(),
                    messages=[
                        {"role": "system", "content": "你是专业的脑肿瘤诊断AI助手，请根据检测结果提供详细的诊断建议。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )

                # 获取回复
                suggestion = response.choices[0].message.content

                elapsed = time.time() - start_time
                print(f"[DEBUG] 诊断API响应时间: {elapsed:.2f}秒")

                # 添加免责声明
                disclaimer = "<hr><p style='color: #F53F3F; font-size: 12px;'><b>⚠️ 重要提示：</b>本结果基于AI模型分析，仅供参考，不能替代专业医生的诊断。请尽快携带检查结果咨询专业医生。</p>"

                # 使用信号槽更新UI
                self.ai_suggestion_signal.emit(suggestion + disclaimer, target_text_edit)

            except ImportError:
                self.ai_suggestion_signal.emit("<p style='color: #F53F3F;'>错误：未安装openai库。</p><p>请运行: pip install openai</p>", target_text_edit)
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    self.ai_suggestion_signal.emit("<p style='color: #F53F3F;'>API请求超时，请稍后重试。</p>", target_text_edit)
                else:
                    self.ai_suggestion_signal.emit(f"<p style='color: #F53F3F;'>API调用失败: {error_msg}</p><p>请检查网络连接和API配置。</p>", target_text_edit)

        # 启动线程执行API调用
        threading.Thread(target=api_call, daemon=True).start()

    # ==================== 批量图像功能 ====================

    def select_image_folder(self):
        """选择图像文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder_path:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            self.batch_file_list = []

            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    self.batch_file_list.append(os.path.join(folder_path, file))

            self.batch_file_list.sort()

            # 更新文件列表
            self.list_image_files.clear()
            for file_path in self.batch_file_list:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                self.list_image_files.addItem(item)

            self.lbl_folder_info.setText(f"找到 {len(self.batch_file_list)} 个图像文件")
            self.btn_batch_run.setEnabled(len(self.batch_file_list) > 0)

    def run_batch_image_detection(self):
        """运行批量检测"""
        if not self.batch_file_list:
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择结果保存目录")
        if not output_dir:
            return

        # 重置批量结果
        self.batch_results = []
        self.batch_table.setRowCount(0)

        # 启动批量处理
        self.batch_worker = BatchWorker(
            self.batch_file_list,
            self.model_classification_path,
            self.model_segmentation_path,
            output_dir,
            is_nii=False
        )
        self.batch_worker.progress_signal.connect(self.on_batch_progress)
        self.batch_worker.result_signal.connect(self.on_batch_result)
        self.batch_worker.finished_signal.connect(self.on_batch_finished)
        self.batch_worker.start()

        # 更新UI
        self.btn_batch_run.setEnabled(False)
        self.btn_batch_stop.setEnabled(True)
        self.btn_save_batch.setEnabled(False)

    def on_batch_progress(self, current, total, filename):
        """批量处理进度"""
        self.batch_progress.setMaximum(total)
        self.batch_progress.setValue(current)
        self.batch_progress_label.setText(f"{current}/{total} 已完成 - {filename}")
        self.status_bar.showMessage(f"正在处理: {filename}")

    def on_batch_result(self, result):
        """单个文件处理结果"""
        # 保存结果到列表
        self.batch_results.append(result)
        
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)

        self.batch_table.setItem(row, 0, QTableWidgetItem(result['filename']))

        if result['status'] == 'success':
            cls_info = ", ".join([f"{d['cls_name']}({d['conf']:.0%})" for d in result['cls_data'][:2]])
            self.batch_table.setItem(row, 1, QTableWidgetItem(cls_info if cls_info else "无"))
            self.batch_table.setItem(row, 2, QTableWidgetItem(str(result['seg_count'])))
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{result['process_time']:.1f}s"))
            self.batch_table.setItem(row, 4, QTableWidgetItem("✅ 成功"))

            self.success_count += 1
        else:
            self.batch_table.setItem(row, 1, QTableWidgetItem("-"))
            self.batch_table.setItem(row, 2, QTableWidgetItem("-"))
            self.batch_table.setItem(row, 3, QTableWidgetItem("-"))
            self.batch_table.setItem(row, 4, QTableWidgetItem(f"❌ {result.get('error', '失败')}"))

            self.error_count += 1

        self.total_processed += 1
        self.update_stats()

    def on_batch_finished(self, results):
        """批量处理完成"""
        self.batch_worker = None
        self.btn_batch_run.setEnabled(True)
        self.btn_batch_stop.setEnabled(False)
        self.btn_save_batch.setEnabled(True)
        self.status_bar.showMessage("批量处理完成")

        # 生成AI统计信息
        self.generate_batch_statistics()

        # 自动显示第一个成功的结果
        if self.batch_results:
            first_success = -1
            for i, result in enumerate(self.batch_results):
                if result['status'] == 'success':
                    first_success = i
                    break
            if first_success >= 0:
                self.show_batch_preview(first_success)
                self.batch_table.selectRow(first_success)

        QMessageBox.information(
            self,
            "完成",
            f"批量处理完成!\n成功: {self.success_count} 个\n失败: {self.error_count} 个"
        )

    def generate_batch_statistics(self):
        """生成批量检测的AI统计分析"""
        if not self.batch_results:
            return

        # 统计各类数据
        total = len(self.batch_results)
        healthy_count = 0
        tumor_count = 0
        error_count = 0
        
        tumor_types = {}
        total_tumors = 0

        for result in self.batch_results:
            if result['status'] != 'success':
                error_count += 1
                continue

            if result.get('has_tumor', False):
                tumor_count += 1
                # 统计肿瘤类型
                for cls_data in result.get('cls_data', []):
                    cls_name = cls_data.get('cls_name', 'Unknown')
                    if cls_name.lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']:
                        tumor_types[cls_name] = tumor_types.get(cls_name, 0) + 1
                # 统计肿瘤数量
                total_tumors += result.get('seg_count', 0)
            else:
                healthy_count += 1

        # 生成统计报告
        stats_html = f"""
        <p><b>📊 总体统计</b></p>
        <ul>
        <li>总图像数: <b>{total}</b> 张</li>
        <li>健康大脑: <b style="color: #00B42A;">{healthy_count}</b> 张 ({healthy_count/total*100:.1f}%)</li>
        <li>检测到肿瘤: <b style="color: #F53F3F;">{tumor_count}</b> 张 ({tumor_count/total*100:.1f}%)</li>
        <li>处理失败: <b>{error_count}</b> 张</li>
        </ul>
        """

        if tumor_count > 0:
            stats_html += f"""
        <p><b>🔬 肿瘤详情</b></p>
        <ul>
        <li>总肿瘤数: <b>{total_tumors}</b> 个</li>
        <li>平均每张肿瘤图像: <b>{total_tumors/tumor_count:.1f}</b> 个</li>
        </ul>
        <p><b>📋 肿瘤类型分布:</b></p>
        <ul>
        """
            for tumor_type_en, count in sorted(tumor_types.items(), key=lambda x: x[1], reverse=True):
                percentage = count / tumor_count * 100
                tumor_type_cn = self.get_tumor_name_cn(tumor_type_en)
                stats_html += f"<li>{tumor_type_cn}: <b>{count}</b> 张 ({percentage:.1f}%)</li>\n"
            stats_html += "</ul>"

        # 生成AI建议
        stats_html += "<p><b>💡 AI建议:</b></p><ul>"
        if tumor_count == 0:
            stats_html += "<li>✅ 恭喜！本批次未检测到肿瘤，整体健康状况良好。</li>"
        elif tumor_count / total < 0.3:
            stats_html += f"<li>⚠️ 本批次肿瘤检出率较低 ({tumor_count/total*100:.1f}%)，建议关注阳性病例。</li>"
        elif tumor_count / total < 0.7:
            stats_html += f"<li>⚠️ 本批次肿瘤检出率中等 ({tumor_count/total*100:.1f}%)，建议仔细审查所有结果。</li>"
        else:
            stats_html += f"<li>🚨 本批次肿瘤检出率较高 ({tumor_count/total*100:.1f}%)，建议重点关注并及时就医咨询。</li>"
        
        stats_html += "<li>📌 建议将所有检测结果导出，供专业医生进一步诊断。</li>"
        stats_html += "</ul>"

        # 显示统计信息
        self.batch_stats_text.setHtml(stats_html)
        self.batch_stats_widget.setVisible(True)

    def on_batch_item_clicked(self, item):
        """点击批量结果表格项 - 显示对应图像预览"""
        row = item.row()
        self.show_batch_preview(row)

    def show_batch_preview(self, index):
        """显示指定索引的批量检测结果预览"""
        if index < 0 or index >= len(self.batch_results):
            return
        
        self.current_preview_index = index
        result = self.batch_results[index]
        
        # 更新索引显示
        self.lbl_preview_index.setText(f"{index + 1} / {len(self.batch_results)}")
        
        # 更新导航按钮状态
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < len(self.batch_results) - 1)
        
        # 重置批量分割图像
        self.current_batch_seg_pixmap = None
        
        if result['status'] == 'success':
            # 显示原始图像
            if result.get('file') and os.path.exists(result['file']):
                original_pixmap = QPixmap(result['file'])
                self.batch_image_viewer.set_image(original_pixmap)
            
            # 显示分割结果 - 使用批量检测专用的查看器
            if result.get('seg_path') and os.path.exists(result['seg_path']):
                seg_pixmap = QPixmap(result['seg_path'])
                self.batch_seg_image_viewer.set_image(seg_pixmap)
                self.current_batch_seg_pixmap = seg_pixmap
                
                # 检查是否有分割掩码（通过seg_count或matched_boxes判断）
                has_segmentation = result.get('seg_count', 0) > 0 or len(result.get('matched_boxes', [])) > 0
                if not has_segmentation and result.get('has_tumor', False):
                    # 有肿瘤但没有分割掩码，显示提示
                    self.status_bar.showMessage(f"图像 {result['filename']}: 显示best5分类结果（无分割掩码）")
            else:
                # 如果没有分割结果，显示提示信息
                self.show_batch_no_result_message(result)
        else:
            # 显示错误提示
            self.batch_image_viewer.scene.clear()
            self.batch_seg_image_viewer.scene.clear()
            error_label = QLabel(f"处理失败: {result.get('error', '未知错误')}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: #F53F3F; font-size: 16px;")

    def show_batch_no_result_message(self, result):
        """显示批量检测中无分割结果的提示"""
        from PyQt5.QtGui import QPainter, QFont, QColor
        
        # 创建提示图像
        pixmap = QPixmap(640, 480)
        pixmap.fill(QColor(240, 240, 240))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont("Microsoft YaHei", 14))
        
        # 根据情况显示不同提示
        if not result.get('has_tumor', False):
            message = "未检测到肿瘤\n该图像被分类为健康大脑"
        else:
            # 获取检测到的肿瘤类型信息
            cls_data = result.get('cls_data', [])
            if cls_data:
                tumor_names_en = [item['cls_name'] for item in cls_data 
                              if item['cls_name'].lower() not in ['no tumor', 'notumor', 'healthy', 'normal', '无肿瘤', '正常']]
                if tumor_names_en:
                    unique_tumors_en = list(dict.fromkeys(tumor_names_en))[:3]  # 去重保持顺序
                    unique_tumors_cn = [self.get_tumor_name_cn(name) for name in unique_tumors_en]
                    tumor_type_str = '、'.join(unique_tumors_cn)  # 最多显示3种
                    message = f"检测到: {tumor_type_str}\n但分割模型未能在相同位置识别到肿瘤区域"
                else:
                    message = "检测到肿瘤\n但分割模型未能在相同位置识别到肿瘤区域"
            else:
                message = "检测到肿瘤\n但分割模型未能在相同位置识别到肿瘤区域"
        
        rect = pixmap.rect()
        painter.drawText(rect, Qt.AlignCenter, message)
        painter.end()
        
        # 显示提示图像
        self.batch_seg_image_viewer.set_image(pixmap)
        # 不保存到 current_batch_seg_pixmap，这样全屏查看按钮不会生效

    def show_prev_image(self):
        """显示上一个图像"""
        if self.current_preview_index > 0:
            self.show_batch_preview(self.current_preview_index - 1)
            # 同步更新表格选中行
            self.batch_table.selectRow(self.current_preview_index)

    def show_next_image(self):
        """显示下一个图像"""
        if self.current_preview_index < len(self.batch_results) - 1:
            self.show_batch_preview(self.current_preview_index + 1)
            # 同步更新表格选中行
            self.batch_table.selectRow(self.current_preview_index)

    def save_batch_results(self):
        """保存批量结果"""
        # 实现保存逻辑
        pass

    def stop_batch_processing(self):
        """停止批量处理"""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            self.batch_worker.wait()
            self.status_bar.showMessage("已停止批量处理")

        self.btn_batch_run.setEnabled(True)
        self.btn_batch_stop.setEnabled(False)

    # ==================== NII功能（多模态MRI） ====================

    def select_patient_folder(self):
        """选择患者文件夹（包含5个NII文件）"""
        if not NII_AVAILABLE:
            QMessageBox.warning(self, "警告", "未安装nibabel库\n请运行: pip install nibabel")
            return

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择患者文件夹",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            try:
                # 创建多模态处理器
                self.mri_processor = MultiModalMRIProcessor(folder_path)
                files_found = self.mri_processor.scan_patient_folder()
                
                # 检查是否找到t1ce文件
                if 't1ce' not in files_found:
                    QMessageBox.warning(self, "警告", "未找到T1CE文件\n文件夹中需要包含_t1ce.nii文件")
                    return
                
                # 加载所有模态
                self.mri_processor.load_all_modalities()
                
                # 更新UI
                self.lbl_nii_info.setText(f"患者文件夹:\n{os.path.basename(folder_path)}")
                
                # 显示文件状态
                status_text = "找到的文件:\n"
                modality_names = {
                    'flair': 'FLAIR',
                    't1': 'T1',
                    't1ce': 'T1CE',
                    't2': 'T2',
                    'seg': '分割标签'
                }
                for key, name in modality_names.items():
                    status = "✅" if key in files_found else "❌"
                    status_text += f"{status} {name}\n"
                self.lbl_modality_status.setText(status_text)
                
                # 启用控件
                self.combo_modality.setEnabled(True)
                self.spin_nii_slice.setEnabled(True)
                self.spin_nii_slice.setRange(0, self.mri_processor.shape[2] - 1)
                self.spin_nii_slice.setValue(self.mri_processor.shape[2] // 2)
                self.btn_nii_run.setEnabled(True)
                
                # 启用3D查看按钮
                if hasattr(self, 'btn_3d_viewer') and PV_AVAILABLE:
                    self.btn_3d_viewer.setEnabled(True)
                
                # 保存当前NII图像用于3D查看
                if 't1ce' in files_found:
                    self.current_nii_img = nib.load(files_found['t1ce'])
                
                # 显示中间切片
                self.current_modality = "t1ce"
                self.display_multimodal_slice(self.mri_processor.shape[2] // 2)
                
                self.status_bar.showMessage(f"已加载患者数据: {len(files_found)}个模态, 切片数: {self.mri_processor.shape[2]}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载患者文件夹失败: {str(e)}")
                import traceback
                traceback.print_exc()

    def on_modality_changed(self, index):
        """模态选择改变"""
        modality_map = {
            0: "t1ce",
            1: "flair",
            2: "t1",
            3: "t2",
            4: "fusion",
            5: "seg"
        }
        self.current_modality = modality_map.get(index, "t1ce")
        
        if hasattr(self, 'mri_processor') and hasattr(self, 'current_nii_slice_idx'):
            self.display_multimodal_slice(self.current_nii_slice_idx)

    def _find_and_load_seg_file(self, folder_path):
        """在指定文件夹中查找并加载分割文件"""
        try:
            if not os.path.exists(folder_path):
                return None
            
            # 查找常见的分割文件名模式
            seg_patterns = [
                '*seg.nii', '*seg.nii.gz',
                '*_seg.nii', '*_seg.nii.gz',
                '*Seg.nii', '*Seg.nii.gz',
                '*_Seg.nii', '*_Seg.nii.gz',
                'seg.nii', 'seg.nii.gz',
                'Seg.nii', 'Seg.nii.gz'
            ]
            
            for pattern in seg_patterns:
                seg_files = glob.glob(os.path.join(folder_path, pattern))
                if seg_files:
                    seg_path = seg_files[0]
                    print(f"[3D查看器] 找到分割文件: {os.path.basename(seg_path)}")
                    seg_img = nib.load(seg_path)
                    print(f"[3D查看器] 分割数据形状: {seg_img.shape}")
                    return seg_img
            
            print(f"[3D查看器] 在文件夹中未找到分割文件: {folder_path}")
            return None
            
        except Exception as e:
            print(f"[3D查看器] 加载分割文件失败: {e}")
            return None
    
    def open_3d_viewer(self):
        """打开3D NII查看器 - 支持分割叠加和检测框显示"""
        if not PV_AVAILABLE:
            QMessageBox.warning(self, "提示", "未安装 pyvista 和 pyvistaqt\n请运行: pip install pyvista pyvistaqt")
            return
        
        if not hasattr(self, 'current_nii_img') or self.current_nii_img is None:
            QMessageBox.warning(self, "提示", "请先加载NII文件")
            return
        
        try:
            # 准备分割数据（如果有）
            seg_img = None
            
            # 首先尝试从mri_processor获取
            if hasattr(self, 'mri_processor') and hasattr(self.mri_processor, 'data') and 'seg' in self.mri_processor.data:
                seg_data = self.mri_processor.data['seg']
                affine = self.mri_processor.affine
                seg_img = nib.Nifti1Image(seg_data, affine)
                print("[3D查看器] 从mri_processor获取分割数据")
            
            # 如果没有，尝试从当前患者文件夹自动查找
            if seg_img is None and hasattr(self, 'current_patient_folder') and self.current_patient_folder:
                seg_img = self._find_and_load_seg_file(self.current_patient_folder)
            
            # 如果还是没有，尝试从当前NII文件路径推断
            if seg_img is None and hasattr(self, 'current_nii_path') and self.current_nii_path:
                nii_dir = os.path.dirname(self.current_nii_path)
                seg_img = self._find_and_load_seg_file(nii_dir)
            
            # 准备检测框数据（如果有检测结果）
            detection_boxes = []
            if hasattr(self, 'last_nii_detection_result'):
                result = self.last_nii_detection_result
                if result and result.get('has_tumor', False):
                    for cls_item in result.get('cls_data', []):
                        if 'box' in cls_item:
                            detection_boxes.append({
                                'x1': cls_item['box'][0],
                                'y1': cls_item['box'][1],
                                'x2': cls_item['box'][2],
                                'y2': cls_item['box'][3],
                                'cls_name': cls_item.get('cls_name', 'Unknown'),
                                'conf': cls_item.get('conf', 0)
                            })
            
            # 创建并显示3D查看器
            self.viewer_3d = NII3DViewer(
                self.current_nii_img, 
                seg_img=seg_img,
                detection_boxes=detection_boxes,
                parent=self
            )
            self.viewer_3d.show()
            self.status_bar.showMessage("已打开3D查看器")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开3D查看器失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def display_multimodal_slice(self, slice_idx):
        """显示多模态切片"""
        if not hasattr(self, 'mri_processor'):
            return
        
        # 确保 current_modality 有默认值
        if not hasattr(self, 'current_modality') or self.current_modality is None:
            self.current_modality = "t1ce"
        
        try:
            if self.current_modality == "fusion":
                # 显示融合图像
                slice_data = self.mri_processor.create_fusion_image(slice_idx)
                height, width = slice_data.shape[:2]
                bytes_per_line = width * 3
                q_image = QImage(slice_data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif self.current_modality == "seg":
                # 显示分割标签（彩色）
                seg_slice = self.mri_processor.get_slice('seg', slice_idx, normalize=False)
                # 创建彩色标签图
                color_seg = np.zeros((*seg_slice.shape, 3), dtype=np.uint8)
                colors = {
                    0: [0, 0, 0],      # 背景-黑
                    1: [255, 0, 0],    # 坏死-红
                    2: [0, 255, 0],    # 水肿-绿
                    4: [0, 0, 255]     # 增强肿瘤-蓝
                }
                for label, color in colors.items():
                    mask = (seg_slice == label)
                    for c in range(3):
                        color_seg[:, :, c][mask] = color[c]
                
                height, width = color_seg.shape[:2]
                bytes_per_line = width * 3
                q_image = QImage(color_seg.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                # 显示单模态灰度图
                slice_data = self.mri_processor.get_slice(self.current_modality, slice_idx)
                height, width = slice_data.shape
                bytes_per_line = width
                q_image = QImage(slice_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(q_image)
            self.nii_image_viewer.set_image(pixmap)
            self.current_nii_slice_idx = slice_idx
            
            # 分析当前切片的肿瘤信息
            tumor_info = self.mri_processor.analyze_tumor_info(slice_idx)
            if tumor_info['has_tumor']:
                region_text = " | ".join([
                    f"{r['name']}: {r['pixel_count']}像素" 
                    for r in tumor_info['regions'].values()
                ])
                self.status_bar.showMessage(f"切片 {slice_idx}: 发现肿瘤 - {region_text}")
            else:
                self.status_bar.showMessage(f"切片 {slice_idx}: 无肿瘤")
                
        except Exception as e:
            QMessageBox.warning(self, "警告", f"显示切片失败: {str(e)}")

    def on_nii_slice_changed(self, value):
        """NII切片改变"""
        if hasattr(self, 'mri_processor'):
            self.display_multimodal_slice(value)

    def run_nii_detection(self):
        """运行NII检测 - 仅对T1CE模态"""
        if not hasattr(self, 'mri_processor'):
            return
        
        try:
            # 获取当前T1CE切片
            t1ce_slice = self.mri_processor.get_slice('t1ce', self.current_nii_slice_idx)
            
            # 创建临时图像（使用tempfile目录但手动管理删除）
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"tumor_nii_slice_{self.current_nii_slice_idx}_{os.getpid()}.png")
            
            slice_rgb = np.stack([t1ce_slice] * 3, axis=-1)
            Image.fromarray(slice_rgb).save(temp_path)
            
            self.current_image_path = temp_path
            self.run_all_detection()
            
            # 保存NII检测结果用于3D查看器
            if hasattr(self, 'last_detection_result') and self.last_detection_result is not None:
                import copy
                self.last_nii_detection_result = copy.deepcopy(self.last_detection_result)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {str(e)}")

    def switch_to_nii_ai_chat(self):
        """切换到NII模式的AI实时对话界面"""
        # 获取分割器并调整左右比例，让右侧AI建议区域显示更多
        if hasattr(self, 'nii_splitter'):
            total_width = self.nii_splitter.width()
            # 设置左侧占40%，右侧占60%
            self.nii_splitter.setSizes([int(total_width * 0.4), int(total_width * 0.6)])
        
        # 聚焦到AI对话输入框
        if hasattr(self, 'nii_diagnosis_input'):
            self.nii_diagnosis_input.setFocus()
        
        # 显示提示信息
        self.status_bar.showMessage("已切换到AI实时对话界面，您可以直接在右侧输入问题与AI交流")
        
        # 如果还没有AI建议，自动触发一次AI分析
        if hasattr(self, 'nii_diagnosis_text'):
            current_text = self.nii_diagnosis_text.toPlainText()
            if not current_text or "请先进行肿瘤检测" in current_text or len(current_text) < 50:
                # 如果有检测结果，触发AI分析
                if hasattr(self, 'last_detection_result') and self.last_detection_result:
                    result = self.last_detection_result
                    has_tumor = result.get('has_tumor', False)
                    tumor_type = result.get('tumor_type_en', 'No Tumor')
                    confidence = result.get('confidence', 0)
                    
                    # 显示提示
                    self.nii_diagnosis_text.setHtml("""
                    <div style="background-color: #E8F0FF; border-left: 4px solid #165DFF; padding: 12px; margin: 10px 0; border-radius: 4px;">
                        <p style="margin: 0; color: #4E5969;">🤖 正在调用AI分析当前切片，请稍候...</p>
                    </div>
                    """)
                    
                    # 调用AI分析
                    nii_slice_idx = self.current_nii_slice_idx if hasattr(self, 'current_nii_slice_idx') else 0
                    self.generate_diagnosis_suggestion(tumor_type, confidence, has_tumor, True, nii_slice_idx, result)

    def generate_patient_report(self):
        """生成患者肿瘤报告 - 多模态综合分析"""
        if not hasattr(self, 'mri_processor'):
            return
        
        try:
            summary = self.mri_processor.get_patient_summary()
            
            # 创建报告文本
            report = []
            report.append("=" * 70)
            report.append("患者脑部肿瘤分析报告（多模态MRI综合诊断）")
            report.append("=" * 70)
            report.append(f"\n患者文件夹: {os.path.basename(summary['folder'])}")
            report.append(f"数据维度: {summary['shape']}")
            report.append(f"\n发现的模态文件: {', '.join(summary['files_found'])}")
            
            if summary['slices_with_tumor']:
                report.append(f"\n【肿瘤检测概况】")
                report.append(f"包含肿瘤的切片数: {len(summary['slices_with_tumor'])}")
                report.append(f"肿瘤切片范围: {min(summary['slices_with_tumor'])} - {max(summary['slices_with_tumor'])}")
                report.append(f"估计肿瘤总体积: {summary['total_tumor_volume_mm3']:.2f} mm³")
                
                report.append(f"\n【各区域详细统计】")
                for label_value, stats in summary['tumor_statistics'].items():
                    label_info = MultiModalMRIProcessor.SEG_LABELS.get(label_value, {})
                    report.append(f"\n{label_info.get('name', '未知')} ({label_info.get('en_name', '')}):")
                    report.append(f"  - 总像素数: {stats['total_pixels']}")
                    report.append(f"  - 涉及切片数: {stats['slice_count']}")
                    report.append(f"  - 占比: {stats['total_pixels'] / sum(s['total_pixels'] for s in summary['tumor_statistics'].values()) * 100:.1f}%")
                
                # 多模态综合医学解读
                report.append(f"\n【多模态MRI综合医学解读】")
                report.append("-" * 70)
                
                has_edema = 2 in summary['tumor_statistics']
                has_enhancing = 4 in summary['tumor_statistics']
                has_necrosis = 1 in summary['tumor_statistics']
                
                # T1CE增强分析
                report.append("\n1. T1CE（增强T1加权）分析:")
                if has_enhancing:
                    report.append("   ✓ 检测到明显强化区域，提示血脑屏障破坏")
                    report.append("   ✓ 增强模式对判断肿瘤恶性程度具有重要价值")
                    if has_necrosis:
                        report.append("   ⚠ 环形强化伴中央坏死，高度提示高级别胶质瘤（如胶质母细胞瘤）")
                    else:
                        report.append("   • 均匀强化或不规则强化，需结合其他序列进一步评估")
                else:
                    report.append("   • 未见明显强化，可能为低级别病变或无血脑屏障破坏")
                
                # FLAIR分析
                report.append("\n2. FLAIR序列分析:")
                if has_edema:
                    edema_pixels = summary['tumor_statistics'][2]['total_pixels']
                    report.append(f"   ✓ 检测到瘤周水肿（FLAIR高信号），范围约{edema_pixels}像素")
                    report.append("   ✓ FLAIR能清晰显示肿瘤周围水肿带，对确定肿瘤边界有重要价值")
                    report.append("   • 水肿程度反映肿瘤对周围组织的浸润和压迫")
                else:
                    report.append("   • 未见明显瘤周水肿")
                
                # T1加权分析
                report.append("\n3. T1加权分析:")
                report.append("   • 提供大脑解剖结构参考基准")
                if has_enhancing:
                    report.append("   • T1低信号区域在T1CE上强化，提示肿瘤实质")
                if has_necrosis:
                    report.append("   • T1低信号坏死区与周围组织对比明显")
                
                # T2加权分析
                report.append("\n4. T2加权分析:")
                total_tumor = sum(s['total_pixels'] for s in summary['tumor_statistics'].values())
                report.append(f"   • T2高信号区域约{total_tumor}像素，包含水肿、肿瘤和坏死")
                if has_edema and has_enhancing:
                    report.append("   • T2高信号范围通常大于T1CE强化范围，差异即为水肿区域")
                
                # 专家标注验证
                report.append("\n5. 专家分割标注分析:")
                report.append("   • 标注区域包含以下病理成分：")
                if has_necrosis:
                    report.append("     - 坏死/非增强肿瘤（红色）：肿瘤核心缺血坏死区")
                if has_edema:
                    report.append("     - 瘤周水肿（绿色）：肿瘤周围血管源性水肿")
                if has_enhancing:
                    report.append("     - 增强肿瘤（蓝色）：血脑屏障破坏的活跃肿瘤区")
                
                # 综合诊断建议
                report.append(f"\n【综合诊断建议】")
                report.append("-" * 70)
                
                # 根据多模态特征给出分级建议
                if has_necrosis and has_enhancing and has_edema:
                    report.append("\n🔴 高度怀疑：高级别胶质瘤（WHO III-IV级）")
                    report.append("   依据：")
                    report.append("   • T1CE显示明显强化伴中央坏死（典型环形强化）")
                    report.append("   • FLAIR显示广泛瘤周水肿")
                    report.append("   • T2显示大范围高信号病变")
                    report.append("   • 以上特征符合胶质母细胞瘤（GBM）或间变性胶质瘤")
                    
                elif has_enhancing and has_edema and not has_necrosis:
                    report.append("\n🟠 怀疑：中高级别胶质瘤（WHO II-III级）")
                    report.append("   依据：")
                    report.append("   • T1CE显示强化但无明确坏死区")
                    report.append("   • FLAIR显示瘤周水肿")
                    report.append("   • 可能为间变性星形细胞瘤或少突胶质细胞瘤")
                    
                elif has_enhancing and not has_edema and not has_necrosis:
                    report.append("\n🟡 考虑：低级别胶质瘤或其他病变（WHO I-II级）")
                    report.append("   依据：")
                    report.append("   • T1CE显示强化但范围局限")
                    report.append("   • 无明显瘤周水肿和坏死")
                    report.append("   • 可能为低级别胶质瘤、转移瘤或其他良性病变")
                    
                elif has_edema and not has_enhancing:
                    report.append("\n🟡 考虑：非增强性病变或炎症/脱髓鞘病变")
                    report.append("   依据：")
                    report.append("   • FLAIR/T2显示高信号但T1CE无强化")
                    report.append("   • 可能为低级别胶质瘤、脑炎、脱髓鞘或其他非肿瘤病变")
                    report.append("   • 建议结合临床病史和进一步检查")
                    
                else:
                    report.append("\n🟢 病变特征不典型，建议进一步评估")
                
                # 多模态互补价值说明
                report.append(f"\n【各模态互补诊断价值】")
                report.append("-" * 70)
                report.append("• T1CE（增强T1）：显示血脑屏障破坏区域，确定肿瘤活性范围")
                report.append("• FLAIR：清晰显示瘤周水肿，帮助判断肿瘤浸润边界")
                report.append("• T1：提供解剖结构参考，显示肿瘤与正常组织对比")
                report.append("• T2：显示病变整体范围（肿瘤+水肿+坏死）")
                report.append("• 专家标注：提供病理区域精确分割，辅助定量分析")
                
                # 建议后续检查
                report.append(f"\n【建议】")
                report.append("-" * 70)
                report.append("1. 结合患者临床症状、病史和神经系统查体")
                report.append("2. 建议多学科会诊（MDT），包括神经外科、肿瘤科、放疗科")
                report.append("3. 考虑MRS（磁共振波谱）进一步评估肿瘤代谢特征")
                report.append("4. 必要时行PET-CT评估肿瘤代谢活性")
                report.append("5. 制定治疗方案前建议活检或手术切除获取病理诊断")
                
            else:
                report.append("\n【结果】未检测到肿瘤")
                report.append("\n多模态MRI未见明显异常信号，脑实质结构正常。")
            
            report.append("\n" + "=" * 70)
            report.append("报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            report.append("=" * 60)
            
            report_text = "\n".join(report)
            
            # 显示报告对话框
            from PyQt5.QtWidgets import QTextEdit, QDialog, QVBoxLayout, QPushButton
            dialog = QDialog(self)
            dialog.setWindowTitle("患者肿瘤分析报告")
            dialog.setMinimumSize(600, 500)
            
            layout = QVBoxLayout(dialog)
            
            text_edit = QTextEdit()
            text_edit.setPlainText(report_text)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)
            
            btn_save = QPushButton("保存报告")
            btn_save.clicked.connect(lambda: self.save_report(report_text))
            layout.addWidget(btn_save)
            
            btn_close = QPushButton("关闭")
            btn_close.clicked.connect(dialog.close)
            layout.addWidget(btn_close)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成报告失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_report(self, report_text):
        """保存报告到文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存报告",
            f"tumor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                QMessageBox.information(self, "成功", "报告已保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def generate_ai_patient_report(self):
        """生成AI患者肿瘤报告 - 微信聊天对话形式"""
        if not hasattr(self, 'mri_processor'):
            return
        
        try:
            summary = self.mri_processor.get_patient_summary()
            
            # 创建微信风格的报告对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("AI医生报告分析")
            dialog.setMinimumSize(700, 600)
            dialog.setStyleSheet("background-color: #EDEDED;")
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            
            # 标题栏
            header = QFrame()
            header.setStyleSheet("background-color: #EDEDED; border-bottom: 1px solid #D6D6D6;")
            header.setFixedHeight(50)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(15, 0, 15, 0)
            
            title_label = QLabel("AI医生报告分析")
            title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #000;")
            title_label.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(title_label)
            
            layout.addWidget(header)
            
            # 聊天区域
            chat_scroll = QScrollArea()
            chat_scroll.setWidgetResizable(True)
            chat_scroll.setStyleSheet("background-color: #EDEDED; border: none;")
            chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            chat_container = QWidget()
            chat_layout = QVBoxLayout(chat_container)
            chat_layout.setAlignment(Qt.AlignTop)
            chat_layout.setSpacing(15)
            chat_layout.setContentsMargins(15, 15, 15, 15)
            
            chat_scroll.setWidget(chat_container)
            layout.addWidget(chat_scroll)
            
            # 输入区域
            input_frame = QFrame()
            input_frame.setStyleSheet("background-color: #F7F7F7; border-top: 1px solid #D6D6D6;")
            input_frame.setFixedHeight(60)
            input_layout = QHBoxLayout(input_frame)
            input_layout.setContentsMargins(15, 10, 15, 10)
            input_layout.setSpacing(10)
            
            input_field = QLineEdit()
            input_field.setPlaceholderText("请输入您的问题...")
            input_field.setStyleSheet("""
                QLineEdit {
                    background-color: white;
                    border: 1px solid #D6D6D6;
                    border-radius: 4px;
                    padding: 8px 12px;
                    font-size: 14px;
                }
                QLineEdit:focus {
                    border: 1px solid #07C160;
                }
            """)
            input_layout.addWidget(input_field)
            
            send_btn = QPushButton("发送")
            send_btn.setStyleSheet("""
                QPushButton {
                    background-color: #07C160;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 20px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #06AD56;
                }
                QPushButton:pressed {
                    background-color: #059A4C;
                }
            """)
            send_btn.setCursor(Qt.PointingHandCursor)
            input_layout.addWidget(send_btn)
            
            layout.addWidget(input_frame)
            
            # 存储聊天历史
            chat_history = []
            
            # 构建患者数据
            has_edema = 2 in summary['tumor_statistics']
            has_enhancing = 4 in summary['tumor_statistics']
            has_necrosis = 1 in summary['tumor_statistics']
            
            # 构建初始报告数据
            report_data = {
                'folder': os.path.basename(summary['folder']),
                'shape': summary['shape'],
                'modalities': summary['files_found'],
                'slices_with_tumor': summary['slices_with_tumor'],
                'total_volume': summary['total_tumor_volume_mm3'],
                'has_edema': has_edema,
                'has_enhancing': has_enhancing,
                'has_necrosis': has_necrosis,
                'tumor_statistics': summary['tumor_statistics']
            }
            
            def add_message(text, is_user=False):
                """添加消息气泡"""
                msg_widget = QWidget()
                msg_layout = QHBoxLayout(msg_widget)
                msg_layout.setContentsMargins(0, 0, 0, 0)
                msg_layout.setSpacing(10)
                
                # 头像
                avatar = QLabel()
                avatar.setFixedSize(40, 40)
                if is_user:
                    avatar.setStyleSheet("""
                        background-color: #07C160;
                        border-radius: 4px;
                        color: white;
                        font-size: 12px;
                        font-weight: bold;
                    """)
                    avatar.setText("医生")
                    avatar.setAlignment(Qt.AlignCenter)
                else:
                    avatar.setStyleSheet("""
                        background-color: #07C160;
                        border-radius: 4px;
                        color: white;
                        font-size: 12px;
                        font-weight: bold;
                    """)
                    avatar.setText("AI")
                    avatar.setAlignment(Qt.AlignCenter)
                
                # 消息气泡
                bubble = QTextEdit()
                bubble.setReadOnly(True)
                bubble.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                bubble.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                
                # 计算文本高度
                font = QFont("Microsoft YaHei", 14)
                bubble.setFont(font)
                
                # 设置气泡样式 - 微信风格
                if is_user:
                    # 用户消息：蓝色气泡，白色文字，右对齐
                    bubble.setStyleSheet("""
                        QTextEdit {
                            background-color: #0078D4;
                            border-radius: 8px;
                            padding: 10px;
                            border: none;
                            color: white;
                        }
                    """)
                    # 用户消息：白色文字
                    escaped_text = text.replace('<', '&lt;').replace('>', '&gt;')
                    escaped_text = escaped_text.replace('\n', '<br>')
                    bubble.setHtml(f"<p style='line-height: 1.5; margin: 0; color: white;'>{escaped_text}</p>")
                else:
                    # AI消息：白色气泡，黑色文字，左对齐
                    bubble.setStyleSheet("""
                        QTextEdit {
                            background-color: white;
                            border-radius: 8px;
                            padding: 10px;
                            border: none;
                            color: black;
                        }
                    """)
                    # AI消息：黑色文字
                    escaped_text = text.replace('<', '&lt;').replace('>', '&gt;')
                    escaped_text = escaped_text.replace('\n', '<br>')
                    bubble.setHtml(f"<p style='line-height: 1.5; margin: 0; color: black;'>{escaped_text}</p>")
                
                # 计算合适的高度
                doc = bubble.document()
                doc.setTextWidth(400)
                height = doc.size().height() + 20
                bubble.setFixedHeight(int(height))
                bubble.setFixedWidth(420)
                
                if is_user:
                    msg_layout.addStretch()
                    msg_layout.addWidget(bubble)
                    msg_layout.addWidget(avatar)
                else:
                    msg_layout.addWidget(avatar)
                    msg_layout.addWidget(bubble)
                    msg_layout.addStretch()
                
                chat_layout.addWidget(msg_widget)
                
                # 滚动到底部
                QTimer.singleShot(100, lambda: chat_scroll.verticalScrollBar().setValue(
                    chat_scroll.verticalScrollBar().maximum()
                ))
                
                return bubble
            
            def send_message():
                """发送消息"""
                message = input_field.text().strip()
                if not message:
                    return
                
                # 显示用户消息
                add_message(message, is_user=True)
                chat_history.append({"role": "user", "content": message})
                input_field.clear()
                
                # 显示AI正在输入
                loading_bubble = add_message("🤔 思考中...", is_user=False)
                
                # 调用AI API
                self._call_ai_report_api(message, chat_history, report_data, loading_bubble, add_message)
            
            send_btn.clicked.connect(send_message)
            input_field.returnPressed.connect(send_message)
            
            # 显示欢迎消息和初始报告
            if report_data['slices_with_tumor']:
                welcome_msg = f"""您好，我是AI医生助手。

我已分析了患者 {report_data['folder']} 的MRI数据：
• 数据维度: {report_data['shape']}
• 发现模态: {', '.join(report_data['modalities'])}
• 肿瘤切片: {len(report_data['slices_with_tumor'])} 层
• 肿瘤体积: {report_data['total_volume']:.2f} mm³

检测到以下病理区域："""
                
                for label_value, stats in report_data['tumor_statistics'].items():
                    label_info = MultiModalMRIProcessor.SEG_LABELS.get(label_value, {})
                    welcome_msg += f"\n• {label_info.get('name', '未知')}: {stats['total_pixels']} 像素"
                
                welcome_msg += "\n\n请问您想了解什么？例如：\n- 肿瘤的严重程度\n- 治疗方案建议\n- 预后评估"
            else:
                welcome_msg = f"""您好，我是AI医生助手。

我已分析了患者 {report_data['folder']} 的MRI数据：
• 数据维度: {report_data['shape']}
• 发现模态: {', '.join(report_data['modalities'])}

✓ 未检测到肿瘤，脑实质结构正常。

如有其他问题，请随时询问。"""
            
            add_message(welcome_msg, is_user=False)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成报告失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _call_ai_report_api(self, message, chat_history, report_data, loading_bubble, add_message_callback):
        """调用AI API生成报告回复 - 优化速度"""
        import threading
        
        def api_call():
            try:
                from openai import OpenAI
                import time
                start_time = time.time()
                
                # 初始化OpenAI客户端 - 使用短超时
                client = OpenAI(
                    base_url=get_ai_api_url(),
                    api_key=get_ai_api_key(),
                    timeout=60.0  # 60秒超时
                )
                
                # 构建精简的系统提示词
                has_tumor = len(report_data['slices_with_tumor']) > 0
                tumor_info = "有肿瘤" if has_tumor else "无肿瘤"
                
                system_prompt = f"""你是脑肿瘤AI医生助手。患者：{report_data['folder']}，{tumor_info}，体积{report_data['total_volume']:.1f}mm³。回答医生问题，专业简洁，200字内。"""
                
                # 构建消息列表 - 只保留最近3轮对话以加快速度
                messages = [{"role": "system", "content": system_prompt}]
                
                # 只保留最近3轮对话历史
                recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
                for msg in recent_history[:-1]:  # 排除最后一条用户消息
                    messages.append(msg)
                
                # 调用API - 使用流式响应加快首字速度
                response = client.chat.completions.create(
                    model=get_ai_model_name(),
                    messages=messages,
                    temperature=0.2,  # 更低温度，更快响应
                    max_tokens=300,   # 减少token数
                    stream=False      # 非流式，直接获取完整响应
                )
                
                reply = response.choices[0].message.content
                
                elapsed = time.time() - start_time
                print(f"[DEBUG] API响应时间: {elapsed:.2f}秒")
                
                # 在主线程更新UI
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self, 
                    "_update_report_reply",
                    Qt.QueuedConnection,
                    Q_ARG(str, reply),
                    Q_ARG(object, loading_bubble),
                    Q_ARG(object, add_message_callback)
                )
                    
            except Exception as e:
                error_msg = f"抱歉，请求失败: {str(e)}"
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self,
                    "_update_report_reply",
                    Qt.QueuedConnection,
                    Q_ARG(str, error_msg),
                    Q_ARG(object, loading_bubble),
                    Q_ARG(object, add_message_callback)
                )
        
        # 启动线程执行API调用
        threading.Thread(target=api_call, daemon=True).start()

    def _update_report_reply(self, reply, loading_bubble, add_message_callback):
        """更新报告回复到UI"""
        # 删除加载提示
        loading_bubble.parent().deleteLater()
        # 添加AI回复
        add_message_callback(reply, is_user=False)

    # 保留旧方法以兼容
    def select_nii_file(self):
        """选择单个NII文件（旧方法，保留兼容）"""
        self.select_patient_folder()

    def display_nii_slice(self, slice_idx):
        """显示NII切片（旧方法，保留兼容）"""
        if hasattr(self, 'mri_processor'):
            self.display_multimodal_slice(slice_idx)

    def nms_boxes(self, boxes_data, iou_threshold=0.5):
        """对检测框进行NMS处理，去除重叠的框，保留置信度高的
        
        Args:
            boxes_data: 包含框信息的列表，每个元素是 {'cls_id', 'cls_name', 'conf', 'box'}
            iou_threshold: IOU阈值，超过此值认为是重叠
        
        Returns:
            过滤后的框列表
        """
        if not boxes_data:
            return []
        
        # 按置信度降序排序
        sorted_boxes = sorted(boxes_data, key=lambda x: x['conf'], reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, box_i in enumerate(sorted_boxes):
            if i in suppressed:
                continue
            
            keep.append(box_i)
            
            # 检查后续所有框
            for j in range(i + 1, len(sorted_boxes)):
                if j in suppressed:
                    continue
                
                box_j = sorted_boxes[j]
                iou = self.calculate_iou(box_i['box'], box_j['box'])
                
                # 如果IOU超过阈值，抑制（去除）后面的框
                if iou > iou_threshold:
                    suppressed.add(j)
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """计算两个框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        # 计算IOU
        if union_area == 0:
            return 0
        return inter_area / union_area

    # ==================== 实时摄像头检测功能 ====================

    def refresh_camera_list(self):
        """刷新摄像头列表"""
        self.combo_camera.clear()
        available_cameras = []

        # 检测可用的摄像头
        for i in range(5):  # 检测前5个摄像头
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append((f"摄像头 {i}", i))
                cap.release()

        if available_cameras:
            for name, idx in available_cameras:
                self.combo_camera.addItem(name, idx)
            self.lbl_camera_status.setText(f"找到 {len(available_cameras)} 个摄像头")
            self.lbl_camera_status.setStyleSheet(f"color: {Theme.SUCCESS};")
        else:
            self.combo_camera.addItem("未检测到摄像头", -1)
            self.lbl_camera_status.setText("未检测到摄像头")
            self.lbl_camera_status.setStyleSheet(f"color: {Theme.WARNING};")
            QMessageBox.warning(self, "警告", "未检测到可用的摄像头\n请检查摄像头连接")

    def start_camera_detection(self):
        """开始摄像头检测"""
        if self.is_camera_running:
            return

        camera_id = self.combo_camera.currentData()
        if camera_id == -1:
            QMessageBox.warning(self, "警告", "请先选择有效的摄像头")
            return

        # 固定使用综合检测模式
        detection_mode = "综合检测"

        # 创建摄像头工作线程
        self.camera_worker = CameraWorker(
            camera_id=camera_id,
            model_cls_path=self.model_classification_path,
            model_seg_path=self.model_segmentation_path,
            detection_mode=detection_mode
        )

        # 连接信号
        self.camera_worker.frame_signal.connect(self.update_camera_frame)
        self.camera_worker.result_signal.connect(self.update_camera_result)
        self.camera_worker.status_signal.connect(self.update_camera_status)
        self.camera_worker.error_signal.connect(self.handle_camera_error)

        # 启动线程
        self.camera_worker.start()
        self.is_camera_running = True

        # 更新UI状态
        self.btn_camera_start.setEnabled(False)
        self.btn_camera_stop.setEnabled(True)
        self.combo_camera.setEnabled(False)
        self.lbl_camera_mode_display.setText("模式: 综合检测")

        self.status_bar.showMessage("摄像头检测已启动")

    def stop_camera_detection(self):
        """停止摄像头检测"""
        if not self.is_camera_running:
            return

        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None

        self.is_camera_running = False

        # 更新UI状态
        self.btn_camera_start.setEnabled(True)
        self.btn_camera_stop.setEnabled(False)
        self.combo_camera.setEnabled(True)
        self.lbl_camera_status.setText("摄像头状态: 已停止")
        self.lbl_camera_mode_display.setText("模式: 未启动")
        self.camera_video_label.setText("摄像头未启动\n点击「开始检测」按钮启动")

        # 清空结果显示
        self.camera_cls_status.setText("未检测")
        self.camera_cls_status.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            padding: 40px;
            border-radius: 12px;
            background-color: #F7F8FA;
            color: #86909C;
        """)
        self.camera_confidence_label.setText("")
        self.camera_result_list.clear()

        self.status_bar.showMessage("摄像头检测已停止")

    def update_camera_frame(self, frame):
        """更新摄像头视频帧"""
        # 将OpenCV图像转换为QPixmap
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 缩放以适应标签大小
        scaled_pixmap = pixmap.scaled(
            self.camera_video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_video_label.setPixmap(scaled_pixmap)

    def update_camera_result(self, result):
        """更新摄像头检测结果"""
        if not result:
            return

        has_tumor = result.get('has_tumor', False)
        tumor_type = result.get('tumor_type', 'Unknown')
        confidence = result.get('confidence', 0)
        all_results = result.get('all_results', [])

        # 转换为中文
        tumor_type_cn = self.get_tumor_name_cn(tumor_type)

        # 更新状态显示
        if has_tumor:
            self.camera_cls_status.setText(f"检测到: {tumor_type_cn}")
            self.camera_cls_status.setStyleSheet(f"""
                font-size: 24px;
                font-weight: 600;
                padding: 40px;
                border-radius: 12px;
                background-color: {Theme.WARNING_LIGHT};
                color: {Theme.WARNING};
            """)
        else:
            self.camera_cls_status.setText("未检测到肿瘤")
            self.camera_cls_status.setStyleSheet(f"""
                font-size: 24px;
                font-weight: 600;
                padding: 40px;
                border-radius: 12px;
                background-color: {Theme.SUCCESS_LIGHT};
                color: {Theme.SUCCESS};
            """)

        # 更新置信度
        self.camera_confidence_label.setText(f"置信度: {confidence:.1%}")

        # 更新详细结果列表
        result_text = "Top 3 检测结果:\n"
        for item in all_results:
            name_cn = self.get_tumor_name_cn(item['cls_name'])
            result_text += f"{item['rank']}. {name_cn}: {item['conf']:.1%}\n"
        self.camera_result_list.setText(result_text)

        # 启用发送给AI分析按钮
        if hasattr(self, 'btn_camera_send_ai'):
            self.btn_camera_send_ai.setEnabled(True)
        
        # 保存当前检测结果
        self.last_camera_result = {
            'has_tumor': has_tumor,
            'tumor_type': tumor_type_cn,
            'confidence': confidence,
            'all_results': all_results
        }

        # 生成AI诊断建议
        self.generate_camera_diagnosis(tumor_type_cn, confidence, has_tumor)

    def generate_camera_diagnosis(self, tumor_type, confidence, has_tumor):
        """生成摄像头检测的AI诊断建议"""
        if not has_tumor:
            suggestion = """<h3>✅ 诊断结果：健康</h3>
<p><b>分析：</b>本次实时检测未发现明显的肿瘤病变迹象。</p>
<p><b>建议：</b></p>
<ul>
<li>继续保持良好的生活习惯</li>
<li>定期进行健康体检</li>
<li>如有不适症状请及时就医</li>
</ul>
<p style="color: #86909C;"><i>注：本结果基于实时视频分析，仅供参考，不能替代专业医生的诊断。</i></p>"""
        else:
            # 根据肿瘤类型生成建议和典型位置
            type_advice = {
                '脑膜瘤': {
                    'description': '脑膜瘤是最常见的原发性脑肿瘤之一，通常为良性。',
                    'typical_location': '大脑凸面、矢状窦旁、蝶骨嵴或鞍结节',
                    'suggestion': [
                        '建议尽快咨询神经外科专家',
                        '根据肿瘤大小和位置决定治疗方案',
                        '手术切除是主要治疗方式，预后通常良好',
                        '定期复查MRI监测肿瘤变化'
                    ]
                },
                '胶质瘤': {
                    'description': '胶质瘤是起源于脑胶质细胞的肿瘤，恶性程度不一。',
                    'typical_location': '大脑半球白质区（额叶、颞叶常见）',
                    'suggestion': [
                        '建议尽快到神经外科或肿瘤科就诊',
                        '需要进一步的病理检查确定肿瘤分级',
                        '治疗方案可能包括手术、放疗和化疗',
                        '早期诊断和治疗对预后至关重要'
                    ]
                },
                '垂体瘤': {
                    'description': '垂体瘤是发生在垂体腺的肿瘤，多为良性。',
                    'typical_location': '鞍区、垂体窝',
                    'suggestion': [
                        '建议咨询内分泌科和神经外科专家',
                        '检查激素水平是否异常',
                        '根据肿瘤类型选择药物或手术治疗',
                        '定期复查监测肿瘤变化'
                    ]
                }
            }

            advice = type_advice.get(tumor_type, {
                'description': '检测到肿瘤病变，需要进一步检查确认。',
                'typical_location': '需进一步检查确定',
                'suggestion': [
                    '建议尽快咨询专业医生',
                    '进行进一步的影像学检查（如MRI多模态扫描）',
                    '根据检查结果制定治疗方案',
                    '保持积极心态，配合医生治疗'
                ]
            })

            confidence_text = f"{confidence:.1%}"
            typical_location = advice['typical_location']
            suggestion = f"""<h3>⚠️ 诊断结果：检测到 {tumor_type}</h3>
<p><b>肿瘤类型：</b>{tumor_type}</p>
<p><b>置信度：</b>{confidence_text}</p>
<p><b>典型好发部位：</b>{typical_location}</p>
<p><b>说明：</b>{advice['description']}</p>
<p><b>建议：</b></p>
<ul>
"""
            for item in advice['suggestion']:
                suggestion += f"<li>{item}</li>\n"
            suggestion += """</ul>
<p style="color: #F53F3F;"><b>⚠️ 重要提示：</b>本结果基于实时视频的AI分析，仅供参考。建议尽快携带检查结果咨询专业医生，必要时进行MRI多模态扫描获取更全面的诊断信息。</p>"""

        self.camera_diagnosis_text.setHtml(suggestion)

    def send_camera_result_to_ai(self):
        """发送摄像头检测结果给AI进行分析"""
        if not hasattr(self, 'last_camera_result') or not self.last_camera_result:
            QMessageBox.warning(self, "提示", "请先进行摄像头检测")
            return
        
        result = self.last_camera_result
        has_tumor = result['has_tumor']
        tumor_type = result['tumor_type']
        confidence = result['confidence']
        all_results = result['all_results']
        
        # 构建检测结果文本
        result_text = "摄像头实时检测结果：\n"
        if has_tumor:
            result_text += f"检测到肿瘤：{tumor_type}\n"
            result_text += f"置信度：{confidence:.1%}\n\n"
            result_text += "Top 3 检测结果：\n"
            for item in all_results:
                name_cn = self.get_tumor_name_cn(item['cls_name'])
                result_text += f"{item['rank']}. {name_cn}: {item['conf']:.1%}\n"
        else:
            result_text += "未检测到肿瘤，影像表现正常。\n"
        
        # 发送到右侧AI聊天区
        prompt = f"""我是基于实时摄像头检测的结果，请分析以下检测数据并提供专业的诊断建议：

{result_text}

请根据以上实时检测结果，提供：
1. 对检测结果的解读
2. 可能的临床意义
3. 下一步的建议（如需要进一步检查或就医）
4. 注意事项

请注意这是基于实时视频流的分析结果，仅供参考。"""
        
        # 切换到右侧AI聊天区并发送消息
        if hasattr(self, 'chat_widget'):
            # 添加用户消息
            self.chat_widget.add_message(prompt, is_user=True)
            
            # 显示加载中
            self.chat_widget.add_message("🤔 正在分析摄像头检测结果...", is_user=False)
            
            # 启动API线程
            from PyQt5.QtCore import QThread
            self.ai_thread = AIRequestThread(prompt, self.chat_widget.API_URL, self.chat_widget.API_KEY, self.chat_widget.MODEL_NAME)
            self.ai_thread.reply_finished.connect(lambda reply: self.on_camera_ai_reply(reply))
            self.ai_thread.error_occurred.connect(lambda error: self.on_camera_ai_error(error))
            self.ai_thread.start()
    
    def on_camera_ai_reply(self, reply):
        """处理摄像头检测的AI回复"""
        # 移除加载提示
        if hasattr(self, 'chat_widget'):
            # 重新发送AI回复
            self.chat_widget.add_message(reply, is_user=False)
            # 保存最后一条回复用于语音播报（清理Markdown格式）
            self.chat_widget.last_ai_reply = self.chat_widget._clean_text_for_voice(reply)
            # 自动语音播报
            if len(reply) > 20:
                self.chat_widget.play_voice()
    
    def on_camera_ai_error(self, error):
        """处理摄像头检测的AI错误"""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.add_message(f"❌ AI分析出错：{error}", is_user=False)

    def update_camera_status(self, status):
        """更新摄像头状态"""
        self.lbl_camera_status.setText(f"摄像头状态: {status}")
        self.status_bar.showMessage(status)

    def handle_camera_error(self, error_msg):
        """处理摄像头错误"""
        QMessageBox.critical(self, "摄像头错误", error_msg)
        self.stop_camera_detection()

    def closeEvent(self, event):
        """关闭事件"""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            self.batch_worker.wait()

        # 停止摄像头检测
        if self.is_camera_running:
            self.stop_camera_detection()

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = TumorDetectionApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
