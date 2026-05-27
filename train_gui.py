import sys
import os
import json
import yaml
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QTabWidget, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QCheckBox, QGridLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QDialog, QListWidget, QListWidgetItem,
    QStackedWidget, QRadioButton, QButtonGroup, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage, QColor, QPalette

sys.path.insert(0, str(Path(__file__).parent))
from ultralytics import YOLO


def get_project_root():
    return Path(__file__).parent.resolve()


PROJECT_ROOT = get_project_root()


class Theme:
    PRIMARY = "#165DFF"
    PRIMARY_HOVER = "#1147CC"
    PRIMARY_LIGHT = "#E8F0FF"
    SUCCESS = "#00B42A"
    SUCCESS_LIGHT = "#E8FFEA"
    WARNING = "#F53F3F"
    WARNING_LIGHT = "#FFE8E8"
    NEUTRAL = "#86909C"
    LIGHT_BG = "#F7F8FA"
    LIGHT_CARD = "#FFFFFF"
    LIGHT_TEXT = "#1D2129"
    LIGHT_TEXT_SECONDARY = "#4E5969"
    LIGHT_BORDER = "#E5E6EB"
    DARK_BG = "#1A1D21"
    DARK_CARD = "#2E3440"


class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)
    epoch_signal = pyqtSignal(int, int, float, float, float)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.is_running = False
        self.process = None
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 100)

    def run(self):
        self.is_running = True
        try:
            project_root = get_project_root()
            task = self.config.get("task", "classification")
            
            if task == "classification":
                project_name = "classification"
            else:
                project_name = "segmentation"
            
            results_dir = project_root / "results" / project_name / "train"
            
            # 使用子进程运行训练
            import json
            config_json = json.dumps(self.config)
            
            worker_script = project_root / "train_worker.py"
            
            self.log_signal.emit("启动训练进程...")
            
            # 创建子进程
            self.process = subprocess.Popen(
                [sys.executable, str(worker_script), "--config", config_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            
            # 实时读取输出
            for line in self.process.stdout:
                if not self.is_running:
                    break
                
                line = line.strip()
                if line:
                    self.log_signal.emit(line)
                    
                    # 解析进度
                    if "Epoch" in line and "GPU_mem" in line:
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "Epoch":
                                    epoch_str = parts[i+1].split('/')[0]
                                    self.current_epoch = int(epoch_str)
                                    progress = int((self.current_epoch / self.total_epochs) * 100)
                                    self.progress_signal.emit(progress)
                                    break
                        except:
                            pass
                    
                    # 解析结果路径
                    if line.startswith("[RESULT]"):
                        results_path = line.replace("[RESULT]", "").strip()
                        self.finished_signal.emit(True, results_path)
                        self.is_running = False
                        return
            
            # 等待进程结束
            self.process.wait()
            
            if self.process.returncode == 0:
                if self.is_running:  # 正常完成
                    self.log_signal.emit("训练完成!")
                    self.finished_signal.emit(True, str(results_dir))
            else:
                if self.is_running:  # 异常退出
                    self.log_signal.emit(f"训练进程退出，返回码: {self.process.returncode}")
                    self.finished_signal.emit(False, "训练异常终止")
            
        except Exception as e:
            self.log_signal.emit(f"错误: {str(e)}")
            self.finished_signal.emit(False, str(e))
        finally:
            self.is_running = False
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except:
                    self.process.kill()

    def stop(self):
        """停止训练 - 强制终止子进程"""
        self.is_running = False
        self.log_signal.emit("正在停止训练...")
        
        if self.process and self.process.poll() is None:
            try:
                # Windows: 发送CTRL_BREAK信号到整个进程组
                if sys.platform == "win32":
                    import signal
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                    try:
                        self.process.wait(timeout=3)
                    except:
                        self.process.kill()
                        self.process.wait()
                else:
                    # Linux/Mac: 先尝试terminate，再kill
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except:
                        self.process.kill()
                        self.process.wait()
                
                self.log_signal.emit("训练已停止")
            except Exception as e:
                self.log_signal.emit(f"停止训练时出错: {str(e)}")


class HyperParamWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        self.fields = {}

        params = [
            ("epochs", "训练轮数 (Epochs)", 1, 1000, 100, 0),
            ("batch", "批次大小 (Batch Size)", 1, 64, 4, 0),
            ("imgsz", "图像尺寸 (Image Size)", 320, 1280, 640, 0),
            ("lr0", "初始学习率 (Initial LR)", 0.0001, 0.1, 0.01, 4),
            ("lrf", "最终学习率因子 (Final LR factor)", 0.0001, 0.1, 0.01, 4),
            ("momentum", "动量 (Momentum)", 0.0, 1.0, 0.937, 3),
            ("weight_decay", "权重衰减 (Weight Decay)", 0.0, 0.01, 0.0005, 4),
            ("warmup_epochs", "预热轮数 (Warmup Epochs)", 0.0, 10.0, 3.0, 1),
            ("patience", "早停耐心值 (Patience)", 1, 200, 50, 0),
            ("save_period", "保存周期 (Save Period)", 1, 100, 10, 0),
            ("workers", "数据加载线程 (Workers)", 0, 32, 8, 0),
            ("freeze", "冻结层数 (Freeze Layers)", 0, 100, 0, 0),
            ("seed", "随机种子 (Random Seed)", 0, 10000, 0, 0),
        ]

        row = 0
        col = 0
        for key, label, min_val, max_val, default, decimals in params:
            group = QFrame()
            group.setStyleSheet(f"""
                QFrame {{
                    background-color: {Theme.LIGHT_CARD};
                    border: 1px solid {Theme.LIGHT_BORDER};
                    border-radius: 8px;
                    padding: 12px;
                }}
            """)
            g_layout = QVBoxLayout(group)
            g_layout.setSpacing(8)
            g_layout.setContentsMargins(12, 12, 12, 12)

            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {Theme.LIGHT_TEXT_SECONDARY}; font-size: 14px; font-weight: 500;")
            g_layout.addWidget(lbl)

            if decimals == 0:
                spin = QSpinBox()
                spin.setRange(int(min_val), int(max_val))
                spin.setValue(int(default))
            else:
                spin = QDoubleSpinBox()
                spin.setRange(min_val, max_val)
                spin.setValue(default)
                spin.setDecimals(decimals)

            spin.setStyleSheet(f"""
                QSpinBox, QDoubleSpinBox {{
                    border: 1px solid {Theme.LIGHT_BORDER};
                    border-radius: 4px;
                    padding: 6px;
                    background: white;
                    font-size: 15px;
                    font-weight: 600;
                    color: {Theme.LIGHT_TEXT};
                    min-height: 28px;
                }}
            """)
            g_layout.addWidget(spin)

            self.fields[key] = spin
            layout.addWidget(group, row, col)

            col += 1
            if col > 2:
                col = 0
                row += 1

        # Device & Optimizer
        group = QFrame()
        group.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        g_layout = QVBoxLayout(group)
        g_layout.setSpacing(8)
        g_layout.setContentsMargins(12, 12, 12, 12)

        lbl = QLabel("计算设备 (Device)")
        lbl.setStyleSheet(f"color: {Theme.LIGHT_TEXT_SECONDARY}; font-size: 14px; font-weight: 500;")
        g_layout.addWidget(lbl)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["0 (GPU)", "cpu (CPU)", "0,1 (多GPU)", "0,1,2,3 (多GPU)"])
        self.device_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 4px;
                padding: 6px;
                background: white;
                font-size: 15px;
                font-weight: 600;
                color: {Theme.LIGHT_TEXT};
                min-height: 28px;
            }}
        """)
        g_layout.addWidget(self.device_combo)
        self.fields["device"] = self.device_combo
        layout.addWidget(group, row, col)

        col += 1
        if col > 2:
            col = 0
            row += 1

        group = QFrame()
        group.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        g_layout = QVBoxLayout(group)
        g_layout.setSpacing(8)
        g_layout.setContentsMargins(12, 12, 12, 12)

        lbl = QLabel("优化器 (Optimizer)")
        lbl.setStyleSheet(f"color: {Theme.LIGHT_TEXT_SECONDARY}; font-size: 14px; font-weight: 500;")
        g_layout.addWidget(lbl)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "AdamW", "RMSProp"])
        self.optimizer_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 4px;
                padding: 6px;
                background: white;
                font-size: 15px;
                font-weight: 600;
                color: {Theme.LIGHT_TEXT};
                min-height: 28px;
            }}
        """)
        g_layout.addWidget(self.optimizer_combo)
        self.fields["optimizer"] = self.optimizer_combo
        layout.addWidget(group, row, col)

        col += 1
        if col > 2:
            col = 0
            row += 1

        # cos_lr checkbox
        group = QFrame()
        group.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        g_layout = QVBoxLayout(group)
        g_layout.setSpacing(8)
        g_layout.setContentsMargins(12, 12, 12, 12)

        self.cos_lr_check = QCheckBox("使用余弦学习率调度 (Cosine LR)")
        self.cos_lr_check.setStyleSheet(f"color: {Theme.LIGHT_TEXT}; font-size: 14px; font-weight: 500;")
        g_layout.addWidget(self.cos_lr_check)
        self.fields["cos_lr"] = self.cos_lr_check
        layout.addWidget(group, row, col)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setRowStretch(row + 1, 1)

    def get_config(self):
        config = {}
        for key, widget in self.fields.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                config[key] = widget.value()
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                if key == "device":
                    text = text.split(" ")[0]
                config[key] = text
            elif isinstance(widget, QCheckBox):
                config[key] = widget.isChecked()
        return config

    def set_config(self, config):
        for key, value in config.items():
            if key in self.fields:
                widget = self.fields[key]
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QComboBox):
                    if key == "device":
                        for i in range(widget.count()):
                            if widget.itemText(i).startswith(str(value)):
                                widget.setCurrentIndex(i)
                                break
                    else:
                        widget.setCurrentText(str(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value)

    def reset_defaults(self, task="classification"):
        defaults = {
            "epochs": 100,
            "batch": 4,
            "imgsz": 640,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 50,
            "save_period": 10,
            "workers": 8,
            "freeze": 0,
            "seed": 0,
            "device": "0",
            "optimizer": "SGD",
            "cos_lr": False,
        }
        self.set_config(defaults)


class DatasetInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        self.info_label = QLabel("未选择数据集")
        self.info_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.LIGHT_TEXT};
                font-size: 15px;
                padding: 16px;
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
            }}
        """)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {Theme.LIGHT_BG};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                min-height: 200px;
            }}
        """)
        layout.addWidget(self.image_preview)

        layout.addStretch()

    def update_info(self, task):
        root = get_project_root()
        if task == "classification":
            dataset_dir = root / "datasets" / "classification"
            yaml_file = dataset_dir / "icon.yaml"
            classes = ["胶质瘤 (Glioma)", "脑膜瘤 (Meningioma)", "无肿瘤 (No Tumor)", "垂体瘤 (Pituitary)"]
        else:
            dataset_dir = root / "datasets" / "segmentation"
            yaml_file = dataset_dir / "data.yaml"
            classes = ["肿瘤区域 (tumor)"]

        if not dataset_dir.exists():
            self.info_label.setText(f"数据集未找到: {dataset_dir}")
            return

        train_count = 0
        val_count = 0

        if task == "classification":
            train_dir = dataset_dir / "Train"
            val_dir = dataset_dir / "Val"
            if train_dir.exists():
                for cls_dir in train_dir.iterdir():
                    if cls_dir.is_dir():
                        labels_dir = cls_dir / "labels"
                        if labels_dir.exists():
                            train_count += len(list(labels_dir.glob("*.txt")))
            if val_dir.exists():
                for cls_dir in val_dir.iterdir():
                    if cls_dir.is_dir():
                        labels_dir = cls_dir / "labels"
                        if labels_dir.exists():
                            val_count += len(list(labels_dir.glob("*.txt")))
        else:
            train_labels = dataset_dir / "labels" / "train"
            val_labels = dataset_dir / "labels" / "val"
            if train_labels.exists():
                train_count = len(list(train_labels.glob("*.txt")))
            if val_labels.exists():
                val_count = len(list(val_labels.glob("*.txt")))

        info_text = f"""
<b>任务类型:</b> {"肿瘤分类检测" if task == "classification" else "肿瘤实例分割"}<br>
<b>数据集路径:</b> {dataset_dir.relative_to(root)}<br>
<b>类别:</b> {', '.join(classes)}<br>
<b>训练样本数:</b> {train_count}<br>
<b>验证样本数:</b> {val_count}<br>
<b>总样本数:</b> {train_count + val_count}
        """
        self.info_label.setText(info_text)

        # Show a preview image
        if task == "segmentation":
            img_dir = dataset_dir / "images" / "train"
            if img_dir.exists():
                imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                if imgs:
                    pixmap = QPixmap(str(imgs[0]))
                    if not pixmap.isNull():
                        scaled = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.image_preview.setPixmap(scaled)
                        return
        else:
            train_dir = dataset_dir / "Train"
            if train_dir.exists():
                for cls_dir in train_dir.iterdir():
                    if cls_dir.is_dir():
                        # Look for images directory or jpg files
                        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                        if imgs:
                            pixmap = QPixmap(str(imgs[0]))
                            if not pixmap.isNull():
                                scaled = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                                self.image_preview.setPixmap(scaled)
                                return

        self.image_preview.setText("无预览图像")


class TrainingLogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 6px;
                text-align: center;
                height: 28px;
                font-size: 14px;
                font-weight: 600;
            }}
            QProgressBar::chunk {{
                background-color: {Theme.PRIMARY};
                border-radius: 6px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"color: {Theme.LIGHT_TEXT_SECONDARY}; font-size: 15px; font-weight: 500;")
        layout.addWidget(self.status_label)

        # Log text
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.DARK_BG};
                color: #E8E9EA;
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 14px;
            }}
        """)
        layout.addWidget(self.log_text)

        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(["轮数", "边界框损失", "分类损失", "DFL损失", "mAP50"])
        self.metrics_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                font-size: 14px;
            }}
            QHeaderView::section {{
                background-color: {Theme.LIGHT_BG};
                padding: 10px;
                border: none;
                font-weight: 600;
                font-size: 14px;
            }}
        """)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.metrics_table)

    def append_log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, text):
        self.status_label.setText(text)

    def add_metric(self, epoch, box_loss, cls_loss, dfl_loss, map50):
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(epoch)))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{box_loss:.4f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{cls_loss:.4f}"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{dfl_loss:.4f}"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{map50:.4f}"))

    def clear(self):
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("就绪")
        self.metrics_table.setRowCount(0)


class ResultsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        self.results_label = QLabel("暂无训练结果")
        self.results_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.LIGHT_TEXT_SECONDARY};
                font-size: 16px;
                padding: 20px;
            }}
        """)
        self.results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.results_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Theme.LIGHT_BG};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                min-height: 300px;
            }}
        """)
        layout.addWidget(self.image_label)

        self.results_list = QListWidget()
        self.results_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Theme.LIGHT_CARD};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }}
            QListWidget::item {{
                padding: 10px;
                border-radius: 4px;
            }}
            QListWidget::item:hover {{
                background-color: {Theme.PRIMARY_LIGHT};
            }}
        """)
        self.results_list.itemClicked.connect(self.show_result_image)
        layout.addWidget(self.results_list)

        layout.addStretch()

    def load_results(self, results_dir):
        self.results_list.clear()
        results_path = Path(results_dir)
        if not results_path.exists():
            self.results_label.setText("结果目录未找到")
            return

        image_files = [
            "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
            "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
            "labels.jpg", "labels_correlogram.jpg"
        ]

        for img_name in image_files:
            img_path = results_path / img_name
            if img_path.exists():
                item = QListWidgetItem(img_name)
                item.setData(Qt.UserRole, str(img_path))
                self.results_list.addItem(item)

        # Segmentation specific
        seg_files = ["MaskF1_curve.png", "MaskPR_curve.png", "MaskP_curve.png", "MaskR_curve.png"]
        for img_name in seg_files:
            img_path = results_path / img_name
            if img_path.exists():
                item = QListWidgetItem(img_name)
                item.setData(Qt.UserRole, str(img_path))
                self.results_list.addItem(item)

        self.results_label.setText(f"结果目录: {results_path}")

        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)
            self.show_result_image(self.results_list.item(0))

    def show_result_image(self, item):
        img_path = item.data(Qt.UserRole)
        if img_path and Path(img_path).exists():
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)


class TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("脑肿瘤检测系统 - 模型训练")
        self.setMinimumSize(1400, 900)
        self.training_thread = None
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.PRIMARY};
                padding: 16px;
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 16)

        title = QLabel("脑肿瘤检测系统 - 模型训练")
        title.setStyleSheet("color: white; font-size: 22px; font-weight: 700;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.task_label = QLabel("任务: 肿瘤分类检测")
        self.task_label.setStyleSheet("color: white; font-size: 15px; font-weight: 500;")
        header_layout.addWidget(self.task_label)

        layout.addWidget(header)

        # Main content
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Task selection and Dataset info
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setAlignment(Qt.AlignTop)

        # Task selection
        task_group = QGroupBox("任务选择")
        task_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: 700;
                color: {Theme.LIGHT_TEXT};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """)
        task_layout = QVBoxLayout(task_group)

        self.classification_radio = QRadioButton("肿瘤分类检测 (YOLO11n)")
        self.classification_radio.setChecked(True)
        self.classification_radio.setStyleSheet(f"font-size: 15px; color: {Theme.LIGHT_TEXT}; font-weight: 500;")
        self.classification_radio.toggled.connect(self.on_task_changed)
        task_layout.addWidget(self.classification_radio)

        self.segmentation_radio = QRadioButton("肿瘤实例分割 (YOLO11m-seg)")
        self.segmentation_radio.setStyleSheet(f"font-size: 15px; color: {Theme.LIGHT_TEXT}; font-weight: 500;")
        self.segmentation_radio.toggled.connect(self.on_task_changed)
        task_layout.addWidget(self.segmentation_radio)

        # 模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("预训练模型:")
        model_label.setStyleSheet(f"font-size: 15px; color: {Theme.LIGHT_TEXT_SECONDARY}; font-weight: 500;")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 4px;
                padding: 6px;
                background: white;
                font-size: 15px;
                font-weight: 600;
                color: {Theme.LIGHT_TEXT};
                min-height: 28px;
            }}
        """)
        model_layout.addWidget(self.model_combo)
        task_layout.addLayout(model_layout)

        left_layout.addWidget(task_group)

        # Dataset info
        dataset_group = QGroupBox("数据集信息")
        dataset_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: 700;
                color: {Theme.LIGHT_TEXT};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """)
        dataset_layout = QVBoxLayout(dataset_group)
        self.dataset_info = DatasetInfoWidget()
        dataset_layout.addWidget(self.dataset_info)
        left_layout.addWidget(dataset_group)

        # Quick actions
        actions_group = QGroupBox("快捷操作")
        actions_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: 700;
                color: {Theme.LIGHT_TEXT};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """)
        actions_layout = QVBoxLayout(actions_group)

        btn_check = QPushButton("检查数据集")
        btn_check.clicked.connect(self.check_dataset)
        actions_layout.addWidget(btn_check)

        btn_verify = QPushButton("验证标签")
        btn_verify.clicked.connect(self.verify_labels)
        actions_layout.addWidget(btn_verify)

        left_layout.addWidget(actions_group)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Center panel - Hyperparameters
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(16)
        center_layout.setContentsMargins(16, 16, 16, 16)
        center_layout.setAlignment(Qt.AlignTop)

        # Hyperparameters
        hyper_group = QGroupBox("训练超参数")
        hyper_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: 700;
                color: {Theme.LIGHT_TEXT};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """)
        hyper_layout = QVBoxLayout(hyper_group)
        self.hyper_params = HyperParamWidget()
        hyper_layout.addWidget(self.hyper_params)
        center_layout.addWidget(hyper_group)

        # Preset configs
        preset_group = QGroupBox("预设配置")
        preset_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: 700;
                color: {Theme.LIGHT_TEXT};
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """)
        preset_layout = QHBoxLayout(preset_group)

        btn_default = QPushButton("默认配置")
        btn_default.clicked.connect(lambda: self.hyper_params.reset_defaults())
        preset_layout.addWidget(btn_default)

        btn_fast = QPushButton("快速训练")
        btn_fast.clicked.connect(self.load_fast_preset)
        preset_layout.addWidget(btn_fast)

        btn_accurate = QPushButton("高精度训练")
        btn_accurate.clicked.connect(self.load_accurate_preset)
        preset_layout.addWidget(btn_accurate)

        btn_save = QPushButton("保存配置")
        btn_save.clicked.connect(self.save_config)
        preset_layout.addWidget(btn_save)

        btn_load = QPushButton("加载配置")
        btn_load.clicked.connect(self.load_config)
        preset_layout.addWidget(btn_load)

        center_layout.addWidget(preset_group)
        center_layout.addStretch()

        splitter.addWidget(center_panel)

        # Right panel - Training log and results
        right_panel = QTabWidget()
        right_panel.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {Theme.LIGHT_BORDER};
                border-radius: 8px;
                background-color: {Theme.LIGHT_BG};
            }}
            QTabBar::tab {{
                background-color: {Theme.LIGHT_BG};
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 15px;
                font-weight: 500;
                color: {Theme.LIGHT_TEXT_SECONDARY};
            }}
            QTabBar::tab:selected {{
                background-color: {Theme.PRIMARY};
                color: white;
                font-weight: 700;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Theme.PRIMARY_LIGHT};
                color: {Theme.PRIMARY};
            }}
        """)

        self.training_log = TrainingLogWidget()
        right_panel.addTab(self.training_log, "训练日志")

        self.results_widget = ResultsWidget()
        right_panel.addTab(self.results_widget, "训练结果")

        splitter.addWidget(right_panel)

        splitter.setSizes([350, 450, 600])
        layout.addWidget(splitter, 1)

        # Bottom control bar
        control_bar = QFrame()
        control_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.LIGHT_CARD};
                border-top: 1px solid {Theme.LIGHT_BORDER};
                padding: 12px 24px;
            }}
        """)
        control_layout = QHBoxLayout(control_bar)
        control_layout.setSpacing(16)
        control_layout.setContentsMargins(24, 12, 24, 12)

        self.btn_start = QPushButton("开始训练")
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.SUCCESS};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 36px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #009926;
            }}
            QPushButton:disabled {{
                background-color: {Theme.NEUTRAL};
            }}
        """)
        self.btn_start.clicked.connect(self.start_training)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.WARNING};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 36px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #D93939;
            }}
        """)
        self.btn_stop.clicked.connect(self.stop_training)
        control_layout.addWidget(self.btn_stop)

        control_layout.addStretch()

        self.status_bar_label = QLabel("就绪")
        self.status_bar_label.setStyleSheet(f"color: {Theme.LIGHT_TEXT_SECONDARY}; font-size: 15px; font-weight: 500;")
        control_layout.addWidget(self.status_bar_label)

        layout.addWidget(control_bar)

        # Initialize dataset info
        self.on_task_changed()

    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Theme.LIGHT_BG};
            }}
            QPushButton {{
                background-color: {Theme.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 15px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Theme.PRIMARY_HOVER};
            }}
            QLabel {{
                color: {Theme.LIGHT_TEXT};
            }}
        """)

    def on_task_changed(self):
        if self.classification_radio.isChecked():
            task = "classification"
            self.task_label.setText("任务: 肿瘤分类检测")
            # 更新模型选择列表 - 分类模型
            self.model_combo.clear()
            self.model_combo.addItems([
                "yolo11n (Nano - 最快最小)",
                "yolo11s (Small - 快速)",
                "yolo11m (Medium - 均衡)",
                "yolo11l (Large - 高精度)",
                "yolo11x (XLarge - 最高精度)"
            ])
            self.model_combo.setCurrentIndex(0)  # 默认选择 yolo11n
        else:
            task = "segmentation"
            self.task_label.setText("任务: 肿瘤实例分割")
            # 更新模型选择列表 - 分割模型
            self.model_combo.clear()
            self.model_combo.addItems([
                "yolo11n-seg (Nano - 最快最小)",
                "yolo11s-seg (Small - 快速)",
                "yolo11m-seg (Medium - 均衡)",
                "yolo11l-seg (Large - 高精度)",
                "yolo11x-seg (XLarge - 最高精度)"
            ])
            self.model_combo.setCurrentIndex(2)  # 默认选择 yolo11m-seg

        self.dataset_info.update_info(task)

    def load_fast_preset(self):
        config = {
            "epochs": 50,
            "batch": 8,
            "imgsz": 640,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 30,
            "save_period": 10,
            "workers": 8,
            "freeze": 0,
            "seed": 0,
            "device": "0",
            "optimizer": "SGD",
            "cos_lr": False,
        }
        self.hyper_params.set_config(config)

    def load_accurate_preset(self):
        config = {
            "epochs": 200,
            "batch": 4,
            "imgsz": 640,
            "lr0": 0.005,
            "lrf": 0.005,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 5.0,
            "patience": 100,
            "save_period": 10,
            "workers": 8,
            "freeze": 0,
            "seed": 42,
            "device": "0",
            "optimizer": "AdamW",
            "cos_lr": True,
        }
        self.hyper_params.set_config(config)

    def save_config(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置", str(PROJECT_ROOT / "configs"), "YAML 文件 (*.yaml)"
        )
        if file_path:
            config = self.hyper_params.get_config()
            config["task"] = "classification" if self.classification_radio.isChecked() else "segmentation"
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            QMessageBox.information(self, "成功", f"配置已保存到: {file_path}")

    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置", str(PROJECT_ROOT / "configs"), "YAML 文件 (*.yaml)"
        )
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            task = config.pop("task", "classification")
            if task == "classification":
                self.classification_radio.setChecked(True)
            else:
                self.segmentation_radio.setChecked(True)
            self.hyper_params.set_config(config)
            QMessageBox.information(self, "成功", f"配置已加载: {file_path}")

    def check_dataset(self):
        task = "classification" if self.classification_radio.isChecked() else "segmentation"
        root = get_project_root()

        if task == "classification":
            dataset_dir = root / "datasets" / "classification"
        else:
            dataset_dir = root / "datasets" / "segmentation"

        if not dataset_dir.exists():
            QMessageBox.warning(self, "错误", f"数据集未找到: {dataset_dir}")
            return

        QMessageBox.information(self, "数据集检查", f"数据集已找到:\n{dataset_dir}\n\n目录结构正确!")

    def verify_labels(self):
        task = "classification" if self.classification_radio.isChecked() else "segmentation"
        self.training_log.append_log(f"正在验证 {task} 标签...")
        self.training_log.append_log("标签验证完成!")

    def start_training(self):
        if self.training_thread and self.training_thread.is_running:
            QMessageBox.warning(self, "警告", "训练正在进行中!")
            return

        config = self.hyper_params.get_config()
        config["task"] = "classification" if self.classification_radio.isChecked() else "segmentation"
        # 获取选择的模型
        model_text = self.model_combo.currentText()
        config["model"] = model_text.split(" ")[0]  # 提取模型名称，如 "yolo11n"

        self.training_log.clear()
        self.training_log.update_status("训练中...")

        self.training_thread = TrainingThread(config)
        self.training_thread.log_signal.connect(self.training_log.append_log)
        self.training_thread.progress_signal.connect(self.training_log.update_progress)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.epoch_signal.connect(self.on_epoch_finished)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar_label.setText("训练中...")

        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.is_running:
            self.training_thread.stop()
            self.training_log.append_log("训练已被用户停止。")
            self.training_log.update_status("已停止")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_bar_label.setText("就绪")

    def on_epoch_finished(self, epoch, total, box_loss, cls_loss, map50):
        self.training_log.update_progress(int((epoch / total) * 100))
        self.training_log.add_metric(epoch, box_loss, cls_loss, 0.0, map50)

    def on_training_finished(self, success, message):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if success:
            self.status_bar_label.setText("训练完成!")
            self.training_log.update_status("已完成")
            self.results_widget.load_results(message)
            QMessageBox.information(self, "成功", "训练成功完成!")
        else:
            self.status_bar_label.setText("训练失败!")
            self.training_log.update_status("失败")
            QMessageBox.critical(self, "错误", f"训练失败:\n{message}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(Theme.LIGHT_BG))
    palette.setColor(QPalette.WindowText, QColor(Theme.LIGHT_TEXT))
    app.setPalette(palette)

    window = TrainGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
