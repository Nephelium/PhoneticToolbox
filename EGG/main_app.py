# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.colors as mcolors

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QLabel, QLineEdit, QGridLayout, QMessageBox,
    QSizePolicy, QSpacerItem, QCheckBox, QSlider, QDialog, QFormLayout,
    QDialogButtonBox, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
import sounddevice as sd
import warnings

# --- NEW: Import parselmouth for Praat F0 algorithm ---
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("*"*50)
    print("警告: 未找到 'parselmouth' 库。")
    print("pip install praat-parselmouth")
    print("基频 (F0) 相关功能将不可用。")
    print("*"*50)


# 从本地模块导入
from gui_components import MplCanvas
from analysis_algorithms import (
    calculate_cq_sq,
    apply_highpass_filter,
    apply_lowpass_filter, find_gci_goi_peak_min_criterion,
)
# 从 config 导入常量
from config import (
    DEFAULT_PEAK_PROMINENCE, DEFAULT_VALLEY_PROMINENCE,
    DEFAULT_ZOOM_WINDOW_MS,
    DEFAULT_ROI_START, DEFAULT_ROI_DURATION,
    DEFAULT_HIGHPASS_CUTOFF, DEFAULT_LOWPASS_CUTOFF,
    DARK_STYLESHEET, MATPLOTLIB_STYLE_SETTINGS, DEFAULT_SPEC_WINDOW_MS,
    DEFAULT_SPEC_VMIN, DEFAULT_SPEC_VMAX
)
# 从 inverse_filtering 导入 (保持不变)
from inverse_filtering import apply_simplified_cp_inverse_filtering, plot_inverse_filtering_results

# --- Main Application Window ---
class EGGAnalysisApp(QMainWindow):
    plotClicked = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EGG & Audio Analysis Tool")
        self.setGeometry(100, 100, 1350, 920)
        icon_path = r"EGG.ico"
        icon = QIcon(icon_path)
        self.setWindowIcon(icon)
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.setWindowIcon(icon)

        # --- State Variables ---
        self.fs = None
        self.egg_signal_processed = None
        self.egg_signal_raw = None
        self.egg_detrended = None
        self.audio_signal = None
        self.time_vector = None
        self.cq_times = None
        self.cq_values = None
        self.sq_times = None
        self.sq_values = None
        self.all_gci_times = None
        self.all_goi_times = None
        self.all_peak_times = None
        self.current_roi_start = DEFAULT_ROI_START
        self.current_roi_duration = DEFAULT_ROI_DURATION
        self.zoom_window_ms = DEFAULT_ZOOM_WINDOW_MS
        self.last_clicked_time = None
        self.file_duration = 0.0
        self.timeline_roi_patch = None
        self.glottal_event_lines = []
        self.current_peak_prominence = DEFAULT_PEAK_PROMINENCE
        self.current_valley_prominence = DEFAULT_VALLEY_PROMINENCE
        self.current_spec_window_ms = DEFAULT_SPEC_WINDOW_MS
        self.spec_colorbar = None
        self.current_filepath = None # For re-loading
        # --- NEW: State variables for new features ---
        self.channels_flipped = False # Feature 1
        self.show_f0 = False # Feature 2
        self.f0_corrected = False # Feature 3
        self.current_spec_vmin = DEFAULT_SPEC_VMIN # Feature 4
        self.current_spec_vmax = DEFAULT_SPEC_VMAX # Feature 4
        self.audio_f0_times = None # Praat F0
        self.audio_f0_values = None # Praat F0
        self.gci_f0_times = None # GCI-derived F0
        self.gci_f0_values = None # GCI-derived F0
        self.f0_contour_line = None # Plotted F0 line reference
        self.spec_ax_f0 = None # 用于 F0 专用坐标轴的引用
        self._spec_dragging = False
        self._spec_last_x = None
        self.gci_to_goi_map = {}
        self.goi_to_gci_map = {}
        self.highpass_cutoff_current = DEFAULT_HIGHPASS_CUTOFF
        self.show_filtered_egg = True
        self.egg_highpassed = None
        self.gci_method = "slope"
        self.goi_method = "slope"
        self.show_glottal_movement = False
        self.glottal_movement_times = []

        # --- Menu ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件(&F)')
        open_action = QAction('打开 WAV 文件(&O)...', self)
        open_action.triggered.connect(self.load_wav_file)
        file_menu.addAction(open_action)
        exit_action = QAction('退出(&E)', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Tools Menu ---
        tools_menu = menubar.addMenu('工具(&T)')
        self.swap_action = QAction('交换左右声道', self, checkable=True)
        self.swap_action.setStatusTip("交换左右声道并重新加载文件（EGG 与音频对调）")
        self.swap_action.triggered.connect(self.toggle_channel_flip)
        tools_menu.addAction(self.swap_action)

        # --- Main Layout ---
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Top Grid for Plots ---
        top_grid_widget = QWidget()
        top_grid_layout = QGridLayout(top_grid_widget)
        top_grid_layout.setSpacing(15)
        top_grid_layout.setVerticalSpacing(20) # Added vertical spacing
        main_layout.addWidget(top_grid_widget, stretch=2)

        # Top-Left Plot (CQ/SQ)
        self.cq_canvas = MplCanvas(self)
        self.cq_ax = self.cq_canvas.get_axes()
        self.cq_ax_sq = self.cq_ax.twinx()
        self.cq_canvas.mpl_connect('button_press_event', self.on_left_plot_click)
        top_grid_layout.addWidget(self.cq_canvas, 0, 0)
        pass

        # Top-Right Plot (Audio Zoom)
        self.audio_zoom_canvas = MplCanvas(self)
        self.audio_zoom_ax = self.audio_zoom_canvas.get_axes()
        top_grid_layout.addWidget(self.audio_zoom_canvas, 0, 1)

        # Bottom-Left Plot (Spectrogram)
        self.spec_canvas = MplCanvas(self)
        self.spec_ax = self.spec_canvas.get_axes()
        self.spec_canvas.mpl_connect('button_press_event', self.on_left_plot_click)
        # Removed drag and scroll interactions to improve stability
        top_grid_layout.addWidget(self.spec_canvas, 1, 0)
        self.spec_vline = None

        # Bottom-Right Plot (EGG Zoom + Toolbar)
        self.egg_zoom_container = QWidget()
        self.egg_zoom_layout = QVBoxLayout(self.egg_zoom_container)
        self.egg_zoom_layout.setContentsMargins(0, 0, 0, 0)
        self.egg_zoom_layout.setSpacing(4)

        egg_toolbar = QWidget()
        egg_toolbar_layout = QHBoxLayout(egg_toolbar)
        egg_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        egg_toolbar_layout.setSpacing(8)

        self.egg_display_toggle_button = QPushButton("显示：滤波波形")
        self.egg_display_toggle_button.setToolTip("在原始与滤波波形之间切换显示")
        self.egg_display_toggle_button.clicked.connect(self.toggle_egg_display_mode)
        egg_toolbar_layout.addWidget(self.egg_display_toggle_button)

        egg_toolbar_layout.addWidget(QLabel("高通频率(Hz):"))
        self.highpass_slider = QSlider(Qt.Orientation.Horizontal)
        self.highpass_slider.setMinimum(1)
        self.highpass_slider.setMaximum(50)
        self.highpass_slider.setSingleStep(1)
        self.highpass_slider.setValue(self.highpass_cutoff_current)
        self.highpass_slider.valueChanged.connect(self.handle_highpass_slider_change)
        egg_toolbar_layout.addWidget(self.highpass_slider)
        self.highpass_label = QLabel(f"{self.highpass_cutoff_current}")
        egg_toolbar_layout.addWidget(self.highpass_label)

        self.egg_zoom_layout.addWidget(egg_toolbar)

        self.egg_zoom_canvas = MplCanvas(self)
        self.egg_zoom_ax = self.egg_zoom_canvas.get_axes()
        self.egg_zoom_canvas.mpl_connect('button_press_event', self.on_zoom_plot_click)
        self.egg_zoom_layout.addWidget(self.egg_zoom_canvas)
        top_grid_layout.addWidget(self.egg_zoom_container, 1, 1)

        # --- Bottom Widget for Controls and Timeline ---
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setSpacing(10)
        main_layout.addWidget(bottom_widget, stretch=1)

        # --- Controls Layout ---
        controls_container = QWidget()
        self.controls_container = controls_container
        controls_layout = QGridLayout(controls_container)
        controls_layout.setSpacing(10)
        bottom_layout.addWidget(controls_container)
        
        # Row 0: ROI and Zoom Controls
        controls_layout.addWidget(QLabel("开始 (s):"), 0, 0)
        self.start_time_input = QLineEdit(f"{self.current_roi_start:.2f}")
        self.start_time_input.setToolTip("ROI 起始时间（秒）")
        self.start_time_input.setFixedWidth(60)
        controls_layout.addWidget(self.start_time_input, 0, 1)
        controls_layout.addWidget(QLabel("时长 (s):"), 0, 2)
        self.duration_input = QLineEdit(f"{self.current_roi_duration:.2f}")
        self.duration_input.setToolTip("ROI 持续时长（秒）")
        self.duration_input.setFixedWidth(60)
        controls_layout.addWidget(self.duration_input, 0, 3)
        controls_layout.addWidget(QLabel("缩放 (ms):"), 0, 4)
        self.zoom_duration_input = QLineEdit(str(int(self.zoom_window_ms)))
        self.zoom_duration_input.setFixedWidth(50)
        self.zoom_duration_input.setToolTip("缩放窗口时长（毫秒，范围10–200）")
        controls_layout.addWidget(self.zoom_duration_input, 0, 5)
        update_roi_button = QPushButton("更新视图")
        update_roi_button.setToolTip("应用上述参数并刷新右侧视图")
        update_roi_button.clicked.connect(self.update_roi_plots)
        controls_layout.addWidget(update_roi_button, 0, 6)
        self.play_button = QPushButton("播放音频")
        self.play_button.setToolTip("播放当前 ROI 音频片段")
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button, 0, 7)
        self.stop_button = QPushButton("停止播放")
        self.stop_button.setToolTip("停止播放")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button, 0, 8)

        # --- Inverse Filter Button (Moved here) ---
        self.inverse_filter_button = QPushButton("逆滤波")
        self.inverse_filter_button.setToolTip("对当前音频 ROI 进行逆滤波并弹出新窗口")
        self.inverse_filter_button.clicked.connect(self.run_inverse_filtering)
        self.inverse_filter_button.setEnabled(False)
        controls_layout.addWidget(self.inverse_filter_button, 0, 11)

        controls_layout.addWidget(QLabel("IF阶数:"), 0, 12)
        self.if_order_input = QLineEdit("")
        self.if_order_input.setFixedWidth(45)
        self.if_order_input.setToolTip("逆滤波 LPC 阶数（为空则自动根据采样率设置）")
        self.if_order_input.editingFinished.connect(self.handle_if_order_change)
        controls_layout.addWidget(self.if_order_input, 0, 13)
        
        # --- NEW: (FEATURE 1) Save Analysis Button ---
        self.save_analysis_button = QPushButton("保存分析")
        self.save_analysis_button.setToolTip("保存当前 ROI 的数据（CSV）与图像（PNG）")
        self.save_analysis_button.clicked.connect(self.save_analysis)
        self.save_analysis_button.setEnabled(False)
        controls_layout.addWidget(self.save_analysis_button, 0, 9)
        # --- END NEW ---

        self.batch_button = QPushButton("批量处理")
        self.batch_button.setToolTip("批量处理整个文件夹的音频并导出结果")
        self.batch_button.clicked.connect(self.open_batch_dialog)
        controls_layout.addWidget(self.batch_button, 1, 15)
        
        # Row 1: Analysis Parameter Controls
        controls_layout.addWidget(QLabel("峰值门限:"), 1, 0)
        self.prominence_input = QLineEdit(f"{self.current_peak_prominence:.3f}")
        self.prominence_input.setFixedWidth(60)
        self.prominence_input.setToolTip("峰值检测门限（影响 GCI/GOI/CQ/SQ）。输入后按回车或失去焦点生效")
        self.prominence_input.editingFinished.connect(self.handle_peak_prominence_change)
        controls_layout.addWidget(self.prominence_input, 1, 1)

        # --- NEW: "Auto" Checkbox for Peak Prominence ---
        self.auto_prom_checkbox = QCheckBox("自动")
        self.auto_prom_checkbox.setChecked(True) # 默认开启自动模式
        self.auto_prom_checkbox.setToolTip("自动根据当前 ROI 计算峰值门限；取消勾选可锁定当前值")
        controls_layout.addWidget(self.auto_prom_checkbox, 1, 2)
        # --- END NEW ---

        pass

        # --- MODIFIED: Shifted layout for Valley and Spec Win ---
        controls_layout.addWidget(QLabel("谷值门限:"), 1, 3)
        self.valley_prominence_input = QLineEdit(f"{self.current_valley_prominence:.3f}")
        self.valley_prominence_input.setFixedWidth(60)
        self.valley_prominence_input.setToolTip("谷值检测门限（影响 GCI/GOI/CQ/SQ）。输入后按回车或失去焦点生效")
        self.valley_prominence_input.editingFinished.connect(self.handle_valley_prominence_change)
        controls_layout.addWidget(self.valley_prominence_input, 1, 4)
        
        controls_layout.addWidget(QLabel("语谱窗长 (ms):"), 1, 5)
        self.spec_window_input = QLineEdit(str(int(self.current_spec_window_ms)))
        self.spec_window_input.setFixedWidth(50)
        self.spec_window_input.setToolTip("语谱图 FFT 窗长（毫秒），影响时频分辨率。输入后按回车或失去焦点生效")
        self.spec_window_input.editingFinished.connect(self.handle_spec_window_change)
        controls_layout.addWidget(self.spec_window_input, 1, 6)
        # --- END MODIFIED ---

        # --- NEW: Row 2: New Feature Controls ---
        # Feature: Glottal Movement Detection (New Button)
        self.detect_glottal_button = QPushButton("声门移动")
        self.detect_glottal_button.setCheckable(True)
        self.detect_glottal_button.setToolTip("检测低频高能量点并标记（黄色竖线）")
        self.detect_glottal_button.toggled.connect(self.toggle_glottal_detection)
        controls_layout.addWidget(self.detect_glottal_button, 0, 10)

        # Feature 2: Show F0
        self.show_f0_checkbox = QCheckBox("显示 F0")
        self.show_f0_checkbox.setToolTip("在语谱图上叠加或隐藏 Praat F0 曲线")
        self.show_f0_checkbox.toggled.connect(self.toggle_f0_visibility)
        controls_layout.addWidget(self.show_f0_checkbox, 1, 7)

        # Feature 3: Correct F0
        self.correct_f0_checkbox = QCheckBox("校正 F0 (GCI)")
        self.correct_f0_checkbox.setToolTip("使用 GCI 间隔推导的 F0 进行校正（适合声门破裂等情况）")
        self.correct_f0_checkbox.toggled.connect(self.toggle_f0_correction)
        controls_layout.addWidget(self.correct_f0_checkbox, 1, 8)

        self.gci_method_button = QPushButton("GCI：斜率")
        self.gci_method_button.setToolTip("在斜率法与尺度法（25%）之间切换 GCI 标定")
        self.gci_method_button.clicked.connect(self.toggle_gci_method)
        controls_layout.addWidget(self.gci_method_button, 1, 9)

        self.goi_method_button = QPushButton("GOI：斜率")
        self.goi_method_button.setToolTip("在斜率法与尺度法（25%）之间切换 GOI 标定")
        self.goi_method_button.clicked.connect(self.toggle_goi_method)
        controls_layout.addWidget(self.goi_method_button, 1, 10)

        # Feature 4: Spectrogram VMin/VMax
        controls_layout.addWidget(QLabel("语谱最小dB:"), 1, 11)
        self.spec_vmin_input = QLineEdit(f"{self.current_spec_vmin:.1f}")
        self.spec_vmin_input.setFixedWidth(50)
        self.spec_vmin_input.setToolTip("语谱图最小 dB（颜色范围），在“Update ROI”时应用")
        controls_layout.addWidget(self.spec_vmin_input, 1, 12)
        
        controls_layout.addWidget(QLabel("语谱最大dB:"), 1, 13)
        self.spec_vmax_input = QLineEdit(f"{self.current_spec_vmax:.1f}")
        self.spec_vmax_input.setFixedWidth(50)
        self.spec_vmax_input.setToolTip("语谱图最大 dB（颜色范围），在“Update ROI”时应用")
        controls_layout.addWidget(self.spec_vmax_input, 1, 14)
        
        # Disable F0 controls if parselmouth is missing
        if not PARSELMOUTH_AVAILABLE:
            self.show_f0_checkbox.setEnabled(False)
            self.correct_f0_checkbox.setEnabled(False)
            self.show_f0_checkbox.setToolTip("F0 features disabled. 'parselmouth' library not found.")
            self.correct_f0_checkbox.setToolTip("F0 features disabled. 'parselmouth' library not found.")

        # Add stretch
        # --- MODIFIED: Adjusted spacer column ---
        controls_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum), 0, 15)
        controls_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum), 1, 15)
        controls_layout.setColumnStretch(12, 1)
        controls_layout.setColumnStretch(15, 1)
        # --- END MODIFIED ---

        # --- Timeline ---
        self.timeline_canvas = MplCanvas(self, height=1.5, dpi=80)
        self.timeline_ax = self.timeline_canvas.get_axes()
        self.timeline_canvas.mpl_connect('button_press_event', self.on_timeline_click)
        bottom_layout.addWidget(self.timeline_canvas)

        # --- NEW: Timeline window and slider ---
        self.timeline_window_s = 60.0
        self.timeline_offset_s = 0.0
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setSingleStep(1)
        self.timeline_slider.setPageStep(5)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setVisible(False)
        self.timeline_slider.setToolTip("拖动定位底部时间轴的一分钟视窗起始位置")
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_change)
        bottom_layout.addWidget(self.timeline_slider)

        # --- Initial Setup ---
        self._setup_initial_plots()
        self.plotClicked.connect(self.handle_plot_click)

    def _setup_initial_plots(self):
        common_style = {'color': 'lightgray'}
        cq_color = 'cyan'
        sq_color = 'yellow'

        # CQ/SQ Plot (Primary Axis: CQ)
        self.cq_ax.set_title("EGG CQ & SQ (Peak-Min Method)", **common_style)
        self.cq_ax.set_xlabel("Time (s)", **common_style)
        # self.cq_ax.set_ylabel("Contact Quotient (CQ)", color=cq_color) # <-- REMOVED Label
        self.cq_ax.tick_params(axis='x', colors='lightgray')
        self.cq_ax.tick_params(axis='y', colors=cq_color, labelcolor=cq_color)
        self.cq_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        self.cq_ax.set_ylim(0, 1)

        # CQ/SQ Plot (Secondary Axis: SQ)
        # self.cq_ax_sq.set_ylabel("Speed Quotient (SQ)", color=sq_color) # <-- REMOVED Label
        self.cq_ax_sq.tick_params(axis='y', colors=sq_color, labelcolor=sq_color)
        self.cq_ax_sq.set_ylim(-1.1, 1.1)
        self.cq_ax_sq.spines['right'].set_color(sq_color)
        self.cq_ax_sq.spines['left'].set_color(cq_color)
        self.cq_ax_sq.grid(False) # Ensure twin axis grid is off

        # Spectrogram Plot
        self.spec_ax.set_title("Spectrogram", **common_style)
        self.spec_ax.set_xlabel("Time (s)", **common_style)
        self.spec_ax.set_ylabel("Frequency (Hz)", **common_style)
        self.spec_ax.tick_params(axis='both', colors='lightgray')
        # self.spec_ax.grid(True, linestyle=':', alpha=0.4, color='gray') # Grid might look busy with colorbar
        self.spec_ax.set_ylim(0, 5000)

        # 为 F0 创建并设置 twinx 坐标轴
        if hasattr(self, 'spec_ax_f0') and self.spec_ax_f0:
             try: self.spec_ax_f0.remove() # 移除旧的
             except Exception: pass
        self.spec_ax_f0 = self.spec_ax.twinx()
        self.spec_ax_f0.tick_params(axis='y', colors='lightgray')
        self.spec_ax_f0.set_ylim(50, 500) # 设置 F0 的默认 Y 轴范围
        self.spec_ax_f0.grid(False) # F0 轴不需要网格
        self.spec_ax_f0.spines['right'].set_color('lightgray') # 显示右侧的轴
        self.spec_ax_f0.spines['left'].set_visible(False) # 隐藏与主轴重叠的左轴
        self.spec_ax_f0.spines['top'].set_visible(False)
        self.spec_ax_f0.spines['bottom'].set_visible(False)

        # Audio Zoom Plot
        self.audio_zoom_ax.set_title(f"Audio (+/- {self.zoom_window_ms/2:.0f}ms)", **common_style)
        self.audio_zoom_ax.set_xlabel("Time relative to click (ms)", **common_style)
        self.audio_zoom_ax.tick_params(axis='x', colors='lightgray')
        self.audio_zoom_ax.tick_params(axis='y', colors='lightgray')
        self.audio_zoom_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        self.audio_zoom_ax.set_xlim(-self.zoom_window_ms/2, self.zoom_window_ms/2)

        # EGG Zoom Plot
        self.egg_zoom_ax.set_title(f"EGG (+/- {self.zoom_window_ms/2:.0f}ms)", **common_style)
        self.egg_zoom_ax.set_xlabel("Time relative to click (ms)", **common_style)
        self.egg_zoom_ax.tick_params(axis='x', colors='lightgray')
        self.egg_zoom_ax.tick_params(axis='y', colors='lightgray')
        self.egg_zoom_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        self.egg_zoom_ax.set_xlim(-self.zoom_window_ms/2, self.zoom_window_ms/2)

        # Timeline Plot
        self.timeline_ax.set_title("Timeline Overview (Click to set Start Time)", **common_style)
        self.timeline_ax.set_xlabel("Time (s)", **common_style)
        self.timeline_ax.set_yticks([])
        self.timeline_ax.tick_params(axis='x', colors='lightgray')
        self.timeline_ax.set_xlim(0, 10) # Initial limit

        # Apply layout (try constrained first) and draw all
        for canvas in [self.cq_canvas, self.spec_canvas, self.audio_zoom_canvas, self.egg_zoom_canvas, self.timeline_canvas]:
            try:
                canvas.fig.set_layout_engine('constrained')
            except Exception:
                try: canvas.fig.tight_layout()
                except Exception: pass # Fallback if layout fails
            canvas.draw()

    # --- MODIFIED: Refactored file loading logic ---
    def load_wav_file(self):
        """Presents the file dialog and triggers loading."""
        filepath, _ = QFileDialog.getOpenFileName(self, "打开 WAV 文件", "", "WAV 文件 (*.wav)")
        if not filepath: return
        # Reset flip state for new file
        self.channels_flipped = False 
        if hasattr(self, 'swap_action'):
            self.swap_action.setChecked(False)
        self._load_data_from_path(filepath)

    def toggle_channel_flip(self, checked=False):
        """(FEATURE 1) Toggles the channel flip state and reloads the file."""
        if self.current_filepath is None:
            QMessageBox.warning(self, "无文件", "请先加载文件。")
            if hasattr(self, 'swap_action'):
                self.swap_action.setChecked(self.channels_flipped)
            return
            
        self.channels_flipped = checked
        
        QMessageBox.information(self, 
                                "声道已交换", 
                                f"声道已交换。EGG 现在位于{'右' if self.channels_flipped else '左'}声道。\n重新加载文件...")
        self._load_data_from_path(self.current_filepath)

    def _load_data_from_path(self, filepath):
        """Internal function to load and process the file."""
        try:
            self.current_filepath = filepath
            sd.stop()
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", wavfile.WavFileWarning)
                 self.fs, data = wavfile.read(filepath)

            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("WAV 文件必须是立体声（2 声道）。左=EGG，右=音频。")

            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                if max_val == 0: max_val = 1
                data = data.astype(np.float32) / max_val
            elif np.issubdtype(data.dtype, np.floating):
                 data = data.astype(np.float32)
            else:
                 raise ValueError(f"不支持的数据类型: {data.dtype}")

            # --- MODIFIED: (FEATURE 1) Normalize channels independently to -0.7~0.7 ---
            # Avoid division by zero
            max_val_0 = np.max(np.abs(data[:, 0]))
            if max_val_0 > 0:
                data[:, 0] = (data[:, 0] / max_val_0) * 0.7
            
            max_val_1 = np.max(np.abs(data[:, 1]))
            if max_val_1 > 0:
                data[:, 1] = (data[:, 1] / max_val_1) * 0.7
            # --- END MODIFICATION ---

            # --- MODIFIED: (FEATURE 1) Assign channels based on flip state ---
            egg_channel_idx = 0 if not self.channels_flipped else 1
            audio_channel_idx = 1 if not self.channels_flipped else 0
            self.egg_signal_raw = data[:, egg_channel_idx]
            self.audio_signal = data[:, audio_channel_idx]
            print(f"Channels Flipped: {self.channels_flipped}. EGG=Channel {egg_channel_idx}, Audio=Channel {audio_channel_idx}")
            # --- END MODIFICATION ---

            # --- NEW: (FEATURE 2) Calculate global Praat F0 ---
            self._calculate_global_f0()
            # --- END NEW ---

            self.egg_detrended = signal.detrend(self.egg_signal_raw)
            egg_highpassed = apply_highpass_filter(self.egg_detrended,
                                                   cutoff_freq=self.highpass_cutoff_current,
                                                   fs=self.fs)
            self.egg_highpassed = egg_highpassed
            self.egg_signal_processed = apply_lowpass_filter(self.egg_highpassed,
                                                             cutoff_freq=DEFAULT_LOWPASS_CUTOFF,
                                                             fs=self.fs)
            self.time_vector = np.arange(len(self.egg_signal_processed)) / self.fs
            self.file_duration = self.time_vector[-1] if len(self.time_vector) > 0 else 0.0

            if hasattr(self, 'auto_prom_checkbox'): # 确保复选框已创建
                self.auto_prom_checkbox.setChecked(True)

            print(f"Loaded file: {filepath}")
            print(f"Sample Rate: {self.fs} Hz")
            print(f"Duration: {self.file_duration:.2f} seconds")
            try:
                default_if_order = int(self.fs / 1000) + 6
                self.if_order_input.setText(f"{default_if_order}")
            except Exception:
                self.if_order_input.setText("")

            # (保留 Valley Prom 和 Spec Win 的 try/except 块)
            try:
                self.current_valley_prominence = float(self.valley_prominence_input.text())
                if self.current_valley_prominence <= 0: raise ValueError("Valley Prominence must be positive")
            except ValueError:
                 print(f"Warning: Invalid initial valley prominence. Using default {DEFAULT_VALLEY_PROMINENCE:.3f}.")
                 self.current_valley_prominence = DEFAULT_VALLEY_PROMINENCE
                 self.valley_prominence_input.setText(f"{self.current_valley_prominence:.3f}")
            try: # Read Spec Win
                self.current_spec_window_ms = float(self.spec_window_input.text())
                if self.current_spec_window_ms <= 0: raise ValueError("Spec Window must be positive")
            except ValueError:
                 print(f"Warning: Invalid initial Spec window. Using default {DEFAULT_SPEC_WINDOW_MS:.1f}.")
                 self.current_spec_window_ms = DEFAULT_SPEC_WINDOW_MS
                 self.spec_window_input.setText(f"{self.current_spec_window_ms:.1f}")

            # --- (MODIFIED) 1. Define initial ROI ---
            self.current_roi_start = DEFAULT_ROI_START
            self.current_roi_duration = min(DEFAULT_ROI_DURATION, self.file_duration)
            self.start_time_input.setText(f"{self.current_roi_start:.2f}")
            self.duration_input.setText(f"{self.current_roi_duration:.2f}")

            # --- (NEW) 2. Calculate dynamic default peak prominence based on initial ROI ---
            start_idx = max(0, int(self.current_roi_start * self.fs))
            end_idx = min(len(self.egg_signal_processed), int(self.current_roi_start + self.current_roi_duration) * self.fs)
            
            new_peak_prom = DEFAULT_PEAK_PROMINENCE # Fallback
            if start_idx < end_idx:
                initial_roi_egg = self.egg_signal_processed[start_idx:end_idx]
                if len(initial_roi_egg) > 0:
                    try:
                        max_val = np.max(initial_roi_egg)
                        min_val = np.min(initial_roi_egg)
                        # 你的逻辑：(abs(max) + abs(min)) / 2
                        calc_prom = (np.abs(max_val) + np.abs(min_val)) / 2.0
                        
                        # 合理性检查 (例如，确保它在 0.001 和 0.5 之间)
                        if 1e-5 < calc_prom < 0.5: 
                            new_peak_prom = calc_prom
                            print(f"Calculated dynamic Peak Prominence: {new_peak_prom:.3f}")
                        else:
                            print(f"Calculated prom ({calc_prom:.3f}) out of sane range, using default {DEFAULT_PEAK_PROMINENCE}.")
                            new_peak_prom = DEFAULT_PEAK_PROMINENCE
                    except Exception as e:
                        print(f"Warning: Could not calc dynamic prom: {e}. Using default.")
            
            # 3. Set the state variable AND the UI text box
            self.current_peak_prominence = new_peak_prom
            self.prominence_input.setText(f"{self.current_peak_prominence:.3f}")

            # 4. Now, read/set the *other* defaults (Valley Prom, Spec Win)
            try:
                self.current_valley_prominence = float(self.valley_prominence_input.text())
                if self.current_valley_prominence <= 0: raise ValueError("Valley Prominence must be positive")
            except ValueError:
                 print(f"Warning: Invalid initial valley prominence. Using default {DEFAULT_VALLEY_PROMINENCE:.3f}.")
                 self.current_valley_prominence = DEFAULT_VALLEY_PROMINENCE
                 self.valley_prominence_input.setText(f"{self.current_valley_prominence:.3f}")
            
            try: # Read Spec Win
                self.current_spec_window_ms = float(self.spec_window_input.text())
                if self.current_spec_window_ms <= 0: raise ValueError("Spec Window must be positive")
            except ValueError:
                 print(f"Warning: Invalid initial Spec window. Using default {DEFAULT_SPEC_WINDOW_MS:.1f}.")
                 self.current_spec_window_ms = DEFAULT_SPEC_WINDOW_MS
                 self.spec_window_input.setText(f"{self.current_spec_window_ms:.1f}")
            
            # --- Initial GCI/GOI Calculation (Triggers GCI F0) ---
            # (现在这一步会使用我们刚计算出的 self.current_peak_prominence)
            self._calculate_global_events()

            # --- Reset UI and plot ---
            self.last_clicked_time = None
            # --- NEW: Reset F0 checkboxes ---
            self.show_f0 = False
            self.f0_corrected = False
            self.show_f0_checkbox.setChecked(False)
            self.correct_f0_checkbox.setChecked(False)
            if self.f0_contour_line:
                try: self.f0_contour_line.remove()
                except Exception: pass
                self.f0_contour_line = None
            # --- END NEW ---
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.inverse_filter_button.setEnabled(True)
            self.save_analysis_button.setEnabled(True) # --- NEW: Enable save button
            self.plot_timeline()
            # Configure timeline slider visibility and range
            try:
                if self.file_duration > self.timeline_window_s:
                    self.timeline_slider.setEnabled(True)
                    self.timeline_slider.setVisible(True)
                    max_start = max(0, int(self.file_duration - self.timeline_window_s))
                    self.timeline_slider.setMaximum(max_start)
                    self.timeline_slider.setMinimum(0)
                    self.timeline_slider.setValue(0)
                    self.timeline_offset_s = 0.0
                    self.update_timeline_view_window()
                else:
                    self.timeline_slider.setEnabled(False)
                    self.timeline_slider.setVisible(False)
                    self.timeline_offset_s = 0.0
                    self.update_timeline_view_window()
            except Exception as e:
                print(f"Error configuring timeline slider: {e}")
            self.update_roi_plots()
            self.clear_zoom_plots()

        except FileNotFoundError:
             QMessageBox.critical(self, "Error Loading File", f"File not found:\n{filepath}")
             self._reset_app_state()
        except ValueError as e:
             QMessageBox.critical(self, "Error Loading File", f"Invalid WAV file format or data:\n{e}")
             self._reset_app_state()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"An unexpected error occurred:\n{e}")
            import traceback
            traceback.print_exc()
            self._reset_app_state()

    # --- MODIFIED: (FEATURE 2) Adjusted audio scaling ---
    def run_inverse_filtering(self):
        if self.audio_signal is None or self.egg_signal_processed is None or self.fs is None:
            QMessageBox.warning(self, "Error", "No EGG/Audio data loaded.")
            return
        if self.all_gci_times is None:
             QMessageBox.warning(self, "Error", "Global GCI times not calculated. Cannot perform CP-based filtering.")
             return
        start_s = self.current_roi_start
        end_s = start_s + self.current_roi_duration
        start_idx = max(0, int(start_s * self.fs))
        end_idx = min(len(self.audio_signal), int(end_s * self.fs))
        if start_idx >= end_idx:
            QMessageBox.warning(self, "Error", "Invalid or zero-duration ROI selected.")
            return
        roi_audio = np.array(self.audio_signal[start_idx:end_idx])
        roi_egg = np.array(self.egg_signal_processed[start_idx:end_idx])
        gci_times_np = np.array(self.all_gci_times)
        gci_indices_in_roi = (gci_times_np >= start_s) & (gci_times_np < end_s)
        gci_times_for_roi = gci_times_np[gci_indices_in_roi]
        gci_times_relative_to_roi_start = gci_times_for_roi - start_s
        print(f"Running inverse filtering for ROI: {start_s:.2f}s - {end_s:.2f}s")
        print(f"Found {len(gci_times_relative_to_roi_start)} GCIs within the ROI.")
        order_text = self.if_order_input.text().strip()
        lp_order_val = None
        if order_text != "":
            try:
                o = int(order_text)
                if o <= 0:
                    raise ValueError("invalid")
                lp_order_val = o
            except Exception:
                QMessageBox.warning(self, "输入错误", "IF 阶数无效，将使用自动设置。")
                self.if_order_input.setText("")
                lp_order_val = None
        filtered_roi_audio = apply_simplified_cp_inverse_filtering(
            roi_audio, self.fs, gci_times_relative_to_roi_start, lp_order=lp_order_val, closed_phase_duration_ms=3.0)
        if filtered_roi_audio is None:
            QMessageBox.warning(self, "Filtering Failed", "Inverse filtering could not be performed. See console for details.")
            return

        print("Saving inverse filtering audio segments...")
        try:
            # 1. 获取原始文件名和路径
            # self.current_filepath 是加载文件时保存的完整路径
            output_dir = os.path.dirname(self.current_filepath)
            base_name_full = os.path.basename(self.current_filepath)
            base_name = os.path.splitext(base_name_full)[0]
            
            # 2. 格式化时间戳 (将 '.' 替换为 '_', 避免文件名问题)
            start_str = f"{start_s:.2f}s".replace('.', '_')
            end_str = f"{end_s:.2f}s".replace('.', '_')

            # 3. 构建新的文件名
            output_filename_orig = f"{base_name}_{start_str}_{end_str}_ORIG.wav"
            output_filename_filt = f"{base_name}_{start_str}_{end_str}_IF.wav"
            
            output_path_orig = os.path.join(output_dir, output_filename_orig)
            output_path_filt = os.path.join(output_dir, output_filename_filt)

            # --- 4. (FEATURE 2 MODIFIED) 准备音频数据 (按原始幅度缩放) ---
            
            # 原始音频 (float, -1 to 1)
            audio_orig_float = roi_audio
            # 原始音频的峰值 (float)
            orig_max_abs_float = np.max(np.abs(audio_orig_float))
            if orig_max_abs_float < 1e-9: orig_max_abs_float = 1.0 # 处理静音

            # 逆滤波后的音频 (float, 任意幅度)
            audio_filt_float = filtered_roi_audio
            # 逆滤波后的峰值 (float)
            filt_max_abs_float = np.max(np.abs(audio_filt_float))
            if filt_max_abs_float < 1e-9: filt_max_abs_float = 1.0 # 处理静音

            # 计算缩放因子，使滤波后的峰值与原始峰值相同
            scale_factor = orig_max_abs_float / filt_max_abs_float
            audio_filt_float_rescaled = audio_filt_float * scale_factor

            # 将 *两个* 缩放后的 float 信号转换为 16-bit PCM
            # (确保原始音频也被正确转换)
            audio_orig_16bit = np.int16(np.clip(audio_orig_float * 32767.0, -32768, 32767))
            audio_filt_16bit = np.int16(np.clip(audio_filt_float_rescaled * 32767.0, -32768, 32767))
            
            # --- END (FEATURE 2 MODIFIED) ---

            # 5. 写入 WAV 文件
            wavfile.write(output_path_orig, self.fs, audio_orig_16bit)
            wavfile.write(output_path_filt, self.fs, audio_filt_16bit)

            print(f"Successfully saved files:\n{output_path_orig}\n{output_path_filt}")
            # 弹窗通知用户
            QMessageBox.information(self, "Files Saved", 
                                    f"已保存音频片段至:\n{output_dir}")

        except Exception as e:
            print(f"Error saving audio files: {e}")
            QMessageBox.warning(self, "Save Error", 
                                f"无法保存音频文件:\n{e}")

        try:
            plot_inverse_filtering_results(
                roi_audio, filtered_roi_audio, roi_egg, self.fs, start_s, end_s, MATPLOTLIB_STYLE_SETTINGS)
        except Exception as e:
             QMessageBox.critical(self, "Plotting Error", f"Failed to plot inverse filtering results:\n{e}")
             import traceback
             traceback.print_exc()

    # --- NEW: (FEATURE 1) Save Analysis Function ---
    def save_analysis(self):
        """
        (FEATURE 1) Saves CSV data (CQ, SQ, F0) and PNG plots (Spec, CQ, Waveforms)
        for the current ROI with a white background.
        """
        if self.current_filepath is None or self.fs is None:
            QMessageBox.warning(self, "Error", "No data loaded to save.")
            return

        print("Starting analysis save process...")
        try:
            # 1. Get ROI and Naming
            start_s = float(self.start_time_input.text())
            duration_s = float(self.duration_input.text())
            end_s = start_s + duration_s
            
            output_dir = os.path.dirname(self.current_filepath)
            base_name_full = os.path.basename(self.current_filepath)
            base_name = os.path.splitext(base_name_full)[0]
            start_str = f"{start_s:.2f}s".replace('.', '_')
            end_str = f"{end_s:.2f}s".replace('.', '_')
            base_name_ts = f"{base_name}_{start_str}_{end_str}"

            # 2. Save CSV Data
            csv_path = os.path.join(output_dir, f"{base_name_ts}_DATA.csv")
            self._save_csv_data(start_s, end_s, csv_path)
            
            # 3. Save Plots
            plot_spec_path = os.path.join(output_dir, f"{base_name_ts}_SPEC_F0.png")
            plot_cq_path = os.path.join(output_dir, f"{base_name_ts}_CQ_SQ.png")
            plot_wave_path = os.path.join(output_dir, f"{base_name_ts}_WAVEFORMS.png")
            self._save_plots(start_s, end_s, plot_spec_path, plot_cq_path, plot_wave_path)

            print(f"Successfully saved analysis files to {output_dir}")
            QMessageBox.information(self, "Analysis Saved", 
                                    f"已保存分析数据和图像至:\n{output_dir}")
        except Exception as e:
            print(f"Error during analysis saving: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Save Error", f"保存分析时出错:\n{e}")

    def _save_csv_data(self, start_s, end_s, csv_path):
        """Helper to save CQ, SQ, F0, and Glottal Movement data for the ROI to a CSV."""
        print(f"Saving CSV data to {csv_path}...")
        
        # Create DataFrames for each data type
        cq_sq_df = pd.DataFrame()
        if self.cq_times is not None:
            cq_sq_df = pd.DataFrame({
                'Time_CQ_SQ': self.cq_times,
                'CQ': self.cq_values,
                'SQ': self.sq_values
            }).set_index('Time_CQ_SQ')

        f0_praat_df = pd.DataFrame()
        if self.audio_f0_times is not None:
            f0_praat_df = pd.DataFrame({
                'Time_F0_Praat': self.audio_f0_times,
                'F0_Praat (Hz)': self.audio_f0_values
            }).set_index('Time_F0_Praat')

        f0_gci_df = pd.DataFrame()
        if self.gci_f0_times is not None:
            f0_gci_df = pd.DataFrame({
                'Time_F0_GCI': self.gci_f0_times,
                'F0_GCI (Hz)': self.gci_f0_values
            }).set_index('Time_F0_GCI')

        glottal_df = pd.DataFrame()
        if self.glottal_movement_times:
            # Separate times and types
            g_times = [x[0] for x in self.glottal_movement_times]
            g_types = [x[1] for x in self.glottal_movement_times]
            glottal_df = pd.DataFrame({
                'Time_Glottal': g_times,
                'Glottal_Movement': g_types
            }).set_index('Time_Glottal')

        # Combine all dataframes using an outer join
        # Note: join works on index.
        df = cq_sq_df.join(f0_praat_df, how='outer').join(f0_gci_df, how='outer').join(glottal_df, how='outer')
        
        # Filter to ROI
        df_roi = df[(df.index >= start_s) & (df.index <= end_s)]
        df_roi.sort_index(inplace=True)
        
        # Save to CSV
        df_roi.to_csv(csv_path, na_rep='NaN', index_label="Time (s)")

    def _save_plots(self, start_s, end_s, spec_path, cq_path, wave_path):
        """Helper to re-plot and save plots with a white background."""
        print("Generating and saving plots with white background...")

        # Define the light style settings
        LIGHT_STYLE = {
            'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
            'xtick.color': 'black', 'ytick.color': 'black',
            'grid.color': '#DDDDDD', 'grid.linestyle': ':',
            'figure.facecolor': 'white', 'axes.facecolor': 'white',
            'savefig.facecolor': 'white', 'text.color': 'black',
            'lines.color': 'black', 'patch.edgecolor': 'black'
        }
        
        # Use style.context to temporarily change styles
        with plt.style.context(LIGHT_STYLE):
            
            # --- 1. Save CQ/SQ Plot ---
            try:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1_sq = ax1.twinx()
                ax1.set_title(f"EGG CQ & SQ ({start_s:.2f}s - {end_s:.2f}s)")
                
                # Plot CQ
                if self.cq_times is not None:
                    valid_cq = ~np.isnan(self.cq_values)
                    roi_mask_cq = (self.cq_times >= start_s) & (self.cq_times <= end_s) & valid_cq
                    if np.any(roi_mask_cq):
                        ax1.plot(self.cq_times[roi_mask_cq], self.cq_values[roi_mask_cq],
                                 label='CQ', color='blue', marker='.', linestyle='', markersize=5)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Contact Quotient (CQ)", color='blue')
                ax1.set_ylim(0, 1)
                ax1.tick_params(axis='y', colors='blue', labelcolor='blue')
                
                # Plot SQ
                if self.cq_times is not None and self.sq_values is not None:
                    valid_sq = ~np.isnan(self.sq_values)
                    roi_mask_sq = (self.cq_times >= start_s) & (self.cq_times <= end_s) & valid_sq
                    if np.any(roi_mask_sq):
                        ax1_sq.plot(self.cq_times[roi_mask_sq], self.sq_values[roi_mask_sq],
                                    label='SQ', color='green', marker='x', linestyle='', markersize=5)
                ax1_sq.set_ylabel("Speed Quotient (SQ)", color='green')
                ax1_sq.set_ylim(-1.1, 1.1)
                ax1_sq.tick_params(axis='y', colors='green', labelcolor='green')
                
                ax1.set_xlim(start_s, end_s)
                ax1.grid(True)
                fig1.tight_layout()
                fig1.savefig(cq_path, dpi=150)
                plt.close(fig1)
                print(f"Saved CQ/SQ plot: {cq_path}")
            except Exception as e:
                print(f"Warning: Failed to save CQ/SQ plot: {e}")

            # --- 2. Save Spectrogram Plot ---
            try:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                # --- MODIFICATION 1: 使用 Constrained Layout ---
                # 这比 tight_layout() 能更好地处理 twinx 和 colorbar
                try:
                    fig2.set_layout_engine('constrained')
                except Exception:
                    print("Note: Constrained layout failed, falling back to manual adjustment.")
                    fig2.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9) # 留出更多右侧空间

                ax2_f0 = ax2.twinx()
                ax2.set_title(f"Spectrogram ({start_s:.2f}s - {end_s:.2f}s)")
                
                # Get ROI Audio
                start_idx = max(0, int(start_s * self.fs))
                end_idx = min(len(self.audio_signal), int(end_s * self.fs))
                roi_audio = self.audio_signal[start_idx:end_idx]

                # Plot Spectrogram (use gray_r cmap for black-on-white)
                window_s = self.current_spec_window_ms / 1000.0
                nfft = int(self.fs * window_s)
                noverlap = int(nfft * 0.75)
                nfft = min(nfft, len(roi_audio))
                if noverlap >= nfft: noverlap = max(0, nfft - 1)
                
                im = None # 初始化 im
                if nfft > 0 and len(roi_audio) > nfft:
                    Pxx, freqs, bins, im = ax2.specgram(
                        roi_audio, NFFT=nfft, Fs=self.fs, noverlap=noverlap,
                        cmap='gray_r', scale='dB', mode='magnitude', 
                        vmin=self.current_spec_vmin, vmax=self.current_spec_vmax)
                    im.set_extent([start_s, end_s, freqs[0], freqs[-1]])
                
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Frequency (Hz)")
                ax2.set_ylim(0, 5000)
                
                # Plot F0 (whichever is active or preferred)
                f0_times, f0_values = (self.gci_f0_times, self.gci_f0_values) if self.f0_corrected else (self.audio_f0_times, self.audio_f0_values)
                
                # --- MODIFICATION 2: 复制主界面的自适应 Y 轴逻辑 ---
                valid_f0_values_roi = []
                if f0_times is not None:
                    roi_mask_f0 = (f0_times >= start_s) & (f0_times <= end_s)
                    if np.any(roi_mask_f0):
                        f0_values_roi = f0_values[roi_mask_f0]
                        ax2_f0.plot(f0_times[roi_mask_f0], f0_values_roi,
                                    color='black', lw=1.5)
                        # Get valid values for ylim calculation
                        valid_f0_values_roi = f0_values_roi[~np.isnan(f0_values_roi)]

                ax2_f0.set_ylabel("F0 (Hz)")
                
                # Apply adaptive Y-lim
                if len(valid_f0_values_roi) > 0:
                    f0_min = np.min(valid_f0_values_roi)
                    f0_max = np.max(valid_f0_values_roi)
                    padding = max(10.0, (f0_max - f0_min) * 0.1)
                    if (f0_max - f0_min) < 1e-6:
                         ax2_f0.set_ylim(f0_min - 20, f0_max + 20)
                    else:
                         ax2_f0.set_ylim(f0_min - padding, f0_max + padding)
                else:
                    ax2_f0.set_ylim(50, 500) # Fallback
                # --- END MODIFICATION 2 ---
                
                ax2.set_xlim(start_s, end_s)
                
                # --- MODIFICATION 3: 将 Colorbar 附加到 figure ---
                # 这可以帮助 constrained_layout 更好地管理空间
                if im is not None:
                    fig2.colorbar(im, ax=ax2, label='Magnitude (dB)')
                
                # --- MODIFICATION 4: 移除 tight_layout() ---
                # fig2.tight_layout(rect=[0, 0.03, 1, 0.95]) # REMOVED
                
                fig2.savefig(spec_path, dpi=150) # constrained_layout 会自动生效
                plt.close(fig2)
                print(f"Saved Spectrogram plot: {spec_path}")
            except Exception as e:
                print(f"Warning: Failed to save Spectrogram plot: {e}")
                import traceback
                traceback.print_exc()

            # --- 3. Save Waveforms Plot ---
            try:
                fig3, (ax_audio, ax_egg) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                fig3.suptitle(f"Waveforms ({start_s:.2f}s - {end_s:.2f}s)")
                
                # Get ROI data
                start_idx = max(0, int(start_s * self.fs))
                end_idx = min(len(self.egg_signal_processed), int(end_s * self.fs))
                roi_egg = self.egg_signal_processed[start_idx:end_idx]
                roi_audio = self.audio_signal[start_idx:end_idx]
                roi_time = self.time_vector[start_idx:end_idx]
                
                # Plot Audio
                ax_audio.plot(roi_time, roi_audio, color='black', lw=0.5)
                ax_audio.set_ylabel("Audio Amplitude")
                ax_audio.grid(True)
                
                # Plot EGG
                ax_egg.plot(roi_time, roi_egg, color='black', lw=0.5)
                ax_egg.set_ylabel("EGG Amplitude")
                ax_egg.set_xlabel("Time (s)")
                ax_egg.grid(True)
                
                ax_egg.set_xlim(start_s, end_s)
                fig3.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
                fig3.savefig(wave_path, dpi=150)
                plt.close(fig3)
                print(f"Saved Waveforms plot: {wave_path}")
            except Exception as e:
                print(f"Warning: Failed to save Waveforms plot: {e}")

    def _calculate_global_f0(self):
        """Calculates F0 (Praat) for the entire audio signal."""
        if not PARSELMOUTH_AVAILABLE or self.audio_signal is None or self.fs is None:
            self.audio_f0_times = None
            self.audio_f0_values = None
            return
            
        print("Calculating global F0 (Praat)...")
        try:
            # Create a parselmouth Sound object
            audio_parsel = parselmouth.Sound(self.audio_signal, self.fs)
            
            # Standard Praat pitch settings (75-600 Hz)
            pitch = praat_call(audio_parsel, "To Pitch...", 0.0, 75, 600)
            
            self.audio_f0_times = pitch.xs()
            self.audio_f0_values = pitch.selected_array['frequency']
            
            # Praat uses 0 for unvoiced, convert to NaN for plotting
            self.audio_f0_values[self.audio_f0_values == 0] = np.nan
            print(f"Found {len(self.audio_f0_times)} F0 (Praat) points.")
        except Exception as e:
            print(f"Error calculating Praat F0: {e}")
            self.audio_f0_times = None
            self.audio_f0_values = None

    # --- MODIFIED: (FEATURE 3) Triggers GCI-F0 calculation ---
    def _calculate_global_events(self):
        if self.egg_signal_processed is None or self.fs is None:
            self.all_gci_times, self.all_goi_times, self.all_peak_times = [], [], []
            self._calculate_gci_f0() # Clear GCI F0
            return
        print(f"Calculating global GCI/GOI (PeakProm={self.current_peak_prominence:.3f}, ValleyProm={self.current_valley_prominence:.3f})...")
        self.all_gci_times, self.all_goi_times, self.all_peak_times = find_gci_goi_peak_min_criterion(
             self.egg_signal_processed, self.fs,
             min_f0=50, max_f0=500, criterion_level=0.25,
             peak_prominence=self.current_peak_prominence,
             valley_prominence=self.current_valley_prominence,
             use_local_prominence=bool(self.auto_prom_checkbox.isChecked()),
            local_window_s=0.2, local_hop_s=0.1, min_auto_prom=DEFAULT_PEAK_PROMINENCE,
            gci_method=self.gci_method, goi_method=self.goi_method)
        if self.all_gci_times is not None and self.all_goi_times is not None:
             print(f"Found {len(self.all_gci_times)} global GCIs and {len(self.all_goi_times)} global GOIs.")
        else:
             print("Warning: Failed to calculate global GCI/GOI events.")
             self.all_gci_times, self.all_goi_times, self.all_goi_times = [], [], []
        
        # --- NEW: (FEATURE 3) Calculate GCI-F0 after GCI events are found ---
        self._calculate_gci_f0()
        self._build_gci_goi_associations()

    # --- NEW: (FEATURE 3) Calculate F0 from GCI intervals ---
    def _calculate_gci_f0(self):
        """Calculates F0 based on GCI time intervals."""
        if self.all_gci_times is None or len(self.all_gci_times) < 2:
            self.gci_f0_times = None
            self.gci_f0_values = None
            if PARSELMOUTH_AVAILABLE:
                self.correct_f0_checkbox.setEnabled(False)
            return

        print("Calculating GCI-derived F0...")
        if PARSELMOUTH_AVAILABLE:
            self.correct_f0_checkbox.setEnabled(True)
            
        try:
            gci_np = np.array(self.all_gci_times)
            periods = np.diff(gci_np)
            f0_values = 1.0 / periods
            f0_times = gci_np[:-1] + periods / 2.0
            vals = np.array(f0_values, dtype=float)
            times = np.array(f0_times, dtype=float)
            if len(vals) > 2:
                med = np.median(vals)
                mad = np.median(np.abs(vals - med))
                thr = 3.0 * mad if mad > 0 else max(1e-12, 3.0 * np.std(vals))
                base = vals.copy()
                base[np.abs(base - med) >= thr] = np.nan
                # Force-keep rule: if corrected F0 < 100 Hz, always keep the point
                force_keep = vals < 100.0
                base[force_keep] = vals[force_keep]
                nan_arr = np.isnan(base)
                left_nan = np.concatenate(([False], nan_arr[:-1]))
                right_nan = np.concatenate((nan_arr[1:], [False]))
                keep = (force_keep) | ((~nan_arr) & (~(left_nan & right_nan)))
                self.gci_f0_times = times[keep]
                self.gci_f0_values = base[keep]
            else:
                self.gci_f0_times = times
                self.gci_f0_values = vals
            print(f"Calculated {len(self.gci_f0_times)} GCI-F0 points")

        except Exception as e:
            print(f"Error calculating GCI F0: {e}")
            self.gci_f0_times = None
            self.gci_f0_values = None
            if PARSELMOUTH_AVAILABLE:
                self.correct_f0_checkbox.setEnabled(False)

    # --- MODIFIED: (FEATURE 4) Reset logic ---
    def _reset_app_state(self):
        sd.stop()
        self.fs = None; self.egg_signal_processed = None; self.egg_signal_raw = None
        self.audio_signal = None; self.time_vector = None; self.file_duration = 0.0
        self.cq_times = None; self.cq_values = None
        self.sq_times = None; self.sq_values = None
        self.all_gci_times = None; self.all_goi_times = None; self.all_goi_times = None
        self.timeline_roi_patch = None
        self.last_clicked_time = None
        self.current_filepath = None
        self.spec_colorbar = None # Reset colorbar reference
        # --- NEW: Reset new state variables ---
        self.channels_flipped = False
        self.show_f0 = False
        self.f0_corrected = False
        self.audio_f0_times = None; self.audio_f0_values = None
        self.gci_f0_times = None; self.gci_f0_values = None
        self.spec_ax_f0 = None
        self.peak_prom_is_auto = True
        if self.f0_contour_line:
            try: self.f0_contour_line.remove()
            except Exception: pass
            self.f0_contour_line = None
        self.current_spec_vmin = DEFAULT_SPEC_VMIN
        self.current_spec_vmax = DEFAULT_SPEC_VMAX
        # --- END NEW ---
        
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.inverse_filter_button.setEnabled(False)
        self.save_analysis_button.setEnabled(False) # --- NEW: Disable save button
        
        # --- Update UI ---
        self.prominence_input.setText(f"{DEFAULT_PEAK_PROMINENCE:.3f}")
        self.valley_prominence_input.setText(f"{DEFAULT_VALLEY_PROMINENCE:.3f}")
        self.start_time_input.setText(f"{DEFAULT_ROI_START:.2f}")
        self.duration_input.setText(f"{DEFAULT_ROI_DURATION:.2f}")
        self.zoom_duration_input.setText(str(int(DEFAULT_ZOOM_WINDOW_MS)))
        self.spec_window_input.setText(f"{DEFAULT_SPEC_WINDOW_MS:.1f}")
        # --- NEW: Reset new UI elements ---
        if hasattr(self, 'swap_action'):
            self.swap_action.setChecked(False)
        self.show_f0_checkbox.setChecked(False)
        self.correct_f0_checkbox.setChecked(False)
        if PARSELMOUTH_AVAILABLE:
            self.correct_f0_checkbox.setEnabled(True)
        self.spec_vmin_input.setText(f"{DEFAULT_SPEC_VMIN:.1f}")
        self.spec_vmax_input.setText(f"{DEFAULT_SPEC_VMAX:.1f}")
        # --- END NEW ---
        
        # --- Clear plots ---
        self._setup_initial_plots()
        self.timeline_ax.cla()
        self.timeline_ax.set_xlim(0, 10)
        self.timeline_ax.set_title("Timeline Overview (Click to set Start Time)", color='lightgray')
        self.timeline_ax.set_xlabel("Time (s)", color='lightgray')
        self.timeline_ax.set_yticks([])
        self.timeline_ax.tick_params(axis='x', colors='lightgray')
        self.timeline_canvas.draw()

    # plot_timeline remains the same
    def plot_timeline(self):
        if self.egg_signal_processed is None or self.time_vector is None: return
        self.timeline_ax.cla()
        self.timeline_roi_patch = None
        time_vec = np.array(self.time_vector)
        egg_sig = np.array(self.egg_signal_processed)
        self.timeline_ax.plot(time_vec, egg_sig, lw=0.7, color='darkgrey')
        self.timeline_ax.set_title("Timeline Overview (Click to set Start Time)", color='lightgray')
        self.timeline_ax.set_xlabel("Time (s)", color='lightgray')
        self.timeline_ax.set_yticks([])
        self.update_timeline_view_window()
        self.timeline_ax.tick_params(axis='x', colors='lightgray')
        self.update_timeline_roi_visual()
        try: self.timeline_canvas.fig.set_layout_engine('constrained')
        except Exception:
            try: self.timeline_canvas.fig.tight_layout()
            except Exception: pass
        self.timeline_canvas.draw()

    # update_timeline_roi_visual remains the same
    def update_timeline_roi_visual(self):
        if self.timeline_ax is None: return
        if self.timeline_roi_patch:
            try: self.timeline_roi_patch.remove()
            except ValueError: pass
            self.timeline_roi_patch = None
        if self.file_duration > 0 and self.current_roi_duration > 0:
            roi_start = self.current_roi_start
            roi_end = min(roi_start + self.current_roi_duration, self.file_duration)
            ylim = self.timeline_ax.get_ylim()
            if ylim[1] > ylim[0]:
                self.timeline_roi_patch = Rectangle((roi_start, ylim[0]), roi_end - roi_start, ylim[1] - ylim[0],
                                                    color='teal', alpha=0.3, zorder=-1, transform=self.timeline_ax.transData)
                self.timeline_ax.add_patch(self.timeline_roi_patch)

    def update_timeline_view_window(self):
        if self.file_duration is None:
            return
        start = float(self.timeline_offset_s)
        max_start = max(0.0, float(self.file_duration) - float(self.timeline_window_s))
        if start > max_start:
            start = max_start
        end = min(float(self.file_duration), start + float(self.timeline_window_s))
        if end <= start:
            end = min(float(self.file_duration), start + 1.0)
        self.timeline_ax.set_xlim(start, end)

    def on_timeline_slider_change(self, value):
        try:
            self.timeline_offset_s = float(value)
            self.update_timeline_view_window()
            self.update_timeline_roi_visual()
            self.timeline_canvas.draw_idle()
        except Exception as e:
            print(f"Error updating timeline slider: {e}")

    # --- MODIFIED: (FEATURE 2, 3, 4) Main plotting logic ---
    def update_roi_plots(self):
        if self.egg_signal_processed is None or self.fs is None: return
        if self.all_gci_times is None or self.all_goi_times is None:
            print("Global GCI/GOI times not available. Cannot update ROI plots.")
            self.cq_ax.cla(); self.cq_ax_sq.cla(); self.spec_ax.cla()
            self._setup_initial_plots()
            self.cq_canvas.draw(); self.spec_canvas.draw()
            return

        try:
            start_s = float(self.start_time_input.text())
            duration_s = float(self.duration_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Start time and duration must be numbers."); return
        if duration_s <= 0:
             QMessageBox.warning(self, "Invalid Input", "Duration must be positive."); return

        recalc_needed = False
        if hasattr(self, 'auto_prom_checkbox') and self.auto_prom_checkbox.isChecked():
            recalc_needed = True
        
        # 3. Trigger recalculation if needed (e.g., first run, or auto-prom change)
        if recalc_needed:
            print("Auto-Prominence or first-run triggered GCI/GOI recalculation...")
            # Also read the *other* analysis params from the UI
            try:
                self.current_valley_prominence = float(self.valley_prominence_input.text())
            except ValueError:
                self.current_valley_prominence = DEFAULT_VALLEY_PROMINENCE
                self.valley_prominence_input.setText(f"{self.current_valley_prominence:.3f}")
            
            # Run the core analysis
            self._calculate_global_events()
        # --- END NEW BLOCK ---


        # --- NEW: (FEATURE 4) Read VMin/VMax from UI ---
        try:
            vm_in = float(self.spec_vmin_input.text())
            vx_in = float(self.spec_vmax_input.text())
            if vm_in >= vx_in:
                raise ValueError("VMin must be less than VMax")
            self.current_spec_vmin = vm_in
            self.current_spec_vmax = vx_in
            print(f"Using manual spec range: vmin={self.current_spec_vmin}, vmax={self.current_spec_vmax}")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Spec Range", f"Invalid VMin/VMax: {e}\nReverting to defaults.")
            self.current_spec_vmin = DEFAULT_SPEC_VMIN
            self.current_spec_vmax = DEFAULT_SPEC_VMAX
            self.spec_vmin_input.setText(f"{self.current_spec_vmin:.1f}")
            self.spec_vmax_input.setText(f"{self.current_spec_vmax:.1f}")
        # --- END NEW ---

        start_s = max(0.0, min(start_s, self.file_duration))
        end_s = min(start_s + duration_s, self.file_duration)
        actual_duration = end_s - start_s
        if actual_duration <= 0 and self.file_duration > 0:
             end_s = min(start_s + 0.01, self.file_duration)
             actual_duration = end_s - start_s
             if actual_duration <= 0:
                  start_s = max(0.0, end_s - 0.01)
                  actual_duration = end_s - start_s

        if abs(start_s - self.current_roi_start) > 1e-3 or abs(actual_duration - self.current_roi_duration) > 1e-3:
            self.start_time_input.setText(f"{start_s:.2f}")
            self.duration_input.setText(f"{actual_duration:.2f}")
            self.current_roi_start = start_s
            self.current_roi_duration = actual_duration

        self.update_timeline_roi_visual()
        try:
            # self.timeline_canvas.fig.tight_layout() # Use constrained layout if possible
            self.timeline_canvas.draw()
        except Exception: pass

        start_idx = int(start_s * self.fs); end_idx = int(end_s * self.fs)
        start_idx = max(0, start_idx)
        end_idx = min(len(self.egg_signal_processed), end_idx)

        if start_idx >= end_idx:
             print("Warning: ROI resulted in zero or negative samples after adjustment.")
             self.cq_ax.cla(); self.cq_ax_sq.cla(); self.spec_ax.cla()
             self._setup_initial_plots()
             self.cq_ax.set_xlim(start_s, end_s); self.spec_ax.set_xlim(start_s, end_s)
             self.cq_canvas.draw(); self.spec_canvas.draw(); return

        roi_audio = self.audio_signal[start_idx:end_idx]

        # --- 1. Calculate CQ and SQ ---
        print(f"Calculating CQ & SQ (per GCI) using global events...")
        cq_t, cq_v, sq_v = calculate_cq_sq(
            self.all_gci_times, self.all_goi_times, self.all_peak_times
        )

        if cq_t is not None and cq_v is not None and sq_v is not None:
            self.cq_times = np.array(cq_t)
            self.cq_values = np.array(cq_v)
            self.sq_values = np.array(sq_v)
            valid_cq_indices = ~np.isnan(self.cq_values)
            valid_sq_indices = ~np.isnan(self.sq_values)
            num_valid_cq = np.sum(valid_cq_indices)
            num_valid_sq = np.sum(valid_sq_indices)
            print(f"CQ/SQ calculation complete. Found {num_valid_cq} valid CQ points and {num_valid_sq} valid SQ points globally.")
        else:
            self.cq_times, self.cq_values, self.sq_values = None, None, None
            print("CQ/SQ calculation failed.")

        # --- 2. Plot CQ and SQ ---
        self.cq_ax.cla(); self.cq_ax_sq.cla()
        cq_plot_success = False; sq_plot_success = False
        cq_color = 'cyan'; sq_color = 'yellow'
        lines = []

        if self.cq_times is not None and self.cq_values is not None:
            valid_cq = ~np.isnan(self.cq_values)
            roi_indices_cq = (self.cq_times >= start_s) & (self.cq_times <= end_s) & valid_cq
            if np.any(roi_indices_cq):
                 line_cq, = self.cq_ax.plot(
                     self.cq_times[roi_indices_cq], self.cq_values[roi_indices_cq],
                     label='CQ', color=cq_color, marker='.', linestyle='', markersize=5)
                 lines.append(line_cq)
                 cq_plot_success = True

        if self.cq_times is not None and self.sq_values is not None:
            valid_sq = ~np.isnan(self.sq_values)
            roi_indices_sq = (self.cq_times >= start_s) & (self.cq_times <= end_s) & valid_sq
            if np.any(roi_indices_sq):
                 line_sq, = self.cq_ax_sq.plot(
                     self.cq_times[roi_indices_sq], self.sq_values[roi_indices_sq],
                     label='SQ', color=sq_color, marker='x', linestyle='', markersize=5)
                 lines.append(line_sq)
                 sq_plot_success = True

        # Setup primary CQ axis
        self.cq_ax.set_xlim(start_s, end_s)
        self.cq_ax.set_ylim(0, 1)
        self.cq_ax.set_title("EGG CQ & SQ (Peak-Min Method)", color='lightgray')
        self.cq_ax.set_xlabel("Time (s)", color='lightgray')
        # self.cq_ax.set_ylabel("", color=cq_color) # Label removed
        self.cq_ax.tick_params(axis='x', colors='lightgray')
        self.cq_ax.tick_params(axis='y', colors=cq_color, labelcolor=cq_color)
        self.cq_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        self.cq_ax.spines['left'].set_color(cq_color)

        # Setup secondary SQ axis
        self.cq_ax_sq.set_ylim(-1.1, 1.1)
        # self.cq_ax_sq.set_ylabel("", color=sq_color) # Label removed
        self.cq_ax_sq.tick_params(axis='y', colors=sq_color, labelcolor=sq_color)
        self.cq_ax_sq.spines['right'].set_color(sq_color)
        self.cq_ax_sq.grid(False)

        if not cq_plot_success and not sq_plot_success:
            self.cq_ax.text(0.5, 0.5, 'CQ/SQ data unavailable in ROI', ha='center', va='center', transform=self.cq_ax.transAxes, color='gray')

        if lines:
            labels = [l.get_label() for l in lines]
            self.cq_ax.legend(lines, labels, loc='upper right', fontsize='small',
                              facecolor='#444444', edgecolor='gray', labelcolor='lightgray')

        self.cq_vline = None
        try:
            self.cq_canvas.fig.set_layout_engine('constrained') # Use constrained layout
            self.cq_canvas.draw()
        except Exception as e:
            print(f"Warning: CQ/SQ plot layout failed: {e}")
            try:
                self.cq_canvas.fig.tight_layout()
                self.cq_canvas.draw()
            except Exception:
                self.cq_canvas.draw() # Draw anyway

        # --- 3. Plot Spectrogram with Colorbar ---
        self.spec_ax.cla()

        # --- MODIFIED: (Stacking Fix) 
        # F0 轴的 *重建* 必须在主轴 cla() *之后*
        # 并且 *必须* 在 _update_f0_contour() 之前完成
        
        # 重建 F0 坐标轴 (因为 cla() 会销毁它)
        self.f0_contour_line = None # 清除 F0 线的引用
        pass
        
        # 检查是否 *已经* 存在 (以防万一)，如果存在就用 cla()
        if hasattr(self, 'spec_ax_f0') and self.spec_ax_f0:
            try:
                self.spec_ax_f0.cla()
            except Exception: # 如果轴无效 (例如已被销毁)，则重建
                self.spec_ax_f0 = self.spec_ax.twinx()
        else: # 如果不存在，则创建
            self.spec_ax_f0 = self.spec_ax.twinx()
            
        # *每次* 都重新应用样式 (因为 cla() 或 twinx() 都需要)
        self.spec_ax_f0.tick_params(axis='y', colors='lightgray')
        self.spec_ax_f0.grid(False)
        self.spec_ax_f0.spines['right'].set_color('lightgray')
        self.spec_ax_f0.spines['left'].set_visible(False)
        self.spec_ax_f0.spines['top'].set_visible(False)
        self.spec_ax_f0.spines['bottom'].set_visible(False)
        # --- END MODIFIED (Stacking Fix) ---

        # --- MODIFIED: Remove previous colorbar if it exists ---
        if self.spec_colorbar is not None:
            try:
                self.spec_colorbar.remove()
            except Exception as e:
                 print(f"Note: Error removing previous colorbar: {e}")
            self.spec_colorbar = None
        # --- END MODIFICATION ---
        plot_success = False

        if len(roi_audio) > 1:
            window_s = self.current_spec_window_ms / 1000.0
            nfft = int(self.fs * window_s)
            noverlap = int(nfft * 0.75)
            nfft = min(nfft, len(roi_audio))
            if noverlap >= nfft: noverlap = max(0, nfft - 1)

            if nfft > 0:
                print(f"Debug - Attempting specgram: roi_audio len={len(roi_audio)}, nfft={nfft}, noverlap={noverlap}, fs={self.fs}")
                try:
                    # --- MODIFIED: (FEATURE 4) Use manual vmin/vmax ---
                    # 移除了原有的 vmin/vmax 自动计算逻辑
                    vmin_to_use = self.current_spec_vmin
                    vmax_to_use = self.current_spec_vmax
                    # --- END MODIFICATION ---
                    
                    Pxx, freqs, bins, im = self.spec_ax.specgram(
                        np.array(roi_audio).flatten(), NFFT=nfft, Fs=self.fs, noverlap=noverlap,
                        cmap='jet', scale='dB', mode='magnitude', 
                        vmin=vmin_to_use, vmax=vmax_to_use) # Use manual values

                    if freqs is not None and len(freqs) > 0:
                        im.set_extent([start_s, end_s, freqs[0], freqs[-1]])
                    else:
                         print("Warning: specgram did not return valid frequencies.")

                    self.spec_ax.set_ylim(0, 5000)
                    self.spec_ax.set_aspect('auto')
                    plot_success = True

                    # --- MODIFIED: Add a narrow colorbar ---
                    try:
                        self.spec_colorbar = self.spec_canvas.fig.colorbar(im, ax=self.spec_ax, fraction=0.02, pad=0.04) # Simple version
                        self.spec_colorbar.ax.tick_params(colors='lightgray') # Style colorbar ticks
                    except Exception as cbar_e:
                        print(f"Warning: Failed to add colorbar: {cbar_e}")
                        self.spec_colorbar = None
                    # --- END MODIFICATION ---

                except Exception as spec_e:
                    import traceback
                    print("\n--- Spectrogram Error Traceback ---")
                    traceback.print_exc()
                    print("--- End Traceback ---")
                    print(f"Original error message caught: {spec_e}\n")
                    self.spec_ax.text(0.5, 0.5, 'Spectrogram Error\n(See console for details)',
                                      ha='center', va='center', transform=self.spec_ax.transAxes,
                                      color='red', fontsize=9)
            else:
                print(f"Error: NFFT calculated as {nfft} (must be > 0). Cannot compute spectrogram.")
                self.spec_ax.text(0.5, 0.5, f'Invalid NFFT ({nfft})\nWindow too small or ROI too short?',
                                  ha='center', va='center', transform=self.spec_ax.transAxes,
                                  color='orange', fontsize=9)
        else:
            self.spec_ax.text(0.5, 0.5, 'Not enough audio data\n(Need > 1 sample)',
                              ha='center', va='center', transform=self.spec_ax.transAxes,
                              color='gray', fontsize=9)

        self.spec_ax.set_title("Spectrogram", color='lightgray')
        self.spec_ax.set_xlabel("Time (s)", color='lightgray')
        self.spec_ax.set_ylabel("Frequency (Hz)", color='lightgray')
        self.spec_ax.tick_params(axis='both', colors='lightgray')
        # self.spec_ax.grid(True, linestyle=':', alpha=0.4, color='gray') # Grid removed for cleaner look with colorbar
        self.spec_ax.set_xlim(start_s, end_s)
        pass

        try:
            self.spec_canvas.fig.set_layout_engine('constrained') # Use constrained layout
            self.spec_canvas.draw()
        except Exception as e:
            print(f"Warning: Spectrogram plot layout failed: {e}")
            try:
                # May need manual adjustment if constrained fails with colorbar
                self.spec_canvas.fig.tight_layout(rect=[0, 0, 0.97, 1]) # Leave space for colorbar
                self.spec_canvas.draw()
            except Exception:
                self.spec_canvas.draw() # Draw anyway

        # --- NEW: (FEATURE 2/3) Update F0 contour after spec is drawn ---
        self._update_f0_contour()
        
        # --- NEW: Draw Glottal Movement Lines ---
        self._draw_glottal_movement_lines()
        
        try:
            if self.last_clicked_time is not None:
                t = float(self.last_clicked_time)
                xmin_spec, xmax_spec = self.spec_ax.get_xlim()
                if xmin_spec <= t <= xmax_spec:
                    self.spec_vline = self.spec_ax.axvline(t, color='red', lw=1.5, linestyle='-', zorder=10)
                xmin_cq, xmax_cq = self.cq_ax.get_xlim()
                if xmin_cq <= t <= xmax_cq:
                    self.cq_vline = self.cq_ax.axvline(t, color='red', lw=1.5, linestyle='-', zorder=10)
                self.spec_canvas.draw_idle()
                self.cq_canvas.draw_idle()
        except Exception as e:
            print(f"Error restoring vlines: {e}")
        # --- END NEW ---

        print(f"Updated ROI plots for {start_s:.2f}s to {end_s:.2f}s")

    # --- Click Handlers ---
    def on_left_plot_click(self, event):
        # This handler correctly identifies the source axes (cq_ax or spec_ax)
        if event.inaxes in [self.cq_ax, self.spec_ax, self.spec_ax_f0] and event.xdata is not None and event.button == 1:
            ax = event.inaxes
            xmin, xmax = ax.get_xlim()
            # Check if click is within the data range of the specific axis clicked
            if xmin <= event.xdata <= xmax:
                clicked_time_s = event.xdata
                print(f"Clicked '{ax.get_title()}' at time: {clicked_time_s:.4f}s")
                # Emit signal with the clicked time
                self.plotClicked.emit(clicked_time_s)
            else:
                print(f"Click ({event.xdata:.4f}s) outside current plot X-axis limits ({xmin:.2f}-{xmax:.2f}). Ignoring.")

    def on_timeline_click(self, event):
        # (No changes needed here)
        if event.inaxes == self.timeline_ax and event.xdata is not None and event.button == 1 and self.file_duration > 0:
            clicked_time_s = event.xdata
            clicked_time_s = max(0.0, min(clicked_time_s, self.file_duration))
            print(f"Timeline clicked at: {clicked_time_s:.4f}s. Setting as Start Time.")
            self.start_time_input.setText(f"{clicked_time_s:.2f}")
            sd.stop()
            try:
                duration_s = float(self.duration_input.text())
                if duration_s <= 0: duration_s = self.current_roi_duration
            except ValueError:
                duration_s = self.current_roi_duration
            self.current_roi_start = clicked_time_s
            self.current_roi_duration = duration_s
            end_s = min(self.current_roi_start + self.current_roi_duration, self.file_duration)
            self.current_roi_duration = end_s - self.current_roi_start
            self.duration_input.setText(f"{self.current_roi_duration:.2f}")
            self.update_roi_plots()

    def handle_plot_click(self, clicked_time_s):
        # --- MODIFIED: Draw vline based on which plot was clicked ---
        if self.egg_signal_processed is None or self.fs is None: return
        if not (0 <= clicked_time_s <= self.file_duration): return

        self.last_clicked_time = clicked_time_s
        line_color = 'red'
        redraw_cq = False
        redraw_spec = False

        cq_xmin, cq_xmax = self.cq_ax.get_xlim()
        spec_xmin, spec_xmax = self.spec_ax.get_xlim()

        # Draw/Update line on CQ/SQ plot if click is within its range
        if cq_xmin <= clicked_time_s <= cq_xmax:
            if self.cq_vline is None:
                self.cq_vline = self.cq_ax.axvline(clicked_time_s, color=line_color, lw=1.5, linestyle='-', zorder=10)
            else:
                try:
                    self.cq_vline.set_xdata([clicked_time_s, clicked_time_s])
                except Exception:
                    self.cq_vline = self.cq_ax.axvline(clicked_time_s, color=line_color, lw=1.5, linestyle='-', zorder=10)
            redraw_cq = True

        # Draw/Update line on Spectrogram plot if click is within its range
        if spec_xmin <= clicked_time_s <= spec_xmax:
            if self.spec_vline is None:
                self.spec_vline = self.spec_ax.axvline(clicked_time_s, color=line_color, lw=1.5, linestyle='-', zorder=10)
            else:
                try:
                    self.spec_vline.set_xdata([clicked_time_s, clicked_time_s])
                except Exception:
                    self.spec_vline = self.spec_ax.axvline(clicked_time_s, color=line_color, lw=1.5, linestyle='-', zorder=10)
            redraw_spec = True

        # Redraw canvases that had lines added/updated
        if redraw_cq:
            self.cq_canvas.draw_idle()
        if redraw_spec:
            self.spec_canvas.draw_idle()

        # Update zoom plots regardless
        self.update_zoom_plots(clicked_time_s)
        # --- END MODIFICATION ---

    # update_zoom_plots remains the same
    def update_zoom_plots(self, center_time_s):
        if self.egg_signal_processed is None or self.fs is None: return
        if self.all_gci_times is None or self.all_goi_times is None:
            self.clear_zoom_plots()
            return
        try:
            zoom_ms = float(self.zoom_duration_input.text())
            zoom_min_ms, zoom_max_ms = 10.0, 200.0
            if not (zoom_min_ms <= zoom_ms <= zoom_max_ms):
                zoom_ms = max(zoom_min_ms, min(zoom_max_ms, zoom_ms))
                print(f"Warning: Zoom duration out of range ({zoom_min_ms}-{zoom_max_ms}). Clamped to {zoom_ms:.0f}ms.")
                self.zoom_duration_input.setText(str(int(zoom_ms)))
            if abs(zoom_ms - self.zoom_window_ms) > 1e-3:
                 self.zoom_window_ms = zoom_ms
        except ValueError:
            print(f"Warning: Invalid zoom duration input. Using {self.zoom_window_ms}ms.")
            zoom_ms = self.zoom_window_ms
            self.zoom_duration_input.setText(str(int(zoom_ms)))

        zoom_half_window_s = (zoom_ms / 1000.0) / 2.0
        start_s = center_time_s - zoom_half_window_s
        end_s = center_time_s + zoom_half_window_s
        start_idx = max(0, int(start_s * self.fs))
        end_idx = min(len(self.egg_signal_processed), int(end_s * self.fs))

        if start_idx >= end_idx:
            self.clear_zoom_plots()
            print("Warning: Zoom window resulted in zero samples.")
            return

        source_egg = self.egg_signal_processed if self.show_filtered_egg else self.egg_signal_raw
        zoom_egg = np.array(source_egg)[start_idx:end_idx]
        zoom_audio = np.array(self.audio_signal)[start_idx:end_idx]
        zoom_time_ms = (np.arange(start_idx, end_idx) / self.fs - center_time_s) * 1000.0

        self.audio_zoom_ax.cla()
        self.egg_zoom_ax.cla()
        self.glottal_event_lines.clear()

        if len(zoom_audio) > 0:
            self.audio_zoom_ax.plot(zoom_time_ms, zoom_audio, color='wheat', linewidth=1.0)
        self.audio_zoom_ax.set_title(f"Audio (+/- {zoom_ms/2:.0f}ms)", color='lightgray')
        self.audio_zoom_ax.set_xlabel("Time relative to click (ms)", color='lightgray')
        self.audio_zoom_ax.tick_params(axis='x', colors='lightgray')
        self.audio_zoom_ax.tick_params(axis='y', colors='lightgray')
        self.audio_zoom_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        if len(zoom_time_ms) > 0: self.audio_zoom_ax.set_xlim(zoom_time_ms[0], zoom_time_ms[-1])
        else: self.audio_zoom_ax.set_xlim(-zoom_ms/2, zoom_ms/2)

        if len(zoom_egg) > 0:
            self.egg_zoom_ax.plot(zoom_time_ms, zoom_egg, color='wheat', linewidth=1.0)
        mode_text = "滤波" if self.show_filtered_egg else "原始"
        self.egg_zoom_ax.set_title(f"EGG（+/- {zoom_ms/2:.0f}ms)", color='lightgray')
        self.egg_zoom_ax.set_xlabel("Time relative to click (ms)", color='lightgray')
        self.egg_zoom_ax.tick_params(axis='x', colors='lightgray')
        self.egg_zoom_ax.tick_params(axis='y', colors='lightgray')
        self.egg_zoom_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        if len(zoom_time_ms) > 0: self.egg_zoom_ax.set_xlim(zoom_time_ms[0], zoom_time_ms[-1])
        else: self.egg_zoom_ax.set_xlim(-zoom_ms/2, zoom_ms/2)

        event_color = 'lime'; gci_linestyle = '-'; goi_linestyle = '--'
        gci_count = 0; goi_count = 0
        all_gci = np.array(self.all_gci_times)
        all_goi = np.array(self.all_goi_times)

        gci_in_zoom = all_gci[(all_gci >= start_s) & (all_gci <= end_s)]
        goi_in_zoom = all_goi[(all_goi >= start_s) & (all_goi <= end_s)]

        for t_gci_abs in gci_in_zoom:
            t_gci_ms_rel_click = (t_gci_abs - center_time_s) * 1000.0
            line_egg = self.egg_zoom_ax.axvline(t_gci_ms_rel_click, color=event_color, lw=1.5, linestyle=gci_linestyle, zorder=10, label='GCI' if gci_count == 0 else None)
            line_audio = self.audio_zoom_ax.axvline(t_gci_ms_rel_click, color=event_color, lw=1.5, linestyle=gci_linestyle, zorder=10)
            self.glottal_event_lines.extend([line_egg, line_audio])
            gci_count += 1
        for t_goi_abs in goi_in_zoom:
            t_goi_ms_rel_click = (t_goi_abs - center_time_s) * 1000.0
            line_egg = self.egg_zoom_ax.axvline(t_goi_ms_rel_click, color=event_color, lw=1.5, linestyle=goi_linestyle, zorder=10, label='GOI' if goi_count == 0 else None)
            line_audio = self.audio_zoom_ax.axvline(t_goi_ms_rel_click, color=event_color, lw=1.5, linestyle=goi_linestyle, zorder=10)
            self.glottal_event_lines.extend([line_egg, line_audio])
            goi_count += 1

        print(f"  Displayed {gci_count} GCI(s) and {goi_count} GOI(s) in zoom window.")

        if gci_count > 0 or goi_count > 0:
             handles, labels = self.egg_zoom_ax.get_legend_handles_labels()
             if handles:
                 self.egg_zoom_ax.legend(handles=handles, labels=labels, loc='upper right', fontsize='x-small',
                                         facecolor='#444444', edgecolor='gray', labelcolor='lightgray')

        try: self.audio_zoom_canvas.fig.set_layout_engine('constrained')
        except Exception:
            try: self.audio_zoom_canvas.fig.tight_layout()
            except Exception: pass
        self.audio_zoom_canvas.draw_idle()
        try: self.egg_zoom_canvas.fig.set_layout_engine('constrained')
        except Exception:
            try: self.egg_zoom_canvas.fig.tight_layout()
            except Exception: pass
        self.egg_zoom_canvas.draw_idle()

    # clear_zoom_plots remains the same
    def clear_zoom_plots(self):
        self.audio_zoom_ax.cla()
        self.egg_zoom_ax.cla()
        self.glottal_event_lines.clear()
        common_style = {'color': 'lightgray'}
        zoom_ms = self.zoom_window_ms
        zoom_half_ms = zoom_ms / 2.0
        self.audio_zoom_ax.set_title(f"Audio (+/- {zoom_half_ms:.0f}ms)", **common_style)
        self.audio_zoom_ax.set_xlabel("Time relative to click (ms)", **common_style)
        self.audio_zoom_ax.tick_params(axis='x', colors='lightgray')
        self.audio_zoom_ax.tick_params(axis='y', colors='lightgray')
        self.audio_zoom_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        self.audio_zoom_ax.set_xlim(-zoom_half_ms, zoom_half_ms)
        self.egg_zoom_ax.set_title(f"EGG (+/- {zoom_half_ms:.0f}ms)", **common_style)
        self.egg_zoom_ax.set_xlabel("Time relative to click (ms)", **common_style)
        self.egg_zoom_ax.tick_params(axis='x', colors='lightgray')
        self.egg_zoom_ax.tick_params(axis='y', colors='lightgray')
        self.egg_zoom_ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        self.egg_zoom_ax.set_xlim(-zoom_half_ms, zoom_half_ms)
        try: self.audio_zoom_canvas.fig.set_layout_engine('constrained')
        except Exception:
            try: self.audio_zoom_canvas.fig.tight_layout()
            except Exception: pass
        self.audio_zoom_canvas.draw()
        try: self.egg_zoom_canvas.fig.set_layout_engine('constrained')
        except Exception:
            try: self.egg_zoom_canvas.fig.tight_layout()
            except Exception: pass
        self.egg_zoom_canvas.draw()

    # --- Audio Playback Functions (Remain the same) ---
    def play_audio(self):
        if self.audio_signal is None or self.fs is None:
            QMessageBox.warning(self, "Playback Error", "No audio data loaded.")
            return
        start_s = self.current_roi_start
        end_s = start_s + self.current_roi_duration
        start_idx = max(0, int(start_s * self.fs))
        end_idx = min(len(self.audio_signal), int(end_s * self.fs))
        if start_idx >= end_idx:
            QMessageBox.warning(self, "Playback Error", "Invalid or zero-duration ROI for playback.")
            return
        audio_segment = np.array(self.audio_signal)[start_idx:end_idx]
        try:
            sd.stop()
            print(f"Playing audio from {start_s:.2f}s to {end_s:.2f}s")
            sd.play(audio_segment, self.fs)
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", f"Could not play audio:\n{e}")
            print(f"Error during audio playback: {e}")

    def stop_audio(self):
        try:
            sd.stop()
            print("Audio playback stopped.")
        except Exception as e:
            print(f"Note: Error stopping audio (may be expected if nothing was playing): {e}")

    # --- NEW: (FEATURE 2/3) F0 Toggling Handlers ---
    def toggle_f0_visibility(self, checked):
        """(FEATURE 2) Handles the 'Show F0' checkbox."""
        if not PARSELMOUTH_AVAILABLE: return
        self.show_f0 = checked
        print(f"Show F0 toggled: {self.show_f0}")
        self._update_f0_contour()

    def toggle_f0_correction(self, checked):
        """(FEATURE 3) Handles the 'Correct F0 (GCI)' checkbox."""
        if not PARSELMOUTH_AVAILABLE: return
        self.f0_corrected = checked
        print(f"F0 Correction toggled: {self.f0_corrected}")
        self._update_f0_contour()

    def toggle_gci_method(self):
        self.gci_method = "scale" if self.gci_method == "slope" else "slope"
        self.gci_method_button.setText("GCI：尺度" if self.gci_method == "scale" else "GCI：斜率")
        self._trigger_recalculation()

    def toggle_goi_method(self):
        self.goi_method = "scale" if self.goi_method == "slope" else "slope"
        self.goi_method_button.setText("GOI：尺度" if self.goi_method == "scale" else "GOI：斜率")
        self._trigger_recalculation()

    # --- NEW: Glottal Movement Detection Logic ---
    def toggle_glottal_detection(self, checked):
        """Toggles glottal movement detection."""
        self.show_glottal_movement = checked
        print(f"Glottal movement detection toggled: {self.show_glottal_movement}")
        
        if self.show_glottal_movement:
             # Always re-detect to ensure latest settings/audio
             self._detect_glottal_movement()
        
        # Redraw lines
        self.update_roi_plots()

    def _detect_glottal_movement(self):
        """
        Detects glottal movement based solely on Praat pitch slope.
        User requirements:
        - Two consecutive points (no NaN in between).
        - Slope > 50 Hz/0.05s (1000 Hz/s) -> Rise (Yellow)
        - Slope < -50 Hz/0.05s (-1000 Hz/s) -> Fall (Pink)
        - Arbitrary two 'Rise' distance >= 100ms.
        - Arbitrary two 'Fall' distance >= 100ms.
        """
        if self.audio_signal is None:
            return

        try:
            # Ensure F0 is calculated (Praat)
            if self.audio_f0_values is None:
                self._calculate_global_f0()
            
            if self.audio_f0_values is None or len(self.audio_f0_values) < 2:
                print("Praat F0 not available or too short for glottal detection.")
                self.glottal_movement_times = []
                return

            f0_vals = self.audio_f0_values
            f0_times = self.audio_f0_times
            
            candidates = []
            
            # Slope threshold (Hz/s)
            # 50 Hz per 0.05s = 1000 Hz/s
            SLOPE_THRESHOLD = 1000.0
            
            # Calculate differences
            dt = np.diff(f0_times)
            df0 = np.diff(f0_vals)
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                slopes = df0 / dt
            
            # Iterate through slopes to check validity (no NaN) and threshold
            # We use a loop to be explicit about "consecutive" logic (though diff implies consecutive in array)
            # We must ensure that the points used for slope calc are valid numbers.
            # np.diff(f0_vals) will be NaN if either point is NaN.
            
            # Filter where slope is not NaN
            valid_indices = np.where(~np.isnan(slopes))[0]
            
            for i in valid_indices:
                slope = slopes[i]
                t_start = f0_times[i]
                
                # Check user logic
                if slope > SLOPE_THRESHOLD:
                    candidates.append((t_start, "Rise"))
                elif slope < -SLOPE_THRESHOLD:
                    candidates.append((t_start, "Fall"))

            # Filter Candidates by Distance (100ms constraint per type)
            self.glottal_movement_times = []
            
            # Process Rise events
            rise_events = sorted([x for x in candidates if x[1] == "Rise"], key=lambda x: x[0])
            last_rise_time = -1.0
            for t, m_type in rise_events:
                if last_rise_time < 0 or (t - last_rise_time >= 0.1):
                    self.glottal_movement_times.append((t, m_type))
                    last_rise_time = t
            
            # Process Fall events
            fall_events = sorted([x for x in candidates if x[1] == "Fall"], key=lambda x: x[0])
            last_fall_time = -1.0
            for t, m_type in fall_events:
                if last_fall_time < 0 or (t - last_fall_time >= 0.1):
                    self.glottal_movement_times.append((t, m_type))
                    last_fall_time = t

            # Sort final list by time
            self.glottal_movement_times.sort(key=lambda x: x[0])

            print(f"Detected {len(self.glottal_movement_times)} glottal movement events (Slope > {SLOPE_THRESHOLD}).")
            
        except Exception as e:
            print(f"Error in glottal movement detection: {e}")
            import traceback
            traceback.print_exc()
            self.glottal_movement_times = []

    def _draw_glottal_movement_lines(self):
        """Draws vertical lines for glottal movement (Yellow=Rise, Pink=Fall)."""
        if not self.show_glottal_movement or not self.glottal_movement_times:
            return

        start_s, end_s = self.spec_ax.get_xlim()
        
        # Filter events in ROI
        events_in_roi = [e for e in self.glottal_movement_times if start_s <= e[0] <= end_s]
        
        print(f"Drawing {len(events_in_roi)} glottal lines in ROI {start_s:.2f}-{end_s:.2f}s")

        for t, m_type in events_in_roi:
            # Color based on type
            color = 'yellow' if m_type == "Rise" else '#FF69B4' # HotPink
            
            # Draw line
            self.spec_ax.axvline(t, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
            
            # Add label
            y_lim = self.spec_ax.get_ylim()
            y_pos = y_lim[1] * 0.9 if m_type == "Rise" else y_lim[1] * 0.1
            self.spec_ax.text(t, y_pos, m_type, color=color, fontsize=8, rotation=90, va='center', ha='right')
        
        self.spec_canvas.draw_idle()
    # --- END NEW ---

    # --- MODIFIED: (Stacking Fix) 彻底修复 F0 堆叠问题 ---
    def _update_f0_contour(self):
        """(FEATURE 2/3) Helper function to draw, remove, or update the F0 line."""
        
        # --- 1. (关键修改) 彻底清除 F0 轴 ---
        # 检查 F0 轴 (self.spec_ax_f0) 是否存在
        if not hasattr(self, 'spec_ax_f0') or self.spec_ax_f0 is None:
            # 这种情况 *不应该* 发生，因为 update_roi_plots 总是会创建它
            print("Warning: F0 axis (self.spec_ax_f0) not found. Recreating.")
            try:
                self.spec_ax_f0 = self.spec_ax.twinx()
            except Exception as e:
                print(f"Error recreating F0 axis: {e}")
                return
        
        # 你的代码中已经有了这一步，这是正确的，也是修复堆叠问题的关键
        self.spec_ax_f0.cla() 
        self.f0_contour_line = None # 确保旧的线条引用被清除

        # --- 2. (关键) 重建 F0 轴的样式 ---
        # 因为 cla() 清除了所有样式，我们必须在这里重建它们
        self.spec_ax_f0.tick_params(axis='y', colors='lightgray')
        self.spec_ax_f0.grid(False) 
        self.spec_ax_f0.spines['right'].set_color('lightgray') 
        self.spec_ax_f0.spines['left'].set_visible(False) 
        self.spec_ax_f0.spines['top'].set_visible(False)
        self.spec_ax_f0.spines['bottom'].set_visible(False)
        # 确保 F0 轴的 X 范围与主轴一致
        self.spec_ax_f0.set_xlim(self.spec_ax.get_xlim())

        # 3. Check if we should draw a new line
        if not self.show_f0 or self.spec_ax is None:
            self.spec_canvas.draw() # Redraw to ensure removal
            return

        # 4. Select data source based on correction flag
        if self.f0_corrected:
            times = self.gci_f0_times
            values = self.gci_f0_values
            color = 'black' # Per request, GCI-F0 is also black
        else:
            times = self.audio_f0_times
            values = self.audio_f0_values
            color = 'black' # Per request

        if times is None or values is None or len(times) == 0:
            self.spec_canvas.draw()
            return

        # 5. Filter data to current ROI
        start_s, end_s = self.spec_ax.get_xlim()
        roi_mask = (times >= start_s) & (times <= end_s)
        
        if not np.any(roi_mask):
            self.spec_canvas.draw()
            return
            
        times_roi = times[roi_mask]
        values_roi = values[roi_mask]

        # 6. Plot the line (in three layers for visibility) 
        # --- 第1层: 绘制细的连接线 (无标记) ---
        self.spec_ax_f0.plot(times_roi, values_roi, 
                             color=color, lw=0.75, # <-- 很细的线
                             marker='None', linestyle='-')

        # --- 第2层: 绘制外层的黑点 (大) ---
        self.spec_ax_f0.plot(times_roi, values_roi,
                             color=color, # 'black'
                             marker='.', markersize=7, # <-- 外层的黑点
                             linestyle='None')

        # --- 第3层: 绘制内层的白点 (小) ---
        # (我们把这个引用赋给 self.f0_contour_line，用于 _reset_app_state)
        line, = self.spec_ax_f0.plot(times_roi, values_roi,
                                     color='white',
                                     marker='.', markersize=2.5, # <-- 内层的白点
                                     linestyle='None')

        self.f0_contour_line = line

        # 8. (自适应 Y 轴范围)
        # 过滤掉 NaN 值以进行 Y 轴范围计算
        valid_values_roi = values_roi[~np.isnan(values_roi)]
        
        if len(valid_values_roi) > 0:
            f0_min = np.min(valid_values_roi)
            f0_max = np.max(valid_values_roi)
            # 增加一些 padding，避免曲线贴边
            padding = max(10.0, (f0_max - f0_min) * 0.1) # 至少 10Hz 或 10% 的 padding
            
            if (f0_max - f0_min) < 1e-6: # 如果 F0 是平的
                 self.spec_ax_f0.set_ylim(f0_min - 20, f0_max + 20)
            else:
                 self.spec_ax_f0.set_ylim(f0_min - padding, f0_max + padding)
        else:
            # 如果 ROI 内没有 F0 数据，回退到默认值
            self.spec_ax_f0.set_ylim(50, 500) 
        
        self.spec_canvas.draw()
    # --- END MODIFIED ---

    def on_spec_button_press(self, event):
        if event.inaxes in [self.spec_ax, self.spec_ax_f0] and event.button == 1 and event.xdata is not None:
            self._spec_dragging = True
            self._spec_last_x = float(event.xdata)

    def on_spec_mouse_move(self, event):
        if self._spec_dragging and event.inaxes in [self.spec_ax, self.spec_ax_f0] and event.xdata is not None:
            xmin, xmax = self.spec_ax.get_xlim()
            dx = float(event.xdata) - float(self._spec_last_x)
            new_xmin = xmin - dx
            new_xmax = xmax - dx
            dur = float(self.file_duration) if self.file_duration else 0.0
            if dur > 0:
                span = new_xmax - new_xmin
                new_xmin = max(0.0, new_xmin)
                new_xmax = min(dur, new_xmax)
                if new_xmax - new_xmin < span:
                    shift = span - (new_xmax - new_xmin)
                    if new_xmin <= 0.0:
                        new_xmax = min(dur, new_xmax + shift)
                    elif new_xmax >= dur:
                        new_xmin = max(0.0, new_xmin - shift)
            self.spec_ax.set_xlim(new_xmin, new_xmax)
            if self.spec_ax_f0 is not None:
                self.spec_ax_f0.set_xlim(new_xmin, new_xmax)
            self._spec_last_x = float(event.xdata)
            self.spec_canvas.draw_idle()

    def on_spec_button_release(self, event):
        if event.button == 1:
            self._spec_dragging = False
            self._spec_last_x = None
            try:
                if hasattr(self, 'auto_prom_checkbox') and self.auto_prom_checkbox.isChecked():
                    self._update_prominence_from_view()
                    self._trigger_recalculation()
            except Exception as e:
                print(f"Error updating prominence after drag: {e}")

    def on_spec_scroll(self, event):
        if event.inaxes in [self.spec_ax, self.spec_ax_f0] and event.xdata is not None:
            xmin, xmax = self.spec_ax.get_xlim()
            xcenter = float(event.xdata)
            span = xmax - xmin
            if span <= 0:
                return
            factor = 0.85 if event.step > 0 else 1.15
            new_span = max(0.02, span * factor)
            half = new_span / 2.0
            new_xmin = xcenter - half
            new_xmax = xcenter + half
            dur = float(self.file_duration) if self.file_duration else 0.0
            if dur > 0:
                if new_xmin < 0.0:
                    new_xmax += -new_xmin
                    new_xmin = 0.0
                if new_xmax > dur:
                    over = new_xmax - dur
                    new_xmin -= over
                    new_xmax = dur
                if new_xmin < 0.0:
                    new_xmin = 0.0
            self.spec_ax.set_xlim(new_xmin, new_xmax)
            if self.spec_ax_f0 is not None:
                self.spec_ax_f0.set_xlim(new_xmin, new_xmax)
            self.spec_canvas.draw_idle()
            try:
                if hasattr(self, 'auto_prom_checkbox') and self.auto_prom_checkbox.isChecked():
                    self._update_prominence_from_view()
                    self._trigger_recalculation()
            except Exception as e:
                print(f"Error updating prominence on scroll: {e}")

    def _update_prominence_from_view(self):
        xmin, xmax = self.spec_ax.get_xlim()
        if self.egg_signal_processed is None or self.fs is None:
            return
        start_idx = max(0, int(xmin * self.fs))
        end_idx = min(len(self.egg_signal_processed), int(xmax * self.fs))
        if end_idx - start_idx < 2:
            return
        roi = self.egg_signal_processed[start_idx:end_idx]
        peak_amp = float(np.max(np.abs(roi))) if len(roi) > 0 else 0.0
        prom = max(DEFAULT_PEAK_PROMINENCE, 0.6 * peak_amp)
        if abs(prom - self.current_peak_prominence) > 1e-6:
            self.current_peak_prominence = prom
            self.prominence_input.setText(f"{self.current_peak_prominence:.3f}")

    def _build_gci_goi_associations(self):
        self.gci_to_goi_map = {}
        self.goi_to_gci_map = {}
        try:
            gci = np.array(self.all_gci_times) if self.all_gci_times is not None else np.array([])
            goi = np.array(self.all_goi_times) if self.all_goi_times is not None else np.array([])
            if len(gci) < 2 or len(goi) == 0:
                return
            for k in range(len(gci) - 1):
                g0 = float(gci[k]); g1 = float(gci[k+1])
                idxs = np.where((goi > g0) & (goi < g1))[0]
                if len(idxs) > 0:
                    t_goi = float(goi[idxs[0]])
                    self.gci_to_goi_map[g0] = t_goi
                    self.goi_to_gci_map[t_goi] = g0
        except Exception as e:
            print(f"Error building GCI-GOI associations: {e}")

    def on_zoom_plot_click(self, event):
        try:
            if event.inaxes == self.egg_zoom_ax and event.xdata is not None and event.button == 3:
                if self.last_clicked_time is None:
                    return
                t_abs = float(self.last_clicked_time) + float(event.xdata) / 1000.0
                self._delete_nearest_event_pair(t_abs)
            elif event.inaxes == self.egg_zoom_ax and event.xdata is not None and event.button == 1:
                center_time = self._get_center_time_for_zoom()
                self.update_zoom_plots(center_time)
        except Exception as e:
            print(f"Error handling zoom right-click: {e}")

    def _get_center_time_for_zoom(self):
        if self.last_clicked_time is not None:
            return float(self.last_clicked_time)
        if self.file_duration and self.current_roi_duration:
            return float(self.current_roi_start + self.current_roi_duration / 2.0)
        return 0.0

    def toggle_egg_display_mode(self):
        self.show_filtered_egg = not self.show_filtered_egg
        self.egg_display_toggle_button.setText("显示：滤波波形" if self.show_filtered_egg else "显示：原始波形")
        try:
            center_time = self._get_center_time_for_zoom()
            self.update_zoom_plots(center_time)
        except Exception as e:
            print(f"Error toggling display mode: {e}")

    def handle_highpass_slider_change(self, value):
        prev_cutoff = int(self.highpass_cutoff_current)
        try:
            new_cutoff = int(value)
            if not (1 <= new_cutoff <= 50):
                raise ValueError("High-pass cutoff out of range")
            self.highpass_cutoff_current = new_cutoff
            self.highpass_label.setText(f"{self.highpass_cutoff_current}")
            if self.egg_detrended is None:
                return
            self.egg_highpassed = apply_highpass_filter(self.egg_detrended,
                                                        cutoff_freq=self.highpass_cutoff_current,
                                                        fs=self.fs)
            processed = apply_lowpass_filter(self.egg_highpassed,
                                             cutoff_freq=DEFAULT_LOWPASS_CUTOFF,
                                             fs=self.fs)
            if processed is None or len(processed) != len(self.egg_highpassed):
                raise ValueError("Invalid processed waveform")
            self.egg_signal_processed = processed
            self._trigger_recalculation()
        except Exception as e:
            print(f"Error applying high-pass cutoff: {e}")
            QMessageBox.warning(self, "滤波错误",
                                f"高通滤波失败，已保留上次有效频率（{prev_cutoff} Hz）。\n错误: {e}")
            self.highpass_cutoff_current = prev_cutoff
            self.highpass_slider.setValue(prev_cutoff)
            self.highpass_label.setText(f"{self.highpass_cutoff_current}")

    def _delete_nearest_event_pair(self, time_s):
        try:
            if self.all_gci_times is None or self.all_goi_times is None:
                return
            gci = np.array(self.all_gci_times)
            goi = np.array(self.all_goi_times)
            if len(gci) == 0 and len(goi) == 0:
                return
            nearest_type = None
            nearest_time = None
            if len(gci) > 0:
                idx_g = int(np.argmin(np.abs(gci - time_s)))
                nearest_time = float(gci[idx_g]); nearest_type = 'gci'
            if len(goi) > 0:
                idx_o = int(np.argmin(np.abs(goi - time_s)))
                if nearest_time is None or abs(goi[idx_o] - time_s) < abs(nearest_time - time_s):
                    nearest_time = float(goi[idx_o]); nearest_type = 'goi'
            if nearest_time is None:
                return
            if nearest_type == 'gci':
                pair = self.gci_to_goi_map.get(nearest_time, None)
                self.all_gci_times = [t for t in self.all_gci_times if t != nearest_time]
                if pair is not None:
                    self.all_goi_times = [t for t in self.all_goi_times if t != pair]
                print(f"Deleted GCI {nearest_time:.6f}s and paired GOI {pair if pair is not None else 'None'}")
            else:
                pair = self.goi_to_gci_map.get(nearest_time, None)
                self.all_goi_times = [t for t in self.all_goi_times if t != nearest_time]
                if pair is not None:
                    self.all_gci_times = [t for t in self.all_gci_times if t != pair]
                print(f"Deleted GOI {nearest_time:.6f}s and paired GCI {pair if pair is not None else 'None'}")
            self._build_gci_goi_associations()
            self._calculate_gci_f0()
            self.update_roi_plots()
            if self.last_clicked_time is not None:
                self.update_zoom_plots(self.last_clicked_time)
        except Exception as e:
            print(f"Error deleting event pair: {e}")

    # --- Parameter Change Handlers ---
    # handle_peak_prominence_change remains the same
    def handle_peak_prominence_change(self):
        if self.egg_signal_processed is None: return
        try:
            new_prominence = float(self.prominence_input.text())
            if new_prominence <= 0: raise ValueError("Prominence must be positive.")
            
            # --- NEW: User has taken manual control, uncheck the box ---
            if hasattr(self, 'auto_prom_checkbox'):
                self.auto_prom_checkbox.setChecked(False)
            print("Peak Prominence set manually. Auto-mode disabled.")
            # --- END NEW ---
            if abs(new_prominence - self.current_peak_prominence) > 1e-6:
                print(f"Peak Prominence changed to: {new_prominence:.3f}")
                self.current_peak_prominence = new_prominence
                self._trigger_recalculation()
            else:
                self.prominence_input.setText(f"{self.current_peak_prominence:.3f}")
        except ValueError as e:
            QMessageBox.warning(self, "输入无效", f"无效的峰值门限: {e}")
            self.prominence_input.setText(f"{self.current_peak_prominence:.3f}")

    # handle_valley_prominence_change remains the same
    def handle_valley_prominence_change(self):
        if self.egg_signal_processed is None: return
        try:
            new_prominence = float(self.valley_prominence_input.text())
            if new_prominence <= 0: raise ValueError("Prominence must be positive.")
            if abs(new_prominence - self.current_valley_prominence) > 1e-6:
                print(f"Valley Prominence changed to: {new_prominence:.3f}")
                self.current_valley_prominence = new_prominence
                self._trigger_recalculation()
            else:
                self.valley_prominence_input.setText(f"{self.current_valley_prominence:.3f}")
        except ValueError as e:
            QMessageBox.warning(self, "输入无效", f"无效的谷值门限: {e}")
            self.valley_prominence_input.setText(f"{self.current_valley_prominence:.3f}")

    # handle_spec_window_change remains the same
    def handle_spec_window_change(self):
        if self.audio_signal is None: return
        try:
            new_window_ms = float(self.spec_window_input.text())
            min_win_ms = 5.0; max_win_ms = 100.0
            if not (min_win_ms <= new_window_ms <= max_win_ms):
                clamped_window_ms = max(min_win_ms, min(max_win_ms, new_window_ms))
                QMessageBox.warning(self, "输入警告",
                                    f"语谱窗长已限制为 {clamped_window_ms:.1f} ms。\n"
                                    f"(允许范围: {min_win_ms}-{max_win_ms} ms)")
                new_window_ms = clamped_window_ms
                self.spec_window_input.setText(f"{new_window_ms:.1f}")
            if abs(new_window_ms - self.current_spec_window_ms) > 1e-3:
                print(f"Spectrogram Window changed to: {new_window_ms:.1f} ms")
                self.current_spec_window_ms = new_window_ms
                self.update_roi_plots()
            else:
                 self.spec_window_input.setText(f"{self.current_spec_window_ms:.1f}")
        except ValueError as e:
            QMessageBox.warning(self, "输入无效", f"无效的语谱窗长: {e}")
            self.spec_window_input.setText(f"{self.current_spec_window_ms:.1f}")

    def handle_if_order_change(self):
        try:
            txt = self.if_order_input.text().strip()
            if txt == "":
                return
            val = int(txt)
            if val <= 0:
                raise ValueError("invalid")
            self.if_order_input.setText(f"{val}")
        except Exception:
            QMessageBox.warning(self, "输入错误", "IF 阶数必须为正整数或留空自动。")
            self.if_order_input.setText("")

    # _trigger_recalculation remains the same
    def _trigger_recalculation(self):
        print("Recalculating GCI/GOI and updating plots...")
        self._calculate_global_events() # This now triggers GCI-F0 calc
        self.update_roi_plots() # This now triggers F0 contour update
        if self.last_clicked_time is not None:
            self.update_zoom_plots(self.last_clicked_time)

    # closeEvent remains the same
    def closeEvent(self, event):
        self.stop_audio()
        event.accept()

    def open_batch_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("批量处理参数")
        layout = QFormLayout(dlg)
        folder_input = QLineEdit()
        browse_btn = QPushButton("选择文件夹")
        flip_checkbox = QCheckBox("翻转左右声道")
        energy_input = QLineEdit("0.01")
        hpf_input = QLineEdit(str(int(self.highpass_cutoff_current)))
        layout.addRow("根文件夹", folder_input)
        layout.addRow("", browse_btn)
        layout.addRow("静音能量阈值", energy_input)
        layout.addRow("高通滤波(Hz)", hpf_input)
        layout.addRow("选项", flip_checkbox)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addRow(btns)
        def on_browse():
            p = QFileDialog.getExistingDirectory(self, "选择根文件夹", "")
            if p:
                folder_input.setText(p)
        browse_btn.clicked.connect(on_browse)
        def on_ok():
            root = folder_input.text().strip()
            if not root:
                QMessageBox.warning(self, "参数错误", "请选择根文件夹")
                return
            try:
                thr = float(energy_input.text().strip())
                if thr < 0:
                    raise ValueError("invalid")
            except Exception:
                QMessageBox.warning(self, "参数错误", "静音能量阈值必须为非负数")
                return
            try:
                hpf = int(float(hpf_input.text().strip()))
                if not (1 <= hpf <= 1000):
                    raise ValueError("invalid")
            except Exception:
                QMessageBox.warning(self, "参数错误", "高通滤波频率无效")
                return
            dlg.accept()
            self.run_batch_processing(root, flip_checkbox.isChecked(), thr, hpf)
        btns.accepted.connect(on_ok)
        btns.rejected.connect(dlg.reject)
        dlg.exec()

    def run_batch_processing(self, root_folder, flip_lr, energy_threshold, highpass_cutoff):
        try:
            wav_paths = []
            for dirpath, dirnames, filenames in os.walk(root_folder):
                for fn in filenames:
                    if fn.lower().endswith('.wav'):
                        wav_paths.append(os.path.join(dirpath, fn))
            if len(wav_paths) == 0:
                QMessageBox.information(self, "无文件", "未在该文件夹下找到 WAV 文件")
                return
            methods = [
                ("斜率法", "slope", "slope"),
                ("尺度法", "scale", "scale"),
                ("斜率-尺度法", "slope", "scale"),
            ]
            parent_dir = os.path.dirname(root_folder)
            out_roots = {}
            for name, gm, om in methods:
                out_dir = os.path.join(parent_dir, f"{name}-{energy_threshold}-{highpass_cutoff}")
                out_roots[(gm, om)] = out_dir
                os.makedirs(out_dir, exist_ok=True)
            total = len(wav_paths) * len(methods)
            progress = QProgressDialog("批量处理中...", "取消", 0, total, self)
            progress.setWindowTitle("批量处理进度")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            done = 0
            canceled = False
            for wav_path in wav_paths:
                if canceled:
                    break
                rel_path = os.path.relpath(wav_path, root_folder)
                rel_dir = os.path.dirname(rel_path)
                base_full = os.path.basename(wav_path)
                base = os.path.splitext(base_full)[0]
                for name, gm, om in methods:
                    if progress.wasCanceled():
                        canceled = True
                        break
                    out_root = out_roots[(gm, om)]
                    out_dir = os.path.join(out_root, rel_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    progress.setLabelText(f"{name}: {rel_path} ({done+1}/{total})")
                    QApplication.processEvents()
                    self._process_single_wav_batch(wav_path, flip_lr, energy_threshold, highpass_cutoff, gm, om, out_dir, base)
                    done += 1
                    progress.setValue(done)
                    QApplication.processEvents()
            progress.close()
            if canceled:
                QMessageBox.warning(self, "已取消", "批量处理已取消")
            else:
                QMessageBox.information(self, "批量完成", "批量处理已完成")
        except Exception as e:
            QMessageBox.critical(self, "批量错误", f"批量处理失败: {e}")

    def _process_single_wav_batch(self, wav_path, flip_lr, energy_threshold, highpass_cutoff, gci_method, goi_method, out_dir, base_name):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", wavfile.WavFileWarning)
                fs, data = wavfile.read(wav_path)
            if data.ndim != 2 or data.shape[1] != 2:
                return
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                if max_val == 0:
                    max_val = 1
                data = data.astype(np.float32) / max_val
            elif np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
            
            # --- MODIFIED: Normalize channels independently to -0.7~0.7 ---
            max_val_0 = np.max(np.abs(data[:, 0]))
            if max_val_0 > 0:
                data[:, 0] = (data[:, 0] / max_val_0) * 0.7
            
            max_val_1 = np.max(np.abs(data[:, 1]))
            if max_val_1 > 0:
                data[:, 1] = (data[:, 1] / max_val_1) * 0.7
            # --- END MODIFICATION ---

            egg_idx = 0 if not flip_lr else 1
            aud_idx = 1 if not flip_lr else 0
            egg_raw = data[:, egg_idx]
            audio = data[:, aud_idx]
            egg_det = signal.detrend(egg_raw)
            egg_hp = apply_highpass_filter(egg_det, cutoff_freq=highpass_cutoff, fs=fs)
            egg_proc = apply_lowpass_filter(egg_hp, cutoff_freq=DEFAULT_LOWPASS_CUTOFF, fs=fs)
            tvec = np.arange(len(egg_proc)) / fs
            gci, goi, peaks = find_gci_goi_peak_min_criterion(
                egg_proc, fs,
                min_f0=50, max_f0=500, criterion_level=0.25,
                peak_prominence=DEFAULT_PEAK_PROMINENCE,
                valley_prominence=self.current_valley_prominence if hasattr(self, 'current_valley_prominence') else DEFAULT_VALLEY_PROMINENCE,
                use_local_prominence=True,
                local_window_s=0.2, local_hop_s=0.1, min_auto_prom=DEFAULT_PEAK_PROMINENCE,
                gci_method=gci_method, goi_method=goi_method
            )
            cq_t, cq_v, sq_v = calculate_cq_sq(gci, goi, peaks)

            # --- NEW: Apply Silence Threshold ---
            if energy_threshold > 0 and cq_t is not None and len(cq_t) > 0:
                try:
                    from scipy.ndimage import uniform_filter1d
                    # Calculate RMS energy profile
                    win_samples = int(0.03 * fs) # 30ms window
                    if win_samples < 1: win_samples = 1
                    audio_sq = audio.astype(np.float64) ** 2
                    energy_profile = np.sqrt(uniform_filter1d(audio_sq, size=win_samples, mode='constant', cval=0.0))
                    
                    cq_t_arr = np.array(cq_t)
                    cq_v_arr = np.array(cq_v)
                    sq_v_arr = np.array(sq_v)
                    
                    cq_indices = (cq_t_arr * fs).astype(int)
                    cq_indices = np.clip(cq_indices, 0, len(energy_profile) - 1)
                    
                    cq_energies = energy_profile[cq_indices]
                    mask = cq_energies >= energy_threshold
                    
                    cq_t = cq_t_arr[mask]
                    cq_v = cq_v_arr[mask]
                    sq_v = sq_v_arr[mask]
                except Exception as e:
                    print(f"Warning: Failed to apply silence threshold: {e}")
            # --- END NEW ---

            audio_f0_t, audio_f0_v = None, None
            try:
                if PARSELMOUTH_AVAILABLE:
                    snd = parselmouth.Sound(audio, fs)
                    pitch = praat_call(snd, "To Pitch...", 0.0, 75, 600)
                    audio_f0_t = pitch.xs()
                    audio_f0_v = pitch.selected_array['frequency']
                    audio_f0_v[audio_f0_v == 0] = np.nan
            except Exception:
                audio_f0_t, audio_f0_v = None, None
            gci_f0_t, gci_f0_v = None, None
            try:
                if gci is not None and len(gci) > 1:
                    gci_np = np.array(gci)
                    periods = np.diff(gci_np)
                    f0_vals = 1.0 / periods
                    f0_times = gci_np[:-1] + periods / 2.0
                    vals = np.array(f0_vals, dtype=float)
                    times = np.array(f0_times, dtype=float)
                    if len(vals) > 2:
                        med = np.median(vals)
                        mad = np.median(np.abs(vals - med))
                        thr = 3.0 * mad if mad > 0 else max(1e-12, 3.0 * np.std(vals))
                        base = vals.copy()
                        base[np.abs(base - med) >= thr] = np.nan
                        force_keep = vals < 100.0
                        base[force_keep] = vals[force_keep]
                        nan_arr = np.isnan(base)
                        left_nan = np.concatenate(([False], nan_arr[:-1]))
                        right_nan = np.concatenate((nan_arr[1:], [False]))
                        keep = (force_keep) | ((~nan_arr) & (~(left_nan & right_nan)))
                        gci_f0_t = times[keep]
                        gci_f0_v = base[keep]
                    else:
                        gci_f0_t = times
                        gci_f0_v = vals
            except Exception:
                gci_f0_t, gci_f0_v = None, None
            cq_df = pd.DataFrame()
            if cq_t is not None and cq_v is not None and sq_v is not None:
                cq_df = pd.DataFrame({'Time': np.array(cq_t), 'CQ': np.array(cq_v), 'SQ': np.array(sq_v)}).set_index('Time')
            f0_praat_df = pd.DataFrame()
            if audio_f0_t is not None and audio_f0_v is not None:
                f0_praat_df = pd.DataFrame({'Time': np.array(audio_f0_t), 'F0_Praat (Hz)': np.array(audio_f0_v)}).set_index('Time')
            f0_corr_df = pd.DataFrame()
            if gci_f0_t is not None and gci_f0_v is not None:
                f0_corr_df = pd.DataFrame({'Time': np.array(gci_f0_t), 'F0_Corrected (Hz)': np.array(gci_f0_v)}).set_index('Time')
            df = cq_df.join(f0_praat_df, how='outer').join(f0_corr_df, how='outer')
            df.sort_index(inplace=True)
            csv_path = os.path.join(out_dir, f"{base_name}.csv")
            df.to_csv(csv_path, na_rep='NaN', index_label='Time (s)')
            cq_only_path = os.path.join(out_dir, f"{base_name}_CQ_SQ.csv")
            if cq_t is not None and cq_v is not None and sq_v is not None and len(cq_t) > 0:
                cq_only_df = pd.DataFrame({'Time': np.array(cq_t), 'CQ': np.array(cq_v), 'SQ': np.array(sq_v)}).set_index('Time')
                cq_only_df.sort_index(inplace=True)
                cq_only_df.to_csv(cq_only_path, na_rep='NaN', index_label='Time (s)')
            else:
                empty_df = pd.DataFrame(columns=['CQ', 'SQ'])
                empty_df.index.name = 'Time (s)'
                empty_df.to_csv(cq_only_path)
            spec_path = os.path.join(out_dir, f"{base_name}_SPEC_F0.png")
            cq_path = os.path.join(out_dir, f"{base_name}_CQ_SQ.png")
            wave_path = os.path.join(out_dir, f"{base_name}_WAVEFORMS.png")
            self._save_plots_batch(audio, egg_proc, fs, 0.0, len(audio)/fs, audio_f0_t, audio_f0_v, cq_t, cq_v, sq_v, spec_path, cq_path, wave_path)
        except Exception:
            pass

    def _save_plots_batch(self, audio_signal, egg_signal, fs, start_s, end_s,
                           f0_times, f0_values, cq_times, cq_values, sq_values,
                           spec_path, cq_path, wave_path):
        LIGHT_STYLE = {
            'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
            'xtick.color': 'black', 'ytick.color': 'black',
            'grid.color': '#DDDDDD', 'grid.linestyle': ':',
            'figure.facecolor': 'white', 'axes.facecolor': 'white',
            'savefig.facecolor': 'white', 'text.color': 'black',
            'lines.color': 'black', 'patch.edgecolor': 'black'
        }
        with plt.style.context(LIGHT_STYLE):
            try:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1_sq = ax1.twinx()
                ax1.set_title(f"EGG CQ & SQ ({start_s:.2f}s - {end_s:.2f}s)")
                if cq_times is not None and cq_values is not None:
                    valid_cq = ~np.isnan(np.array(cq_values))
                    roi_mask_cq = (np.array(cq_times) >= start_s) & (np.array(cq_times) <= end_s) & valid_cq
                    if np.any(roi_mask_cq):
                        ax1.plot(np.array(cq_times)[roi_mask_cq], np.array(cq_values)[roi_mask_cq],
                                 label='CQ', color='blue', marker='.', linestyle='', markersize=5)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Contact Quotient (CQ)", color='blue')
                ax1.set_ylim(0, 1)
                ax1.tick_params(axis='y', colors='blue', labelcolor='blue')
                if cq_times is not None and sq_values is not None:
                    valid_sq = ~np.isnan(np.array(sq_values))
                    roi_mask_sq = (np.array(cq_times) >= start_s) & (np.array(cq_times) <= end_s) & valid_sq
                    if np.any(roi_mask_sq):
                        ax1_sq.plot(np.array(cq_times)[roi_mask_sq], np.array(sq_values)[roi_mask_sq],
                                    label='SQ', color='green', marker='x', linestyle='', markersize=5)
                ax1_sq.set_ylabel("Speed Quotient (SQ)", color='green')
                ax1_sq.set_ylim(-1.1, 1.1)
                ax1_sq.tick_params(axis='y', colors='green', labelcolor='green')
                ax1.set_xlim(start_s, end_s)
                ax1.grid(True)
                fig1.tight_layout()
                fig1.savefig(cq_path, dpi=150)
                plt.close(fig1)
            except Exception:
                pass
            try:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                try:
                    fig2.set_layout_engine('constrained')
                except Exception:
                    fig2.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)
                ax2_f0 = ax2.twinx()
                ax2.set_title(f"Spectrogram ({start_s:.2f}s - {end_s:.2f}s)")
                nfft = int(fs * (self.current_spec_window_ms / 1000.0)) if hasattr(self, 'current_spec_window_ms') else int(fs * 0.03)
                noverlap = int(nfft * 0.75)
                nfft = min(nfft, len(audio_signal))
                if noverlap >= nfft:
                    noverlap = max(0, nfft - 1)
                im = None
                if nfft > 0 and len(audio_signal) > nfft:
                    Pxx, freqs, bins, im = ax2.specgram(
                        audio_signal, NFFT=nfft, Fs=fs, noverlap=noverlap,
                        cmap='gray_r', scale='dB', mode='magnitude',
                        vmin=self.current_spec_vmin if hasattr(self, 'current_spec_vmin') else -60,
                        vmax=self.current_spec_vmax if hasattr(self, 'current_spec_vmax') else 0)
                    im.set_extent([start_s, end_s, freqs[0], freqs[-1]])
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Frequency (Hz)")
                ax2.set_ylim(0, 5000)
                if f0_times is not None and f0_values is not None:
                    mask_f0 = (np.array(f0_times) >= start_s) & (np.array(f0_times) <= end_s)
                    if np.any(mask_f0):
                        f0_vals = np.array(f0_values)[mask_f0]
                        ax2_f0.plot(np.array(f0_times)[mask_f0], f0_vals, color='black', lw=1.5)
                        valid_vals = f0_vals[~np.isnan(f0_vals)]
                        if valid_vals.size > 0:
                            f0_min = float(np.min(valid_vals))
                            f0_max = float(np.max(valid_vals))
                            padding = max(10.0, (f0_max - f0_min) * 0.1)
                            if (f0_max - f0_min) < 1e-6:
                                ax2_f0.set_ylim(f0_min - 20, f0_max + 20)
                            else:
                                ax2_f0.set_ylim(f0_min - padding, f0_max + padding)
                        else:
                            ax2_f0.set_ylim(50, 500)
                    else:
                        ax2_f0.set_ylim(50, 500)
                else:
                    ax2_f0.set_ylim(50, 500)
                ax2.set_xlim(start_s, end_s)
                if im is not None:
                    fig2.colorbar(im, ax=ax2, label='Magnitude (dB)')
                fig2.savefig(spec_path, dpi=150)
                plt.close(fig2)
            except Exception:
                pass
            try:
                fig3, (ax_audio, ax_egg) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                fig3.suptitle(f"Waveforms ({start_s:.2f}s - {end_s:.2f}s)")
                time = np.arange(len(audio_signal)) / fs
                ax_audio.plot(time, audio_signal, color='black', lw=0.5)
                ax_audio.set_ylabel("Audio Amplitude")
                ax_audio.grid(True)
                ax_egg.plot(time, egg_signal, color='black', lw=0.5)
                ax_egg.set_ylabel("EGG Amplitude")
                ax_egg.set_xlabel("Time (s)")
                ax_egg.grid(True)
                ax_egg.set_xlim(start_s, end_s)
                fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig3.savefig(wave_path, dpi=150)
                plt.close(fig3)
            except Exception:
                pass

# --- Run the Application ---
if __name__ == '__main__':
    # Set constrained_layout globally if desired (can help with colorbars)
    # plt.rcParams['figure.constrained_layout.use'] = True
    plt.style.use('dark_background')
    plt.rcParams.update(MATPLOTLIB_STYLE_SETTINGS)

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)

    main_win = EGGAnalysisApp()
    main_win.show()

    sys.exit(app.exec())
