import os
import glob
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QLineEdit, QLabel, QGroupBox, 
                             QProgressBar, QTextEdit, QMessageBox, QFileDialog,
                             QTableWidget, QTableWidgetItem, QComboBox, QInputDialog,
                             QRadioButton, QCheckBox, QDialog)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QIcon

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from phonetic_toolbox.services.manipulation_service import ManipulationService
from phonetic_toolbox.gui.dialogs.manipulation_dialogs import BatchProcessorDialog, ImportF0Dialog, KnotEditorDialog
from phonetic_toolbox.gui.utils import apply_plot_theme, play_audio_sd
from phonetic_toolbox.utils import get_resource_path, parse_float_list
from phonetic_toolbox.gui.styles import GLOBAL_DARK_STYLESHEET, GLOBAL_LIGHT_STYLESHEET

# Set matplotlib font for Chinese support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PitchManipulationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("基频实验室 (Pitch Lab)")
        self.resize(1400, 900)
        
        self.service = ManipulationService()
        self.is_dark = True  # Default theme

        # --- Data Storage ---
        self.snd_path = ""
        self.snd = None       
        self.snd_part = None 
        self.start_mode = "order"   # full | order | reverse | constant
        self.end_mode = "order"
        self.knot_modes = []        # per-knot modes
        
        self.times = []
        self.original_f0 = []
        self.modified_f0 = [] 
        self.synth_snd = None 
        
        self.ref_lines = [] 
        self.knot_points = []
        
        # Interaction State
        self.is_drawing = False
        self.is_restoring = False
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_xlim = None
        self.last_edit_idx = None
        self.last_edit_freq = None
        self.current_xlim = None 
        self.history_cache = []

        self.init_ui()
        self.apply_theme()  # Apply initial theme

    def init_ui(self):
        # Main Layout
        main_layout = QVBoxLayout(self)

        # --- 1. Top Controls ---
        controls_layout = QVBoxLayout()
        
        # Row 1: File & Playback
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(0)
        self.btn_load = QPushButton("加载音频")
        self.btn_load.clicked.connect(self.load_audio)
        
        self.btn_play_orig = QPushButton("播放当前视野原音")
        self.btn_play_orig.clicked.connect(lambda: play_audio_sd(self.snd_part))
        self.btn_play_orig.setEnabled(False)

        self.btn_synthesize = QPushButton("合成当前视野")
        self.btn_synthesize.clicked.connect(self.synthesize_sound)
        self.btn_synthesize.setEnabled(False)

        # Speed Control
        self.input_speed = QLineEdit("1.0")
        self.input_speed.setFixedWidth(50)
        self.input_speed.setPlaceholderText("语速")
        row1_layout.addWidget(QLabel("语速倍率:"))
        row1_layout.addWidget(self.input_speed)
        row1_layout.addSpacing(20)

        self.btn_play_synth = QPushButton("播放合成音")
        self.btn_play_synth.clicked.connect(lambda: play_audio_sd(self.synth_snd))
        self.btn_play_synth.setEnabled(False)

        self.btn_save_audio = QPushButton("保存并编号")
        self.btn_save_audio.clicked.connect(self.save_audio_smart)
        self.btn_save_audio.setEnabled(False)
        
        self.btn_delete_batch = QPushButton("删除本批次音频")
        self.btn_delete_batch.clicked.connect(self.delete_current_batch_files)
        
        self.btn_rename_batch = QPushButton("批量重命名")
        self.btn_rename_batch.clicked.connect(self.rename_current_batch_files)
        
        row1_layout.addWidget(self.btn_load)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_play_orig)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_synthesize)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_play_synth)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_save_audio)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_delete_batch)
        row1_layout.addSpacing(5)
        row1_layout.addWidget(self.btn_rename_batch)
        row1_layout.addStretch()
        
        # Row 2: Axis & Reference Lines
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(0)
        
        # Y-Axis Range
        row2_layout.addWidget(QLabel("Y轴范围(Hz):"))
        row2_layout.addSpacing(5)
        self.input_ymin = QLineEdit("50")
        self.input_ymin.setFixedWidth(60)
        row2_layout.addWidget(QLabel("Min:"))
        row2_layout.addWidget(self.input_ymin)
        row2_layout.addSpacing(15)
        self.input_ymax = QLineEdit("350")
        self.input_ymax.setFixedWidth(60)
        row2_layout.addWidget(QLabel("Max:"))
        row2_layout.addWidget(self.input_ymax)
        row2_layout.addSpacing(15)
        btn_set_axis = QPushButton("应用范围")
        btn_set_axis.clicked.connect(self.update_axis_range)
        row2_layout.addWidget(btn_set_axis)
        
        row2_layout.addSpacing(20)
        
        # Reference Lines
        row2_layout.addWidget(QLabel("参考线:"))
        self.input_ref = QLineEdit("200")
        self.input_ref.setFixedWidth(60)
        self.input_ref.setPlaceholderText("频率")
        row2_layout.addWidget(self.input_ref)
        row2_layout.addSpacing(15)
        btn_add_ref = QPushButton("添加参考线")
        btn_add_ref.clicked.connect(self.add_ref_line)
        row2_layout.addWidget(btn_add_ref)
        row2_layout.addSpacing(5)
        btn_clear_ref = QPushButton("清除所有")
        btn_clear_ref.clicked.connect(self.clear_ref_lines)
        row2_layout.addWidget(btn_clear_ref)
        
        row2_layout.addSpacing(20)
        
        # Other Tools
        self.btn_save_compare = QPushButton("保存对比图")
        self.btn_save_compare.clicked.connect(self.save_comparison_plot)
        self.btn_save_compare.setEnabled(False)
        
        self.btn_import_f0 = QPushButton("导入基频序列")
        self.btn_import_f0.clicked.connect(self.import_f0_sequence)
        self.btn_import_f0.setEnabled(False)
        
        btn_batch_tool = QPushButton("批量变速变调工具")
        btn_batch_tool.clicked.connect(self.open_batch_tool)

        # Forest-themed Help Button
        btn_help = QPushButton("帮助")
        btn_help.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_help.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57; 
                color: white; 
                font-weight: bold; 
                border-radius: 4px; 
                padding: 5px 10px;
                font-family: "Microsoft YaHei";
            }
            QPushButton:hover {
                background-color: #3CB371;
            }
            QPushButton:pressed {
                background-color: #228B22;
            }
        """)
        btn_help.clicked.connect(self._open_help_doc)

        row2_layout.addWidget(self.btn_save_compare)
        row2_layout.addSpacing(5)
        row2_layout.addWidget(self.btn_import_f0)
        row2_layout.addSpacing(5)
        row2_layout.addWidget(btn_batch_tool)
        row2_layout.addSpacing(5)
        row2_layout.addWidget(btn_help)
        row2_layout.addStretch()

        controls_layout.addLayout(row1_layout)
        controls_layout.addLayout(row2_layout)
        
        # Batch Linear Pitch Change Group
        row5_layout = QHBoxLayout()
        grp_batch = QGroupBox("批量改变基频")
        layout_batch = QVBoxLayout()
        
        rowA = QHBoxLayout()
        rowA.setSpacing(0)
        rowB = QHBoxLayout()
        rowB.setSpacing(0)
        
        self.input_batch_t1 = QLineEdit("0.00")
        self.input_batch_t1.setPlaceholderText("起始时间(s)")
        self.input_batch_t1.setFixedWidth(100)
        self.input_batch_t2 = QLineEdit("1.00")
        self.input_batch_t2.setPlaceholderText("终止时间(s)")
        self.input_batch_t2.setFixedWidth(100)
        
        self.input_batch_f1_list = QLineEdit("")
        self.input_batch_f1_list.setPlaceholderText("起始基频列表(Hz, 逗号分隔)")
        self.input_batch_f1_list.setFixedWidth(220)
        self.input_batch_f2_list = QLineEdit("")
        self.input_batch_f2_list.setPlaceholderText("终止基频列表(Hz, 逗号分隔)")
        self.input_batch_f2_list.setFixedWidth(220)
        
        self.btn_batch_linear_save = QPushButton("批量生成并保存")
        self.btn_batch_linear_save.clicked.connect(self.batch_linear_save)
        self.btn_batch_linear_save.setEnabled(False)
        
        rowA.addWidget(QLabel("起始t(s):"))
        rowA.addWidget(self.input_batch_t1)
        rowA.addSpacing(15)
        rowA.addWidget(QLabel("终止t(s):"))
        rowA.addWidget(self.input_batch_t2)
        rowA.addSpacing(15)
        rowA.addWidget(QLabel("起始F0列表:"))
        rowA.addWidget(self.input_batch_f1_list)
        rowA.addSpacing(15)
        rowA.addWidget(QLabel("终止F0列表:"))
        rowA.addWidget(self.input_batch_f2_list)
        rowA.addStretch()
        
        self.input_knot_time = QLineEdit("")
        self.input_knot_time.setPlaceholderText("拐点时间(s)")
        self.input_knot_time.setFixedWidth(120)
        self.input_knot_freqs = QLineEdit("")
        self.input_knot_freqs.setPlaceholderText("拐点频率列表(Hz, 逗号分隔)")
        self.input_knot_freqs.setFixedWidth(240)
        
        btn_add_knot = QPushButton("添加拐点")
        btn_add_knot.clicked.connect(self.add_knot)
        btn_clear_knots = QPushButton("清除拐点")
        btn_clear_knots.clicked.connect(self.clear_knots)
        
        self.lbl_knots_summary = QLabel("拐点: 0")
        
        self.radio_linear = QRadioButton("直线插值")
        self.radio_linear.setChecked(True)
        
        btn_edit_knots = QPushButton("编辑拐点")
        btn_edit_knots.clicked.connect(self.open_edit_knots)
        
        self.checkbox_offset = QCheckBox("使用升降偏移模式")
        
        rowB.addWidget(QLabel("拐点t:"))
        rowB.addWidget(self.input_knot_time)
        rowB.addSpacing(15)
        rowB.addWidget(QLabel("拐点F0列表:"))
        rowB.addWidget(self.input_knot_freqs)
        rowB.addSpacing(15)
        rowB.addWidget(btn_add_knot)
        rowB.addSpacing(5)
        rowB.addWidget(btn_clear_knots)
        rowB.addSpacing(5)
        rowB.addWidget(btn_edit_knots)
        rowB.addSpacing(10)
        rowB.addWidget(self.lbl_knots_summary)
        rowB.addSpacing(10)
        rowB.addWidget(self.radio_linear)
        rowB.addSpacing(10)
        rowB.addWidget(self.checkbox_offset)
        rowB.addSpacing(10)
        rowB.addWidget(self.btn_batch_linear_save)
        rowB.addStretch()
        
        layout_batch.addLayout(rowA)
        layout_batch.addLayout(rowB)
        grp_batch.setLayout(layout_batch)
        row5_layout.addWidget(grp_batch)
        
        controls_layout.addLayout(row5_layout)
        main_layout.addLayout(controls_layout)

        # --- 2. Plot Area (2x2 Grid) ---
        self.figure = Figure(figsize=(12, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        self.gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
        
        self.ax_wave_orig = self.figure.add_subplot(self.gs[0, 0])
        self.ax_pitch = self.figure.add_subplot(self.gs[0, 1])
        self.ax_wave_synth = self.figure.add_subplot(self.gs[1, 0])
        self.ax_compare = self.figure.add_subplot(self.gs[1, 1])

        # Connect Events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.reset_plots()

    def _open_help_doc(self):
        help_file = Path(get_resource_path(r"Phonetic_Export\index.html"))
        if not help_file.exists():
            QMessageBox.warning(self, "帮助", f"未找到帮助文件：{help_file}")
            return
        url = QUrl.fromLocalFile(str(help_file))
        url.setFragment("s1764313121711")
        opened = QDesktopServices.openUrl(url)
        if not opened:
            QMessageBox.warning(self, "帮助", "帮助页面打开失败。")

    def get_downsampled_slice(self, times, values, xmin, xmax):
        """Downsample waveform data to 5000-10000 points in view."""
        if times is None or len(times) == 0:
            return np.array([]), np.array([])
            
        # Find index range
        if xmin is None or xmax is None:
            i0, i1 = 0, len(times)
        else:
            i0 = np.searchsorted(times, xmin)
            i1 = np.searchsorted(times, xmax)
            
        n_points = i1 - i0
        if n_points <= 0:
            return np.array([]), np.array([])
            
        # If total points in view is already small enough, return as is
        if n_points <= 10000:
            return times[i0:i1], values[i0:i1]
            
        step = 1
        # Keep points between 5000 and 10000
        # If > 10000, step *= 2
        current_count = n_points
        while current_count > 10000:
            step *= 2
            current_count = n_points // step
            
        return times[i0:i1:step], values[i0:i1:step]

    def apply_theme(self):
        """Apply dark/light theme to the widget and plots."""
        if self.is_dark:
            self.setStyleSheet(GLOBAL_DARK_STYLESHEET)
            self.figure.patch.set_facecolor('black')
        else:
            self.setStyleSheet(GLOBAL_LIGHT_STYLESHEET)
            self.figure.patch.set_facecolor('white')
            
        if self.snd is None:
            self.reset_plots()
        else:
            self.draw_all()

    def reset_plots(self):
        """Initialize/Clear plot styles."""
        # 1. Original Waveform
        self.ax_wave_orig.clear()
        apply_plot_theme(self.ax_wave_orig, self.is_dark)
        self.ax_wave_orig.set_title("原始波形", fontsize=10)
        self.ax_wave_orig.set_xticks([])
        
        # 2. Pitch Editor
        self.ax_pitch.clear()
        apply_plot_theme(self.ax_pitch, self.is_dark)
        self.ax_pitch.set_title("基频编辑 - 滚轮缩放X轴 / 左键拖拽X轴 / Shift绘制 / Ctrl恢复", fontsize=10)
        self.ax_pitch.set_ylabel("Frequency (Hz)")
        grid_color = 'gray' if self.is_dark else '#cccccc'
        self.ax_pitch.grid(True, linestyle='--', alpha=0.3, color=grid_color)
        
        # 3. Synthesized Waveform
        self.ax_wave_synth.clear()
        apply_plot_theme(self.ax_wave_synth, self.is_dark)
        self.ax_wave_synth.set_title("合成波形", fontsize=10)
        self.ax_wave_synth.set_xlabel("Time (s)")
        
        # 4. Comparison
        self.ax_compare.clear()
        apply_plot_theme(self.ax_compare, self.is_dark)
        self.ax_compare.set_title("历史F0对比（0起点对齐）", fontsize=10)
        self.ax_compare.set_xlabel("Relative Time (s)")
        self.ax_compare.grid(True, linestyle=':', alpha=0.3, color=grid_color)

        self.canvas.draw()

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "Audio Files (*.wav *.mp3 *.flac)")
        if not file_path:
            return

        try:
            times, f0, xmin, xmax = self.service.load_audio(file_path)
            self.snd_path = file_path
            self.snd = self.service.snd
            self.snd_part = self.service.get_sound_part(xmin, xmax)
            
            self.times = times
            self.original_f0 = f0
            self.modified_f0 = f0.copy()
            self.current_xlim = (xmin, xmax)
            
            # Update UI state
            self.btn_play_orig.setEnabled(True)
            self.btn_synthesize.setEnabled(True)
            self.btn_play_synth.setEnabled(False)
            self.btn_save_audio.setEnabled(False)
            self.btn_save_compare.setEnabled(True)
            self.btn_import_f0.setEnabled(True)
            self.btn_batch_linear_save.setEnabled(True)
            
            # Update time inputs
            # Find range where pitch exists (non-zero F0)
            valid_indices = np.where(f0 > 0)[0]
            if len(valid_indices) > 0:
                first_t = times[valid_indices[0]]
                last_t = times[valid_indices[-1]]
                self.input_batch_t1.setText(f"{first_t:.2f}")
                self.input_batch_t2.setText(f"{last_t:.2f}")
            else:
                self.input_batch_t1.setText(f"{xmin:.2f}")
                self.input_batch_t2.setText(f"{xmax:.2f}")
            
            self.draw_all()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载音频失败: {str(e)}")

    def update_axis_range(self):
        try:
            ymin = float(self.input_ymin.text())
            ymax = float(self.input_ymax.text())
            if ymin >= ymax:
                raise ValueError("Min must be less than Max")
            self.ax_pitch.set_ylim([ymin, ymax])
            self.ax_compare.set_ylim([ymin, ymax])
            self.canvas.draw_idle()
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的数字范围")

    def add_ref_line(self):
        try:
            val = float(self.input_ref.text())
            self.ref_lines.append(val)
            self.draw_pitch_curve_content()
            self.canvas.draw_idle()
        except ValueError:
            pass

    def clear_ref_lines(self):
        self.ref_lines = []
        self.draw_pitch_curve_content()
        self.canvas.draw_idle()

    def draw_all(self):
        if self.snd is None: return

        # 1. Original Waveform
        self.draw_orig_wave_content()

        # 2. Pitch Curve
        self.draw_pitch_curve_content()
        
        # 3. Synth Wave
        self.draw_synth_wave_content()

        # 4. Comparison
        self.update_comparison_plot()

        self.canvas.draw()

    def draw_orig_wave_content(self):
        self.ax_wave_orig.clear()
        apply_plot_theme(self.ax_wave_orig, self.is_dark)
        
        xmin, xmax = self.current_xlim if self.current_xlim else (None, None)
        
        if self.snd:
            # Downsample for display
            xs, ys = self.get_downsampled_slice(self.snd.xs(), self.snd.values.T, xmin, xmax)
            
            color = 'white' if self.is_dark else 'blue'
            self.ax_wave_orig.plot(xs, ys, color=color, alpha=0.6, linewidth=0.5)
            
        self.ax_wave_orig.set_title("原始波形")
        self.ax_wave_orig.set_xticks([])
        if self.current_xlim: self.ax_wave_orig.set_xlim(self.current_xlim)

    def draw_pitch_curve_content(self):
        self.ax_pitch.clear()
        apply_plot_theme(self.ax_pitch, self.is_dark)
        
        # Original
        plot_orig = self.original_f0.copy()
        plot_orig[plot_orig == 0] = np.nan
        color_orig = 'cyan' if self.is_dark else 'blue'
        self.ax_pitch.plot(self.times, plot_orig, color=color_orig, linestyle=':', label='Original', alpha=0.5)

        # Modified
        plot_mod = self.modified_f0.copy()
        plot_mod[plot_mod == 0] = np.nan
        self.line_mod, = self.ax_pitch.plot(
            self.times,
            plot_mod,
            color='red',
            linewidth=1.5,
            marker='o',
            markersize=2.2,
            markeredgewidth=0,
            markevery=1,
            label='Modified',
        )
        
        # Ref lines
        line_color = 'gray' if self.is_dark else 'black'
        text_color = 'white' if self.is_dark else 'black'
        for ref in self.ref_lines:
            self.ax_pitch.axhline(y=ref, color=line_color, linestyle='--', alpha=0.8)
            self.ax_pitch.text(self.times[0], ref, f"{int(ref)}Hz", color=text_color, fontsize=8, verticalalignment='bottom')

        self.ax_pitch.set_title("基频编辑")
        self.ax_pitch.set_ylabel("Frequency (Hz)")
        grid_color = 'gray' if self.is_dark else '#cccccc'
        self.ax_pitch.grid(True, linestyle='--', alpha=0.3, color=grid_color)
        
        legend_face = 'black' if self.is_dark else 'white'
        legend_text = 'white' if self.is_dark else 'black'
        self.ax_pitch.legend(loc='upper right', facecolor=legend_face, labelcolor=legend_text)
        
        self.update_axis_range()
        if self.current_xlim: self.ax_pitch.set_xlim(self.current_xlim)

    def draw_synth_wave_content(self):
        self.ax_wave_synth.clear()
        apply_plot_theme(self.ax_wave_synth, self.is_dark)
        color = 'lime' if self.is_dark else 'green'
        
        xmin, xmax = self.current_xlim if self.current_xlim else (None, None)
        
        if self.synth_snd:
            # Downsample for display
            xs, ys = self.get_downsampled_slice(self.synth_snd.xs(), self.synth_snd.values.T, xmin, xmax)
            self.ax_wave_synth.plot(xs, ys, color=color, alpha=0.6, linewidth=0.5)
            
        self.ax_wave_synth.set_title("合成波形")
        self.ax_wave_synth.set_xlabel("Time (s)")
        if self.current_xlim: self.ax_wave_synth.set_xlim(self.current_xlim)

    def update_comparison_plot(self):
        self.ax_compare.clear()
        apply_plot_theme(self.ax_compare, self.is_dark)
        self.ax_compare.set_title("历史F0对比")
        self.ax_compare.set_xlabel("Time (s)")
        grid_color = 'gray' if self.is_dark else '#cccccc'
        self.ax_compare.grid(True, linestyle=':', alpha=0.3, color=grid_color)
        
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2'])
        linestyles = cycle(['--', '-.', ':'])

        if self.snd_path:
            folder = os.path.dirname(self.snd_path)
            filename = os.path.basename(self.snd_path)
            stem, _ = os.path.splitext(filename)
            
            if self.current_xlim:
                search_path = os.path.join(folder, f"{stem}_*_modified_*.wav")
                found_files = glob.glob(search_path)
                
                def get_index(fname):
                    match = re.search(r'_(\d+)\.wav$', fname)
                    return int(match.group(1)) if match else 0
                found_files.sort(key=get_index)
                
                for fpath in found_files:
                    try:
                        # We use ManipulationService just to load quickly or use parselmouth directly
                        # Using parselmouth directly for speed here as we don't need service state update
                        import parselmouth
                        h_snd = parselmouth.Sound(fpath)
                        h_pitch = h_snd.to_pitch()
                        h_times = h_pitch.xs()
                        h_vals = h_pitch.selected_array['frequency']
                        h_vals[h_vals == 0] = np.nan
                        
                        idx = get_index(os.path.basename(fpath))
                        c = next(colors)
                        ls = next(linestyles)
                        
                        self.ax_compare.plot(h_times, h_vals, color=c, linestyle=ls, 
                                             linewidth=1.5, alpha=0.8, label=f'Ver {idx}')
                        
                    except Exception as e:
                        print(f"Skipped {fpath}: {e}")

        try:
            handles, labels = self.ax_compare.get_legend_handles_labels()
            if len(labels) <= 10:
                legend_face = 'black' if self.is_dark else 'white'
                legend_text = 'white' if self.is_dark else 'black'
                self.ax_compare.legend(loc='upper right', fontsize='small', framealpha=0.5, facecolor=legend_face, labelcolor=legend_text)
        except:
            pass
        
        try:
            ymin = float(self.input_ymin.text())
            ymax = float(self.input_ymax.text())
            self.ax_compare.set_ylim([ymin, ymax])
        except:
            pass
        if self.current_xlim:
            try:
                self.ax_compare.set_xlim(self.current_xlim)
            except:
                pass

    def on_scroll(self, event):
        if event.inaxes not in [self.ax_pitch, self.ax_wave_orig, self.ax_wave_synth]:
            return

        cur_xlim = self.ax_pitch.get_xlim()
        cur_range = cur_xlim[1] - cur_xlim[0]
        xdata = event.xdata 

        if xdata is None: return

        scale_factor = 0.8 if event.button == 'up' else 1.2
        new_range = cur_range * scale_factor
        
        rel_pos = (xdata - cur_xlim[0]) / cur_range
        new_xmin = xdata - new_range * rel_pos
        new_xmax = xdata + new_range * (1 - rel_pos)
        
        new_xmin = max(self.snd.xmin, new_xmin)
        new_xmax = min(self.snd.xmax, new_xmax)
        
        if new_xmax - new_xmin < 0.05: return

        self.current_xlim = (new_xmin, new_xmax)
        
        # Re-draw waveforms with new sampling density
        self.draw_orig_wave_content()
        self.draw_synth_wave_content()
        
        # Pitch curve doesn't need downsampling, just xlim update
        self.ax_pitch.set_xlim(self.current_xlim)
        
        self.update_comparison_plot()
        self.canvas.draw_idle()
        self.snd_part = self.service.get_sound_part(new_xmin, new_xmax)

    def on_mouse_press(self, event):
        if event.button != 1:
            return
        modifiers = QApplication.keyboardModifiers()
        is_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        is_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if event.inaxes == self.ax_pitch and is_shift and not is_ctrl:
            self.is_drawing = True
            self.is_restoring = False
            self.is_panning = False
            self.last_edit_idx = None
            self.last_edit_freq = None
            self.update_pitch_data(event.xdata, event.ydata)
            return
        if event.inaxes == self.ax_pitch and is_ctrl and not is_shift:
            self.is_restoring = True
            self.is_drawing = False
            self.is_panning = False
            self.last_edit_idx = None
            self.last_edit_freq = None
            self.restore_pitch_data(event.xdata)
            return
        if event.inaxes in [self.ax_pitch, self.ax_wave_orig, self.ax_wave_synth]:
            self.is_panning = True
            self.is_drawing = False
            self.is_restoring = False
            self.pan_start_x = event.xdata
            self.pan_start_xlim = self.current_xlim if self.current_xlim else self.ax_pitch.get_xlim()

    def on_mouse_move(self, event):
        if self.is_drawing and event.inaxes == self.ax_pitch:
            self.update_pitch_data(event.xdata, event.ydata)
            return
        if self.is_restoring and event.inaxes == self.ax_pitch:
            self.restore_pitch_data(event.xdata)
            return
        if self.is_panning and event.inaxes in [self.ax_pitch, self.ax_wave_orig, self.ax_wave_synth]:
            self.pan_x_axis(event.xdata)

    def on_mouse_release(self, event):
        self.is_drawing = False
        self.is_restoring = False
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_xlim = None
        self.last_edit_idx = None
        self.last_edit_freq = None

    def update_pitch_data(self, x_time, y_freq):
        if x_time is None or y_freq is None:
            return
        if len(self.times) == 0:
            return
        idx = int((np.abs(self.times - x_time)).argmin())
        if self.original_f0[idx] <= 0:
            return
        if self.last_edit_idx is None:
            self.modified_f0[idx] = y_freq
            self.last_edit_idx = idx
            self.last_edit_freq = float(y_freq)
        else:
            lo = min(self.last_edit_idx, idx)
            hi = max(self.last_edit_idx, idx)
            interp_vals = np.linspace(float(self.last_edit_freq), float(y_freq), hi - lo + 1)
            valid = self.original_f0[lo:hi + 1] > 0
            if np.any(valid):
                target = self.modified_f0[lo:hi + 1]
                target[valid] = interp_vals[valid]
                self.modified_f0[lo:hi + 1] = target
            self.last_edit_idx = idx
            self.last_edit_freq = float(y_freq)
        plot_data = self.modified_f0.copy()
        plot_data[plot_data == 0] = np.nan
        self.line_mod.set_ydata(plot_data)
        self.canvas.draw_idle()

    def restore_pitch_data(self, x_time):
        if x_time is None:
            return
        if len(self.times) == 0:
            return
        idx = int((np.abs(self.times - x_time)).argmin())
        if self.last_edit_idx is None:
            lo = idx
            hi = idx
        else:
            lo = min(self.last_edit_idx, idx)
            hi = max(self.last_edit_idx, idx)
        self.modified_f0[lo:hi + 1] = self.original_f0[lo:hi + 1]
        self.last_edit_idx = idx
        self.last_edit_freq = float(self.modified_f0[idx]) if self.modified_f0[idx] > 0 else None
        plot_data = self.modified_f0.copy()
        plot_data[plot_data == 0] = np.nan
        self.line_mod.set_ydata(plot_data)
        self.canvas.draw_idle()

    def pan_x_axis(self, x_time):
        if self.snd is None:
            return
        if self.pan_start_x is None or self.pan_start_xlim is None or x_time is None:
            return
        left0, right0 = self.pan_start_xlim
        span = right0 - left0
        if span <= 0:
            return
        shift = x_time - self.pan_start_x
        new_left = left0 - shift
        new_right = right0 - shift
        snd_left = float(self.snd.xmin)
        snd_right = float(self.snd.xmax)
        if new_left < snd_left:
            new_left = snd_left
            new_right = new_left + span
        if new_right > snd_right:
            new_right = snd_right
            new_left = new_right - span
        self.current_xlim = (new_left, new_right)
        self.draw_orig_wave_content()
        self.draw_synth_wave_content()
        self.ax_pitch.set_xlim(self.current_xlim)
        self.update_comparison_plot()
        self.canvas.draw_idle()
        self.snd_part = self.service.get_sound_part(new_left, new_right)

    def synthesize_sound(self):
        if self.snd is None: return
        xmin, xmax = self.ax_pitch.get_xlim()
        
        try:
            try:
                speed_val = float(self.input_speed.text())
            except:
                speed_val = 1.0
            
            self.synth_snd = self.service.synthesize(self.modified_f0, xmin, xmax, speed_val)
            
            self.draw_synth_wave_content()
            self.canvas.draw()
            
            self.btn_play_synth.setEnabled(True)
            self.btn_save_audio.setEnabled(True)
            QMessageBox.information(self, "完成", "合成完毕！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"合成失败: {str(e)}")

    def _get_current_batch_files(self):
        if not self.snd_path:
            return []
        folder = os.path.dirname(self.snd_path)
        stem, _ = os.path.splitext(os.path.basename(self.snd_path))
        xmin, xmax = self.ax_pitch.get_xlim()
        base_pattern = f"{stem}_{xmin:.2f}_{xmax:.2f}_modified"
        return sorted(glob.glob(os.path.join(folder, f"{base_pattern}_*.wav")))

    def delete_current_batch_files(self):
        files = self._get_current_batch_files()
        if not files:
            QMessageBox.information(self, "提示", "当前批次未找到可删除的文件")
            return
        ok = QMessageBox.question(self, "确认删除", f"将删除 {len(files)} 个文件，是否继续？")
        if ok != QMessageBox.StandardButton.Yes:
            return
        deleted = 0
        for f in files:
            try:
                os.remove(f)
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
        QMessageBox.information(self, "完成", f"已删除 {deleted} 个文件")
        self.update_comparison_plot()
        self.canvas.draw_idle()

    def rename_current_batch_files(self):
        files = self._get_current_batch_files()
        if not files:
            QMessageBox.information(self, "提示", "当前批次未找到可重命名的文件")
            return
        basenames = [os.path.basename(p) for p in files]
        common = os.path.commonprefix(basenames)
        if not common:
            QMessageBox.warning(self, "提示", "这些文件没有共同前缀，无法批量重命名")
            return
        new_prefix, ok = QInputDialog.getText(self, "批量重命名", f"当前共同前缀:\n{common}\n请输入新的共同前缀：", text=common)
        if not ok:
            return
        folder = os.path.dirname(self.snd_path)
        mappings = []
        for old in basenames:
            new = new_prefix + old[len(common):]
            mappings.append((os.path.join(folder, old), os.path.join(folder, new)))
        conflicts = [n for _, n in mappings if os.path.exists(n) and n not in [o for o, _ in mappings]]
        if conflicts:
            QMessageBox.warning(self, "提示", "新文件名与现有文件冲突，请修改共同前缀")
            return
        renamed = 0
        try:
            for old, new in mappings:
                os.rename(old, new)
                renamed += 1
            QMessageBox.information(self, "完成", f"已重命名 {renamed} 个文件")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重命名失败: {e}")
        self.update_comparison_plot()
        self.canvas.draw_idle()

    def add_knot(self):
        try:
            t_text = self.input_knot_time.text().strip()
            if not t_text:
                QMessageBox.warning(self, "提示", "请先输入拐点时间")
                return
            kt = float(t_text)
            fl = parse_float_list(self.input_knot_freqs.text())
            if not fl:
                QMessageBox.warning(self, "提示", "拐点频率列表不能为空")
                return
            self.knot_points.append({"time": kt, "freqs": fl})
            self.knot_points.sort(key=lambda x: x["time"])
            self.lbl_knots_summary.setText(f"拐点: {len(self.knot_points)}")
            self.input_knot_time.clear()
            self.input_knot_freqs.clear()
        except Exception as e:
            QMessageBox.warning(self, "提示", f"参数错误: {e}")

    def open_edit_knots(self):
        t1 = 0.0
        try:
            t1 = float(self.input_batch_t1.text()) if self.input_batch_t1.text().strip() else 0.0
        except: pass
        
        t2 = 0.0
        try:
            t2 = float(self.input_batch_t2.text()) if self.input_batch_t2.text().strip() else 0.0
        except: pass
        
        dlg = KnotEditorDialog(
            self, 
            t1=t1, 
            f1_text=self.input_batch_f1_list.text(),
            start_mode=self.start_mode,
            t2=t2,
            f2_text=self.input_batch_f2_list.text(),
            end_mode=self.end_mode,
            knot_points=self.knot_points,
            knot_modes=self.knot_modes
        )
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.result_data
            if data:
                self.input_batch_t1.setText(f"{data['t1']}")
                self.input_batch_t2.setText(f"{data['t2']}")
                self.input_batch_f1_list.setText(",".join([str(x) for x in data['f1']]))
                self.input_batch_f2_list.setText(",".join([str(x) for x in data['f2']]))
                
                self.knot_points = data['knots']
                self.knot_modes = data['knot_modes']
                self.start_mode = data['start_mode']
                self.end_mode = data['end_mode']
                
                self.lbl_knots_summary.setText(f"拐点: {len(self.knot_points)}")
                
                self.draw_pitch_curve_content()
                self.update_comparison_plot()
                self.canvas.draw_idle()

    def clear_knots(self):
        self.knot_points = []
        self.lbl_knots_summary.setText("拐点: 0")

    def batch_linear_save(self):
        if self.snd is None: return
        try:
            t1 = float(self.input_batch_t1.text())
            t2 = float(self.input_batch_t2.text())
            f1_list = parse_float_list(self.input_batch_f1_list.text())
            f2_list = parse_float_list(self.input_batch_f2_list.text())
            
            xmin, xmax = self.ax_pitch.get_xlim()
            offset_mode = self.checkbox_offset.isChecked()
            
            msg = self.service.batch_generate(
                times=self.times,
                original_f0=self.original_f0,
                xmin=xmin,
                xmax=xmax,
                t1=t1,
                t2=t2,
                f1_list=f1_list,
                f2_list=f2_list,
                knot_points=self.knot_points,
                start_mode=self.start_mode,
                end_mode=self.end_mode,
                knot_modes=self.knot_modes,
                offset_mode=offset_mode
            )
            
            QMessageBox.information(self, "完成", msg)
            self.update_comparison_plot()
            self.canvas.draw_idle()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量失败: {e}")

    def save_audio_smart(self):
        if not self.synth_snd or not self.snd_path: return
        try:
            folder = os.path.dirname(self.snd_path)
            filename = os.path.basename(self.snd_path)
            stem, _ = os.path.splitext(filename)
            xmin, xmax = self.ax_pitch.get_xlim()
            
            base_pattern = f"{stem}_{xmin:.2f}_{xmax:.2f}_modified"
            search_path = os.path.join(folder, f"{base_pattern}_*.wav")
            existing_files = glob.glob(search_path)
            
            max_num = 0
            for f in existing_files:
                match = re.search(r'_(\d+)\.wav$', f)
                if match:
                    num = int(match.group(1))
                    if num > max_num: max_num = num
            
            next_num = max_num + 1
            save_name = f"{base_pattern}_{next_num}.wav"
            full_save_path = os.path.join(folder, save_name)
            
            self.synth_snd.save(full_save_path, "WAV")
            QMessageBox.information(self, "保存成功", f"文件已保存。\n对比图已自动更新！")
            
            self.update_comparison_plot()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def save_comparison_plot(self):
        if not self.snd: return
        path, _ = QFileDialog.getSaveFileName(self, "保存对比图", "comparison_plot.png", "PNG Files (*.png)")
        if path:
            try:
                temp_fig = Figure(figsize=(8, 6), dpi=150)
                temp_ax = temp_fig.add_subplot(111)
                
                apply_plot_theme(temp_ax, self.is_dark)
                
                temp_ax.set_title("历史F0对比 (History Comparison)")
                temp_ax.set_xlabel("Relative Time (s)")
                temp_ax.set_ylabel("Frequency (Hz)")
                grid_color = 'gray' if self.is_dark else '#cccccc'
                temp_ax.grid(True, linestyle=':', alpha=0.3, color=grid_color)
                
                try:
                    ymin = float(self.input_ymin.text())
                    ymax = float(self.input_ymax.text())
                    temp_ax.set_ylim([ymin, ymax])
                except: pass

                colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2'])
                linestyles = cycle(['--', '-.', ':'])
                
                folder = os.path.dirname(self.snd_path)
                stem, _ = os.path.splitext(os.path.basename(self.snd_path))
                found_files = glob.glob(os.path.join(folder, f"{stem}_*_modified_*.wav"))
                
                def get_index(fname):
                    match = re.search(r'_(\d+)\.wav$', fname)
                    return int(match.group(1)) if match else 0
                found_files.sort(key=get_index)
                
                for fpath in found_files:
                    import parselmouth
                    h_snd = parselmouth.Sound(fpath)
                    h_pitch = h_snd.to_pitch()
                    h_times = h_pitch.xs()
                    h_vals = h_pitch.selected_array['frequency']
                    h_vals[h_vals == 0] = np.nan
                    temp_ax.plot(h_times, h_vals, color=next(colors), linestyle=next(linestyles), label=f'Ver {get_index(os.path.basename(fpath))}')

                handles, labels = temp_ax.get_legend_handles_labels()
                if len(labels) <= 10:
                    legend_face = 'black' if self.is_dark else 'white'
                    legend_text = 'white' if self.is_dark else 'black'
                    temp_ax.legend(loc='upper right', facecolor=legend_face, labelcolor=legend_text)
                    
                if self.current_xlim:
                    try:
                        temp_ax.set_xlim(self.current_xlim)
                    except:
                        pass
                        
                # Set figure facecolor
                facecolor = 'black' if self.is_dark else 'white'
                temp_fig.savefig(path, facecolor=facecolor, bbox_inches='tight')
                QMessageBox.information(self, "成功", "对比图已单独保存！")
                
            except Exception as e:
                QMessageBox.critical(self, "保存图片失败", str(e))

    def import_f0_sequence(self):
        if self.snd is None or not self.current_xlim:
            return
            
        dlg = ImportF0Dialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_f0 = dlg.get_data()
            if new_f0 is None:
                return
                
            xmin, xmax = self.current_xlim
            
            mask = (self.times >= xmin) & (self.times <= xmax)
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                QMessageBox.warning(self, "错误", "当前视野内没有数据点")
                return
                
            view_f0 = self.modified_f0[indices]
            nonzero_offsets = np.where(view_f0 > 0)[0]
            
            if len(nonzero_offsets) == 0:
                QMessageBox.warning(self, "错误", "当前视野内没有有效基频（全为静音/无声）")
                return
                
            if len(nonzero_offsets) > 1:
                diffs = np.diff(nonzero_offsets)
                if np.any(diffs > 1):
                    QMessageBox.warning(self, "错误", "当前视野包含多段中断的基频曲线。\n请缩放图窗，使视野内只包含一段完整的连续基频曲线。")
                    return
            
            start_offset = nonzero_offsets[0]
            end_offset = nonzero_offsets[-1]
            
            target_len = end_offset - start_offset + 1
            target_indices = indices[start_offset : end_offset + 1]
            
            source_len = len(new_f0)
            if source_len < 2:
                 resampled_f0 = np.full(target_len, new_f0[0])
            else:
                x_old = np.linspace(0, 1, source_len)
                x_new = np.linspace(0, 1, target_len)
                resampled_f0 = np.interp(x_new, x_old, new_f0)
            
            self.modified_f0[target_indices] = resampled_f0
            
            self.draw_pitch_curve_content()
            self.canvas.draw()
            QMessageBox.information(self, "成功", "基频序列已导入并替换")

    def open_batch_tool(self):
        dlg = BatchProcessorDialog(self)
        dlg.exec()
