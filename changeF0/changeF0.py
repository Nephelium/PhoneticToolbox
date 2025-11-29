import sys
import os
import glob
import re
import numpy as np
import parselmouth
from parselmouth.praat import call
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QFileDialog, 
                             QMessageBox, QLabel, QLineEdit, QGroupBox, QRadioButton, QCheckBox,
                             QDialog, QTableWidget, QTableWidgetItem, QComboBox, QInputDialog,
                             QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from itertools import cycle, product
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 (SimHei 是黑体)
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号 (不然负号也会变成方框)

class BatchProcessorWorker(QThread):
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, folder, speed, pitch_ratio, pitch_hz):
        super().__init__()
        self.folder = folder
        self.speed = speed
        self.pitch_ratio = pitch_ratio
        self.pitch_hz = pitch_hz
        self.is_running = True

    def run(self):
        try:
            files = glob.glob(os.path.join(self.folder, "*.wav")) + \
                    glob.glob(os.path.join(self.folder, "*.mp3")) + \
                    glob.glob(os.path.join(self.folder, "*.flac"))
            
            total = len(files)
            if total == 0:
                self.finished.emit("未找到音频文件")
                return

            # 创建输出目录
            out_folder = os.path.join(self.folder, f"processed_s{self.speed}_pr{self.pitch_ratio}_ph{self.pitch_hz}")
            os.makedirs(out_folder, exist_ok=True)

            for i, fpath in enumerate(files):
                if not self.is_running:
                    break
                
                fname = os.path.basename(fpath)
                self.progress.emit(i + 1, total, f"正在处理: {fname}")
                
                try:
                    snd = parselmouth.Sound(fpath)
                    
                    # 1. 变速 (使用 parselmouth 的 lengthen，speed = 1/factor)
                    if abs(self.speed - 1.0) > 0.01:
                        factor = 1.0 / self.speed
                        snd = call(snd, "Lengthen (overlap-add)", 75.0, 600.0, factor)

                    # 2. 变调 (Shift pitch frequencies)
                    if abs(self.pitch_ratio - 1.0) > 0.01 or abs(self.pitch_hz) > 0.01:
                        manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
                        pitch_tier = call(manipulation, "Extract pitch tier")
                        
                        # Apply ratio shift
                        if abs(self.pitch_ratio - 1.0) > 0.01:
                            call(pitch_tier, "Multiply frequencies", snd.xmin, snd.xmax, self.pitch_ratio)
                        
                        # Apply Hz shift
                        if abs(self.pitch_hz) > 0.01:
                            call(pitch_tier, "Shift frequencies", snd.xmin, snd.xmax, self.pitch_hz, "Hz")
                        
                        call([pitch_tier, manipulation], "Replace pitch tier")
                        snd = call(manipulation, "Get resynthesis (overlap-add)")

                    out_path = os.path.join(out_folder, fname)
                    snd.save(out_path, "WAV")
                    
                except Exception as e:
                    print(f"Error processing {fname}: {e}")

            self.finished.emit(f"处理完成！输出目录: {out_folder}")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

class BatchProcessorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量变速变调处理")
        self.resize(500, 400)
        self.layout = QVBoxLayout(self)
        
        # 文件夹选择
        h1 = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择包含音频的文件夹")
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(self.browse_folder)
        h1.addWidget(self.path_edit)
        h1.addWidget(btn_browse)
        self.layout.addLayout(h1)
        
        # 参数设置
        gbox = QGroupBox("处理参数")
        gl = QGridLayout()
        
        self.speed_edit = QLineEdit("1.0")
        gl.addWidget(QLabel("语速倍率 (1.0为原速):"), 0, 0)
        gl.addWidget(self.speed_edit, 0, 1)
        
        self.pitch_ratio_edit = QLineEdit("1.0")
        gl.addWidget(QLabel("音高倍率 (1.0为原调):"), 1, 0)
        gl.addWidget(self.pitch_ratio_edit, 1, 1)
        
        self.pitch_hz_edit = QLineEdit("0.0")
        gl.addWidget(QLabel("音高偏移 (Hz):"), 2, 0)
        gl.addWidget(self.pitch_hz_edit, 2, 1)
        
        gbox.setLayout(gl)
        self.layout.addWidget(gbox)
        
        # 进度显示
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.layout.addWidget(self.log_view)
        
        # 按钮
        h2 = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        h2.addWidget(self.btn_start)
        h2.addWidget(self.btn_close)
        self.layout.addLayout(h2)
        
        self.worker = None

    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if d:
            self.path_edit.setText(d)

    def log(self, msg):
        self.log_view.append(msg)

    def start_processing(self):
        folder = self.path_edit.text().strip()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "提示", "请选择有效的文件夹")
            return
            
        try:
            speed = float(self.speed_edit.text())
            pr = float(self.pitch_ratio_edit.text())
            ph = float(self.pitch_hz_edit.text())
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的数字参数")
            return
            
        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("开始批量处理...")
        
        self.worker = BatchProcessorWorker(folder, speed, pr, ph)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, current, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.log(msg)

    def on_finished(self, msg):
        self.log(msg)
        QMessageBox.information(self, "完成", msg)
        self.btn_start.setEnabled(True)

    def on_error(self, msg):
        self.log(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
        self.btn_start.setEnabled(True)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)

class DraggablePitchEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("基频实验室")
        self.resize(1400, 900)

        # --- 数据存储 ---
        self.snd_path = ""
        self.snd = None       
        self.snd_part = None 
        self.start_mode = "order"   # full | order | reverse | constant
        self.end_mode = "order"
        self.knot_modes = []        # per-knot modes aligned to self.knot_points
        
        self.times = []
        self.original_f0 = []
        self.modified_f0 = [] 
        self.synth_snd = None 
        
        self.ref_lines = [] # 存储参考线的值 [freq1, freq2, ...]
        self.knot_points = []
        
        # 交互状态
        self.is_drawing = False
        self.current_xlim = None 
        self.history_cache = [] # 缓存历史文件路径

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局：顶部是控制面板，下部是2x2图表
        main_layout = QVBoxLayout(main_widget)

        # --- 1. 顶部控制区 (多行) ---
        controls_layout = QVBoxLayout()
        
        btn_batch_tool = QPushButton("批量变速变调工具")
        btn_batch_tool.clicked.connect(self.open_batch_tool)
        btn_batch_tool.setStyleSheet("background-color: #e1f5fe; color: #0277bd;")
        
        # 第一行：文件与播放操作
        row1_layout = QHBoxLayout()
        self.btn_load = QPushButton("加载音频")
        self.btn_load.clicked.connect(self.load_audio)
        
        self.btn_play_orig = QPushButton("播放当前视野原音")
        self.btn_play_orig.clicked.connect(lambda: self.play_audio_sd(self.snd_part))
        self.btn_play_orig.setEnabled(False)

        self.btn_synthesize = QPushButton("合成当前视野")
        self.btn_synthesize.clicked.connect(self.synthesize_sound)
        self.btn_synthesize.setEnabled(False)

        # 语速控制
        self.input_speed = QLineEdit("1.0")
        self.input_speed.setFixedWidth(50)
        self.input_speed.setPlaceholderText("语速")
        row1_layout.addWidget(QLabel("语速倍率:"))
        row1_layout.addWidget(self.input_speed)

        self.btn_play_synth = QPushButton("播放合成音")
        self.btn_play_synth.clicked.connect(lambda: self.play_audio_sd(self.synth_snd))
        self.btn_play_synth.setEnabled(False)

        self.btn_save_audio = QPushButton("保存并编号")
        self.btn_save_audio.clicked.connect(self.save_audio_smart)
        self.btn_save_audio.setEnabled(False)
        self.btn_delete_batch = QPushButton("删除本批次音频")
        self.btn_delete_batch.clicked.connect(self.delete_current_batch_files)
        self.btn_rename_batch = QPushButton("批量重命名")
        self.btn_rename_batch.clicked.connect(self.rename_current_batch_files)
        
        row1_layout.addWidget(self.btn_load)
        row1_layout.addWidget(self.btn_play_orig)
        row1_layout.addWidget(self.btn_synthesize)
        row1_layout.addWidget(self.btn_play_synth)
        row1_layout.addWidget(self.btn_save_audio)
        row1_layout.addWidget(self.btn_delete_batch)
        row1_layout.addWidget(self.btn_rename_batch)
        
        # 第二行：参数设置 (Y轴范围 & 参考线)
        row2_layout = QHBoxLayout()
        
        # Y轴范围设置组
        grp_axis = QGroupBox("Y轴范围设置 (Hz)")
        layout_axis = QHBoxLayout()
        self.input_ymin = QLineEdit("50")
        self.input_ymin.setFixedWidth(60)
        self.input_ymax = QLineEdit("350")
        self.input_ymax.setFixedWidth(60)
        btn_set_axis = QPushButton("应用范围")
        btn_set_axis.clicked.connect(self.update_axis_range)
        
        layout_axis.addWidget(QLabel("Min:"))
        layout_axis.addWidget(self.input_ymin)
        layout_axis.addWidget(QLabel("Max:"))
        layout_axis.addWidget(self.input_ymax)
        layout_axis.addWidget(btn_set_axis)
        grp_axis.setLayout(layout_axis)
        
        # 参考线设置组
        grp_ref = QGroupBox("参考线工具")
        layout_ref = QHBoxLayout()
        self.input_ref = QLineEdit("200")
        self.input_ref.setFixedWidth(60)
        self.input_ref.setPlaceholderText("频率")
        btn_add_ref = QPushButton("添加参考线")
        btn_add_ref.clicked.connect(self.add_ref_line)
        btn_clear_ref = QPushButton("清除所有")
        btn_clear_ref.clicked.connect(self.clear_ref_lines)
        
        layout_ref.addWidget(self.input_ref)
        layout_ref.addWidget(btn_add_ref)
        layout_ref.addWidget(btn_clear_ref)
        grp_ref.setLayout(layout_ref)
        
        # 对比图保存按钮
        self.btn_save_compare = QPushButton("保存对比图")
        self.btn_save_compare.clicked.connect(self.save_comparison_plot)
        self.btn_save_compare.setEnabled(False)
        self.btn_save_compare.setStyleSheet("background-color: #e1f5fe; color: #0277bd;")

        row2_layout.addWidget(grp_axis)
        row2_layout.addWidget(grp_ref)
        row2_layout.addWidget(self.btn_save_compare)
        row2_layout.addWidget(btn_batch_tool)
        row2_layout.addStretch() # 弹簧，把按钮顶到左边

        controls_layout.addLayout(row1_layout)
        controls_layout.addLayout(row2_layout)
        
        
        
        row5_layout = QHBoxLayout()
        grp_batch = QGroupBox("批量改变基频")
        layout_batch = QVBoxLayout()
        rowA = QHBoxLayout()
        rowB = QHBoxLayout()
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
        rowA.addWidget(QLabel("终止t(s):"))
        rowA.addWidget(self.input_batch_t2)
        rowA.addWidget(QLabel("起始F0列表:"))
        rowA.addWidget(self.input_batch_f1_list)
        rowA.addWidget(QLabel("终止F0列表:"))
        rowA.addWidget(self.input_batch_f2_list)
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
        rowB.addWidget(QLabel("拐点F0列表:"))
        rowB.addWidget(self.input_knot_freqs)
        rowB.addWidget(btn_add_knot)
        rowB.addWidget(btn_clear_knots)
        rowB.addWidget(btn_edit_knots)
        rowB.addWidget(self.lbl_knots_summary)
        rowB.addWidget(self.radio_linear)
        rowB.addWidget(self.checkbox_offset)
        rowB.addWidget(self.btn_batch_linear_save)
        layout_batch.addLayout(rowA)
        layout_batch.addLayout(rowB)
        grp_batch.setLayout(layout_batch)
        row5_layout.addWidget(grp_batch)
        controls_layout.addLayout(row5_layout)
        main_layout.addLayout(controls_layout)

        # --- 2. 绘图区 (2x2 Grid) ---
        self.figure = Figure(figsize=(12, 10), dpi=100, facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # 初始化四个子图
        # (0,0) 原音波形  (0,1) 基频编辑
        # (1,0) 合成波形  (1,1) 历史对比
        self.gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
        
        self.ax_wave_orig = self.figure.add_subplot(self.gs[0, 0])
        self.ax_pitch = self.figure.add_subplot(self.gs[0, 1])
        self.ax_wave_synth = self.figure.add_subplot(self.gs[1, 0])
        self.ax_compare = self.figure.add_subplot(self.gs[1, 1])

        # 连接事件
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.reset_plots()

    def apply_dark_theme(self, ax):
        """应用深色主题到坐标轴"""
        ax.set_facecolor('black')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    def reset_plots(self):
        """初始化/清空图表样式"""
        # 1. 原音波形
        self.ax_wave_orig.clear()
        self.apply_dark_theme(self.ax_wave_orig)
        self.ax_wave_orig.set_title("原始波形 (Original Waveform)", fontsize=10)
        self.ax_wave_orig.set_xticks([]) # 隐藏x轴刻度以简洁
        
        # 2. 基频编辑
        self.ax_pitch.clear()
        self.apply_dark_theme(self.ax_pitch)
        self.ax_pitch.set_title("基频编辑 (Pitch Editor) - 滚轮缩放/拖拽修改", fontsize=10)
        self.ax_pitch.set_ylabel("Frequency (Hz)")
        self.ax_pitch.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 3. 合成波形
        self.ax_wave_synth.clear()
        self.apply_dark_theme(self.ax_wave_synth)
        self.ax_wave_synth.set_title("合成波形 (Synthesized Waveform)", fontsize=10)
        self.ax_wave_synth.set_xlabel("Time (s)")
        
        # 4. 历史对比
        self.ax_compare.clear()
        self.apply_dark_theme(self.ax_compare)
        self.ax_compare.set_title("历史F0对比 (History Comparison) - 0起点对齐", fontsize=10)
        self.ax_compare.set_xlabel("Relative Time (s)")
        self.ax_compare.grid(True, linestyle=':', alpha=0.3, color='gray')

        self.canvas.draw()

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "Audio Files (*.wav *.mp3 *.flac)")
        if not file_path:
            return

        try:
            self.snd_path = file_path
            self.snd = parselmouth.Sound(file_path)
            self.snd_part = self.snd 
            
            # 提取基频
            pitch = self.snd.to_pitch(time_step=0.01)
            self.times = pitch.xs()
            self.original_f0 = pitch.selected_array['frequency']
            self.modified_f0 = self.original_f0.copy()

            # 设置初始视野范围
            self.current_xlim = (self.snd.xmin, self.snd.xmax)
            
            # 启用按钮
            self.btn_play_orig.setEnabled(True)
            self.btn_synthesize.setEnabled(True)
            self.btn_play_synth.setEnabled(False)
            self.btn_save_audio.setEnabled(False)
            self.btn_save_compare.setEnabled(True)
            self.btn_batch_linear_save.setEnabled(True)

            # 绘制
            self.draw_all()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载音频失败: {str(e)}")

    def update_axis_range(self):
        """更新Y轴范围"""
        try:
            ymin = float(self.input_ymin.text())
            ymax = float(self.input_ymax.text())
            if ymin >= ymax:
                raise ValueError("Min must be less than Max")
            self.ax_pitch.set_ylim([ymin, ymax])
            self.ax_compare.set_ylim([ymin, ymax]) # 对比图也同步
            self.canvas.draw_idle()
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的数字范围")

    def add_ref_line(self):
        """添加参考线"""
        try:
            val = float(self.input_ref.text())
            self.ref_lines.append(val)
            self.draw_pitch_curve_content() # 仅重绘基频图内容
            self.canvas.draw_idle()
        except ValueError:
            pass

    def clear_ref_lines(self):
        """清除参考线"""
        self.ref_lines = []
        self.draw_pitch_curve_content()
        self.canvas.draw_idle()

    def draw_all(self):
        """绘制所有静态图表"""
        if self.snd is None: return

        # 1. 原始波形
        self.ax_wave_orig.clear()
        self.apply_dark_theme(self.ax_wave_orig)
        self.ax_wave_orig.plot(self.snd.xs(), self.snd.values.T, color='white', alpha=0.6, linewidth=0.5)
        self.ax_wave_orig.set_title("原始波形")
        self.ax_wave_orig.set_xticks([])
        if self.current_xlim: self.ax_wave_orig.set_xlim(self.current_xlim)

        # 2. 基频编辑图 (拆分内容绘制以便复用)
        self.draw_pitch_curve_content()
        
        # 3. 合成波形
        self.draw_synth_wave_content()

        # 4. 历史对比图
        self.update_comparison_plot()

        self.canvas.draw()

    def draw_pitch_curve_content(self):
        """仅重绘基频编辑图的内容"""
        self.ax_pitch.clear()
        self.apply_dark_theme(self.ax_pitch)
        
        # 绘制原始
        plot_orig = self.original_f0.copy()
        plot_orig[plot_orig == 0] = np.nan
        self.ax_pitch.plot(self.times, plot_orig, color='cyan', linestyle=':', label='Original', alpha=0.5)

        # 绘制当前修改
        plot_mod = self.modified_f0.copy()
        plot_mod[plot_mod == 0] = np.nan
        self.line_mod, = self.ax_pitch.plot(self.times, plot_mod, color='red', linewidth=2, label='Modified')
        
        # 绘制参考线
        for ref in self.ref_lines:
            self.ax_pitch.axhline(y=ref, color='gray', linestyle='--', alpha=0.8)
            self.ax_pitch.text(self.times[0], ref, f"{int(ref)}Hz", color='white', fontsize=8, verticalalignment='bottom')

        self.ax_pitch.set_title("基频编辑")
        self.ax_pitch.set_ylabel("Frequency (Hz)")
        self.ax_pitch.grid(True, linestyle='--', alpha=0.3, color='gray')
        self.ax_pitch.legend(loc='upper right', facecolor='black', labelcolor='white')
        
        # 应用范围
        self.update_axis_range()
        
        if self.current_xlim: self.ax_pitch.set_xlim(self.current_xlim)

    def draw_synth_wave_content(self):
        self.ax_wave_synth.clear()
        self.apply_dark_theme(self.ax_wave_synth)
        if self.synth_snd:
            self.ax_wave_synth.plot(self.synth_snd.xs(), self.synth_snd.values.T, color='lime', alpha=0.6, linewidth=0.5)
        self.ax_wave_synth.set_title("合成波形")
        self.ax_wave_synth.set_xlabel("Time (s)")
        if self.current_xlim: self.ax_wave_synth.set_xlim(self.current_xlim)

    def update_comparison_plot(self):
        """实时绘制历史F0对比（核心功能）"""
        self.ax_compare.clear()
        self.apply_dark_theme(self.ax_compare)
        self.ax_compare.set_title("历史F0对比")
        self.ax_compare.set_xlabel("Time (s)")
        self.ax_compare.grid(True, linestyle=':', alpha=0.3, color='gray')
        
        # 样式循环：颜色和线型
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2'])
        linestyles = cycle(['--', '-.', ':'])

        # 不再绘制当前编辑曲线，仅显示历史版本
                    
        # 2. 搜索并加载历史文件
        if self.snd_path:
            folder = os.path.dirname(self.snd_path)
            filename = os.path.basename(self.snd_path)
            stem, _ = os.path.splitext(filename)
            
            # 构造当前视野的搜索模式，只有相同切片的历史记录才有对比意义
            # 浮点数匹配比较麻烦，这里我们简化：搜索所有该文件生成的 modified 文件
            # 并在循环中判断时间范围是否和当前视野大致匹配（可选），或者全部画出来
            # 井井的需求是"除了最后的数字不同，前面全相同"，说明只对比当前这个切片生成的历史
            
            if self.current_xlim:
                # 构造精确的文件名前缀
                search_path = os.path.join(folder, f"{stem}_*_modified_*.wav")
                found_files = glob.glob(search_path)
                
                # 排序
                def get_index(fname):
                    match = re.search(r'_(\d+)\.wav$', fname)
                    return int(match.group(1)) if match else 0
                found_files.sort(key=get_index)
                
                # 绘制每一个历史文件
                for fpath in found_files:
                    try:
                        # 读取
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
            legend_count = 0
            handles, labels = self.ax_compare.get_legend_handles_labels()
            legend_count = len(labels)
            if legend_count <= 10:
                self.ax_compare.legend(loc='upper right', fontsize='small', framealpha=0.5, facecolor='black', labelcolor='white')
        except:
            pass
        
        # 同步坐标轴范围
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
        """鼠标滚轮缩放X轴"""
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
        
        # 同步更新
        self.ax_wave_orig.set_xlim(self.current_xlim)
        self.ax_pitch.set_xlim(self.current_xlim)
        self.ax_wave_synth.set_xlim(self.current_xlim)
        
        # 视野改变时，对比图也需要刷新（因为对比图只显示当前切片的历史）
        self.update_comparison_plot()
        
        self.canvas.draw_idle()
        self.snd_part = self.snd.extract_part(from_time=new_xmin, to_time=new_xmax, preserve_times=True)

    def on_mouse_press(self, event):
        if event.inaxes == self.ax_pitch and event.button == 1:
            self.is_drawing = True
            self.update_pitch_data(event.xdata, event.ydata)

    def on_mouse_move(self, event):
        if self.is_drawing and event.inaxes == self.ax_pitch:
            self.update_pitch_data(event.xdata, event.ydata)

    def on_mouse_release(self, event):
        self.is_drawing = False

    def update_pitch_data(self, x_time, y_freq):
        if x_time is None or y_freq is None: return
        idx = (np.abs(self.times - x_time)).argmin()
        if self.original_f0[idx] > 0:
            self.modified_f0[idx] = y_freq
            plot_data = self.modified_f0.copy()
            plot_data[plot_data == 0] = np.nan
            self.line_mod.set_ydata(plot_data)
            self.canvas.draw_idle()
            
            # 实时拖拽时不建议疯狂重绘对比图（太卡），可以在释放鼠标时重绘，或者这里不重绘对比图

    def synthesize_sound(self):
        """合成当前视野"""
        if self.snd is None: return
        xmin, xmax = self.ax_pitch.get_xlim()
        
        try:
            part_snd = self.snd.extract_part(from_time=xmin, to_time=xmax, preserve_times=True)
            new_pitch_tier = call("Create PitchTier", "modified", xmin, xmax)
            
            mask = (self.times >= xmin) & (self.times <= xmax)
            part_times = self.times[mask]
            part_f0 = self.modified_f0[mask]
            
            for t, f in zip(part_times, part_f0):
                if f > 0:
                    call(new_pitch_tier, "Add point", t, f)
            
            manipulation = call(part_snd, "To Manipulation", 0.01, 75, 600)
            call([manipulation, new_pitch_tier], "Replace pitch tier")
            
            # Speed Logic
            try:
                speed_val = float(self.input_speed.text())
            except:
                speed_val = 1.0
            
            if abs(speed_val - 1.0) > 0.01:
                factor = 1.0 / speed_val
                duration_tier = call("Create DurationTier", "duration", xmin, xmax)
                call(duration_tier, "Add point", xmin, factor)
                call([manipulation, duration_tier], "Replace duration tier")

            self.synth_snd = call(manipulation, "Get resynthesis (overlap-add)")
            self.draw_synth_wave_content()
            self.canvas.draw()
            
            self.btn_play_synth.setEnabled(True)
            self.btn_save_audio.setEnabled(True)
            QMessageBox.information(self, "完成", "合成完毕！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"合成失败: {str(e)}")

    def play_audio_sd(self, snd_obj):
        if snd_obj is None: return
        try:
            fs = snd_obj.sampling_frequency
            data = snd_obj.values.T 
            sd.play(data, fs)
        except Exception as e:
            QMessageBox.critical(self, "播放错误", str(e))

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

    def parse_float_list(self, s):
        try:
            return [float(x) for x in re.split(r"[,，\s]+", s.strip()) if x != ""]
        except:
            return []

    def add_knot(self):
        try:
            t_text = self.input_knot_time.text().strip()
            if not t_text:
                QMessageBox.warning(self, "提示", "请先输入拐点时间")
                return
            kt = float(t_text)
            fl = self.parse_float_list(self.input_knot_freqs.text())
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
        dlg = QDialog(self)
        dlg.setWindowTitle("编辑拐点")
        vbox = QVBoxLayout(dlg)
        # 表格：起点 + 拐点们 + 终点
        total_rows = 1 + len(self.knot_points) + 1
        table = QTableWidget(total_rows, 3, dlg)
        table.setHorizontalHeaderLabels(["时间(s)", "频率列表(Hz,逗号)", "连接方式"])
        # 起点行
        try:
            t1 = float(self.input_batch_t1.text()) if self.input_batch_t1.text().strip() else 0.0
        except:
            t1 = 0.0
        table.setItem(0, 0, QTableWidgetItem(f"{t1:.3f}"))
        table.setItem(0, 1, QTableWidgetItem(self.input_batch_f1_list.text()))
        cb0 = QComboBox(dlg); cb0.addItems(["全连接", "顺序", "逆序", "常量"])
        cb0.setCurrentIndex({"full":0, "order":1, "reverse":2, "constant":3}.get(self.start_mode, 1))
        table.setCellWidget(0, 2, cb0)
        # 拐点行
        for i, kp in enumerate(self.knot_points):
            row = 1 + i
            table.setItem(row, 0, QTableWidgetItem(f"{kp['time']:.3f}"))
            table.setItem(row, 1, QTableWidgetItem(
                ",".join([str(x) for x in kp['freqs']])
            ))
            cb = QComboBox(dlg); cb.addItems(["全连接", "顺序", "逆序", "常量"])
            mode = self.knot_modes[i] if i < len(self.knot_modes) else "order"
            cb.setCurrentIndex({"full":0, "order":1, "reverse":2, "constant":3}.get(mode, 1))
            table.setCellWidget(row, 2, cb)
        # 终点行
        try:
            t2 = float(self.input_batch_t2.text()) if self.input_batch_t2.text().strip() else 0.0
        except:
            t2 = 0.0
        last_row = total_rows - 1
        table.setItem(last_row, 0, QTableWidgetItem(f"{t2:.3f}"))
        table.setItem(last_row, 1, QTableWidgetItem(self.input_batch_f2_list.text()))
        cbn = QComboBox(dlg); cbn.addItems(["全连接", "顺序", "逆序", "常量"])
        cbn.setCurrentIndex({"full":0, "order":1, "reverse":2, "constant":3}.get(self.end_mode, 1))
        table.setCellWidget(last_row, 2, cbn)
        btn_row_add = QPushButton("添加行", dlg)
        btn_row_del = QPushButton("删除选中", dlg)
        btn_save = QPushButton("保存更改", dlg)
        btn_close = QPushButton("关闭", dlg)
        hbox = QHBoxLayout()
        hbox.addWidget(btn_row_add)
        hbox.addWidget(btn_row_del)
        hbox.addWidget(btn_save)
        hbox.addWidget(btn_close)
        vbox.addWidget(table)
        vbox.addLayout(hbox)

        def add_row():
            # 在终点前插入为新的拐点
            end_row = table.rowCount() - 1
            table.insertRow(end_row)
            table.setItem(end_row, 0, QTableWidgetItem(""))
            table.setItem(end_row, 1, QTableWidgetItem(""))
            cb = QComboBox(dlg); cb.addItems(["全连接", "顺序", "逆序", "常量"])
            cb.setCurrentIndex(1)
            table.setCellWidget(end_row, 2, cb)
        def del_rows():
            rows = set([idx.row() for idx in table.selectedIndexes()])
            # 不允许删除起点与终点
            rows = [r for r in rows if r not in (0, table.rowCount()-1)]
            for r in sorted(rows, reverse=True):
                table.removeRow(r)
        def save_changes():
            # 起点
            t1_text = table.item(0,0).text().strip() if table.item(0,0) else ""
            f1_text = table.item(0,1).text().strip() if table.item(0,1) else ""
            mode0 = table.cellWidget(0,2).currentText()
            # 终点
            last = table.rowCount()-1
            t2_text = table.item(last,0).text().strip() if table.item(last,0) else ""
            f2_text = table.item(last,1).text().strip() if table.item(last,1) else ""
            moden = table.cellWidget(last,2).currentText()

            try:
                t1v = float(t1_text)
                t2v = float(t2_text)
                f1v = self.parse_float_list(f1_text)
                f2v = self.parse_float_list(f2_text)
                if not f1v or not f2v:
                    raise ValueError("起止频率列表不能为空")
                if t2v <= t1v:
                    raise ValueError("终止时间必须大于起始时间")
            except Exception as e:
                QMessageBox.warning(self, "提示", f"起止点错误: {e}")
                return

            new_knots = []
            new_modes = []
            for r in range(1, last):
                t_item = table.item(r, 0)
                f_item = table.item(r, 1)
                mode_widget = table.cellWidget(r, 2)
                t_text = t_item.text().strip() if t_item else ""
                f_text = f_item.text().strip() if f_item else ""
                try:
                    kt = float(t_text)
                    fl = self.parse_float_list(f_text)
                    if not fl:
                        raise ValueError("频率列表为空")
                    new_knots.append({"time": kt, "freqs": fl})
                    new_modes.append({"mode": {"全连接":"full","顺序":"order","逆序":"reverse","常量":"constant"}[mode_widget.currentText()]})
                except Exception as e:
                    QMessageBox.warning(self, "提示", f"第{r+1}行错误: {e}")
                    return
            # 排序并应用
            order = np.argsort([k['time'] for k in new_knots])
            new_knots = [new_knots[i] for i in order]
            new_modes = [new_modes[i]["mode"] for i in order]
            # 检查时间重复且在(t1,t2)内
            times = [k['time'] for k in new_knots]
            if len(set(times)) != len(times):
                QMessageBox.warning(self, "提示", "拐点时间不能重复")
                return
            for kt in times:
                if kt <= t1v or kt >= t2v:
                    QMessageBox.warning(self, "提示", "拐点时间需在起止时间之间")
                    return
            # 写回数据
            self.input_batch_t1.setText(f"{t1v}")
            self.input_batch_t2.setText(f"{t2v}")
            self.input_batch_f1_list.setText(",".join([str(x) for x in f1v]))
            self.input_batch_f2_list.setText(",".join([str(x) for x in f2v]))
            self.knot_points = new_knots
            self.knot_modes = new_modes
            self.start_mode = {"全连接":"full","顺序":"order","逆序":"reverse","常量":"constant"}[mode0]
            self.end_mode = {"全连接":"full","顺序":"order","逆序":"reverse","常量":"constant"}[moden]
            self.lbl_knots_summary.setText(f"拐点: {len(self.knot_points)}")
            dlg.accept()
            self.draw_pitch_curve_content()
            self.update_comparison_plot()
            self.canvas.draw_idle()

        btn_row_add.clicked.connect(add_row)
        btn_row_del.clicked.connect(del_rows)
        btn_save.clicked.connect(save_changes)
        btn_close.clicked.connect(dlg.reject)
        dlg.exec()

    def clear_knots(self):
        self.knot_points = []
        self.lbl_knots_summary.setText("拐点: 0")

    def batch_linear_save(self):
        if self.snd is None: return
        try:
            t1 = float(self.input_batch_t1.text())
            t2 = float(self.input_batch_t2.text())
            f1_list = self.parse_float_list(self.input_batch_f1_list.text())
            f2_list = self.parse_float_list(self.input_batch_f2_list.text())
            if t2 <= t1:
                raise ValueError("终止时间必须大于起始时间")
            if not f1_list or not f2_list:
                raise ValueError("起止基频列表不能为空")
            xmin, xmax = self.ax_pitch.get_xlim()
            part_snd = self.snd.extract_part(from_time=xmin, to_time=xmax, preserve_times=True)
            folder = os.path.dirname(self.snd_path)
            stem, _ = os.path.splitext(os.path.basename(self.snd_path))
            base_pattern = f"{stem}_{xmin:.2f}_{xmax:.2f}_modified"
            existing = glob.glob(os.path.join(folder, f"{base_pattern}_*.wav"))
            def get_index(fname):
                m = re.search(r"_(\d+)\.wav$", fname)
                return int(m.group(1)) if m else 0
            cur_max = max([get_index(f) for f in existing], default=0)
            idx_counter = cur_max
            knot_times = [kp["time"] for kp in self.knot_points]
            knot_freq_lists = [kp["freqs"] for kp in self.knot_points]
            all_times = [t1] + knot_times + [t2]
            all_lists = [f1_list] + knot_freq_lists + [f2_list]
            for kt in knot_times:
                if kt <= t1 or kt >= t2:
                    raise ValueError("拐点时间需在起止时间之间")
            if len(set(all_times)) != len(all_times):
                raise ValueError("时间点不能重复")
            # 组合构建依据每点的连接方式：起点、各拐点、终点
            point_lists = [f1_list] + knot_freq_lists + [f2_list]
            point_modes = [self.start_mode] + [m if i < len(self.knot_modes) else "order" for i, m in enumerate(self.knot_modes + [None]*max(0, len(knot_freq_lists)-len(self.knot_modes)))] + [self.end_mode]
            # 将常量模式折叠为单值
            processed_lists = []
            for lst, mode in zip(point_lists, point_modes):
                if mode == "constant":
                    if not lst:
                        raise ValueError("常量模式频率列表不能为空")
                    processed_lists.append([lst[0]])
                else:
                    processed_lists.append(lst)
            # 计算对角长度
            diag_indices = [i for i, m in enumerate(point_modes) if m in ("order", "reverse")]
            if diag_indices:
                base_len = len(processed_lists[diag_indices[0]])
                for di in diag_indices[1:]:
                    if len(processed_lists[di]) != base_len:
                        raise ValueError("对角模式的列表长度必须一致")
            else:
                base_len = 1
            # 全连接维度
            full_indices = [i for i, m in enumerate(point_modes) if m == "full" and len(processed_lists[i]) > 1]
            full_product_source = [processed_lists[i] for i in full_indices]
            full_product = list(product(*full_product_source)) if full_indices else [()]
            # 生成最终组合（保持原始点的顺序）
            combos = []
            for k in range(base_len):
                for fp in full_product:
                    fp_iter = iter(fp)
                    path = []
                    for idx in range(len(processed_lists)):
                        mode = point_modes[idx]
                        if mode in ("order", "reverse"):
                            dlst = processed_lists[idx]
                            sorted_vals = sorted(dlst)
                            if mode == "reverse":
                                sorted_vals = list(reversed(sorted_vals))
                            path.append(sorted_vals[k])
                        elif mode == "full":
                            # 若该维只有一个值，也可作为常量使用
                            if len(processed_lists[idx]) == 1:
                                path.append(processed_lists[idx][0])
                            else:
                                path.append(next(fp_iter))
                        else:  # constant
                            path.append(processed_lists[idx][0])
                    combos.append(tuple(path))
            mask_win = (self.times >= xmin) & (self.times <= xmax)
            win_times = self.times[mask_win]
            mask_seg = (self.times >= t1) & (self.times <= t2)
            seg_times = self.times[mask_seg]
            if seg_times.size == 0:
                raise ValueError("时间范围无数据")
            offset_mode = self.checkbox_offset.isChecked()
            # 已移除顺序/倒序检测；仅按连接方式构造组合
            # 线性插值实现（样条已移除）
            for i, vals in enumerate(combos, start=1):
                ctrl_vals = list(vals)
                new_pitch_tier = call("Create PitchTier", "modified", xmin, xmax)
                delta_curve = np.zeros_like(seg_times)
                for si in range(len(all_times) - 1):
                    ta = all_times[si]
                    tb = all_times[si + 1]
                    va = ctrl_vals[si]
                    vb = ctrl_vals[si + 1]
                    seg_mask = (seg_times >= ta) & (seg_times <= tb)
                    if tb - ta > 0:
                        delta_curve[seg_mask] = va + (vb - va) * (seg_times[seg_mask] - ta) / (tb - ta)
                base_curve_win = self.original_f0[mask_win].copy()
                seg_in_win = (win_times >= t1) & (win_times <= t2)
                if offset_mode:
                    seg_final = self.original_f0[mask_seg] + delta_curve
                else:
                    seg_final = delta_curve
                base_curve_win[seg_in_win] = seg_final
                for t, f in zip(win_times, base_curve_win):
                    call(new_pitch_tier, "Add point", t, f)
                for si, ta in enumerate(all_times):
                    base_val = self.original_f0[(np.abs(self.times - ta)).argmin()]
                    final_val = ctrl_vals[si] if not offset_mode else (base_val + ctrl_vals[si])
                    call(new_pitch_tier, "Add point", ta, final_val)
                manipulation = call(part_snd, "To Manipulation", 0.01, 75, 600)
                call([manipulation, new_pitch_tier], "Replace pitch tier")
                out_snd = call(manipulation, "Get resynthesis (overlap-add)")
                idx_counter += 1
                mode_tag = "lin"
                off_tag = "_offset" if offset_mode else ""
                knot_detail = ""
                if knot_times:
                    pairs = [f"k{kt:.2f}_{kv:.1f}Hz" for kt, kv in zip(knot_times, ctrl_vals[1:-1])]
                    knot_detail = "_" + "_".join(pairs)
                freq_path = "-".join([f"{v:.1f}" for v in ctrl_vals])
                tag = f"seg_{t1:.2f}-{t2:.2f}_kn{len(knot_times)}_{mode_tag}{off_tag}_Fpath_{freq_path}{knot_detail}_combo{i}"
                save_name = f"{base_pattern}_{tag}_{idx_counter}.wav"
                out_snd.save(os.path.join(folder, save_name), "WAV")
            QMessageBox.information(self, "完成", "批量插值保存完成")
            self.update_comparison_plot()
            self.canvas.draw_idle()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量失败: {e}")

    def save_audio_smart(self):
        """保存并触发对比图更新"""
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
            
            # 保存后，强制更新对比图，把刚保存的这条线画上去
            self.update_comparison_plot()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def save_comparison_plot(self):
        """保存右下角的对比图"""
        if not self.snd: return
        path, _ = QFileDialog.getSaveFileName(self, "保存对比图", "comparison_plot.png", "PNG Files (*.png)")
        if path:
            # 技巧：只保存 ax_compare 这个子图
            # 但matplotlib保存子图比较麻烦，这里保存整个窗口的截图，或者临时创建一个figure保存
            # 为了简单有效，我们利用 extent 变换，或者直接保存整个 Figure
            # 井井可能只需要那张图，我们这里用一个变通方法：
            # 将其他子图暂时隐藏，保存，再显示？不，太闪烁。
            # 我们这里保存整张大图，或者只保存右下角的部分。
            # 最稳妥的方式：新建一个不可见的Figure，把数据重新画一遍保存。
            
            try:
                # 创建临时图表用于保存，保证清晰度
                temp_fig = Figure(figsize=(8, 6), dpi=150)
                temp_ax = temp_fig.add_subplot(111)
                
                # 把 ax_compare 的内容复制过去（重新绘制一遍）
                temp_ax.set_title("历史F0对比 (History Comparison)")
                temp_ax.set_xlabel("Relative Time (s)")
                temp_ax.set_ylabel("Frequency (Hz)")
                temp_ax.grid(True, linestyle=':', alpha=0.3)
                
                # 复制 Y 轴范围
                ymin = float(self.input_ymin.text())
                ymax = float(self.input_ymax.text())
                temp_ax.set_ylim([ymin, ymax])

                # 重新执行绘制逻辑 (简化版)
                colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2'])
                linestyles = cycle(['--', '-.', ':'])
                
                # 不绘制当前编辑曲线，只绘制历史版本
                
                # 画历史线
                folder = os.path.dirname(self.snd_path)
                stem, _ = os.path.splitext(os.path.basename(self.snd_path))
                found_files = glob.glob(os.path.join(folder, f"{stem}_*_modified_*.wav"))
                
                def get_index(fname):
                    match = re.search(r'_(\d+)\.wav$', fname)
                    return int(match.group(1)) if match else 0
                found_files.sort(key=get_index)
                
                for fpath in found_files:
                    h_snd = parselmouth.Sound(fpath)
                    h_pitch = h_snd.to_pitch()
                    h_times = h_pitch.xs()
                    h_vals = h_pitch.selected_array['frequency']
                    h_vals[h_vals == 0] = np.nan
                    temp_ax.plot(h_times, h_vals, color=next(colors), linestyle=next(linestyles), label=f'Ver {get_index(os.path.basename(fpath))}')

                handles, labels = temp_ax.get_legend_handles_labels()
                if len(labels) <= 10:
                    temp_ax.legend()
                if self.current_xlim:
                    try:
                        temp_ax.set_xlim(self.current_xlim)
                    except:
                        pass
                temp_fig.savefig(path, facecolor='white', bbox_inches='tight')
                QMessageBox.information(self, "成功", "对比图已单独保存！")
                
            except Exception as e:
                QMessageBox.critical(self, "保存图片失败", str(e))

    def open_batch_tool(self):
        dlg = BatchProcessorDialog(self)
        dlg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DraggablePitchEditor()
    window.show()
    sys.exit(app.exec())
