import logging
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from phonetic_toolbox.services.acoustic_service import (
    AcousticAnalysisService,
    PARAMETER_MAPPING,
)
from phonetic_toolbox.models.config import AcousticConfig
from phonetic_toolbox.services.settings_service import SettingsService
from phonetic_toolbox.services.io.textgrid import parse_textgrid, write_textgrid, TextGrid
from phonetic_toolbox.services.io.excel import load_excel, save_excel
import scipy.io.wavfile as wavfile
import warnings
from phonetic_toolbox.gui.dialogs.settings_dialog import SettingsDialog
from phonetic_toolbox.gui.dialogs.parameter_tools_dialog import ParameterSelectionDialog, ParameterHelpDialog

log = logging.getLogger(__name__)

class PEWorker(QThread):
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)
    finished_sig = pyqtSignal()

    def __init__(self, items: List[str], input_dir: Path, output_dir: Path, 
                 lip_data_map: Dict[str, Path], config: AcousticConfig):
        super().__init__()
        self.items = items
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.lip_data_map = lip_data_map
        self.config = config
        self.service = AcousticAnalysisService()
        self._is_interrupted = False

    def run(self):
        self.service.analyze_batch(
            files=self.items,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            lip_data_map=self.lip_data_map,
            config=self.config,
            progress_callback=self._emit_progress,
            stop_check=self.isInterruptionRequested
        )
        self.finished_sig.emit()

    def _emit_progress(self, progress: int, name: str):
        self.progress_sig.emit(progress, name)

class ParameterEstimationWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.service = AcousticAnalysisService()
        self._textgrid_cache: Dict[str, TextGrid] = {}
        self._lip_data_map: Dict[str, Path] = {}
        self._selected_tier_name: Optional[str] = None
        self._selectable_param_items = [(k, v) for k, v in PARAMETER_MAPPING.items()]
        self._selected_param_keys = [k for k, _ in self._selectable_param_items]
        self._player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)
        self.is_dark = True # Default
        
        # Load last input dir
        settings = SettingsService()
        last_input = settings.get("last_input_dir")
        
        self.init_ui()
        
        if last_input and Path(last_input).exists():
            self.edit_input.setText(last_input)
            if self.chk_same_dir.isChecked():
                self.edit_output.setText(last_input)
            self._refresh_files()
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- Top: Directories ---
        dir_group = QtWidgets.QGroupBox("请选择输入/输出目录并开始处理")
        dir_layout = QtWidgets.QGridLayout()
        
        self.edit_input = QtWidgets.QLineEdit()
        self.btn_browse_input = QtWidgets.QPushButton("浏览...")
        self.edit_output = QtWidgets.QLineEdit()
        self.btn_browse_output = QtWidgets.QPushButton("浏览...")
        self.chk_same_dir = QtWidgets.QCheckBox("xlsx 与 wav 同目录")
        self.chk_same_dir.setChecked(True)
        self.edit_output.setEnabled(False)
        self.btn_browse_output.setEnabled(False)
        
        dir_layout.addWidget(QtWidgets.QLabel("输入目录"), 0, 0)
        dir_layout.addWidget(self.edit_input, 0, 1)
        dir_layout.addWidget(self.btn_browse_input, 0, 2)
        
        dir_layout.addWidget(QtWidgets.QLabel("输出目录"), 1, 0)
        dir_layout.addWidget(self.edit_output, 1, 1)
        dir_layout.addWidget(self.btn_browse_output, 1, 2)
        
        dir_layout.addWidget(self.chk_same_dir, 2, 0, 1, 3)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # --- Middle: Controls ---
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("播放选中音频")
        self.btn_stop = QtWidgets.QPushButton("停止播放")
        self.btn_refresh = QtWidgets.QPushButton("刷新文件列表")
        self.btn_settings = QtWidgets.QPushButton("设置")
        self.btn_help = QtWidgets.QPushButton("帮助")
        green_btn_style = """
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                border: 1px solid #1f7a33;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #34c759;
            }
            QPushButton:pressed {
                background-color: #1f7a33;
            }
        """
        self.btn_settings.setStyleSheet(green_btn_style)
        self.btn_help.setStyleSheet(green_btn_style)
        settings_help_widget = QtWidgets.QWidget()
        settings_help_layout = QtWidgets.QHBoxLayout(settings_help_widget)
        settings_help_layout.setContentsMargins(0, 0, 0, 0)
        settings_help_layout.setSpacing(0)
        settings_help_layout.addWidget(self.btn_settings, 1)
        settings_help_layout.addWidget(self.btn_help, 1)
        for btn in [self.btn_play, self.btn_stop, self.btn_refresh, self.btn_settings, self.btn_help]:
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        
        ctrl_layout.addWidget(self.btn_play, 1)
        ctrl_layout.addWidget(self.btn_stop, 1)
        ctrl_layout.addWidget(self.btn_refresh, 1)
        ctrl_layout.addWidget(settings_help_widget, 2)
        layout.addLayout(ctrl_layout)
        
        # --- Middle: Advanced Controls ---
        adv_layout = QtWidgets.QHBoxLayout()
        self.btn_param_select = QtWidgets.QPushButton("参数选择")
        self.btn_param_help = QtWidgets.QPushButton("参数说明")
        self.btn_read_tg = QtWidgets.QPushButton("读取TextGrid")
        self.btn_tg_seg = QtWidgets.QPushButton("TextGrid切分")
        self.btn_save_seg = QtWidgets.QPushButton("保存切分音频")
        self.btn_read_lip = QtWidgets.QPushButton("读取唇形数据")
        
        adv_layout.addWidget(self.btn_param_select)
        adv_layout.addWidget(self.btn_param_help)
        adv_layout.addWidget(self.btn_read_tg)
        adv_layout.addWidget(self.btn_tg_seg)
        adv_layout.addWidget(self.btn_save_seg)
        adv_layout.addWidget(self.btn_read_lip)
        layout.addLayout(adv_layout)
        
        # --- Content: List and Waveform ---
        content_layout = QtWidgets.QHBoxLayout()
        
        self.list_files = QtWidgets.QListWidget()
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_files.setMaximumWidth(300)
        
        # Matplotlib Canvas
        self.figure = Figure(facecolor='#1e1e1e') # Dark background match
        self.canvas = Canvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        
        content_layout.addWidget(self.list_files)
        content_layout.addWidget(self.canvas)
        layout.addLayout(content_layout)
        
        # --- Bottom: Start ---
        self.btn_start = QtWidgets.QPushButton("开始处理")
        # Refined style for primary action, keeping it green but smaller/cleaner
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; 
                color: white; 
                font-weight: bold;
                border: 1px solid #1b5e20;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #1b5e20;
            }
        """)
        layout.addWidget(self.btn_start)
        
        # --- Connections ---
        self.btn_browse_input.clicked.connect(self._browse_input)
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.chk_same_dir.toggled.connect(self._toggle_output_dir)
        self.btn_refresh.clicked.connect(self._refresh_files)
        self.btn_settings.clicked.connect(self._open_settings)
        self.btn_help.clicked.connect(self._open_help)
        self.list_files.itemSelectionChanged.connect(self._on_selection_changed)
        self.btn_play.clicked.connect(self._play_audio)
        self.btn_stop.clicked.connect(self._stop_audio)
        self.btn_start.clicked.connect(self._start_processing)
        
        self.btn_read_tg.clicked.connect(self._read_textgrid)
        self.btn_tg_seg.clicked.connect(self._toggle_segmentation)
        self.btn_save_seg.clicked.connect(self._save_segmented_audio)
        self.btn_read_lip.clicked.connect(self._read_lip_data)
        self.btn_param_select.clicked.connect(self._open_parameter_selection)
        self.btn_param_help.clicked.connect(self._open_parameter_help)
        
        # Interaction variables
        self._full_y: Optional[np.ndarray] = None
        self._fs: int = 16000
        self._press_x: Optional[float] = None
        self._pan_active = False

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _browse_input(self):
        start_dir = self.edit_input.text() if self.edit_input.text() else ""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输入目录", start_dir)
        if d:
            self.edit_input.setText(d)
            # Save to settings
            SettingsService().set("last_input_dir", d)
            
            if self.chk_same_dir.isChecked():
                self.edit_output.setText(d)
            self._refresh_files()

    def _browse_output(self):
        start_dir = self.edit_output.text() if self.edit_output.text() else ""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", start_dir)
        if d:
            self.edit_output.setText(d)

    def _toggle_output_dir(self, checked):
        self.edit_output.setEnabled(not checked)
        self.btn_browse_output.setEnabled(not checked)
        if checked:
            self.edit_output.setText(self.edit_input.text())

    def _refresh_files(self):
        self.list_files.clear()
        self._textgrid_cache = {}
        self._lip_data_map = {}
        
        p = Path(self.edit_input.text())
        if not p.exists() or not p.is_dir():
            return
            
        # Only simple glob for now, no recursive
        for wav in p.glob("*.wav"):
            self.list_files.addItem(wav.name)

    def _on_selection_changed(self):
        items = self.list_files.selectedItems()
        if not items:
            return
        
        # Show waveform of first selected
        name = items[0].text()
        p = Path(self.edit_input.text()) / name
        self._plot_waveform(p)

    def _plot_waveform(self, path: Path):
        self.ax.clear()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", wavfile.WavFileWarning)
                fs, data = wavfile.read(str(path))
            
            # Convert to float
            if data.dtype == np.int16:
                y = data.astype(np.float64) / 32768.0
            elif data.dtype == np.int32:
                y = data.astype(np.float64) / 2147483648.0
            elif data.dtype == np.uint8:
                y = (data.astype(np.float64) - 128) / 128.0
            else:
                y = data.astype(np.float64)
            
            # Convert to mono
            if y.ndim > 1:
                y = np.mean(y, axis=1)
                
            self._full_y = y
            self._fs = fs
            
            # Initial plot - full view
            total_duration = len(data) / fs
            self.ax.set_xlim(0, total_duration)
            self._update_plot()
            
        except Exception as e:
            log.error(f"Plot error: {e}")

    def _update_plot(self):
        if self._full_y is None:
            return
            
        xlim = self.ax.get_xlim()
        fs = self._fs
        
        # Determine indices
        start_idx = int(max(0, xlim[0] * fs))
        end_idx = int(min(len(self._full_y), xlim[1] * fs))
        
        if end_idx <= start_idx:
             return
             
        # Extract chunk
        chunk = self._full_y[start_idx:end_idx]
        
        # Dynamic downsampling
        step = 1
        while len(chunk) // step > 10000:
            step *= 2
            
        plot_y = chunk[::step]
        plot_t = np.linspace(max(0, xlim[0]), min(len(self._full_y)/fs, xlim[1]), len(plot_y))
        
        color = '#00aaff' if self.is_dark else '#007acc'
        
        # Clear previous lines but keep limits
        # Actually ax.clear() wipes limits, so we manage lines directly
        if self.ax.lines:
             self.ax.lines[0].set_data(plot_t, plot_y)
        else:
             self.ax.plot(plot_t, plot_y, color=color, linewidth=0.5)
             
        # Restore limits (set_data doesn't change limits, but just in case)
        self.ax.set_xlim(xlim)
        
        # Y-axis auto-scale based on visible data? User said "Y-axis range unchanged (max(abs(y)))"
        # So we fix Y to full data max
        max_val = np.max(np.abs(self._full_y)) if len(self._full_y) > 0 else 1.0
        if max_val == 0: max_val = 1.0
        self.ax.set_ylim(-max_val * 1.1, max_val * 1.1)
        
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        
        cur_xlim = self.ax.get_xlim()
        cur_range = cur_xlim[1] - cur_xlim[0]
        xdata = event.xdata
        
        scale_factor = 0.8 if event.button == 'up' else 1.25
        
        new_range = cur_range * scale_factor
        
        # Center zoom on mouse cursor
        rel_pos = (xdata - cur_xlim[0]) / cur_range
        new_left = xdata - new_range * rel_pos
        new_right = new_left + new_range
        
        # Clamp
        total_dur = len(self._full_y) / self._fs if self._full_y is not None else 1.0
        
        # Optional: prevent zooming out beyond full file?
        # User didn't specify, but usually desirable.
        if new_range > total_dur:
            new_left = 0
            new_right = total_dur
        else:
            if new_left < 0: 
                new_left = 0
                new_right = new_range
            if new_right > total_dur:
                new_right = total_dur
                new_left = total_dur - new_range
                
        self.ax.set_xlim(new_left, new_right)
        self._update_plot()

    def _on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1: # Left click
            self._pan_active = True
            self._press_x = event.xdata

    def _on_release(self, event):
        self._pan_active = False
        self._press_x = None

    def _on_motion(self, event):
        if not self._pan_active or self._press_x is None: return
        if event.inaxes != self.ax: return
        
        dx = event.xdata - self._press_x
        cur_xlim = self.ax.get_xlim()
        
        new_left = cur_xlim[0] - dx
        new_right = cur_xlim[1] - dx
        
        # Clamp
        total_dur = len(self._full_y) / self._fs if self._full_y is not None else 1.0
        width = new_right - new_left
        
        if new_left < 0:
            new_left = 0
            new_right = width
        if new_right > total_dur:
            new_right = total_dur
            new_left = total_dur - width
            
        self.ax.set_xlim(new_left, new_right)
        self._update_plot()

    def _play_audio(self):
        items = self.list_files.selectedItems()
        if not items:
            return
        name = items[0].text()
        p = Path(self.edit_input.text()) / name
        
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(p)))
        self._player.setPosition(0) # Ensure rewind to start
        self._player.play()

    def _stop_audio(self):
        self._player.stop()

    def _read_textgrid(self):
        p = Path(self.edit_input.text())
        if not p.exists():
            return
        
        count = 0
        self._textgrid_cache = {}
        for i in range(self.list_files.count()):
            name = self.list_files.item(i).text()
            wav_path = p / name
            tg_path = wav_path.with_suffix(".TextGrid")
            if tg_path.exists():
                tg = parse_textgrid(tg_path)
                if tg:
                    self._textgrid_cache[name] = tg
                    count += 1
        
        QtWidgets.QMessageBox.information(self, "TextGrid", f"已读取 {count} 个TextGrid文件")

    def _toggle_segmentation(self):
        items = self.list_files.selectedItems()
        if not items:
            QtWidgets.QMessageBox.warning(self, "TextGrid切分", "请先选中一个文件")
            return
        
        name = items[0].text()
        tg = self._textgrid_cache.get(name)
        if not tg:
            # Try to load on demand
            p = Path(self.edit_input.text()) / name
            tg_path = p.with_suffix(".TextGrid")
            if tg_path.exists():
                tg = parse_textgrid(tg_path)
                if tg:
                    self._textgrid_cache[name] = tg
        
        if not tg or not tg.tiers:
            QtWidgets.QMessageBox.warning(self, "TextGrid切分", "未找到TextGrid数据")
            return

        # Cycle through tiers
        current_idx = -1
        if self._selected_tier_name:
            for i, t in enumerate(tg.tiers):
                if t.name == self._selected_tier_name:
                    current_idx = i
                    break
        
        next_idx = current_idx + 1
        if next_idx >= len(tg.tiers):
            self._selected_tier_name = None
            self.btn_tg_seg.setText("TextGrid切分")
        else:
            self._selected_tier_name = tg.tiers[next_idx].name
            self.btn_tg_seg.setText(f"item [{next_idx+1}]：{self._selected_tier_name}")

    def _save_segmented_audio(self):
        if not self._selected_tier_name:
            QtWidgets.QMessageBox.warning(self, "保存", "请先选择切分层级")
            return
            
        items = self.list_files.selectedItems()
        if not items:
            return

        input_dir = Path(self.edit_input.text())
        output_dir = Path(self.edit_output.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for item in items:
            name = item.text()
            tg = self._textgrid_cache.get(name)
            if not tg: continue
            
            tier = next((t for t in tg.tiers if t.name == self._selected_tier_name), None)
            if not tier: continue
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", wavfile.WavFileWarning)
                    fs, data = wavfile.read(str(input_dir / name))
                stem = Path(name).stem
                
                for interval in tier.intervals:
                    txt = interval.text.strip()
                    if not txt or txt.lower() in ["sil", "eps", "<sil>", "<eps>", ""]:
                        continue
                        
                    start = int(interval.xmin * fs)
                    end = int(interval.xmax * fs)
                    if start >= len(data) or end <= start: continue
                    
                    seg = data[start:end]
                    new_name = f"{stem}_{tier.name}_{txt}_{interval.xmin:.3f}_{interval.xmax:.3f}.wav"
                    # Sanitize filename
                    new_name = "".join([c for c in new_name if c.isalnum() or c in "._-"])
                    
                    wavfile.write(str(output_dir / new_name), fs, seg)
                    count += 1
            except Exception as e:
                log.error(f"Error segmenting {name}: {e}")
                
        QtWidgets.QMessageBox.information(self, "保存", f"已保存 {count} 个片段")

    def _read_lip_data(self):
        p = Path(self.edit_input.text())
        if not p.exists():
            return
            
        count = 0
        self._lip_data_map = {}
        
        for i in range(self.list_files.count()):
            name = self.list_files.item(i).text()
            wav_path = p / name
            pkl_path = wav_path.with_suffix(".pkl")
            if pkl_path.exists():
                self._lip_data_map[name] = pkl_path
                count += 1
        
        if count > 0:
            QtWidgets.QMessageBox.information(self, "唇形数据", f"已关联 {count} 个唇形数据文件")
        else:
            QtWidgets.QMessageBox.warning(self, "唇形数据", "未找到同名 .pkl 文件")

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            # Settings saved
            pass

    def _open_help(self):
        help_file = Path(r"d:\PhoneticToolbox\PhoneticToolbox_v2\Phonetic_Export\index.html")
        if not help_file.exists():
            QtWidgets.QMessageBox.warning(self, "帮助", f"未找到帮助文件：{help_file}")
            return
        url = QUrl.fromLocalFile(str(help_file))
        url.setFragment("s1764306595966")
        opened = QtGui.QDesktopServices.openUrl(url)
        if not opened:
            QtWidgets.QMessageBox.warning(self, "帮助", "帮助页面打开失败。")

    def _open_parameter_selection(self):
        dialog = ParameterSelectionDialog(self._selectable_param_items, self._selected_param_keys, self)
        if dialog.exec():
            selected = dialog.selected_keys()
            if not selected:
                QtWidgets.QMessageBox.warning(self, "参数选择", "至少选择一个参数。")
                return
            self._selected_param_keys = selected
            QtWidgets.QMessageBox.information(self, "参数选择", f"已选择 {len(selected)} 个参数。")

    def _open_parameter_help(self):
        dialog = ParameterHelpDialog(self)
        dialog.exec()

    def _start_processing(self):
        items = [self.list_files.item(i).text() for i in range(self.list_files.count())]
        if not items:
            QtWidgets.QMessageBox.warning(self, "参数估计", "文件列表为空")
            return
            
        input_dir = Path(self.edit_input.text())
        output_dir = Path(self.edit_output.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        settings = SettingsService()
        config = settings.get_config_object()
        config.selected_parameter_keys = list(self._selected_param_keys)
        
        self.btn_start.setEnabled(False)
        self.progress = QtWidgets.QProgressDialog("处理中...", "取消", 0, 100, self)
        self.progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress.show()
        
        self.worker = PEWorker(items, input_dir, output_dir, self._lip_data_map, config)
        self.worker.progress_sig.connect(self._on_progress)
        self.worker.finished_sig.connect(self._on_finished)
        self.progress.canceled.connect(self.worker.requestInterruption)
        self.worker.start()
        
    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        if is_dark:
            facecolor = '#1e1e1e'
            axis_color = 'white'
            plot_color = '#00aaff'
        else:
            facecolor = '#f0f0f0'
            axis_color = 'black'
            plot_color = '#007acc'
            
        self.figure.set_facecolor(facecolor)
        self.ax.set_facecolor(facecolor)
        self.ax.tick_params(colors=axis_color, which='both')
        self.ax.xaxis.label.set_color(axis_color)
        self.ax.yaxis.label.set_color(axis_color)
        
        for spine in self.ax.spines.values():
            spine.set_color(axis_color)
            
        # Replot if there's data
        if self.ax.lines:
            self.ax.lines[0].set_color(plot_color)
            
        self.canvas.draw()
        
    def _on_progress(self, val, name):
        self.progress.setValue(val)
        self.progress.setLabelText(f"处理中: {name}")
        
    def _on_finished(self):
        self.progress.close()
        self.btn_start.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "完成", "批处理完成")
