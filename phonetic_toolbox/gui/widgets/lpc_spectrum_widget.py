from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from phonetic_toolbox.models.lpc_models import LPCSpectrumConfig
from phonetic_toolbox.services.io.textgrid import TextGrid
from phonetic_toolbox.services.lpc_service import LPCSpectrumService
from phonetic_toolbox.services.settings_service import SettingsService


class LPCSpectrumWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.service = LPCSpectrumService()
        self.is_dark = True
        self._player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)
        self._player.positionChanged.connect(self._on_player_position_changed)
        self._current_wav_path: Optional[Path] = None
        self._full_y: Optional[np.ndarray] = None
        self._fs: int = 16000
        self._press_x: Optional[float] = None
        self._pan_start_xlim: Optional[tuple[float, float]] = None
        self._pan_active = False
        self._select_active = False
        self._select_start_x: Optional[float] = None
        self._selection_range_sec: Optional[tuple[float, float]] = None
        self._selection_patch = None
        self._textgrid_cache: dict[str, TextGrid] = {}
        self._selected_tier_name: Optional[str] = None
        self._current_plot_mode = "wave"
        self._last_lpc_result = None
        self._last_wave_xlim: Optional[tuple[float, float]] = None
        self._play_start_ms = 0
        self._play_end_ms = 0
        self.init_ui()

        settings = SettingsService()
        last_input = settings.get("last_input_dir")
        if last_input and Path(last_input).exists():
            self.edit_input.setText(last_input)
            self.edit_output.setText(last_input)
            self._refresh_files()

    def init_ui(self):
        self.setWindowTitle("LPC 谱图")
        root = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("请选择输入/输出目录并开始处理")
        root.addWidget(title)

        dir_layout = QtWidgets.QGridLayout()
        self.edit_input = QtWidgets.QLineEdit()
        self.edit_output = QtWidgets.QLineEdit()
        self.btn_browse_input = QtWidgets.QPushButton("浏览...")
        self.btn_browse_output = QtWidgets.QPushButton("浏览...")
        dir_layout.addWidget(QtWidgets.QLabel("音频目录"), 0, 0)
        dir_layout.addWidget(self.edit_input, 0, 1)
        dir_layout.addWidget(self.btn_browse_input, 0, 2)
        dir_layout.addWidget(QtWidgets.QLabel("输出目录"), 1, 0)
        dir_layout.addWidget(self.edit_output, 1, 1)
        dir_layout.addWidget(self.btn_browse_output, 1, 2)
        root.addLayout(dir_layout)

        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("播放音频")
        self.btn_stop = QtWidgets.QPushButton("停止播放")
        self.btn_refresh = QtWidgets.QPushButton("刷新文件列表")
        self.btn_read_tg = QtWidgets.QPushButton("读取TextGrid")
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_refresh)
        ctrl_layout.addWidget(self.btn_read_tg)
        root.addLayout(ctrl_layout)

        lpc_group = QtWidgets.QGroupBox("LPC 参数")
        lpc_layout = QtWidgets.QHBoxLayout(lpc_group)
        self.spin_lpc_order = QtWidgets.QSpinBox()
        self.spin_lpc_order.setRange(1, 200)
        self.spin_lpc_order.setValue(50)
        self.spin_lpc_order.setPrefix("阶数: ")
        self.spin_freq_max = QtWidgets.QSpinBox()
        self.spin_freq_max.setRange(100, 48000)
        self.spin_freq_max.setValue(8000)
        self.spin_freq_max.setPrefix("频率上限: ")
        self.spin_freq_max.setSuffix(" Hz")
        self.spin_amp_min = QtWidgets.QDoubleSpinBox()
        self.spin_amp_min.setRange(-200.0, 100.0)
        self.spin_amp_min.setValue(-5.0)
        self.spin_amp_min.setPrefix("Min: ")
        self.spin_amp_min.setSuffix(" dB")
        self.spin_amp_max = QtWidgets.QDoubleSpinBox()
        self.spin_amp_max.setRange(-200.0, 100.0)
        self.spin_amp_max.setValue(35.0)
        self.spin_amp_max.setPrefix("Max: ")
        self.spin_amp_max.setSuffix(" dB")
        self.chk_dynamic_y = QtWidgets.QCheckBox("动态设置y轴范围")
        lpc_layout.addWidget(self.spin_lpc_order)
        lpc_layout.addWidget(self.spin_freq_max)
        lpc_layout.addWidget(self.spin_amp_min)
        lpc_layout.addWidget(self.spin_amp_max)
        lpc_layout.addWidget(self.chk_dynamic_y)
        root.addWidget(lpc_group)

        content_layout = QtWidgets.QHBoxLayout()
        self.list_files = QtWidgets.QListWidget()
        self.list_files.setMaximumWidth(280)
        self.figure = Figure(figsize=(12, 6.75), facecolor="white")
        self.canvas = Canvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        content_layout.addWidget(self.list_files)
        content_layout.addWidget(self.canvas, 1)
        root.addLayout(content_layout)

        action_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("开始处理")
        self.btn_back_wave = QtWidgets.QPushButton("返回波形图")
        self.btn_back_wave.setEnabled(False)
        self.btn_help = QtWidgets.QPushButton("帮助")
        self.btn_help.setFixedSize(50, 23)
        self.btn_help.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; border: none; border-radius: 4px;")
        action_layout.addWidget(self.btn_start, stretch=1)
        action_layout.addWidget(self.btn_back_wave, stretch=1)
        action_layout.addWidget(self.btn_help)
        root.addLayout(action_layout)

        self.btn_browse_input.clicked.connect(self._browse_input)
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.btn_refresh.clicked.connect(self._refresh_files)
        self.btn_play.clicked.connect(self._play_current_file)
        self.btn_stop.clicked.connect(self._stop_audio)
        self.btn_read_tg.clicked.connect(self._on_textgrid_button_clicked)
        self.list_files.itemSelectionChanged.connect(self._on_file_selected)
        self.btn_start.clicked.connect(self._start_lpc_processing)
        self.btn_back_wave.clicked.connect(self._return_to_waveform)
        self.btn_help.clicked.connect(self._open_help)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._ensure_plot_margins()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ensure_plot_margins()
        self.canvas.draw_idle()

    def _browse_input(self):
        start_dir = self.edit_input.text() if self.edit_input.text() else ""
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择音频目录", start_dir)
        if not selected:
            return
        self.edit_input.setText(selected)
        if not self.edit_output.text():
            self.edit_output.setText(selected)
        SettingsService().set("last_input_dir", selected)
        self._refresh_files()

    def _browse_output(self):
        start_dir = self.edit_output.text() if self.edit_output.text() else ""
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", start_dir)
        if selected:
            self.edit_output.setText(selected)

    def _refresh_files(self):
        self.list_files.clear()
        input_dir = Path(self.edit_input.text())
        if not input_dir.exists() or not input_dir.is_dir():
            return
        for wav_path in input_dir.glob("*.wav"):
            self.list_files.addItem(wav_path.name)

    def _on_file_selected(self):
        items = self.list_files.selectedItems()
        if not items:
            return
        input_dir = Path(self.edit_input.text())
        wav_path = input_dir / items[0].text()
        if not wav_path.exists():
            return
        self._current_wav_path = wav_path
        fs, y = self.service.load_audio(wav_path)
        self._fs = fs
        self._full_y = y
        self._selection_range_sec = None
        self._last_lpc_result = None
        self._selection_patch = None
        self._select_start_x = None
        self._pan_active = False
        self._pan_start_xlim = None
        self._select_active = False
        self._last_wave_xlim = None
        self.btn_back_wave.setEnabled(False)
        self._plot_waveform()
        self._textgrid_cache.pop(wav_path.name, None)
        self._selected_tier_name = None
        self.btn_read_tg.setText("读取TextGrid")

    def _plot_waveform(self, keep_view: bool = False):
        if self._full_y is None or self._full_y.size == 0:
            return
        duration_sec = self._full_y.size / self._fs
        previous_xlim = self.ax.get_xlim()
        self._current_plot_mode = "wave"
        self.ax.clear()
        max_points = 20000
        step = max(1, int(np.ceil(self._full_y.size / max_points)))
        y = self._full_y[::step]
        t = np.arange(y.size) * step / self._fs
        color = "#00aaff" if self.is_dark else "#007acc"
        self.ax.plot(t, y, color=color, linewidth=0.8)
        self.ax.set_xlim(0.0, duration_sec)
        max_val = float(np.max(np.abs(self._full_y))) if self._full_y.size else 1.0
        max_val = max(max_val, 1e-6)
        self.ax.set_ylim(-1.1 * max_val, 1.1 * max_val)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        if keep_view:
            left = max(0.0, min(float(previous_xlim[0]), duration_sec))
            right = max(0.0, min(float(previous_xlim[1]), duration_sec))
            if right - left > 0:
                self.ax.set_xlim(left, right)
        self._draw_textgrid()
        self._draw_selection_overlay()
        self._apply_axes_theme()
        self._ensure_plot_margins()
        self.canvas.draw_idle()

    def _draw_selection_overlay(self):
        if self._current_plot_mode != "wave" or self._selection_range_sec is None:
            return
        start_sec, end_sec = self._selection_range_sec
        if end_sec <= start_sec:
            return
        self._selection_patch = self.ax.axvspan(start_sec, end_sec, alpha=0.3, facecolor="gold")

    def _play_current_file(self):
        if self._current_wav_path is None or self._full_y is None or self._full_y.size == 0:
            QtWidgets.QMessageBox.warning(self, "播放", "请先选中一个音频文件")
            return
        duration_sec = self._full_y.size / self._fs
        if self._current_plot_mode == "wave":
            start_sec, end_sec = self.ax.get_xlim()
        elif self._selection_range_sec is not None:
            start_sec, end_sec = self._selection_range_sec
        else:
            start_sec, end_sec = 0.0, duration_sec
        start_sec = float(max(0.0, min(start_sec, duration_sec)))
        end_sec = float(max(0.0, min(end_sec, duration_sec)))
        if end_sec <= start_sec:
            QtWidgets.QMessageBox.warning(self, "播放", "当前窗口范围无效，无法播放")
            return
        self._play_start_ms = int(start_sec * 1000)
        self._play_end_ms = int(end_sec * 1000)
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(self._current_wav_path)))
        self._audio_output.setVolume(0.9)
        self._player.play()
        self._player.setPosition(self._play_start_ms)

    def _stop_audio(self):
        self._player.stop()
        self._play_start_ms = 0
        self._play_end_ms = 0

    def _on_player_position_changed(self, position_ms: int):
        if self._play_end_ms <= self._play_start_ms:
            return
        if position_ms >= self._play_end_ms:
            self._stop_audio()

    def _on_textgrid_button_clicked(self):
        if self._current_wav_path is None:
            QtWidgets.QMessageBox.warning(self, "TextGrid", "请先选中一个音频文件")
            return
        key = self._current_wav_path.name
        textgrid = self._textgrid_cache.get(key)
        if textgrid is None:
            loaded = self.service.read_sibling_textgrid(self._current_wav_path)
            if loaded is None or not loaded.tiers:
                QtWidgets.QMessageBox.warning(self, "TextGrid", "未找到同名 TextGrid 或解析失败")
                return
            self._textgrid_cache[key] = loaded
            self._selected_tier_name = loaded.tiers[0].name
            self.btn_read_tg.setText(f"切换层级 ({self._selected_tier_name})")
            self._plot_waveform()
            return
        next_tier = self.service.next_tier_name(textgrid, self._selected_tier_name)
        self._selected_tier_name = next_tier
        if next_tier:
            self.btn_read_tg.setText(f"切换层级 ({next_tier})")
        self._plot_waveform()

    def _draw_textgrid(self):
        if self._current_plot_mode != "wave" or self._current_wav_path is None:
            return
        key = self._current_wav_path.name
        textgrid = self._textgrid_cache.get(key)
        if textgrid is None or not self._selected_tier_name:
            return
        tier = next((item for item in textgrid.tiers if item.name == self._selected_tier_name), None)
        if tier is None:
            return
        trans = self.ax.get_xaxis_transform()
        for interval in tier.intervals:
            t_min = interval.xmin
            t_max = interval.xmax
            line_color = "#ff8888" if self.is_dark else "r"
            self.ax.axvline(x=t_min, color=line_color, linestyle="--", alpha=0.6)
            mid = (t_min + t_max) / 2.0
            text = interval.text.strip()
            if text:
                self.ax.text(
                    mid,
                    1.02,
                    text,
                    transform=trans,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="white" if self.is_dark else "black",
                    clip_on=False,
                )

    def _start_lpc_processing(self):
        if self._full_y is None or self._full_y.size == 0 or self._current_wav_path is None:
            QtWidgets.QMessageBox.warning(self, "LPC", "请先选中并加载音频")
            return
        if self._selection_range_sec is None:
            start_sec, end_sec = self.ax.get_xlim()
        else:
            start_sec, end_sec = self._selection_range_sec
        start_idx = int(max(0, start_sec) * self._fs)
        end_idx = int(min(self._full_y.size / self._fs, end_sec) * self._fs)
        if end_idx <= start_idx:
            QtWidgets.QMessageBox.warning(self, "LPC", "选区无效")
            return
        segment = self._full_y[start_idx:end_idx]
        config = LPCSpectrumConfig(
            order=self.spin_lpc_order.value(),
            freq_max_hz=self.spin_freq_max.value(),
            amp_min_db=float(self.spin_amp_min.value()),
            amp_max_db=float(self.spin_amp_max.value()),
            dynamic_y=self.chk_dynamic_y.isChecked(),
        )
        try:
            result = self.service.compute_spectrum(segment, self._fs, config)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "LPC", f"LPC 分析失败: {exc}")
            return
        if self._current_plot_mode == "wave":
            self._last_wave_xlim = self.ax.get_xlim()
        self._last_lpc_result = result
        self._plot_lpc_curve(result, config.freq_max_hz)
        self.btn_back_wave.setEnabled(True)
        self.spin_amp_min.setValue(result.amp_min_db)
        self.spin_amp_max.setValue(result.amp_max_db)

        output_dir = Path(self.edit_output.text()) if self.edit_output.text() else Path(self.edit_input.text())
        textgrid_label = self.service.extract_label_in_range(
            self._textgrid_cache.get(self._current_wav_path.name),
            self._selected_tier_name,
            start_idx / self._fs,
            end_idx / self._fs,
        )
        try:
            saved_path = self.service.save_plot_figure(
                self._create_export_figure(result, config.freq_max_hz),
                output_dir=output_dir,
                wav_stem=self._current_wav_path.stem,
                textgrid_label=textgrid_label,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "完成", f"LPC 图片已保存至:\n{saved_path}")

    def _plot_lpc_curve(self, result, freq_max_hz: int):
        self._current_plot_mode = "lpc"
        self.ax.clear()
        self.ax.plot(result.frequencies_hz, result.magnitude_db, color="#4ba3ff", linewidth=1.8)
        self.ax.set_xlim(0.0, float(freq_max_hz))
        self.ax.set_ylim(float(result.amp_min_db), float(result.amp_max_db))
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.grid(True, alpha=0.3)
        self._apply_axes_theme()
        self._ensure_plot_margins()
        self.canvas.draw_idle()

    def _apply_axes_theme(self):
        axis_color = "white" if self.is_dark else "black"
        face = "#1e1e1e" if self.is_dark else "white"
        self.ax.set_facecolor(face)
        self.ax.tick_params(colors=axis_color, which="both")
        self.ax.xaxis.label.set_color(axis_color)
        self.ax.yaxis.label.set_color(axis_color)
        self.ax.title.set_color(axis_color)
        for spine in self.ax.spines.values():
            spine.set_color(axis_color)

    def _ensure_plot_margins(self):
        self.figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.19)

    def _create_export_figure(self, result, freq_max_hz: int) -> Figure:
        export_figure = Figure(figsize=(8, 4.5), facecolor="white")
        export_ax = export_figure.add_subplot(111)
        export_ax.set_facecolor("white")
        export_ax.plot(result.frequencies_hz, result.magnitude_db, color="black", linewidth=1.8)
        export_ax.set_xlim(0.0, float(freq_max_hz))
        export_ax.set_ylim(float(result.amp_min_db), float(result.amp_max_db))
        export_ax.set_xlabel("Frequency (Hz)", color="black")
        export_ax.set_ylabel("Magnitude (dB)", color="black")
        export_ax.grid(True, alpha=0.3, color="#888888")
        export_ax.tick_params(colors="black", which="both")
        for spine in export_ax.spines.values():
            spine.set_color("black")
        export_figure.tight_layout()
        return export_figure

    def _open_help(self):
        base_path = Path(__file__).parent.parent.parent.parent / "Phonetic_Export" / "index.html"
        if not base_path.exists():
            QtWidgets.QMessageBox.warning(self, "帮助", f"未找到帮助文件：{base_path}")
            return
        url = QUrl.fromLocalFile(str(base_path.resolve()))
        url.setFragment("s1773837409220")
        opened = QtGui.QDesktopServices.openUrl(url)
        if not opened:
            QtWidgets.QMessageBox.warning(self, "帮助", "帮助页面打开失败。")

    def _is_shift_pressed(self) -> bool:
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        return bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

    def _on_scroll(self, event):
        if event.inaxes != self.ax or self._current_plot_mode != "wave" or self._full_y is None:
            return
        current_left, current_right = self.ax.get_xlim()
        current_range = current_right - current_left
        if current_range <= 0:
            return
        pointer_x = event.xdata if event.xdata is not None else (current_left + current_right) / 2.0
        scale = 0.8 if event.button == "up" else 1.25
        new_range = current_range * scale
        duration_sec = self._full_y.size / self._fs
        if new_range > duration_sec:
            new_range = duration_sec
        rel = (pointer_x - current_left) / current_range
        new_left = pointer_x - new_range * rel
        new_right = new_left + new_range
        if new_left < 0:
            new_left = 0
            new_right = new_range
        if new_right > duration_sec:
            new_right = duration_sec
            new_left = duration_sec - new_range
        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or self._current_plot_mode != "wave" or event.button != 1:
            return
        if event.xdata is None:
            return
        if self._is_shift_pressed():
            self._select_active = True
            self._select_start_x = event.xdata
            self._pan_active = False
            self._press_x = None
            if self._selection_patch is not None:
                self._selection_patch.remove()
            self._selection_patch = self.ax.axvspan(event.xdata, event.xdata, alpha=0.3, facecolor="gold")
            self.canvas.draw_idle()
            return
        self._pan_active = True
        self._press_x = event.xdata
        self._pan_start_xlim = self.ax.get_xlim()

    def _on_release(self, event):
        if self._select_active:
            self._select_active = False
            if self._select_start_x is not None and event.inaxes == self.ax and event.xdata is not None:
                start_sec = min(self._select_start_x, event.xdata)
                end_sec = max(self._select_start_x, event.xdata)
                if end_sec > start_sec:
                    self._selection_range_sec = (start_sec, end_sec)
                else:
                    self._selection_range_sec = None
            self._select_start_x = None
            self._plot_waveform(keep_view=True)
        self._pan_active = False
        self._press_x = None
        self._pan_start_xlim = None

    def _on_motion(self, event):
        if (
            self._select_active
            and self._select_start_x is not None
            and event.inaxes == self.ax
            and event.xdata is not None
            and self._current_plot_mode == "wave"
        ):
            start_sec = min(self._select_start_x, event.xdata)
            end_sec = max(self._select_start_x, event.xdata)
            if self._selection_patch is not None:
                self._selection_patch.remove()
            self._selection_patch = self.ax.axvspan(start_sec, end_sec, alpha=0.3, facecolor="gold")
            self.canvas.draw_idle()
            return
        if (
            not self._pan_active
            or self._press_x is None
            or self._pan_start_xlim is None
            or event.inaxes != self.ax
            or self._current_plot_mode != "wave"
            or self._full_y is None
            or event.xdata is None
        ):
            return
        dx = event.xdata - self._press_x
        start_left, start_right = self._pan_start_xlim
        width = start_right - start_left
        duration_sec = self._full_y.size / self._fs
        new_left = start_left - dx
        new_right = start_right - dx
        if new_left < 0:
            new_left = 0
            new_right = width
        if new_right > duration_sec:
            new_right = duration_sec
            new_left = duration_sec - width
        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def _return_to_waveform(self):
        self._plot_waveform(keep_view=False)
        if self._last_wave_xlim is not None and self._full_y is not None and self._full_y.size > 0:
            duration_sec = self._full_y.size / self._fs
            left = max(0.0, min(float(self._last_wave_xlim[0]), duration_sec))
            right = max(0.0, min(float(self._last_wave_xlim[1]), duration_sec))
            if right - left > 0:
                self.ax.set_xlim(left, right)
                self.canvas.draw_idle()
        self.btn_back_wave.setEnabled(False)

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        if is_dark:
            face = "#1e1e1e"
            axis_color = "white"
        else:
            face = "white"
            axis_color = "black"
        self.figure.set_facecolor(face)
        self.ax.set_facecolor(face)
        self.ax.tick_params(colors=axis_color, which="both")
        self.ax.xaxis.label.set_color(axis_color)
        self.ax.yaxis.label.set_color(axis_color)
        self.ax.title.set_color(axis_color)
        for spine in self.ax.spines.values():
            spine.set_color(axis_color)
        if self._current_plot_mode == "wave":
            self._plot_waveform(keep_view=True)
        elif self._current_plot_mode == "lpc" and self._last_lpc_result is not None:
            self._plot_lpc_curve(self._last_lpc_result, self.spin_freq_max.value())
        self.canvas.draw_idle()
