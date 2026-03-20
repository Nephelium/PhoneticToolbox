import csv
from collections import defaultdict
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QDoubleValidator, QIcon, QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly, spectrogram

from phonetic_toolbox.core.synthesis.klatt import (
    DEFAULT_DURATION,
    DEFAULT_FS,
    FADE_MS,
    PARAM_DEFAULTS,
    VOWEL_FORMANTS,
    KlattParam1980,
    SpectralFilter,
    klatt_make,
    parse_vowel_sequence,
)
from phonetic_toolbox.core.acoustic import (
    compute_energy,
    compute_hnr,
    compute_jitter_shimmer,
    compute_praat_f0,
    compute_praat_formants,
    compute_shr,
    compute_spectral_features_batch,
    compute_spectral_slope,
    compute_voiced_mask,
)
from phonetic_toolbox.utils import get_resource_path


PARAM_ORDER = [
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "AV",
    "HNR",
    "SHR",
    "Jitter",
    "Shimmer",
    "Slope",
    "H1H2",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
]

FORMANT_PARAMS = {"F1", "F2", "F3", "F4", "F5"}
FORMANT_Y_RANGES = {
    "F1": (0.0, 2000.0),
    "F2": (0.0, 3000.0),
    "F3": (1500.0, 4500.0),
    "F4": (2500.0, 5500.0),
    "F5": (3000.0, 6000.0),
}

PARAM_STYLES = {
    "F0": {"color": "#4dabf7", "symbol": "o"},
    "F1": {"color": "#ff6b6b", "symbol": "s"},
    "F2": {"color": "#51cf66", "symbol": "t"},
    "F3": {"color": "#ffd43b", "symbol": "d"},
    "F4": {"color": "#845ef7", "symbol": "+"},
    "F5": {"color": "#f06595", "symbol": "x"},
    "AV": {"color": "#74c0fc", "symbol": "o"},
    "HNR": {"color": "#63e6be", "symbol": "s"},
    "SHR": {"color": "#ffa94d", "symbol": "t"},
    "Jitter": {"color": "#b197fc", "symbol": "d"},
    "Shimmer": {"color": "#ff8787", "symbol": "+"},
    "Slope": {"color": "#fcc419", "symbol": "x"},
    "H1H2": {"color": "#22b8cf", "symbol": "o"},
    "A1": {"color": "#20c997", "symbol": "s"},
    "A2": {"color": "#94d82d", "symbol": "t"},
    "A3": {"color": "#ff922b", "symbol": "d"},
    "A4": {"color": "#f76707", "symbol": "+"},
    "A5": {"color": "#868e96", "symbol": "x"},
    "B1": {"color": "#a61e4d", "symbol": "o"},
    "B2": {"color": "#5f3dc4", "symbol": "s"},
    "B3": {"color": "#1864ab", "symbol": "t"},
    "B4": {"color": "#087f5b", "symbol": "d"},
    "B5": {"color": "#e8590c", "symbol": "+"},
}


class ParameterCurve:
    def __init__(self, name: str, default_value: float, min_val: float, max_val: float, duration: float):
        self.name = name
        self.default_value = default_value
        self.min_val = min_val
        self.max_val = max_val
        self.points = [(0.0, default_value), (duration, default_value)]
        self.global_override: Optional[float] = None

    def set_points(self, points: list[tuple[float, float]]):
        self.points = sorted(points, key=lambda item: item[0])
        self.global_override = None

    def get_array(self, duration: float, fs: int) -> np.ndarray:
        n_samples = int(round(duration * fs))
        if n_samples <= 0:
            return np.array([], dtype=float)
        if self.global_override is not None:
            return np.ones(n_samples, dtype=float) * float(self.global_override)
        if not self.points:
            return np.ones(n_samples, dtype=float) * self.default_value
        times = [p[0] for p in self.points]
        values = [p[1] for p in self.points]
        if times[0] > 0.0:
            times.insert(0, 0.0)
            values.insert(0, values[0])
        if times[-1] < duration:
            times.append(duration)
            values.append(values[-1])
        t_grid = np.linspace(0.0, duration, n_samples)
        arr = np.interp(t_grid, times, values)
        return np.clip(arr, self.min_val, self.max_val)


class ParameterEditorViewBox(pg.ViewBox):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor

    def wheelEvent(self, event, axis=None):
        super().wheelEvent(event, axis=0)

    def mouseDragEvent(self, ev, axis=None):
        modifiers = QApplication.keyboardModifiers()
        is_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        is_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.scenePos()
            point = self.editor.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = float(np.clip(point.x(), 0.0, self.editor.duration))
            active_curve = self.editor.active_curve
            factor = self.editor.display_factor(self.editor.active_name)
            y_display = float(np.clip(point.y(), active_curve.min_val * factor, active_curve.max_val * factor))
            y = y_display / factor
            if not is_shift and not is_ctrl:
                super().mouseDragEvent(ev, axis=axis)
                return
            if self.editor.active_curve.global_override is not None:
                return
            ev.accept()
            if is_shift and not is_ctrl:
                if ev.isStart():
                    self.editor._last_draw_t = None
                tol = max(1e-4, 1.0 / max(1.0, float(self.editor.draw_points_per_second)))
                last = self.editor._last_draw_t
                if last is None or abs(x - last) >= tol:
                    points = [p for p in active_curve.points if abs(p[0] - x) > tol]
                    points.append((x, y))
                    points.sort(key=lambda item: item[0])
                    active_curve.points = points
                    self.editor._last_draw_t = x
                    self.editor.update_plot()
                    self.editor.curve_changed.emit(self.editor.active_name)
                if ev.isFinish():
                    self.editor._last_draw_t = None
                return
            if is_ctrl and not is_shift:
                if ev.isStart():
                    self.editor._erase_start_t = x
                if ev.isFinish():
                    lo, hi = sorted([self.editor._erase_start_t, x])
                    points = [p for p in active_curve.points if not (lo <= p[0] <= hi)]
                    points.append((lo, active_curve.default_value))
                    points.append((hi, active_curve.default_value))
                    points.sort(key=lambda item: item[0])
                    active_curve.points = points
                    self.editor.update_plot()
                    self.editor.curve_changed.emit(self.editor.active_name)
                return
        super().mouseDragEvent(ev, axis=axis)


class ParameterWorkspace(QWidget):
    curve_changed = pyqtSignal(str)

    def __init__(self, curves: dict[str, ParameterCurve], duration: float):
        super().__init__()
        self.curves = curves
        self.duration = duration
        self.active_name = "F0"
        self.vowel_boundaries: list[float] = []
        self.hidden_regions: list[tuple[float, float]] = []
        self.draw_points_per_second = 50.0
        self._last_draw_t = None
        self._erase_start_t = 0.0
        self._plot_items: list = []
        self._boundary_items: list = []
        self._hidden_items: list = []
        self._is_dark = True
        layout = QVBoxLayout(self)
        self.title_label = QLabel("参数编辑: F0")
        layout.addWidget(self.title_label)
        self.view_box = ParameterEditorViewBox(self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)
        layout.addWidget(self.plot_widget)
        self.plot_widget.addLegend(offset=(12, 12))
        self.update_plot()

    @property
    def active_curve(self) -> ParameterCurve:
        return self.curves[self.active_name]

    def set_theme(self, is_dark: bool):
        self._is_dark = bool(is_dark)
        bg = "#1e1e1e" if is_dark else "#ffffff"
        fg = "#e0e0e0" if is_dark else "#333333"
        self.plot_widget.setBackground(bg)
        axis_bottom = self.plot_widget.getAxis("bottom")
        axis_left = self.plot_widget.getAxis("left")
        axis_bottom.setTextPen(pg.mkPen(fg))
        axis_left.setTextPen(pg.mkPen(fg))
        axis_bottom.setPen(pg.mkPen(fg))
        axis_left.setPen(pg.mkPen(fg))
        self.update_plot()

    def set_duration(self, duration: float):
        self.duration = duration
        self.update_plot()

    def set_active_parameter(self, name: str):
        if name not in self.curves:
            return
        self.active_name = name
        self.title_label.setText(f"参数编辑: {name}")
        self.update_plot()

    def set_vowel_boundaries(self, boundaries: list[float]):
        self.vowel_boundaries = list(boundaries)
        self.update_plot()

    def set_hidden_regions(self, regions: list[tuple[float, float]]):
        self.hidden_regions = list(regions)
        self.update_plot()

    def _clear_plot_items(self):
        for item in self._plot_items:
            self.plot_widget.removeItem(item)
        self._plot_items = []
        for item in self._boundary_items:
            self.plot_widget.removeItem(item)
        self._boundary_items = []
        for item in self._hidden_items:
            self.plot_widget.removeItem(item)
        self._hidden_items = []
        legend = self.plot_widget.plotItem.legend
        if legend is not None:
            legend.clear()

    def _plot_curve(self, name: str, curve: ParameterCurve, alpha: int, width: int, symbol_size: int, label: str):
        style = PARAM_STYLES.get(name, {"color": "#4dabf7", "symbol": "o"})
        color = pg.mkColor(style["color"])
        color.setAlpha(alpha)
        pen = pg.mkPen(color=color, width=width)
        symbol_brush = pg.mkBrush(color)
        factor = self.display_factor(name)
        if curve.global_override is not None:
            x_data = [0.0, self.duration]
            y_data = [float(curve.global_override) * factor, float(curve.global_override) * factor]
        else:
            x_data = [p[0] for p in curve.points]
            y_data = [p[1] * factor for p in curve.points]
        item = self.plot_widget.plot(
            x_data,
            y_data,
            pen=pen,
            symbol=style["symbol"],
            symbolBrush=symbol_brush,
            symbolSize=symbol_size,
            name=label,
        )
        self._plot_items.append(item)

    def update_plot(self):
        current_x_min, current_x_max = 0.0, self.duration
        view_range = self.plot_widget.plotItem.vb.viewRange()[0]
        if len(view_range) == 2 and np.isfinite(view_range[0]) and np.isfinite(view_range[1]) and view_range[1] > view_range[0]:
            current_x_min = float(view_range[0])
            current_x_max = float(view_range[1])
        self._clear_plot_items()
        x_min = max(0.0, min(float(self.duration), current_x_min))
        x_max = max(0.0, min(float(self.duration), current_x_max))
        if x_max <= x_min + 1e-9:
            x_min, x_max = 0.0, float(self.duration)
        self.plot_widget.setXRange(x_min, x_max, padding=0)
        active = self.active_curve
        factor = self.display_factor(self.active_name)
        if self.active_name in FORMANT_Y_RANGES:
            y_min, y_max = FORMANT_Y_RANGES[self.active_name]
            self.plot_widget.setYRange(y_min, y_max)
        else:
            self.plot_widget.setYRange(active.min_val * factor, active.max_val * factor)
        if self.active_name in FORMANT_PARAMS:
            for name in ["F1", "F2", "F3", "F4", "F5"]:
                if name == self.active_name:
                    continue
                self._plot_curve(name, self.curves[name], alpha=50, width=1, symbol_size=4, label=f"{name} (对比)")
        self._plot_curve(
            self.active_name,
            active,
            alpha=230,
            width=2,
            symbol_size=8,
            label=f"{self.active_name} (可编辑)",
        )
        for t in self.vowel_boundaries:
            line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen("#ff922b", width=1, style=Qt.PenStyle.DashLine))
            self.plot_widget.addItem(line)
            self._boundary_items.append(line)
        for start, end in self.hidden_regions:
            region = pg.LinearRegionItem(values=[start, end], brush=pg.mkBrush(128, 128, 128, 50), pen=pg.mkPen(None))
            region.setMovable(False)
            self.plot_widget.addItem(region)
            self._hidden_items.append(region)

    def display_factor(self, name: str) -> float:
        if name == "Shimmer":
            return 100.0
        return 1.0

    def on_click(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return
        curve = self.active_curve
        if curve.global_override is not None:
            return
        point = self.plot_widget.plotItem.vb.mapSceneToView(ev.scenePos())
        x = float(np.clip(point.x(), 0.0, self.duration))
        factor = self.display_factor(self.active_name)
        y_display = float(np.clip(point.y(), curve.min_val * factor, curve.max_val * factor))
        y = y_display / factor
        points = list(curve.points)
        points.append((x, y))
        points.sort(key=lambda item: item[0])
        curve.points = points
        self.update_plot()
        self.curve_changed.emit(self.active_name)


class AudioPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._is_dark = True
        self._audio = np.array([], dtype=float)
        self._fs = DEFAULT_FS
        self._current_view = "wave"
        self._spec_window_ms = 20
        self._spec_colormap = pg.ColorMap(
            pos=np.array([0.0, 0.2, 0.45, 0.7, 1.0], dtype=float),
            color=np.array(
                [
                    [0, 0, 90, 255],
                    [0, 80, 255, 255],
                    [0, 255, 255, 255],
                    [255, 255, 0, 255],
                    [255, 0, 0, 255],
                ],
                dtype=np.ubyte,
            ),
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.showGrid(x=True, y=True, alpha=0.2)
        self.wave_plot.setMouseEnabled(x=True, y=False)
        self.wave_plot.setYRange(-1.0, 1.0)
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.spec_img = pg.ImageItem()
        self.spec_img.setColorMap(self._spec_colormap)
        self.spec_plot.addItem(self.spec_img)
        self.spec_plot.plotItem.vb.setLimits(yMin=0.0, yMax=5500.0)
        self.spec_plot.plotItem.vb.setYRange(0.0, 5500.0, padding=0)
        self.stack_holder = QWidget()
        self._stack_layout = QVBoxLayout(self.stack_holder)
        self._stack_layout.setContentsMargins(0, 0, 0, 0)
        self._stack_layout.addWidget(self.wave_plot)
        layout.addWidget(self.stack_holder, stretch=1)
        self.show_waveform()

    def set_theme(self, is_dark: bool):
        self._is_dark = bool(is_dark)
        bg = "#1e1e1e" if is_dark else "#ffffff"
        fg = "#e0e0e0" if is_dark else "#333333"
        for plot in [self.wave_plot, self.spec_plot]:
            plot.setBackground(bg)
            axis_bottom = plot.getAxis("bottom")
            axis_left = plot.getAxis("left")
            axis_bottom.setTextPen(pg.mkPen(fg))
            axis_left.setTextPen(pg.mkPen(fg))
            axis_bottom.setPen(pg.mkPen(fg))
            axis_left.setPen(pg.mkPen(fg))

    def show_waveform(self):
        self._current_view = "wave"
        self._set_stack_widget(self.wave_plot)

    def show_spectrogram(self):
        self._current_view = "spec"
        self._set_stack_widget(self.spec_plot)

    def is_showing_spectrogram(self) -> bool:
        return self._current_view == "spec"

    def set_spectrogram_window_ms(self, window_ms: int):
        self._spec_window_ms = max(1, int(window_ms))
        self._update_spectrogram()

    def _set_stack_widget(self, widget: QWidget):
        while self._stack_layout.count():
            child = self._stack_layout.takeAt(0)
            if child.widget() is not None:
                child.widget().setParent(None)
        self._stack_layout.addWidget(widget)

    def set_audio(self, audio: np.ndarray, fs: int):
        self._audio = audio
        self._fs = fs
        self._update_waveform()
        self._update_spectrogram()

    def _update_waveform(self):
        if len(self._audio) == 0:
            self.wave_plot.clear()
            return
        x = np.arange(len(self._audio), dtype=float) / float(self._fs)
        duration = float(x[-1]) if len(x) else 0.0
        step = max(1, len(self._audio) // 12000)
        self.wave_plot.plot(x[::step], self._audio[::step], clear=True, pen="#4dabf7")
        self.wave_plot.plotItem.vb.setXRange(0.0, duration, padding=0)
        self.wave_plot.plotItem.vb.setLimits(xMin=0.0)
        peak = float(np.max(np.abs(self._audio)))
        peak = max(peak, 1e-6)
        self.wave_plot.plotItem.vb.setYRange(-peak, peak, padding=0)

    def _update_spectrogram(self):
        if len(self._audio) == 0:
            self.spec_img.setImage(np.zeros((10, 10)))
            return
        nperseg = int(round(self._fs * self._spec_window_ms / 1000.0))
        nperseg = max(32, min(nperseg, len(self._audio)))
        noverlap = int(round(nperseg * 0.75))
        noverlap = min(max(0, noverlap), max(0, nperseg - 1))
        f, t, sxx = spectrogram(self._audio, fs=self._fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
        sxx = 10.0 * np.log10(sxx + 1e-12)
        max_freq = 5500.0
        f_mask = f <= max_freq
        if np.any(f_mask):
            f = f[f_mask]
            sxx = sxx[f_mask, :]
        lo = float(np.percentile(sxx, 5.0))
        hi = float(np.percentile(sxx, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        self.spec_img.setImage(sxx.T, levels=(lo, hi), autoLevels=False)
        y_min = float(f.min()) if len(f) else 0.0
        y_max = min(max_freq, float(f.max())) if len(f) else max_freq
        x_span = float(t.max()) if len(t) else 1.0
        self.spec_img.setRect(pg.QtCore.QRectF(0.0, y_min, x_span, max(1e-6, y_max - y_min)))
        self.spec_plot.plotItem.vb.setYRange(0.0, max_freq, padding=0)


class SpeechSynthesisWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音合成 - Phonetic Toolbox v2")
        self.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        self.resize(1180, 760)
        self.duration = float(DEFAULT_DURATION)
        self.fs = int(DEFAULT_FS)
        self.is_dark = True
        self.synthesized_audio: Optional[np.ndarray] = None
        self.loaded_audio: Optional[np.ndarray] = None
        self.loaded_audio_path: Optional[str] = None
        self.f0_min_hz = 50.0
        self.f0_max_hz = 500.0
        self.vowel_boundaries: list[float] = []
        self.silence_intervals: list[tuple[float, float]] = []
        self.params: dict[str, ParameterCurve] = {}
        for name, (default_val, min_val, max_val, _) in PARAM_DEFAULTS.items():
            self.params[name] = ParameterCurve(name, float(default_val), float(min_val), float(max_val), self.duration)
        self.param_buttons: dict[str, QPushButton] = {}
        self.param_inputs: dict[str, QLineEdit] = {}
        self.param_rows: dict[str, QWidget] = {}
        self._is_syncing_x = False
        self._spec_window_options = [5, 10, 20, 40]
        self._spec_window_index = self._spec_window_options.index(20)
        self.spec_btn: Optional[QPushButton] = None
        self._build_ui()
        self._build_menu()
        self._connect_x_axis_sync()
        self.apply_theme()
        self._set_active_parameter("F0")
        self._sync_x_range(0.0, self.duration, "init")

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QGridLayout(root)
        main.setContentsMargins(8, 8, 8, 8)
        main.setHorizontalSpacing(8)
        main.setVerticalSpacing(8)
        self.audio_panel = AudioPanel()
        self.audio_panel.setFixedHeight(220)
        top_control = self._build_top_control_panel()
        top_control.setFixedWidth(420)
        left_panel = self._build_left_panel()
        left_panel.setFixedWidth(420)
        self.workspace = ParameterWorkspace(self.params, self.duration)
        self.workspace.curve_changed.connect(self._on_curve_changed)
        main.addWidget(top_control, 0, 0)
        main.addWidget(self.audio_panel, 0, 1)
        main.addWidget(left_panel, 1, 0)
        main.addWidget(self.workspace, 1, 1)
        main.setColumnStretch(1, 1)
        main.setRowStretch(1, 1)
        self.status_label = QLabel("Ready")
        main.addWidget(self.status_label, 2, 0, 1, 2)

    def _build_top_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("时长(s):"))
        self.duration_input = QLineEdit(f"{self.duration:.2f}")
        self.duration_input.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self.duration_input.setFixedWidth(80)
        self.duration_input.editingFinished.connect(self.update_duration_from_input)
        row1.addWidget(self.duration_input)
        row1.addWidget(QLabel("元音(IPA):"))
        self.vowel_input = QLineEdit()
        self.vowel_input.setPlaceholderText("例如: a-i/- ///u+/e++o/-")
        row1.addWidget(self.vowel_input, stretch=1)
        layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("渐入(ms):"))
        self.fade_in_input = QLineEdit("50")
        self.fade_in_input.setValidator(QIntValidator(0, 1000))
        self.fade_in_input.setFixedWidth(70)
        row2.addWidget(self.fade_in_input)
        row2.addWidget(QLabel("渐出(ms):"))
        self.fade_out_input = QLineEdit(str(FADE_MS))
        self.fade_out_input.setValidator(QIntValidator(0, 1000))
        self.fade_out_input.setFixedWidth(70)
        row2.addWidget(self.fade_out_input)
        row2.addWidget(QLabel("平滑点数:"))
        self.smooth_val_label = QLabel("5")
        row2.addWidget(self.smooth_val_label)
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(1, 50)
        self.smooth_slider.setValue(5)
        self.smooth_slider.setFixedWidth(120)
        self.smooth_slider.valueChanged.connect(lambda v: self.smooth_val_label.setText(str(v)))
        row2.addWidget(self.smooth_slider)
        layout.addLayout(row2)
        row_f0 = QHBoxLayout()
        row_f0.addWidget(QLabel("基频范围(Hz):"))
        self.f0_min_input = QLineEdit(f"{self.f0_min_hz:.0f}")
        self.f0_min_input.setValidator(QDoubleValidator(1.0, 2000.0, 1))
        self.f0_min_input.setFixedWidth(70)
        row_f0.addWidget(self.f0_min_input)
        row_f0.addWidget(QLabel("-"))
        self.f0_max_input = QLineEdit(f"{self.f0_max_hz:.0f}")
        self.f0_max_input.setValidator(QDoubleValidator(1.0, 3000.0, 1))
        self.f0_max_input.setFixedWidth(70)
        row_f0.addWidget(self.f0_max_input)
        f0_apply_btn = QPushButton("应用范围")
        f0_apply_btn.clicked.connect(self._apply_f0_range)
        row_f0.addWidget(f0_apply_btn)
        help_btn = QPushButton("帮助")
        help_btn.setStyleSheet("font-weight:700; color:#ffffff; background-color:#2b8a3e;")
        help_btn.clicked.connect(self._on_help_placeholder)
        row_f0.addWidget(help_btn)
        row_f0.addStretch()
        layout.addLayout(row_f0)
        row3 = QHBoxLayout()
        wave_btn = QPushButton("波形图")
        wave_btn.clicked.connect(self.audio_panel.show_waveform)
        row3.addWidget(wave_btn)
        self.spec_btn = QPushButton("")
        self.spec_btn.clicked.connect(self._on_spectrogram_button_clicked)
        metrics = self.spec_btn.fontMetrics()
        max_text_width = max(metrics.horizontalAdvance(f"{ms}ms语谱图") for ms in self._spec_window_options)
        self.spec_btn.setFixedWidth(max_text_width + 24)
        self.spec_btn.setStyleSheet("padding-left: 8px; padding-right: 8px;")
        self._update_spectrogram_button_text()
        row3.addWidget(self.spec_btn)
        reset_btn = QPushButton("重置范围")
        reset_btn.clicked.connect(self.reset_view_ranges)
        row3.addWidget(reset_btn)
        clear_btn = QPushButton("清空参数")
        clear_btn.clicked.connect(self.clear_all_params)
        row3.addWidget(clear_btn)
        layout.addLayout(row3)
        row4 = QHBoxLayout()
        gen_btn = QPushButton("生成元音")
        gen_btn.clicked.connect(self.generate_vowels)
        row4.addWidget(gen_btn)
        synth_btn = QPushButton("合成")
        synth_btn.clicked.connect(self.synthesize)
        row4.addWidget(synth_btn)
        play_btn = QPushButton("播放")
        play_btn.clicked.connect(self.play_audio)
        row4.addWidget(play_btn)
        export_btn = QPushButton("导出音频")
        export_btn.clicked.connect(self.export_audio)
        row4.addWidget(export_btn)
        layout.addLayout(row4)
        layout.addStretch()
        return panel

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        for name in PARAM_ORDER:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            btn = QPushButton(name)
            btn.setFixedWidth(110)
            btn.clicked.connect(lambda _=False, n=name: self._set_active_parameter(n))
            inp = QLineEdit()
            inp.setPlaceholderText("Override curve")
            inp.editingFinished.connect(lambda n=name: self._apply_param_input(n))
            row.addWidget(btn)
            row.addWidget(inp, stretch=1)
            self.param_buttons[name] = btn
            self.param_inputs[name] = inp
            self.param_rows[name] = row_widget
            inner_layout.addWidget(row_widget)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        return container

    def _build_menu(self):
        bar = self.menuBar()
        action_synthesize = bar.addAction("合成")
        action_synthesize.triggered.connect(self.synthesize)
        action_play = bar.addAction("播放")
        action_play.triggered.connect(self.play_audio)
        action_export_audio = bar.addAction("导出音频")
        action_export_audio.triggered.connect(self.export_audio)
        action_load_audio = bar.addAction("加载音频")
        action_load_audio.triggered.connect(self._on_load_audio_placeholder)
        action_extract = bar.addAction("提取参数")
        action_extract.triggered.connect(self._extract_loaded_audio_params)
        action_vowel_rules = bar.addAction("元音规则")
        action_vowel_rules.triggered.connect(self.show_vowel_rules_dialog)
        action_consonant_rules = bar.addAction("辅音规则")
        action_consonant_rules.triggered.connect(self._on_consonant_removed)
        action_preset = bar.addAction("发声类型预设")
        action_preset.triggered.connect(self._on_preset_placeholder)
        action_export_params = bar.addAction("导出参数")
        action_export_params.triggered.connect(self.export_params)
        action_import_params = bar.addAction("导入参数")
        action_import_params.triggered.connect(self.import_params)

    def set_theme(self, is_dark: bool):
        self.is_dark = bool(is_dark)
        self.apply_theme()

    def apply_theme(self):
        self.audio_panel.set_theme(self.is_dark)
        self.workspace.set_theme(self.is_dark)
        self._refresh_button_styles()

    def _refresh_button_styles(self):
        for name, btn in self.param_buttons.items():
            if name == self.workspace.active_name:
                btn.setStyleSheet("font-weight: 700; border: 2px solid #4dabf7;")
            else:
                btn.setStyleSheet("")

    def _set_active_parameter(self, name: str):
        self.workspace.set_active_parameter(name)
        self._refresh_button_styles()

    def _on_curve_changed(self, _name: str):
        self.status_label.setText(f"已更新: {self.workspace.active_name}")

    def _set_f0_range(self):
        self._set_active_parameter("F0")
        self.workspace.plot_widget.setYRange(self.params["F0"].min_val, self.params["F0"].max_val)

    def _apply_f0_range(self):
        try:
            min_f0 = float(self.f0_min_input.text().strip() or self.f0_min_hz)
            max_f0 = float(self.f0_max_input.text().strip() or self.f0_max_hz)
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的基频范围。")
            return
        if min_f0 <= 0 or max_f0 <= min_f0:
            QMessageBox.warning(self, "提示", "基频范围无效，请确保最小值 < 最大值。")
            return
        self.f0_min_hz = min_f0
        self.f0_max_hz = max_f0
        f0_curve = self.params["F0"]
        f0_curve.min_val = min_f0
        f0_curve.max_val = max_f0
        if f0_curve.global_override is not None:
            f0_curve.global_override = float(np.clip(f0_curve.global_override, min_f0, max_f0))
        f0_curve.points = [(t, float(np.clip(v, min_f0, max_f0))) for t, v in f0_curve.points]
        self.workspace.update_plot()
        if self.workspace.active_name == "F0":
            self.workspace.plot_widget.setYRange(min_f0, max_f0)
        self.status_label.setText(f"基频范围已应用: {min_f0:.1f}-{max_f0:.1f} Hz")

    def update_duration_from_input(self):
        try:
            new_duration = float(self.duration_input.text())
        except ValueError:
            return
        if new_duration <= 0:
            return
        ratio = new_duration / self.duration if self.duration > 0 else 1.0
        self.duration = new_duration
        for curve in self.params.values():
            curve.points = [(t * ratio, v) for t, v in curve.points]
            curve.points.sort(key=lambda item: item[0])
        self.workspace.set_duration(self.duration)
        self.workspace.update_plot()
        self._update_x_axis_limits()
        self._sync_x_range(0.0, self.duration, "duration")
        self.status_label.setText(f"时长已更新: {self.duration:.2f}s")

    def _connect_x_axis_sync(self):
        self.workspace.plot_widget.plotItem.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        self.audio_panel.wave_plot.plotItem.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        self.audio_panel.spec_plot.plotItem.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        axis_width = 78
        for plot in [self.audio_panel.wave_plot, self.audio_panel.spec_plot, self.workspace.plot_widget]:
            axis = plot.getAxis("left")
            axis.setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
            axis.setWidth(axis_width)
            plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
            plot.getPlotItem().vb.setDefaultPadding(0.0)
        self._update_x_axis_limits()
        self.audio_panel.wave_plot.getAxis("left").setWidth(max(0, axis_width + 9))

    def _on_x_range_changed(self, _view_box, x_range):
        if self._is_syncing_x:
            return
        x_min = float(x_range[0])
        x_max = float(x_range[1])
        norm_min, norm_max = self._normalize_x_range(x_min, x_max)
        if norm_max <= norm_min:
            return
        self._sync_x_range(norm_min, norm_max, "linked")

    def _sync_x_range(self, x_min: float, x_max: float, source: str):
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return
        x_min, x_max = self._normalize_x_range(x_min, x_max)
        if x_max <= x_min:
            return
        self._is_syncing_x = True
        try:
            self.workspace.plot_widget.plotItem.vb.setXRange(x_min, x_max, padding=0)
            self.audio_panel.wave_plot.plotItem.vb.setXRange(x_min, x_max, padding=0)
            self.audio_panel.spec_plot.plotItem.vb.setXRange(x_min, x_max, padding=0)
        finally:
            self._is_syncing_x = False

    def _normalize_x_range(self, x_min: float, x_max: float) -> tuple[float, float]:
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return 0.0, max(0.1, self.duration)
        max_t = max(0.1, float(self.duration))
        width = x_max - x_min
        if width <= 1e-9:
            return 0.0, max_t
        if width >= max_t:
            return 0.0, max_t
        if x_min < 0.0:
            x_max = x_max - x_min
            x_min = 0.0
        if x_max > max_t:
            shift = x_max - max_t
            x_min = x_min - shift
            x_max = max_t
            if x_min < 0.0:
                x_min = 0.0
        if x_max > max_t:
            x_max = max_t
        return float(x_min), float(x_max)

    def _update_x_axis_limits(self):
        max_t = max(0.1, float(self.duration))
        for plot in [self.audio_panel.wave_plot, self.audio_panel.spec_plot, self.workspace.plot_widget]:
            plot.getPlotItem().vb.setLimits(xMin=0.0, xMax=max_t, maxXRange=max_t)

    def _update_spectrogram_button_text(self):
        if self.spec_btn is None:
            return
        ms = int(self._spec_window_options[self._spec_window_index])
        self.spec_btn.setText(f"{ms}ms语谱图")

    def _on_spectrogram_button_clicked(self):
        if self.audio_panel.is_showing_spectrogram():
            self._spec_window_index = (self._spec_window_index + 1) % len(self._spec_window_options)
        current_ms = int(self._spec_window_options[self._spec_window_index])
        self.audio_panel.set_spectrogram_window_ms(current_ms)
        self.audio_panel.show_spectrogram()
        self._update_spectrogram_button_text()

    def _apply_param_input(self, name: str):
        line = self.param_inputs[name]
        text = line.text().strip()
        curve = self.params[name]
        if not text:
            curve.global_override = None
            self.workspace.update_plot()
            return
        try:
            value = float(text)
            if name == "Shimmer":
                value = value / 100.0
            curve.global_override = value
            self.workspace.update_plot()
            return
        except ValueError:
            pass
        parts = text.replace("，", ",").replace("；", ";").split(";")
        if not parts:
            return
        seg_duration = self.duration / len(parts)
        points: list[tuple[float, float]] = []
        for idx, part in enumerate(parts):
            raw_values = [p for p in part.replace(",", " ").split(" ") if p.strip()]
            if not raw_values:
                continue
            try:
                values = [float(v) for v in raw_values]
            except ValueError:
                return
            if name == "Shimmer":
                values = [v / 100.0 for v in values]
            t0 = idx * seg_duration
            t1 = (idx + 1) * seg_duration
            if len(values) == 1:
                points.append((t0, values[0]))
                points.append((t1, values[0]))
            else:
                times = np.linspace(t0, t1, len(values))
                points.extend((float(t), float(v)) for t, v in zip(times, values))
        if points:
            curve.set_points(points)
            self.workspace.update_plot()

    def _get_neighbor_vowel_formants(self, segments, idx, direction):
        pos = idx + direction
        while 0 <= pos < len(segments):
            seg = segments[pos]
            if seg.type == "vowel":
                return VOWEL_FORMANTS.get(seg.symbol, [500, 1500, 2500])
            pos += direction
        return [500, 1500, 2500]

    def generate_vowels(self):
        text = self.vowel_input.text().strip()
        if not text:
            return
        segments = parse_vowel_sequence(text)
        if not segments:
            QMessageBox.warning(self, "提示", "未解析到有效元音或空格。")
            return
        weights = [seg.duration_modifier for seg in segments]
        total_weight = sum(weights)
        if total_weight <= 0:
            return
        seg_durs = [self.duration * (w / total_weight) for w in weights]
        boundaries = np.cumsum(seg_durs)[:-1]
        n_grid = max(2, int(round(self.duration * 100)) + 1)
        t_grid = np.linspace(0.0, self.duration, n_grid)
        grids = {
            "F1": np.zeros(n_grid, dtype=float),
            "F2": np.zeros(n_grid, dtype=float),
            "F3": np.zeros(n_grid, dtype=float),
            "AV": np.zeros(n_grid, dtype=float),
        }
        self.silence_intervals = []
        self.vowel_boundaries = [float(x) for x in boundaries]
        av_default = self.params["AV"].default_value
        for idx, seg in enumerate(segments):
            start = 0.0 if idx == 0 else float(boundaries[idx - 1])
            end = self.duration if idx == len(segments) - 1 else float(boundaries[idx])
            mask = (t_grid >= start) & (t_grid <= end)
            if not np.any(mask):
                continue
            if seg.type == "silence":
                self.silence_intervals.append((start, end))
                grids["AV"][mask] = 0.0
                prev_f = self._get_neighbor_vowel_formants(segments, idx, -1)
                next_f = self._get_neighbor_vowel_formants(segments, idx, 1)
                local = np.where(mask)[0]
                alpha = np.linspace(0.0, 1.0, len(local))
                for form_idx, key in enumerate(["F1", "F2", "F3"]):
                    grids[key][mask] = prev_f[form_idx] * (1.0 - alpha) + next_f[form_idx] * alpha
            else:
                formants = VOWEL_FORMANTS.get(seg.symbol, [500, 1500, 2500])
                grids["F1"][mask] = formants[0]
                grids["F2"][mask] = formants[1]
                grids["F3"][mask] = formants[2]
                grids["AV"][mask] = av_default
        smooth_size = int(self.smooth_slider.value())
        for key in ["F1", "F2", "F3"]:
            grids[key] = uniform_filter1d(grids[key], size=max(1, smooth_size), mode="nearest")
            grids[key] = uniform_filter1d(grids[key], size=max(1, smooth_size), mode="nearest")
        for key in ["F1", "F2", "F3", "AV"]:
            points = [(float(t), float(v)) for t, v in zip(t_grid, grids[key])]
            self.params[key].set_points(points)
        for key in ["F4", "F5"]:
            dv = self.params[key].default_value
            self.params[key].set_points([(0.0, dv), (self.duration, dv)])
        self.workspace.set_vowel_boundaries(self.vowel_boundaries)
        self.workspace.set_hidden_regions(self.silence_intervals)
        self.workspace.update_plot()
        self.status_label.setText(f"已生成: {text}")

    def _resize_to(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        if target_len <= 0:
            return np.array([], dtype=float)
        if len(arr) == 0:
            return np.zeros(target_len, dtype=float)
        if len(arr) == target_len:
            return arr.copy()
        x_old = np.linspace(0.0, 1.0, len(arr))
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, arr)

    def _slice_array_by_time(self, arr: np.ndarray, start: float, end: float) -> np.ndarray:
        if end <= start or len(arr) == 0:
            return np.array([], dtype=float)
        i0 = int(np.floor(start * self.fs))
        i1 = int(np.ceil(end * self.fs))
        i0 = max(0, min(len(arr), i0))
        i1 = max(0, min(len(arr), i1))
        if i1 <= i0:
            return np.array([], dtype=float)
        return arr[i0:i1].copy()

    def _merge_silence_intervals(self) -> list[tuple[float, float]]:
        if not self.silence_intervals:
            return []
        raw = sorted((max(0.0, float(s)), min(self.duration, float(e))) for s, e in self.silence_intervals if float(e) > float(s))
        if not raw:
            return []
        merged: list[tuple[float, float]] = [raw[0]]
        for s, e in raw[1:]:
            ps, pe = merged[-1]
            if s <= pe + 1e-9:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def _apply_fade(self, audio: np.ndarray, fade_in_ms: int, fade_out_ms: int) -> np.ndarray:
        if len(audio) == 0:
            return audio
        fade_in_len = int(self.fs * max(0, fade_in_ms) / 1000.0)
        fade_out_len = int(self.fs * max(0, fade_out_ms) / 1000.0)
        if fade_in_len > 0:
            n = min(len(audio), fade_in_len)
            t = np.linspace(0.0, 1.0, n)
            in_env = 0.5 - 0.5 * np.cos(np.pi * t)
            audio[:n] *= in_env
        if fade_out_len > 0:
            n = min(len(audio), fade_out_len)
            t = np.linspace(0.0, 1.0, n)
            out_env = 0.5 + 0.5 * np.cos(np.pi * t)
            audio[-n:] *= out_env
        return audio

    def _compute_rms(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def _match_rms(self, audio: np.ndarray, target_rms: float) -> np.ndarray:
        if len(audio) == 0 or target_rms <= 0.0:
            return audio
        current_rms = self._compute_rms(audio)
        if current_rms <= 1e-10:
            return audio
        gain = float(target_rms / current_rms)
        return audio * gain

    def _synthesize_single_segment(self, arrays: dict[str, np.ndarray], effective_f0: np.ndarray, duration: float) -> np.ndarray:
        if duration <= 0:
            return np.array([], dtype=float)
        klatt_fs = 10000
        kp = KlattParam1980(
            FS=klatt_fs,
            DUR=duration,
            F0=float(effective_f0[0]) if len(effective_f0) else float(self.params["F0"].default_value),
            Jitter=0,
            Shimmer=0,
            SHR=0,
            HNR=None,
            Slope=0,
        )
        n_samp = kp.N_SAMP
        kp.F0 = self._resize_to(effective_f0, n_samp)
        kp.Jitter = self._resize_to(arrays["Jitter"], n_samp)
        kp.Shimmer = np.zeros(n_samp, dtype=float)
        kp.SHR = self._resize_to(arrays["SHR"], n_samp)
        kp.Slope = self._resize_to(arrays["Slope"], n_samp)
        kp.AV = self._resize_to(arrays["AV"], n_samp)
        kp.AVS = np.zeros(n_samp, dtype=float)
        kp.AH = self._resize_to(np.maximum(0.0, 130.0 - arrays["HNR"]), n_samp)
        for idx in range(5):
            f_key = f"F{idx + 1}"
            b_key = f"B{idx + 1}"
            kp.FF[idx] = self._resize_to(arrays[f_key], n_samp)
            kp.BW[idx] = self._resize_to(arrays[b_key], n_samp)
        kp.A1 = self._resize_to(arrays["A1"], n_samp)
        kp.A2 = self._resize_to(arrays["A2"], n_samp)
        kp.A3 = self._resize_to(arrays["A3"], n_samp)
        kp.A4 = self._resize_to(arrays["A4"], n_samp)
        kp.A5 = self._resize_to(arrays["A5"], n_samp)
        engine = klatt_make(kp)
        engine.run()
        output = engine.output
        if output is None or len(output) == 0:
            raise ValueError("Synthesis produced empty output")
        if self.fs != klatt_fs:
            ratio_gcd = gcd(int(self.fs), int(klatt_fs))
            up = int(self.fs) // ratio_gcd
            down = int(klatt_fs) // ratio_gcd
            audio = resample_poly(output, up, down)
        else:
            audio = output
        target_len = int(round(duration * self.fs))
        audio = self._resize_to(audio, target_len)

        def resize_audio(name: str) -> np.ndarray:
            return self._resize_to(arrays[name], target_len)

        spec_filter = SpectralFilter(self.fs)
        audio = spec_filter.process(
            audio,
            self._resize_to(effective_f0, target_len),
            resize_audio("H1H2"),
            resize_audio("Slope"),
            resize_audio("HNR"),
        )
        audio = spec_filter.apply_agc(audio, target_rms=0.1)
        audio = spec_filter.apply_shimmer(audio, resize_audio("Shimmer") * 100.0)
        return spec_filter.normalize(audio)

    def synthesize(self):
        self.status_label.setText("Synthesizing...")
        QApplication.processEvents()
        try:
            fade_in_ms = int(self.fade_in_input.text() or FADE_MS)
            fade_out_ms = int(self.fade_out_input.text() or FADE_MS)
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的渐入/渐出毫秒数。")
            self.status_label.setText("Synthesis failed.")
            return
        arrays = {name: curve.get_array(self.duration, self.fs) for name, curve in self.params.items()}
        effective_f0 = arrays["F0"].copy()
        mask = arrays["SHR"] >= 0.2
        effective_f0[: len(mask)][mask] *= 2.0
        try:
            intervals = self._merge_silence_intervals()
            if intervals:
                pieces: list[np.ndarray] = []
                cursor = 0.0
                reference_rms: Optional[float] = None
                for start, end in intervals:
                    if start > cursor:
                        seg_arrays = {name: self._slice_array_by_time(arr, cursor, start) for name, arr in arrays.items()}
                        seg_f0 = self._slice_array_by_time(effective_f0, cursor, start)
                        seg_audio = self._synthesize_single_segment(seg_arrays, seg_f0, start - cursor)
                        seg_audio = self._apply_fade(seg_audio, fade_in_ms, fade_out_ms)
                        seg_rms = self._compute_rms(seg_audio)
                        if reference_rms is None and seg_rms > 1e-10:
                            reference_rms = seg_rms
                        elif reference_rms is not None:
                            seg_audio = self._match_rms(seg_audio, reference_rms)
                        pieces.append(seg_audio)
                    silence_len = int(round((end - start) * self.fs))
                    if silence_len > 0:
                        pieces.append(np.zeros(silence_len, dtype=float))
                    cursor = end
                if cursor < self.duration:
                    seg_arrays = {name: self._slice_array_by_time(arr, cursor, self.duration) for name, arr in arrays.items()}
                    seg_f0 = self._slice_array_by_time(effective_f0, cursor, self.duration)
                    seg_audio = self._synthesize_single_segment(seg_arrays, seg_f0, self.duration - cursor)
                    seg_audio = self._apply_fade(seg_audio, fade_in_ms, fade_out_ms)
                    if reference_rms is not None:
                        seg_audio = self._match_rms(seg_audio, reference_rms)
                    pieces.append(seg_audio)
                if not pieces:
                    raise ValueError("Synthesis produced empty output")
                audio = np.concatenate(pieces)
            else:
                audio = self._synthesize_single_segment(arrays, effective_f0, self.duration)
                audio = self._apply_fade(audio, fade_in_ms, fade_out_ms)

            target_total_len = int(round(self.duration * self.fs))
            if len(audio) != target_total_len:
                audio = self._resize_to(audio, target_total_len)

            mx = np.max(np.abs(audio))
            if mx > 1e-8:
                audio = audio / mx * 0.95
            self.synthesized_audio = audio
            self.audio_panel.set_audio(audio, self.fs)
            self._sync_x_range(0.0, self.duration, "synth")
            self.status_label.setText("Synthesis complete.")
        except Exception as exc:
            QMessageBox.critical(self, "Synthesis Error", str(exc))
            self.status_label.setText("Synthesis failed.")

    def play_audio(self):
        if self.synthesized_audio is None:
            QMessageBox.warning(self, "提示", "暂无合成音频。")
            return
        audio = self.synthesized_audio
        mx = np.max(np.abs(audio))
        if mx > 1e-8:
            audio = audio / mx * 0.99
        sd.play(audio, self.fs)

    def export_audio(self):
        if self.synthesized_audio is None:
            QMessageBox.warning(self, "提示", "暂无合成音频。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出音频", "", "WAV Files (*.wav)")
        if not path:
            return
        sf.write(path, self.synthesized_audio.astype(np.float32), self.fs)
        self.status_label.setText(f"已导出: {path}")

    def export_params(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出参数", "", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Parameter", "Time", "Value", "Global"])
            writer.writerow(["__DURATION__", self.duration, 0, False])
            writer.writerow(["__VOWEL_INPUT__", 0, self.vowel_input.text(), False])
            for name, param in self.params.items():
                if param.global_override is not None:
                    writer.writerow([name, 0, param.global_override, True])
                else:
                    for t, v in param.points:
                        writer.writerow([name, t, v, False])
        self.status_label.setText(f"参数已导出: {path}")

    def import_params(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入参数", "", "CSV Files (*.csv)")
        if not path:
            return
        by_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
        by_global: dict[str, float] = {}
        imported_duration = self.duration
        imported_text = self.vowel_input.text()
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = str(row.get("Parameter", "")).strip()
                if not name:
                    continue
                try:
                    t_val = float(row.get("Time", "0"))
                except ValueError:
                    t_val = 0.0
                value_raw = row.get("Value", "")
                is_global = str(row.get("Global", "False")).lower() in {"1", "true", "yes"}
                if name == "__DURATION__":
                    imported_duration = t_val
                    continue
                if name == "__VOWEL_INPUT__":
                    imported_text = value_raw
                    continue
                if name not in self.params:
                    continue
                try:
                    v_val = float(value_raw)
                except ValueError:
                    continue
                if is_global:
                    by_global[name] = v_val
                else:
                    by_points[name].append((t_val, v_val))
        if imported_duration > 0:
            self.duration = imported_duration
            self.duration_input.setText(f"{self.duration:.2f}")
        self.vowel_input.setText(imported_text)
        for name, curve in self.params.items():
            curve.global_override = None
            if name in by_global:
                curve.global_override = by_global[name]
                display_val = by_global[name] * 100.0 if name == "Shimmer" else by_global[name]
                self.param_inputs[name].setText(str(display_val))
            elif name in by_points and by_points[name]:
                curve.set_points(by_points[name])
                self.param_inputs[name].setText("")
            else:
                curve.set_points([(0.0, curve.default_value), (self.duration, curve.default_value)])
                self.param_inputs[name].setText("")
        self.workspace.set_duration(self.duration)
        self.workspace.update_plot()
        self._update_x_axis_limits()
        self.status_label.setText(f"参数已导入: {path}")

    def reset_view_ranges(self):
        self._sync_x_range(0.0, self.duration, "reset")
        if self.audio_panel is not None and len(self.audio_panel._audio) > 0:
            peak = float(np.max(np.abs(self.audio_panel._audio)))
            peak = max(peak, 1e-6)
            self.audio_panel.wave_plot.setYRange(-peak, peak, padding=0)
        else:
            self.audio_panel.wave_plot.setYRange(-1.0, 1.0)
        active = self.params[self.workspace.active_name]
        factor = self.workspace.display_factor(self.workspace.active_name)
        if self.workspace.active_name in FORMANT_Y_RANGES:
            y_min, y_max = FORMANT_Y_RANGES[self.workspace.active_name]
            self.workspace.plot_widget.setYRange(y_min, y_max)
        else:
            self.workspace.plot_widget.setYRange(active.min_val * factor, active.max_val * factor)

    def clear_all_params(self):
        self.f0_min_hz = 50.0
        self.f0_max_hz = 500.0
        self.f0_min_input.setText("50")
        self.f0_max_input.setText("500")
        for name, curve in self.params.items():
            default_val = float(PARAM_DEFAULTS[name][0])
            min_val = float(PARAM_DEFAULTS[name][1])
            max_val = float(PARAM_DEFAULTS[name][2])
            curve.default_value = default_val
            curve.min_val = min_val
            curve.max_val = max_val
            curve.global_override = None
            curve.set_points([(0.0, default_val), (self.duration, default_val)])
            self.param_inputs[name].setText("")
        self.vowel_boundaries = []
        self.silence_intervals = []
        self.workspace.set_vowel_boundaries([])
        self.workspace.set_hidden_regions([])
        self.workspace.update_plot()
        self.reset_view_ranges()
        self.status_label.setText("所有参数已恢复默认值。")

    def _fit_track_to_len(self, values: Optional[np.ndarray], target_len: int) -> np.ndarray:
        if values is None or target_len <= 0:
            return np.full(max(0, target_len), np.nan, dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return np.full(target_len, np.nan, dtype=float)
        if arr.size == target_len:
            return arr.copy()
        x_new = np.linspace(0.0, 1.0, target_len)
        valid = np.isfinite(arr)
        if np.count_nonzero(valid) == 0:
            return np.full(target_len, np.nan, dtype=float)
        if np.count_nonzero(valid) == 1:
            return np.full(target_len, float(arr[valid][0]), dtype=float)
        x_old = np.linspace(0.0, 1.0, arr.size)[valid]
        y_old = arr[valid]
        return np.interp(x_new, x_old, y_old)

    def _sanitize_track_for_param(self, name: str, values: Optional[np.ndarray], target_len: int) -> np.ndarray:
        arr = self._fit_track_to_len(values, target_len)
        curve = self.params[name]
        if arr.size == 0:
            return np.array([], dtype=float)
        finite = np.isfinite(arr)
        if np.count_nonzero(finite) == 0:
            arr[:] = curve.default_value
        elif np.count_nonzero(finite) < arr.size:
            idx = np.arange(arr.size)
            arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
        return np.clip(arr, curve.min_val, curve.max_val)

    def _apply_track_to_curve(self, name: str, values: Optional[np.ndarray], target_len: int):
        arr = self._sanitize_track_for_param(name, values, target_len)
        if arr.size == 0:
            return
        times = np.linspace(0.0, self.duration, arr.size)
        self.params[name].set_points([(float(t), float(v)) for t, v in zip(times, arr)])

    def _extract_loaded_audio_params(self):
        if self.loaded_audio is None or self.loaded_audio_path is None:
            QMessageBox.warning(self, "提示", "请先加载音频。")
            return
        y = self.loaded_audio
        fs = self.fs
        path = Path(self.loaded_audio_path)
        self.status_label.setText("提取参数中...")
        QApplication.processEvents()
        try:
            frameshift_ms = 10.0
            min_f0 = float(self.f0_min_hz)
            max_f0 = float(self.f0_max_hz)
            target_len = max(2, int(round(self.duration * 1000.0 / frameshift_ms)))
            f0 = compute_praat_f0(path, frameshift_ms, min_f0, max_f0, method="cc")
            f0 = self._fit_track_to_len(f0, target_len)
            voiced_mask = compute_voiced_mask(f0)
            formants = compute_praat_formants(path, frameshift_ms, max_formant=6000.0, num_formants=5, pf0=f0)
            f1 = self._fit_track_to_len(formants.get("pF1"), target_len)
            f2 = self._fit_track_to_len(formants.get("pF2"), target_len)
            f3 = self._fit_track_to_len(formants.get("pF3"), target_len)
            f4 = self._fit_track_to_len(formants.get("pF4"), target_len)
            b1 = self._fit_track_to_len(formants.get("pB1"), target_len)
            b2 = self._fit_track_to_len(formants.get("pB2"), target_len)
            b3 = self._fit_track_to_len(formants.get("pB3"), target_len)
            b4 = self._fit_track_to_len(formants.get("pB4"), target_len)
            js = compute_jitter_shimmer(
                y,
                fs,
                frameshift_ms,
                160,
                voiced_mask=voiced_mask,
                min_f0=min_f0,
                max_f0=max_f0,
            )
            jitter = self._fit_track_to_len(js.get("Jitter_PPQ5"), target_len)
            shimmer = self._fit_track_to_len(js.get("Shimmer_APQ5"), target_len)
            shr = self._fit_track_to_len(
                compute_shr(y, fs, frameshift_ms, f0, min_f0, max_f0, voiced_mask=voiced_mask),
                target_len,
            )
            hnr_res = compute_hnr(y, fs, frameshift_ms, f0, N_periods=5, voiced_mask=voiced_mask)
            hnr_tracks = [self._fit_track_to_len(hnr_res.get(k), target_len) for k in ["HNR05", "HNR15", "HNR25", "HNR35"]]
            hnr_stack = np.vstack(hnr_tracks) if hnr_tracks else np.full((1, target_len), np.nan)
            hnr = np.nanmean(hnr_stack, axis=0)
            slope = self._fit_track_to_len(
                compute_spectral_slope(y, fs, frameshift_ms, f0, min_pitch=min_f0, voiced_mask=voiced_mask),
                target_len,
            )
            energy = self._fit_track_to_len(compute_energy(y, fs, frameshift_ms, f0, energy_window_ms=20.0), target_len)
            spec = compute_spectral_features_batch(y, fs, frameshift_ms, f0, f1, f2, f3, 5, voiced_mask=voiced_mask)
            a1 = self._fit_track_to_len(spec.get("A1"), target_len)
            a2 = self._fit_track_to_len(spec.get("A2"), target_len)
            a3 = self._fit_track_to_len(spec.get("A3"), target_len)
            h1 = self._fit_track_to_len(spec.get("H1"), target_len)
            h2 = self._fit_track_to_len(spec.get("H2"), target_len)
            h1h2 = h1 - h2
            self._apply_track_to_curve("F0", f0, target_len)
            self._apply_track_to_curve("F1", f1, target_len)
            self._apply_track_to_curve("F2", f2, target_len)
            self._apply_track_to_curve("F3", f3, target_len)
            self._apply_track_to_curve("F4", f4, target_len)
            self._apply_track_to_curve("AV", energy, target_len)
            self._apply_track_to_curve("HNR", hnr, target_len)
            self._apply_track_to_curve("SHR", shr, target_len)
            self._apply_track_to_curve("Jitter", jitter, target_len)
            self._apply_track_to_curve("Shimmer", shimmer, target_len)
            self._apply_track_to_curve("Slope", slope, target_len)
            self._apply_track_to_curve("H1H2", h1h2, target_len)
            self._apply_track_to_curve("A1", a1, target_len)
            self._apply_track_to_curve("A2", a2, target_len)
            self._apply_track_to_curve("A3", a3, target_len)
            self._apply_track_to_curve("B1", b1, target_len)
            self._apply_track_to_curve("B2", b2, target_len)
            self._apply_track_to_curve("B3", b3, target_len)
            self._apply_track_to_curve("B4", b4, target_len)
            for name in ["F5", "A4", "A5", "B5"]:
                self.params[name].set_points([(0.0, self.params[name].default_value), (self.duration, self.params[name].default_value)])
            self.workspace.update_plot()
            self._sync_x_range(0.0, self.duration, "extract")
            self.status_label.setText("参数提取完成。")
        except Exception as exc:
            QMessageBox.critical(self, "参数提取失败", str(exc))
            self.status_label.setText("参数提取失败。")

    def show_vowel_rules_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("元音规则")
        dlg.resize(560, 660)
        layout = QVBoxLayout(dlg)
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["元音", "F1 (Hz)", "F2 (Hz)", "F3 (Hz)"])
        table.setRowCount(len(VOWEL_FORMANTS))
        for row, (vowel, formants) in enumerate(VOWEL_FORMANTS.items()):
            v_item = QTableWidgetItem(vowel)
            v_item.setFlags(v_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, v_item)
            for col, value in enumerate(formants):
                f_item = QTableWidgetItem(str(value))
                f_item.setFlags(f_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, col + 1, f_item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        def on_cell_clicked(row: int, _col: int):
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(table.item(row, 0).text())

        table.cellClicked.connect(on_cell_clicked)
        layout.addWidget(table)
        tip = QLabel("点击表格可复制对应元音音标")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tip)
        explanation = QTextEdit()
        explanation.setReadOnly(True)
        explanation.setPlainText(
            "元音韵律语法\n"
            "如果不加任何符号，默认所有元音（或空格）时间等长。使用韵律语法，可以更改每个元音（或空格）的长度。\n"
            "• “+”：当前元音时长增加0.1倍\n"
            "• “-”：当前元音时长减少0.1倍\n"
            "• “*”：当前元音时长变为2倍\n"
            "• “/”：当前元音时长变为1/2倍\n"
            "同一个元音（或空格），后面的韵律语法可以叠加使用，从左至右依次生效。最终按照各个元音（或空格）长度的比例进行赋值。\n"
            "如：\n"
            "①：a-i/- ///u+/e++o/-\n"
            "②：o ///i ///i ///a ///i ///o ///i ///i ///i ///a ///i\n"
            "对于①，元音长度比例分别为：\n"
            "- a：0.9\n"
            "- i：0.4\n"
            "- 空格：0.175\n"
            "- u：0.55\n"
            "- e：1.2\n"
            "- o：0.4\n"
            "按照2秒的总长度为各个元音赋值：\n"
            "- a：0.497秒\n"
            "- i：0.221秒\n"
            "- 空格：0.097秒\n"
            "- u：0.303秒\n"
            "- e：0.662秒\n"
            "- o：0.221秒\n"
            "对于②，元音长度比例分别为：\n"
            "- 元音：1.0\n"
            "- 空格：0.175\n"
            "按照2秒的总长度为各个元音赋值：\n"
            "- 元音：0.157秒\n"
            "- 空格：0.027秒\n"
            "生成后，图中会用红色虚线标注元音（或空格）之间的分界线。此时可以编辑基频曲线为不同的元音赋予不同的基频。\n"
            "点击表格中的元音自动复制到剪贴板上。"
        )
        layout.addWidget(explanation)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dlg.reject)
        button_box.accepted.connect(dlg.accept)
        layout.addWidget(button_box)
        dlg.exec()

    def _on_load_audio_placeholder(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载音频", "", "WAV Files (*.wav)")
        if not path:
            return
        try:
            audio, fs = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            self.loaded_audio = np.asarray(audio, dtype=float)
            self.loaded_audio_path = path
            self.synthesized_audio = self.loaded_audio.copy()
            self.fs = int(fs)
            self.duration = len(self.loaded_audio) / float(self.fs)
            self.duration_input.setText(f"{self.duration:.2f}")
            self.workspace.set_duration(self.duration)
            self.audio_panel.set_audio(self.loaded_audio, self.fs)
            self._update_x_axis_limits()
            self._sync_x_range(0.0, self.duration, "load")
            self._extract_loaded_audio_params()
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def _on_extract_placeholder(self):
        self._extract_loaded_audio_params()

    def _on_consonant_removed(self):
        QMessageBox.information(self, "提示", "辅音合成已按需求移除，当前保留元音合成。")

    def _on_preset_placeholder(self):
        presets = {
            "常态浊声": {"HNR": 40.0, "Slope": -10.0, "H1H2": 0.0, "SHR": 0.0, "Jitter": 0.5, "F0": 120.0, "AV": 200.0},
            "耳语": {"HNR": 40.0, "Slope": -10.0, "H1H2": 0.0, "SHR": 0.0, "Jitter": 0.0, "F0": 120.0, "AV": 120.0},
            "气声": {"HNR": 20.0, "Slope": -10.0, "H1H2": 10.0, "SHR": 0.0, "Jitter": 0.0, "F0": 120.0, "AV": 190.0},
            "嘎裂": {"HNR": 40.0, "Slope": -10.0, "H1H2": -10.0, "SHR": 0.8, "Jitter": 3.0, "F0": 60.0, "AV": 200.0},
            "假声": {"HNR": 40.0, "Slope": -15.0, "H1H2": 0.0, "SHR": 0.0, "Jitter": 0.0, "F0": 300.0, "AV": 200.0},
        }
        dlg = QDialog(self)
        dlg.setWindowTitle("发声类型预设")
        dlg.resize(360, 220)
        layout = QVBoxLayout(dlg)
        for name, values in presets.items():
            btn = QPushButton(name)
            btn.clicked.connect(lambda _=False, n=name, v=values: self._apply_voice_preset(n, v, dlg))
            layout.addWidget(btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dlg.reject)
        layout.addWidget(close_btn)
        dlg.exec()

    def _apply_voice_preset(self, preset_name: str, values: dict[str, float], dlg: Optional[QDialog] = None):
        for name, value in values.items():
            if name not in self.params:
                continue
            curve = self.params[name]
            curve.global_override = float(np.clip(value, curve.min_val, curve.max_val))
            display_val = curve.global_override * 100.0 if name == "Shimmer" else curve.global_override
            self.param_inputs[name].setText(f"{display_val:.4g}")
        self.workspace.update_plot()
        self.status_label.setText(f"已应用预设: {preset_name}")
        if dlg is not None:
            dlg.accept()

    def _on_help_placeholder(self):
        help_file = Path(r"d:\PhoneticToolbox\PhoneticToolbox_v2\Phonetic_Export\index.html")
        if not help_file.exists():
            QMessageBox.warning(self, "帮助", f"未找到帮助文件：{help_file}")
            return
        url = QUrl.fromLocalFile(str(help_file))
        url.setFragment("s1764312609013")
        opened = QDesktopServices.openUrl(url)
        if not opened:
            QMessageBox.warning(self, "帮助", "帮助页面打开失败。")
