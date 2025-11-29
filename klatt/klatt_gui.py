import sys
from pathlib import Path
# Add project root to sys.path to allow importing PhoneticToolbox
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# Also add PhoneticToolbox root just in case
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import parselmouth
import pyqtgraph as pg
import csv
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sounddevice as sd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import soundfile as sf
from PyQt6.QtGui import QAction, QIcon
# Import backend
from tdklatt import KlattParam1980, KlattSynth, klatt_make
from spectral_filter import SpectralFilter
from praat_service import (compute_praat_f0_formants,
    compute_HNR,
    compute_SHR,
    compute_spectral_slope,
    compute_jitter_shimmer,
    compute_harmonics_H1H2H4,
    compute_H1H2_H2H4_corrected,
)

try:
    from .klatt_config import PARAM_DEFAULTS, DEFAULT_DURATION, DEFAULT_FS, FADE_MS
except ImportError:
    from klatt_config import PARAM_DEFAULTS, DEFAULT_DURATION, DEFAULT_FS, FADE_MS

# Vowel Formants (Approximate Male)
VOWEL_FORMANTS = {
    'i': [270, 2290, 3010], 'y': [270, 1800, 2200], 
    'ɨ': [290, 1400, 2100], 'ʉ': [290, 1200, 2100], 
    'ɯ': [300, 1100, 2200], 'u': [300, 870, 2240],
    'ɪ': [390, 1990, 2550], 'ʏ': [390, 1500, 2100], 
    'ʊ': [440, 1020, 2240],
    'e': [390, 2030, 2600], 'ø': [390, 1550, 2200], 
    'ɘ': [390, 1300, 2200], 'ɵ': [390, 1100, 2200], 
    'ɤ': [460, 1100, 2300], 'o': [460, 800, 2250],
    'ə': [500, 1500, 2500],
    'ɛ': [530, 1840, 2480], 'œ': [530, 1300, 2200], 
    'ɜ': [560, 1350, 2200], 'ɞ': [560, 1100, 2200], 
    'ʌ': [640, 1200, 2400], 'ɔ': [570, 840, 2410],
    'æ': [660, 1720, 2410], 'ɐ': [660, 1400, 2300],
    'a': [730, 1090, 2440], 'ɶ': [730, 1000, 2200], 
    'ɑ': [730, 1100, 2400], 'ɒ': [730, 850, 2300],
    # Apical vowels (approximate)
    'ɹ': [400, 1600, 2600],  # zi, ci, si (Dental)
    'ɻ': [400, 1350, 2200],  # zhi, chi, shi (Retroflex)
}

class XOnlyZoomViewBox(pg.ViewBox):
    def wheelEvent(self, event, axis=None):
        # Only zoom X axis (axis=0)
        super().wheelEvent(event, axis=0)
        
    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis=axis)

class CurveEditViewBox(XOnlyZoomViewBox):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        
    def mouseDragEvent(self, ev, axis=None):
        modifiers = QApplication.keyboardModifiers()
        is_shift = modifiers & Qt.KeyboardModifier.ShiftModifier
        is_ctrl = modifiers & Qt.KeyboardModifier.ControlModifier

        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.pos()
            mouse_point = self.editor.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            duration = self.editor.duration_limit
            if x < 0:
                x = 0
            if x > duration:
                x = duration
            if y < self.editor.parameter.min_val:
                y = self.editor.parameter.min_val
            if y > self.editor.parameter.max_val:
                y = self.editor.parameter.max_val

            # No modifier: pan only
            if not is_shift and not is_ctrl:
                super().mouseDragEvent(ev, axis=axis)
                return

            # Prevent edits when globally overridden
            if self.editor.parameter.global_override is not None:
                return

            # Prevent edits in locked regions
            if hasattr(self.editor, 'is_locked_time') and self.editor.is_locked_time(x):
                return

            ev.accept()

            # SHIFT: draw curve along mouse trajectory
            if is_shift and not is_ctrl:
                if ev.isStart():
                    self.editor._draw_start_t = x
                    self.editor._last_draw_t = None
                    self.editor.drag_point_index = None
                # User requested less dense points. 
                # Old: duration / 1000.0 (approx 1ms for 1s)
                # New: duration / 50.0 (approx 20ms for 1s)
                tol = max(1e-4, duration / 50.0)
                last = getattr(self.editor, '_last_draw_t', None)
                if (last is None) or (abs(x - last) >= tol):
                    new_points = [p for p in self.editor.parameter.points if abs(p[0] - x) > tol]
                    new_points.append((x, y))
                    new_points.sort(key=lambda p: p[0])
                    self.editor.parameter.points = new_points
                    self.editor._last_draw_t = x
                    self.editor.update_plot()
                if ev.isFinish():
                    self.editor._draw_start_t = None
                    self.editor._last_draw_t = None
                return

            # CTRL: restore default over dragged time span
            if is_ctrl and not is_shift:
                if ev.isStart():
                    self.editor._erase_start_t = x
                if ev.isFinish():
                    t0 = getattr(self.editor, '_erase_start_t', x)
                    t1 = x
                    lo, hi = sorted([t0, t1])
                    # Clip to unlocked regions by skipping locked times
                    # Simple approach: apply over [lo,hi] if not fully locked
                    if hasattr(self.editor, 'locked_regions'):
                        all_locked = False
                        for (L0, L1) in self.editor.locked_regions:
                            if lo >= L0 and hi <= L1:
                                all_locked = True
                                break
                        if all_locked:
                            self.editor._erase_start_t = None
                            return
                    p = self.editor.parameter
                    pts = [(tx, ty) for tx, ty in p.points if not (lo <= tx <= hi)]
                    dv = p.default_value
                    pts.append((lo, dv))
                    pts.append((hi, dv))
                    pts.sort(key=lambda q: q[0])
                    p.points = pts
                    self.editor.update_plot()
                    self.editor._erase_start_t = None
                return

        super().mouseDragEvent(ev, axis=axis)
        
    def mouseClickEvent(self, ev):
        super().mouseClickEvent(ev)

def apply_plot_style(plot):
    try:
        plot.setLabel('left', '')
        plot.setLabel('bottom', '')
        plot.plotItem.layout.setContentsMargins(10, 10, 10, 10)
        plot.getViewBox().setDefaultPadding(0.0)
    except Exception:
        pass

class ParameterCurve:
    """
    Data structure to hold curve points and generate array.
    """
    def __init__(self, name, default_value=0.0, min_val=0.0, max_val=100.0, unit=""):
        self.name = name
        self.default_value = default_value
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        # Points: List of (time, value) tuples
        self.points = [(0.0, default_value), (1.0, default_value)]
        self.global_override = None # If set, overrides points
        
    def set_points(self, points):
        self.points = sorted(points, key=lambda x: x[0])
        self.global_override = None # Reset override when points are set
        
    def get_array(self, duration, fs):
        """
        Generate numpy array for the given duration and sampling rate.
        """
        n_samples = int(round(duration * fs))
        if n_samples == 0:
            return np.array([])
            
        if self.global_override is not None:
            return np.ones(n_samples) * self.global_override
            
        times = [p[0] for p in self.points]
        values = [p[1] for p in self.points]
        
        # Ensure start and end cover the duration
        if times[0] > 0:
            times.insert(0, 0.0)
            values.insert(0, values[0])
        if times[-1] < duration:
            times.append(duration)
            values.append(values[-1])
            
        # Interpolate
        interp_func = interp1d(times, values, kind='linear', fill_value="extrapolate")
        t_grid = np.linspace(0, duration, n_samples)
        arr = interp_func(t_grid)
        
        # Clip
        arr = np.clip(arr, self.min_val, self.max_val)
        return arr

class CurveEditor(QWidget):
    """
    Widget to edit a parameter curve.
    """
    data_changed = pyqtSignal()
    
    def __init__(self, parameter, duration=1.0):
        super().__init__()
        self.parameter = parameter
        self.duration_limit = duration
        self.layout = QVBoxLayout(self)
        self.locked_regions = []
        
        # Header with Global Input
        header_layout = QHBoxLayout()
        self.layout.addLayout(header_layout)
        
        header_layout.addWidget(QLabel(f"{parameter.name} Global:"))
        self.global_input = QLineEdit()
        self.global_input.setPlaceholderText("Override curve")
        if self.parameter.global_override is not None:
            self.global_input.setText(str(self.parameter.global_override))
        self.global_input.textChanged.connect(self.on_global_input_change)
        header_layout.addWidget(self.global_input)
        
        # Plot Widget
        self.vb = CurveEditViewBox(self)
        self.plot_widget = pg.PlotWidget(viewBox=self.vb)
        apply_plot_style(self.plot_widget)
        self.plot_widget.setYRange(parameter.min_val, parameter.max_val)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=True, y=True) # Zoom x only (handled by ViewBox), Pan x/y allowed
        self.plot_widget.setXRange(0, self.duration_limit)
        self.layout.addWidget(self.plot_widget)
        
        # Curve Item
        self.curve_item = self.plot_widget.plot(symbol='o', pen='y', symbolBrush='y')
        
        # Interaction
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)
        self.drag_point_index = None
        # self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move) # Handled by ViewBox
        
        # Vowel boundary lines
        self.vowel_lines = []

        # Initial draw
        self.update_plot()

    def set_duration(self, duration):
        self.duration_limit = duration
        self.plot_widget.setXRange(0, duration)
        self.update_plot()

    def set_vowel_boundaries(self, boundaries):
        for line in self.vowel_lines:
            self.plot_widget.removeItem(line)
        self.vowel_lines = []
        for t in boundaries:
            line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            self.plot_widget.addItem(line)
            self.vowel_lines.append(line)


        
    def on_global_input_change(self, text):
        if not text.strip():
            self.parameter.global_override = None
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.curve_item.setPen('y')
            self.update_plot()
            self.data_changed.emit()
            return

        # Replace Chinese punctuation
        text = text.replace('，', ',').replace('；', ';')
        
        # Parsing Logic
        # Case 1: Single Value (Global Override)
        # Condition: No semicolon, no comma, no space (after strip)
        # Actually, space might be used as separator.
        
        is_complex = False
        if ';' in text or ',' in text:
            is_complex = True
        else:
            # Check for space separator
            if len(text.split()) > 1:
                is_complex = True
        
        if not is_complex:
            try:
                val = float(text)
                self.parameter.global_override = val
                self.curve_item.setPen('r')
                self.update_plot()
                self.data_changed.emit()
                return
            except ValueError:
                # Might be garbage, ignore or treat as complex?
                pass

        # Case 2: Complex (Segments or Linear)
        # Strategy: Split by ';'. If no ';', it's just one segment.
        parts = text.split(';')
        n_segs = len(parts)
        seg_duration = self.duration_limit / n_segs
        
        new_points = []
        valid = True
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part: continue
            
            t_start = i * seg_duration
            t_end = (i + 1) * seg_duration
            
            # Check for vector in this segment
            # Allow space or comma as separator
            sub_parts = part.replace(',', ' ').split()
            
            try:
                sub_values = [float(p) for p in sub_parts]
            except ValueError:
                valid = False
                break
                
            if not sub_values:
                continue
                
            m = len(sub_values)
            if m == 1:
                # Single value in segment -> Step / Flat for this segment
                val = sub_values[0]
                new_points.append((t_start, val))
                new_points.append((t_end, val))
            else:
                # Multiple values -> Linear interpolation in this segment
                # Distribute m points from t_start to t_end
                # e.g. 2 points: start and end.
                # e.g. 3 points: start, middle, end.
                if m == 1: 
                    # Should be covered above, but just in case
                    new_points.append((t_start, sub_values[0]))
                    new_points.append((t_end, sub_values[0]))
                else:
                    times = np.linspace(t_start, t_end, m)
                    for k in range(m):
                        new_points.append((times[k], sub_values[k]))

        if valid and new_points:
            self.parameter.global_override = None
            self.curve_item.setPen('y')
            new_points.sort(key=lambda x: x[0])
            self.parameter.set_points(new_points)
            self.update_plot()
            self.data_changed.emit()

    def update_plot(self):
        if self.parameter.global_override is not None:
            # Show flat line
            val = self.parameter.global_override
            # Use current view range for time or 0 to 1 if default
            # But parameter doesn't know duration. We use existing points x range?
            # Or just use 0 and max time of points
            times = [p[0] for p in self.parameter.points]
            if not times: times = [0, 1]
            max_t = max(times[-1], 1.0) # Assume at least 1s or max point
            
            # Show a flat line covering the whole range
            self.curve_item.setData([0, max_t], [val, val])
        else:
            times = [p[0] for p in self.parameter.points]
            values = [p[1] for p in self.parameter.points]
            self.curve_item.setData(times, values)
        
    def move_point(self, idx, x, y):
        # Constrain x if not first/last? 
        # Usually first point is at 0, last at duration.
        # Allow moving x? Yes, but maybe keep order?
        
        # Constrain y
        y = max(self.parameter.min_val, min(self.parameter.max_val, y))
        x = max(0, x)
        
        # Update point
        self.parameter.points[idx] = (x, y)
        
        # Sort if x changed?
        # If we allow x change, we should re-sort, but dragging might swap indices.
        # For simplicity, maybe lock x for first/last points?
        if idx == 0:
            x = 0
            self.parameter.points[idx] = (x, y)
        elif idx == len(self.parameter.points) - 1:
            # Maybe last point x is locked to duration?
            # Let's allow moving x for intermediate points.
            pass
            
        self.parameter.points.sort(key=lambda p: p[0])
        # After sort, idx might change. Dragging becomes tricky.
        # If we re-sort, we lose track of which point is being dragged.
        # Solution: Don't sort while dragging? Or find new index?
        # Let's defer sort until release? 
        # But we need to draw lines correctly.
        # Let's just update and sort.
        # If we sort, we need to update drag_point_index.
        
        # Find new index of the point we just moved (it's the one with closest x, y)
        # Or just don't sort during drag.
        
        self.update_plot()
        self.data_changed.emit()
        
    def on_click(self, event):
        if self.parameter.global_override is not None:
            return
        modifiers = QApplication.keyboardModifiers()
        if not (modifiers & Qt.KeyboardModifier.ShiftModifier):
            return
        if self.drag_point_index is not None:
            self.drag_point_index = None
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            if self.is_locked_time(x):
                return
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                nearest_idx = self.find_nearest(x, y)
                if nearest_idx is not None and len(self.parameter.points) > 1:
                    self.parameter.points.pop(nearest_idx)
                    self.update_plot()
                    self.data_changed.emit()
            else:
                self.parameter.points.append((x, y))
                self.parameter.points.sort(key=lambda p: p[0])
                self.update_plot()
                self.data_changed.emit()
    
    def on_mouse_move(self, pos):
        # Handled by ViewBox
        pass

    def find_nearest(self, x, y):
        # Find nearest point index
        min_dist = float('inf')
        idx = None
        # Scale differences by range to make distance meaningful?
        x_range = 1.0 # default
        y_range = self.parameter.max_val - self.parameter.min_val
        if y_range == 0: y_range = 1.0
        
        for i, (px, py) in enumerate(self.parameter.points):
            # Normalized distance
            dist = np.sqrt(((px-x))**2 + ((py-y)/y_range*0.5)**2) # Weighted y
            if dist < min_dist:
                min_dist = dist
                idx = i
        return idx

    def is_locked_time(self, x):
        for (t0, t1) in getattr(self, 'locked_regions', []):
            if t0 <= x <= t1:
                return True
        return False



class AudioVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Waveform Plot
        self.wave_plot = pg.PlotWidget(viewBox=XOnlyZoomViewBox())
        apply_plot_style(self.wave_plot)
        self.wave_plot.setYRange(-1, 1)
        self.wave_plot.showGrid(x=True, y=True)
        self.wave_plot.setMouseEnabled(x=True, y=False)
        self.layout.addWidget(self.wave_plot)
        
        # Spectrogram Plot
        self.spec_plot = pg.PlotWidget(viewBox=XOnlyZoomViewBox())
        apply_plot_style(self.spec_plot)
        self.spec_plot.setXLink(self.wave_plot)
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.layout.addWidget(self.spec_plot)
        
        # Image Item for Spectrogram
        self.img_item = pg.ImageItem()
        self.spec_plot.addItem(self.img_item)
        
        # Color map
        # pg.colormap.get('viridis') or similar
        # Simple grayscale or hot
        self.img_item.setLookupTable(pg.colormap.get('viridis').getLookupTable())

        # Pitch Plot
        self.pitch_plot = pg.PlotWidget(viewBox=XOnlyZoomViewBox())
        apply_plot_style(self.pitch_plot)
        self.pitch_plot.setMouseEnabled(x=True, y=False)
        self.pitch_plot.setXLink(self.wave_plot)
        self.layout.addWidget(self.pitch_plot)

    def update_data(self, audio, fs):
        if audio is None or len(audio) == 0:
            return
            
        duration = len(audio) / fs
        times = np.linspace(0, duration, len(audio))
        
        # Update Waveform
        # Downsample for performance if needed
        step = max(1, len(audio) // 10000)
        self.wave_plot.plot(times[::step], audio[::step], clear=True, pen='c')
        
        # Update Spectrogram
        # Compute spectrogram
        # Simple STFT
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(audio, fs, nperseg=512, noverlap=256)
        
        # Log scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Set Image
        self.img_item.setImage(Sxx_db.T)
        
        # Scale image to axes
        # ImageItem draws from (0,0) to (w, h). We need to scale to (duration, fs/2)
        # Rect: [x, y, w, h]
        self.img_item.setRect(QRectF(0, 0, duration, fs/2))
        self.spec_plot.setYRange(0, 5000)

    def update_pitch(self, times, f0):
        if times is None or f0 is None:
            return
        self.pitch_plot.plot(times, f0, clear=True, pen='m')

class FixedParamWidget(QWidget):
    def __init__(self, params, duration_limit=1.0):
        super().__init__()
        self.params = params
        self.duration_limit = duration_limit
        self.layout = QVBoxLayout(self)
        
        # Combo
        self.combo = QComboBox()
        self.combo.addItems(list(params.keys()))
        self.combo.currentIndexChanged.connect(self.on_change)
        self.layout.addWidget(self.combo)
        
        # Editor Container
        self.editor_container = QWidget()
        self.layout.addWidget(self.editor_container)
        self.editor_layout = QVBoxLayout(self.editor_container)
        self.current_editor = None
        
        # Select first
        self.on_change(0)
        
    def on_change(self, index):
        if self.current_editor:
            self.editor_layout.removeWidget(self.current_editor)
            self.current_editor.deleteLater()
            
        param_name = self.combo.currentText()
        param = self.params[param_name]
        self.current_editor = CurveEditor(param, self.duration_limit)
        self.editor_layout.addWidget(self.current_editor)

    def set_duration(self, duration):
        self.duration_limit = duration
        if self.current_editor:
            self.current_editor.set_duration(duration)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("发声态合成系统 (Voice Quality Synthesizer)")
        self.resize(1400, 900)
        icon_path = r"klatt.ico"
        icon = QIcon(icon_path)
        self.setWindowIcon(icon)
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.setWindowIcon(icon)
        # State
        self.duration = DEFAULT_DURATION
        self.duration_locked = False
        self.fs = DEFAULT_FS
        self.reference_audio = None
        self.synthesized_audio = None
        
        # Parameters
        self.params = {}
        for name, (default_val, min_val, max_val, unit) in PARAM_DEFAULTS.items():
            self.params[name] = ParameterCurve(name, default_val, min_val, max_val, unit)
        
        # Override F0 max for manual drawing as requested
        if "F0" in self.params:
            self.params["F0"].max_val = 600.0

        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        menubar = self.menuBar()
        
        # 1. 合成
        act_synth = QAction("合成", self)
        act_synth.triggered.connect(self.synthesize)
        menubar.addAction(act_synth)
        
        # 2. 播放
        act_play = QAction("播放", self)
        act_play.triggered.connect(self.play_audio)
        menubar.addAction(act_play)
        
        # 3. 导出音频
        act_export_audio = QAction("导出音频", self)
        act_export_audio.triggered.connect(self.export_audio)
        menubar.addAction(act_export_audio)
        
        # 4. 加载音频
        act_load = QAction("加载音频", self)
        act_load.triggered.connect(self.load_audio)
        menubar.addAction(act_load)
        
        # 5. 提取参数
        act_analyze = QAction("提取参数", self)
        act_analyze.triggered.connect(self.analyze_audio)
        menubar.addAction(act_analyze)
        
        # 6. 语谱图
        act_spec = QAction("语谱图", self)
        act_spec.triggered.connect(self.show_spectrogram_dialog)
        menubar.addAction(act_spec)
        
        # 7. 重置范围
        act_reset_view = QAction("重置范围", self)
        act_reset_view.triggered.connect(self.reset_view_ranges)
        menubar.addAction(act_reset_view)
        
        # 8. F4/F5
        self.act_toggle_f4f5 = QAction("F4/F5", self)
        self.act_toggle_f4f5.setCheckable(True)
        self.act_toggle_f4f5.setChecked(False)
        self.act_toggle_f4f5.triggered.connect(self.toggle_f4f5_visibility)
        menubar.addAction(self.act_toggle_f4f5)
        
        # 9. 基频范围
        act_pitch_range = QAction("基频范围", self)
        act_pitch_range.triggered.connect(self.show_pitch_range_dialog)
        menubar.addAction(act_pitch_range)

        # 10. 元音规则
        act_vowel_rules = QAction("元音规则", self)
        act_vowel_rules.triggered.connect(self.show_vowel_rules_dialog)
        menubar.addAction(act_vowel_rules)

        # 11. 发声类型预设
        preset_menu = menubar.addMenu("发声类型预设")
        for p_name in ["常态浊声", "耳语", "气声", "嘎裂", "假声"]:
            act = QAction(p_name, self)
            act.triggered.connect(lambda checked, n=p_name: self.apply_preset(n))
            preset_menu.addAction(act)
            
        # 12. 导出参数
        act_export = QAction("导出参数", self)
        act_export.triggered.connect(self.export_params)
        menubar.addAction(act_export)
        
        # 13. 导入参数
        act_import = QAction("导入参数", self)
        act_import.triggered.connect(self.import_params)
        menubar.addAction(act_import)

        # Top Controls
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)
        
        top_layout.addWidget(QLabel("时长(s):"))
        self.duration_input = QLineEdit(str(self.duration))
        self.duration_input.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self.duration_input.editingFinished.connect(self.update_duration_from_input)
        self.duration_input.setFixedWidth(60)
        top_layout.addWidget(self.duration_input)
        
        top_layout.addWidget(QLabel("元音(IPA):"))
        self.vowel_input = QLineEdit()
        self.vowel_input.setPlaceholderText("e.g. i a u")
        top_layout.addWidget(self.vowel_input)

        # Fade controls
        top_layout.addWidget(QLabel("渐入(ms):"))
        self.fade_in_input = QLineEdit("50")
        self.fade_in_input.setValidator(QIntValidator(0, 1000))
        self.fade_in_input.setFixedWidth(40)
        top_layout.addWidget(self.fade_in_input)

        top_layout.addWidget(QLabel("渐出(ms):"))
        self.fade_out_input = QLineEdit(str(FADE_MS))
        self.fade_out_input.setValidator(QIntValidator(0, 1000))
        self.fade_out_input.setFixedWidth(40)
        top_layout.addWidget(self.fade_out_input)
        
        # Smooth slider
        top_layout.addWidget(QLabel("平滑点数:"))
        self.smooth_val_label = QLabel("5")
        top_layout.addWidget(self.smooth_val_label)
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(1, 50)
        self.smooth_slider.setValue(5)
        self.smooth_slider.setFixedWidth(80)
        self.smooth_slider.valueChanged.connect(lambda v: self.smooth_val_label.setText(str(v)))
        top_layout.addWidget(self.smooth_slider)
        
        vowel_btn = QPushButton("合成元音")
        vowel_btn.clicked.connect(self.generate_vowels)
        top_layout.addWidget(vowel_btn)
        top_layout.addWidget(QLabel("能量阈值"))
        self.energy_spin = QDoubleSpinBox()
        self.energy_spin.setRange(0.0, 1.0)
        self.energy_spin.setDecimals(3)
        self.energy_spin.setSingleStep(0.001)
        self.energy_spin.setValue(0.01)
        self.energy_spin.valueChanged.connect(self.on_energy_threshold_changed)
        top_layout.addWidget(self.energy_spin)
        
        self.lock_mode_check = QCheckBox("锁定共振峰")
        self.lock_mode_check.setToolTip("使用原始音频作为声源，仅应用频域滤波和Jitter/Shimmer/SHR")
        top_layout.addWidget(self.lock_mode_check)
        
        top_layout.addStretch()

        self.visualizer = AudioVisualizer()

        self.main_grid = QGridLayout()
        main_layout.addLayout(self.main_grid)

        # Put waveform on homepage in top-left
        apply_plot_style(self.visualizer.wave_plot)
        self.main_grid.addWidget(self.visualizer.wave_plot, 0, 0)

        # Layout map: Name, Row, Col
        # Col 0 (visualizer at 0,0)
        layout_map = [
            ("F0", 1, 0), ("AV", 2, 0),
            # Col 1
            ("Jitter", 0, 1), ("Shimmer", 1, 1), ("SHR", 2, 1),
            # Col 2
            ("HNR", 0, 2), ("Slope", 1, 2), ("H1H2", 2, 2),
            # Col 3 (F1-F3)
            ("F1", 0, 3), ("F2", 1, 3), ("F3", 2, 3),
            # Col 4 (F4-F5)
            ("F4", 0, 4), ("F5", 1, 4)
        ]

        self.param_editors = []
        self.f4_f5_editors = []
        for name, r, c in layout_map:
            if name in self.params:
                ed = CurveEditor(self.params[name], self.duration)
                apply_plot_style(ed.plot_widget)
                ed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.param_editors.append(ed)
                self.main_grid.addWidget(ed, r, c)
                
                if name in ["F4", "F5"]:
                    ed.setVisible(False)
                    self.f4_f5_editors.append(ed)

        for r in range(3):
            self.main_grid.setRowStretch(r, 1)
        for c in range(5):
            # Col 4 is F4/F5, which is hidden by default
            if c == 4:
                self.main_grid.setColumnStretch(c, 0)
            else:
                self.main_grid.setColumnStretch(c, 1)
        
        # Status
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Init
        self.current_editor = None


        self._syncing = False
        self.setup_sync_links()
        
        # Initial F0 range apply is handled by ParameterCurve defaults
        
    def apply_preset(self, preset_name):
        preset_params = {
            "常态浊声": {"HNR": 40, "Slope": -10, "H1H2": 0, "SHR": 0, "Jitter": 0.5, "F0": 120, "AV": 200},
            "耳语": {"HNR": 40, "Slope": -10, "H1H2": 0, "SHR": 0, "Jitter": 0.0, "F0": 120, "AV": 20},
            "气声": {"HNR": 20, "Slope": -10, "H1H2": 10, "SHR": 0, "Jitter": 0.0, "F0": 120, "AV": 200},
            "嘎裂": {"HNR": 40, "Slope": -10, "H1H2": -10, "SHR": 0.8, "Jitter": 3,"F0": 60, "AV": 200},    
            "假声": {"HNR": 40, "Slope": -15, "H1H2": 0, "SHR": 0, "Jitter": 0, "F0": 300, "AV": 200}
        }
        
        if preset_name in preset_params:
            for param_name, value in preset_params[preset_name].items():
                if param_name in self.params:
                    # Set as global override
                    self.params[param_name].global_override = float(value)
                    
            # Update all editors
            for ed in self.param_editors:
                if ed.parameter.name in preset_params[preset_name]:
                    # Update input text and plot
                    if ed.parameter.global_override is not None:
                        ed.global_input.setText(str(ed.parameter.global_override))
                    ed.update_plot()
            
            self.status_label.setText(f"Applied preset: {preset_name}")

    def show_pitch_range_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("基频范围")
        layout = QFormLayout(dlg)
        
        # F0
        f0_param = self.params["F0"]
        f0_min = QSpinBox()
        f0_min.setRange(0, 1000)
        f0_min.setValue(int(f0_param.min_val))
        
        f0_max = QSpinBox()
        f0_max.setRange(50, 6000)
        f0_max.setValue(int(f0_param.max_val))
        
        layout.addRow("F0 Min (Hz):", f0_min)
        layout.addRow("F0 Max (Hz):", f0_max)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Update F0
            f_min_val = f0_min.value()
            f_max_val = f0_max.value()
            if f_min_val < f_max_val:
                f0_param.min_val = f_min_val
                f0_param.max_val = f_max_val
                # Update editor if exists
                for ed in self.param_editors:
                    if ed.parameter.name == "F0":
                        ed.plot_widget.setYRange(f_min_val, f_max_val)
                        ed.update_plot()

    def toggle_f4f5_visibility(self, checked):
        for ed in self.f4_f5_editors:
            ed.setVisible(checked)
        
        # Col 4
        self.main_grid.setColumnStretch(4, 1 if checked else 0)

    def on_param_select(self, current, previous):
        if not current:
            return
        
        param_name = current.text()
        param = self.params[param_name]
        
        # Clear existing editor
        if self.current_editor:
            self.current_editor_layout.removeWidget(self.current_editor)
            self.current_editor.deleteLater()
            
        # Create new editor
        self.current_editor = CurveEditor(param, self.duration)
        apply_plot_style(self.current_editor.plot_widget)
        self.current_editor.plot_widget.setXLink(self.visualizer.wave_plot)
        self.current_editor_layout.addWidget(self.current_editor)
        try:
            vb = self.current_editor.plot_widget.plotItem.vb
            vb.sigXRangeChanged.connect(lambda: self.on_any_xrange_changed(vb))
        except Exception:
            pass
        
        # Hint
        self.status_label.setText(f"Editing {param_name}. Shift+Click to add point. Ctrl+Click to delete point.")

    def update_duration_from_input(self):
        try:
            val = float(self.duration_input.text())
            if val <= 0: return
            if self.duration_locked:
                self.duration_input.setText(str(self.duration))
                QMessageBox.information(self, "Info", "Duration is locked to audio file.")
                return
            
            old_duration = self.duration
            self.duration = val
            ratio = self.duration / old_duration if old_duration > 0 else 1.0

            # Update all curves (rescale points)
            for p in self.params.values():
                new_points = []
                for t, v in p.points:
                    new_points.append((t * ratio, v))
                new_points.sort(key=lambda x: x[0])
                p.points = new_points
                
            # Update current editor
            if hasattr(self, 'current_editor') and self.current_editor:
                self.current_editor.set_duration(self.duration)
                self.current_editor.update_plot()
            
            # Also update visualizer range
            if hasattr(self, 'visualizer'):
                self.visualizer.wave_plot.setXRange(0, self.duration, padding=0)

            QApplication.processEvents()
            self.status_label.setText(f"Duration updated to {self.duration}s")
            
        except ValueError:
            pass



    def show_vowel_rules_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("元音规则")
        dlg.resize(500, 600)
        layout = QVBoxLayout(dlg)
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["元音", "F1 (Hz)", "F2 (Hz)", "F3 (Hz)"])
        table.setRowCount(len(VOWEL_FORMANTS))
        
        for row, (vowel, formants) in enumerate(VOWEL_FORMANTS.items()):
            item_v = QTableWidgetItem(vowel)
            item_v.setFlags(item_v.flags() ^ Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, item_v)
            
            for col, f_val in enumerate(formants):
                if col < 3:
                    item_f = QTableWidgetItem(str(f_val))
                    item_f.setFlags(item_f.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row, col+1, item_f)
        
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        
        label_tip = QLabel("点击表格任意单元格可复制对应元音音标")
        label_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_tip)

        def on_cell_clicked(row, col):
            vowel_item = table.item(row, 0)
            if vowel_item:
                text = vowel_item.text()
                QApplication.clipboard().setText(text)
                QMessageBox.information(dlg, "提示", f"已复制: {text}")

        table.cellClicked.connect(on_cell_clicked)
        
        explanation = QTextEdit()
        explanation.setReadOnly(True)
        explanation.setPlainText("""元音韵律语法 
如果不加任何符号，默认所有元音（或空格）时间等长。使用韵律语法，可以更改每个元音（或空格）的长度。 
• “+”：当前元音时长增加0.1倍 
• “-”：当前元音时长减少0.1倍 
• “*”：当前元音时长变为2倍 
• “/”：当前元音时长变为1/2倍 
同一个元音（或空格），后面的韵律语法可以叠加使用，从左至右依次生效。最终按照各个元音（或空格）长度的比例进行赋值。 
如： 
①：a-i/- ///u+/e++o/- 
①：o ///i ///i ///a ///i ///o ///i ///i ///i ///a ///i
对于①，元音长度比例分别为：
- a：0.9
- i：0.4
- 空格：0.175
- u：0.55
- e：1.2
- o: 0.4
按照2秒的总长度为各个元音赋值：
- a：0.497秒
- i：0.221秒
- 空格：0.097秒
- u：0.303秒
- e： 0.662秒
- o: 0.221秒
对于②，元音长度比例分别为：
- 元音：1.0
- 空格：0.175
按照2秒的总长度为各个元音赋值：
- 元音：0.157秒
- 空格：0.027秒
生成后，图中会用红色虚线标注元音（或空格）之间的分界线。此时可以编辑基频曲线为不同的元音赋予不同的基频。""")
        layout.addWidget(explanation)
        
        dlg.exec()

    def show_waveform_dialog(self):
        audio = None
        fs = self.fs
        if self.synthesized_audio is not None:
            audio = self.synthesized_audio
        elif hasattr(self, 'sound'):
            audio = self.sound.values
            fs = int(self.sound.sampling_frequency)
        if audio is None:
            QMessageBox.information(self, "提示", "暂无音频数据。")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("音频波形")
        v = QVBoxLayout(dlg)
        pw = pg.PlotWidget(viewBox=XOnlyZoomViewBox())
        apply_plot_style(pw)
        pw.setYRange(-1, 1)
        pw.showGrid(x=True, y=True)
        pw.setMouseEnabled(x=True, y=False)
        duration = len(audio) / fs
        times = np.linspace(0, duration, len(audio))
        step = max(1, len(audio) // 10000)
        pw.plot(times[::step], audio[::step], pen='c')
        v.addWidget(pw)
        dlg.resize(900, 500)
        dlg.exec()

    def show_spectrogram_dialog(self):
        audio = None
        fs = self.fs
        if self.synthesized_audio is not None:
            audio = self.synthesized_audio
        elif hasattr(self, 'sound'):
            audio = self.sound.values
            fs = int(self.sound.sampling_frequency)
        if audio is None:
            QMessageBox.information(self, "提示", "暂无音频数据。")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("语谱图")
        v = QVBoxLayout(dlg)
        pw = pg.PlotWidget(viewBox=XOnlyZoomViewBox())
        apply_plot_style(pw)
        pw.setMouseEnabled(x=True, y=False)
        img = pg.ImageItem()
        pw.addItem(img)
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(audio, fs, nperseg=512, noverlap=256)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        img.setImage(Sxx_db.T)
        duration = len(audio) / fs
        img.setRect(QRectF(0, 0, duration, fs/2))
        pw.setYRange(0, 5000)
        v.addWidget(pw)
        dlg.resize(900, 500)
        dlg.exec()

    def setup_sync_links(self):
        views = []
        try:
            views.append(self.visualizer.wave_plot.plotItem.vb)
        except Exception:
            pass
        for ed in getattr(self, 'param_editors', []):
            try:
                views.append(ed.plot_widget.plotItem.vb)
            except Exception:
                pass
        for vb in views:
            try:
                vb.sigXRangeChanged.connect(lambda vb=vb: self.on_any_xrange_changed(vb))
            except Exception:
                pass

    def on_any_xrange_changed(self, source_vb):
        if self._syncing:
            return
        self._syncing = True
        try:
            x0, x1 = source_vb.viewRange()[0]
            # Sync waveform
            try:
                self.visualizer.wave_plot.plotItem.vb.setXRange(x0, x1, padding=0)
            except Exception:
                pass
            # Sync param editors
            for ed in getattr(self, 'param_editors', []):
                try:
                    ed.plot_widget.plotItem.vb.setXRange(x0, x1, padding=0)
                except Exception:
                    pass

        finally:
            self._syncing = False

    def reset_view_ranges(self):
        try:
            self.visualizer.wave_plot.plotItem.vb.setXRange(0, self.duration, padding=0)
            self.visualizer.wave_plot.setYRange(-1, 1)
        except Exception:
            pass
        try:
            self.visualizer.spec_plot.plotItem.vb.setXRange(0, self.duration, padding=0)
            self.visualizer.spec_plot.setYRange(0, 5000)
        except Exception:
            pass
        try:
            self.visualizer.pitch_plot.plotItem.vb.setXRange(0, self.duration, padding=0)
        except Exception:
            pass
        for ed in getattr(self, 'param_editors', []):
            try:
                ed.plot_widget.plotItem.vb.setXRange(0, self.duration, padding=0)
                ed.plot_widget.setYRange(ed.parameter.min_val, ed.parameter.max_val)
            except Exception:
                pass


    def export_params(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Parameters", "", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Parameter", "Time", "Value", "Global"])
                    for name, param in self.params.items():
                        if param.global_override is not None:
                            writer.writerow([name, 0, param.global_override, True])
                        else:
                            for t, v in param.points:
                                writer.writerow([name, t, v, False])
                self.status_label.setText(f"Exported to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def import_params(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Parameters", "", "CSV Files (*.csv)")
        if path:
            try:
                # Reset all points first?
                for p in self.params.values():
                    p.points = []
                    p.global_override = None
                
                with open(path, 'r') as f:
                    reader = csv.DictReader(f) # Assuming header we just wrote?
                    # Check header
                    # Actually we used csv.writer with header row.
                    # Let's re-read simply.
                    f.seek(0)
                    header = next(csv.reader(f))
                    if header != ["Parameter", "Time", "Value", "Global"]:
                        # Try to adapt or fail
                        pass
                    
                    for row in csv.reader(f):
                        if not row: continue
                        name, t, v, glob = row
                        if name in self.params:
                            if glob == "True":
                                self.params[name].global_override = float(v)
                            else:
                                self.params[name].points.append((float(t), float(v)))
                                
                # Sort points
                for p in self.params.values():
                    if not p.points and p.global_override is None:
                        p.points = [(0.0, p.default_value), (self.duration, p.default_value)]
                    p.points.sort(key=lambda x: x[0])
                    
                # Update UI
                for ed in getattr(self, 'param_editors', []):
                    ed.update_plot()

                         
                self.status_label.setText(f"Imported from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def generate_vowels(self):
        text = self.vowel_input.text().strip()
        if not text:
            return
            
        # Parse tokens
        tokens = []
        i = 0
        while i < len(text):
            ch = text[i].lower()
            # Handle space as silence or Vowel
            is_space = (ch == ' ')
            if is_space or ch in VOWEL_FORMANTS:
                j = i + 1
                mods = []
                while j < len(text) and text[j] in "+-*/":
                    mods.append(text[j])
                    j += 1
                m = 1.0
                for c in mods:
                    if c == '+':
                        m *= 1.1
                    elif c == '-':
                        m *= 0.9
                    elif c == '*':
                        m *= 2.0
                    elif c == '/':
                        m *= 0.5
                if m > 0:
                    tokens.append((ch, m))
                i = j
            else:
                i += 1
                
        if not tokens:
            return
            
        # transition_type = "Smooth" (Default)
        n_vowels = len(tokens)
        weights = [w for _, w in tokens]
        total_w = sum(weights)
        if total_w <= 0:
            return
            
        seg_durs = [self.duration * (w / total_w) for w in weights]
        boundaries_arr = np.cumsum(seg_durs)[:-1]
        
        # Smooth points from slider
        smooth_pts = int(self.smooth_slider.value()) if hasattr(self, 'smooth_slider') else 5
        
        n_grid = max(2, int(np.round(self.duration * 100)) + 1)
        t_grid = np.linspace(0.0, self.duration, n_grid)
        
        # Init grids
        grids = {
            "F1": np.zeros(n_grid), 
            "F2": np.zeros(n_grid), 
            "F3": np.zeros(n_grid),
            "AV": np.zeros(n_grid)
        }
        
        av_default = self.params["AV"].default_value
        
        # Helper to get formant of a token index (searching backward/forward for non-space)
        def get_target_formants(idx, direction):
            curr = idx
            while 0 <= curr < n_vowels:
                t_char = tokens[curr][0]
                if t_char != ' ':
                    return VOWEL_FORMANTS.get(t_char, [500, 1500, 2500])
                curr += direction
            return [500, 1500, 2500] 

        self.silence_intervals = []

        # Fill segments
        for i in range(n_vowels):
            t_start = 0 if i == 0 else boundaries_arr[i-1]
            t_end = self.duration if i == n_vowels-1 else boundaries_arr[i]
            
            mask = (t_grid >= t_start) & (t_grid <= t_end)
            if not np.any(mask):
                continue
                
            token_char = tokens[i][0]
            
            if token_char == ' ':
                # User request: Do not adjust AV (keep it as default/voiced), 
                # enforce silence in time domain later.
                grids["AV"][mask] = av_default 
                
                # Record silence interval for post-processing
                self.silence_intervals.append((t_start, t_end))
                
                prev_f = get_target_formants(i - 1, -1)
                next_f = get_target_formants(i + 1, 1)
                segment_indices = np.where(mask)[0]
                if len(segment_indices) > 0:
                    local_steps = np.linspace(0, 1, len(segment_indices))
                    for k, key in enumerate(["F1", "F2", "F3"]):
                        val_start = prev_f[k]
                        val_end = next_f[k]
                        interp_vals = val_start * (1 - local_steps) + val_end * local_steps
                        grids[key][mask] = interp_vals
            else:
                grids["AV"][mask] = av_default
                fvals = VOWEL_FORMANTS.get(token_char, [500, 1500, 2500])
                grids["F1"][mask] = fvals[0]
                grids["F2"][mask] = fvals[1]
                grids["F3"][mask] = fvals[2]

        # Pass 2: Smoothing (Twice moving average on F1-F3)
        from scipy.ndimage import uniform_filter1d
        for key in ["F1", "F2", "F3"]:
            grids[key] = uniform_filter1d(grids[key], size=smooth_pts, mode='nearest')
            grids[key] = uniform_filter1d(grids[key], size=smooth_pts, mode='nearest')

        # Pass 3: AV Fade Out (Moved to Time Domain in synthesize)

        # Convert grids to Points for ParameterCurve
        boundaries = []
        for i in range(n_vowels - 1):
            boundaries.append(float(boundaries_arr[i]))
            
        for key in ["F1", "F2", "F3", "AV"]:
            pts = [(t_grid[j], float(grids[key][j])) for j in range(n_grid)]
            self.params[key].set_points(pts)
            
        for key in ["F4", "F5"]:
            if key in self.params:
                dv = self.params[key].default_value
                self.params[key].set_points([(0.0, dv), (self.duration, dv)])
                
        self.status_label.setText(f"Generated vowels: {text}")
        self.vowel_boundaries = boundaries
        for ed in getattr(self, 'param_editors', []):
            ed.set_vowel_boundaries(boundaries)
            ed.update_plot()



    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.wav *.mp3)")
        if path:
            self.reference_audio_path = path
            # Load with parselmouth
            self.sound = parselmouth.Sound(path)
            
            # Store original audio (1D)
            vals = self.sound.values
            if vals.ndim > 1:
                vals = vals[0]
            self.original_audio = vals.copy()
            
            self.duration = self.sound.get_total_duration()
            self.duration_locked = True
            self.duration_input.setText(f"{self.duration:.2f}")
            self.duration_input.setDisabled(True) # Lock input
            self.fs = int(self.sound.sampling_frequency)
            self.status_label.setText(f"Loaded {path}. Duration: {self.duration:.2f}s, FS: {self.fs}")
            
            # Set as current synthesized audio so it can be played/viewed immediately
            self.synthesized_audio = self.original_audio.copy()
            
            # Update all curves duration
            for p in self.params.values():
                # Reset points to end at new duration
                p.points = [(0.0, p.default_value), (self.duration, p.default_value)]
                
            for ed in getattr(self, 'param_editors', []):
                ed.set_duration(self.duration)
                ed.update_plot()
            # Draw waveform on homepage
            try:
                # Plot
                if self.synthesized_audio is not None and len(self.synthesized_audio) > 0:
                    duration = len(self.synthesized_audio) / self.fs
                    times = np.linspace(0, duration, len(self.synthesized_audio))
                    step = max(1, len(self.synthesized_audio) // 10000)
                    
                    # Ensure 1D
                    if self.synthesized_audio.ndim > 1:
                        y_plot = self.synthesized_audio[0]
                    else:
                        y_plot = self.synthesized_audio
                        
                    self.visualizer.wave_plot.plot(times[::step], y_plot[::step], clear=True, pen='c')
                
                self.on_energy_threshold_changed(self.energy_spin.value())
            except Exception as e:
                # Log error but don't crash
                print(f"Plotting error: {e}")
                self.status_label.setText(f"Error plotting waveform: {e}")



    def analyze_audio(self):
        if not hasattr(self, 'sound'):
            QMessageBox.warning(self, "Error", "No audio loaded.")
            return
            
        self.status_label.setText("Analyzing...")
        QApplication.processEvents()
        frameshift_ms = 5
        min_f0 = 40
        max_f0 = 500
        res = compute_praat_f0_formants(Path(self.reference_audio_path), frameshift_ms, min_f0, max_f0)
        t_vals = np.arange(res["pF0"].shape[0], dtype=float) * (frameshift_ms / 1000.0)
        f0_arr = np.array(res["pF0"], dtype=float)
        f0_pts = [(float(t), float(v)) for t, v in zip(t_vals, f0_arr) if not np.isnan(v) and v > 0]
        if f0_pts:
            self.params["F0"].set_points(f0_pts)
            try:
                self.visualizer.update_pitch(t_vals, f0_arr)
            except Exception:
                pass
        points_f1 = [(float(t), float(v)) for t, v in zip(t_vals, np.array(res["pF1"], dtype=float)) if not np.isnan(v)]
        points_f2 = [(float(t), float(v)) for t, v in zip(t_vals, np.array(res["pF2"], dtype=float)) if not np.isnan(v)]
        points_f3 = [(float(t), float(v)) for t, v in zip(t_vals, np.array(res["pF3"], dtype=float)) if not np.isnan(v)]
        points_f4 = [(float(t), float(v)) for t, v in zip(t_vals, np.array(res["pF4"], dtype=float)) if not np.isnan(v)]
        if points_f1: self.params["F1"].set_points(points_f1)
        if points_f2: self.params["F2"].set_points(points_f2)
        if points_f3: self.params["F3"].set_points(points_f3)
        if points_f4: self.params["F4"].set_points(points_f4)
        
        y = self.sound.values.astype(float)
        fs = int(self.sound.sampling_frequency)

        
        hnr_dict = compute_HNR(y, fs, frameshift_ms, f0_arr, N_periods=5)
        hnr_key = "HNR15" if "HNR15" in hnr_dict else ("HNR05" if "HNR05" in hnr_dict else None)
        if hnr_key:
            hnr_arr = np.array(hnr_dict[hnr_key], dtype=float)
            hnr_points = [(float(t), float(v)) for t, v in zip(t_vals, hnr_arr) if not np.isnan(v)]
            if hnr_points:
                self.params["HNR"].set_points(hnr_points)

        shr_arr = np.array(compute_SHR(y, fs, frameshift_ms, f0_arr, min_f0, max_f0), dtype=float)
        shr_points = [(float(t), float(v)) for t, v in zip(t_vals, shr_arr) if not np.isnan(v)]
        if shr_points:
            self.params["SHR"].set_points(shr_points)

        slope_arr = np.array(compute_spectral_slope(y, fs, frameshift_ms, f0_arr), dtype=float)
        slope_points = [(float(t), float(v)) for t, v in zip(t_vals, slope_arr) if not np.isnan(v)]
        if slope_points:
            self.params["Slope"].set_points(slope_points)

        jit_arr, shim_arr = compute_jitter_shimmer(y, fs, frameshift_ms, window_ms=40)
        jit_points = [(float(t), float(v) * 100.0) for t, v in zip(t_vals, np.array(jit_arr, dtype=float)) if not np.isnan(v)]
        shim_points = [(float(t), float(v)) for t, v in zip(t_vals, np.array(shim_arr, dtype=float)) if not np.isnan(v)]
        if jit_points:
            self.params["Jitter"].set_points(jit_points)
        if shim_points:
            self.params["Shimmer"].set_points(shim_points)

        harms = compute_harmonics_H1H2H4(y, fs, frameshift_ms, f0_arr, N_periods=5)
        H1 = np.array(harms.get("H1"), dtype=float)
        H2 = np.array(harms.get("H2"), dtype=float)
        H4 = np.array(harms.get("H4"), dtype=float)
        F1_arr = np.array(res["pF1"], dtype=float)
        F2_arr = np.array(res["pF2"], dtype=float)
        B1_arr = self.params["B1"].get_array(self.duration, self.fs)
        B2_arr = self.params["B2"].get_array(self.duration, self.fs)
        corr = compute_H1H2_H2H4_corrected(H1, H2, H4, fs, f0_arr, F1_arr, F2_arr, B1_arr, B2_arr)
        h1h2_points = [(float(t), float(v)) for t, v in zip(t_vals, np.array(corr.get("H1H2c"), dtype=float)) if not np.isnan(v)]
        if h1h2_points:
            self.params["H1H2"].set_points(h1h2_points)
            
        # 4. Amplitudes A1-A3 (Estimated from spectrogram)
        spectrogram = self.sound.to_spectrogram()
        # This is slow, skip for now or use simplified logic
        # We'll rely on default A1-A3 generation in synthesis if not provided?
        # Or set to 0.
        
        
                
        self.status_label.setText("Analysis complete. Curves updated.")
        try:
            self.apply_locks_to_params()
        except Exception:
            pass
        try:
            for ed in getattr(self, 'param_editors', []):
                ed.locked_regions = list(getattr(self, 'locked_regions', []))
                ed.update_plot()
        except Exception:
            pass
        if self.current_editor:
            self.current_editor.update_plot()


    def synthesize(self):
        self.status_label.setText("Synthesizing...")
        QApplication.processEvents()
        
        # Check lock mode
        if hasattr(self, 'lock_mode_check') and self.lock_mode_check.isChecked():
            if not hasattr(self, 'original_audio') or self.original_audio is None:
                QMessageBox.warning(self, "Error", "No audio loaded to lock to.")
                return

            self.status_label.setText("Processing locked audio...")
            QApplication.processEvents()

            try:
                # Generate arrays
                arrays = {}
                for name, param in self.params.items():
                    arrays[name] = param.get_array(self.duration, self.fs)
                
                # Logic for SHR >= 0.2: Double F0
                effective_f0 = arrays["F0"].copy()
                if "SHR" in arrays:
                    shr_arr = arrays["SHR"]
                    min_len = min(len(effective_f0), len(shr_arr))
                    mask = shr_arr[:min_len] >= 0.2
                    effective_f0[:min_len][mask] *= 2.0
                
                # Helper to get array
                def get_arr(name):
                    if name in arrays:
                        return arrays[name]
                    elif name in self.params:
                        return self.params[name].get_array(self.duration, self.fs)
                    else:
                        return np.zeros(len(self.original_audio))

                f0_c = effective_f0
                h1h2_c = get_arr("H1H2")
                slope_c = get_arr("Slope")
                hnr_c = get_arr("HNR")
                
                # Use original audio
                audio = self.original_audio.copy()
                
                # Spectral Filter
                spec_filter = SpectralFilter(self.fs)
                
                # 1. H1-H2, Slope, HNR
                # Ensure contours match audio length
                target_len = len(audio)
                def resize_contour(c):
                    if len(c) != target_len:
                        return np.resize(c, target_len) # Or interp? 
                        # Usually param arrays match if duration/fs match.
                        # But if slight mismatch due to rounding:
                        # interp is better.
                    return c
                    
                # Note: get_array uses self.duration which comes from sound.
                # So lengths should be very close.
                
                audio = spec_filter.process(audio, f0_c, h1h2_c, slope_c, hnr_c)
                
                # 2. Jitter (Time domain resampling)
                jitter_c = get_arr("Jitter")
                audio = spec_filter.apply_jitter(audio, jitter_c, f0_c)

                # 3. AGC
                audio = spec_filter.apply_agc(audio, target_rms=0.1)
                
                # 4. Shimmer (AM)
                shimmer_c = get_arr("Shimmer")
                audio = spec_filter.apply_shimmer(audio, shimmer_c * 100.0)
                
                # 5. Normalize
                audio = spec_filter.normalize(audio)
                
                self.synthesized_audio = audio
                
                # Update Visualizer
                try:
                    duration = len(self.synthesized_audio) / self.fs
                    times = np.linspace(0, duration, len(self.synthesized_audio))
                    step = max(1, len(self.synthesized_audio) // 10000)
                    self.visualizer.wave_plot.plot(times[::step], self.synthesized_audio[::step], clear=True, pen='c')
                except Exception:
                    pass
                
                self.status_label.setText("Processing complete (Locked Mode).")
                
            except Exception as e:
                QMessageBox.critical(self, "Processing Error", str(e))
                self.status_label.setText("Processing failed.")
            return

        # Generate arrays
        arrays = {}
        for name, param in self.params.items():
            arrays[name] = param.get_array(self.duration, self.fs)
            
        # --- Logic for SHR >= 0.2: Double F0 ---
        # When SHR >= 0.2, effective F0 = F0 * 2
        # This effective F0 is used for synthesis and H1-H2 calculation.
        # But UI F0 curve remains unchanged.
        effective_f0 = arrays["F0"].copy()
        if "SHR" in arrays:
            shr_arr = arrays["SHR"]
            # Ensure lengths match
            min_len = min(len(effective_f0), len(shr_arr))
            mask = shr_arr[:min_len] >= 0.2
            effective_f0[:min_len][mask] *= 2.0
            
        # Create Klatt Param
        # Map parameters
        
        # AV amplitude (voicing)
        av_arr = arrays["AV"]

        # Prefer normal voicing; set AVS small or zero to avoid masking AV effect
        avs_arr = np.zeros_like(av_arr)

        # Map HNR to AH independently of AV to keep AV controlling voicing pulses
        # Higher HNR -> lower AH (less noise). Use reference 50 dB.
        # User requested raw HNR usage without mapping. 
        # But to restore "noisy version" (breath sound), we need to map HNR to AH (Aspiration Noise).
        # Formula: AH = 50 - HNR. 
        # If HNR = -10 (noisy), AH = 60 (loud noise).
        # If HNR = 50 (clean), AH = 0 (no noise).
        if "HNR" in arrays:
            ah_arr = np.maximum(0.0, 130.0 - arrays["HNR"])  # dB
        else:
            ah_arr = np.zeros_like(av_arr)
        
        # Initialize KlattParam
        kp = KlattParam1980(
            FS=self.fs,
            DUR=self.duration,
            F0=effective_f0[0], # Use effective F0 initial value
            Jitter=0, Shimmer=0, SHR=0, HNR=None, Slope=0 # Placeholders
        )
        
        # Inject arrays
        kp.F0 = effective_f0 # Use effective F0
        kp.Jitter = arrays["Jitter"]
        # Use Shimmer as fraction (0-1) so KlattVoice can add as dB perturbation internally
        kp.Shimmer = 0 # Disable internal Shimmer (handled in post-processing)
        
        kp.SHR = arrays["SHR"]
        kp.Slope = arrays["Slope"]
        kp.AVS = avs_arr
        kp.AH = ah_arr
        kp.AV = av_arr
        
        # Inject Formants and Bandwidths
        # kp.FF is a list of arrays [F1_arr, F2_arr, ...]
        # kp.BW is a list of arrays [B1_arr, B2_arr, ...]
        # Klatt 1980 supports up to 6 formants in some implementations, usually 5.
        # We handle F1-F5.
        for i in range(5): # F1-F5, B1-B5
            f_key = f"F{i+1}"
            b_key = f"B{i+1}"
            if f_key in arrays:
                if i < len(kp.FF):
                    kp.FF[i] = arrays[f_key]
            if b_key in arrays:
                if i < len(kp.BW):
                    kp.BW[i] = arrays[b_key]
        
        # Inject Amplitudes A1-A5
        if "A1" in arrays: kp.A1 = arrays["A1"]
        if "A2" in arrays: kp.A2 = arrays["A2"]
        if "A3" in arrays: kp.A3 = arrays["A3"]
        if "A4" in arrays: kp.A4 = arrays["A4"]
        if "A5" in arrays: kp.A5 = arrays["A5"]
        
        # Run Synthesis
        try:
            self.synth_engine = klatt_make(kp)
            self.synth_engine.run()
            self.synthesized_audio = self.synth_engine.output
            
            # --- Post-processing: Spectral Filtering ---
            # H1-H2, Slope, HNR adjustment in frequency domain
            try:
                # Prepare contours
                # Using parameter arrays which are already sample-wise
                # SpectralFilter expects sample-wise or similar length arrays
                
                # Note: Parameters might not exist in arrays if not in layout, 
                # but they should be in self.params
                
                # Helper to get array even if not in 'arrays' dict (e.g. if default)
                def get_arr(name):
                    if name in arrays:
                        return arrays[name]
                    elif name in self.params:
                        return self.params[name].get_array(self.duration, self.fs)
                    else:
                        # Should not happen for core params, but fallback
                        return np.zeros(len(self.synthesized_audio))

                f0_c = effective_f0 # Use effective F0
                h1h2_c = get_arr("H1H2")
                slope_c = get_arr("Slope")
                hnr_c = get_arr("HNR")
                
                # Filter
                spec_filter = SpectralFilter(self.fs)
                self.synthesized_audio = spec_filter.process(
                    self.synthesized_audio, 
                    f0_c, 
                    h1h2_c, 
                    slope_c, 
                    hnr_c
                )
                
                # --- New Post-processing (AGC -> Shimmer -> Normalize) ---
                
                # 1. AGC (Stabilize amplitude differences)
                self.synthesized_audio = spec_filter.apply_agc(self.synthesized_audio, target_rms=0.1)
                
                # 2. Shimmer (Post-synthesis AM)
                # Shimmer parameter is 0-0.1 (Fraction), convert to % for apply_shimmer (which expects % and divides by 100)
                shimmer_c = get_arr("Shimmer")
                self.synthesized_audio = spec_filter.apply_shimmer(self.synthesized_audio, shimmer_c * 100.0)
                
                # 3. Normalize (-1 to 1)
                self.synthesized_audio = spec_filter.normalize(self.synthesized_audio)
                
                # --- 4. Time Domain Windowing (Silence & Fade Out) ---
                # User request:
                # 1. Audio end: fade out.
                # 2. Forced silence (spaces): Zero out, and fade out preceding.
                # 3. New: Fade In at start and after silence.
                # 4. New: Use quadratic curve.
                
                audio = self.synthesized_audio
                n_samples = len(audio)
                
                # Get fade durations from UI
                try:
                    fade_in_ms = int(self.fade_in_input.text())
                except ValueError:
                    fade_in_ms = FADE_MS
                    
                try:
                    fade_out_ms = int(self.fade_out_input.text())
                except ValueError:
                    fade_out_ms = FADE_MS
                
                fade_in_len = int(fade_in_ms * self.fs / 1000.0)
                fade_out_len = int(fade_out_ms * self.fs / 1000.0)
                
                # Quadratic Fade Functions
                def apply_fade_in(arr, start_idx, length):
                    if length <= 0: return
                    end_idx = min(len(arr), start_idx + length)
                    actual_len = end_idx - start_idx
                    if actual_len <= 0: return
                    
                    t = np.linspace(0.0, 1.0, actual_len)
                    curve = t ** 2
                    arr[start_idx:end_idx] *= curve
                    
                def apply_fade_out(arr, end_idx, length):
                    if length <= 0: return
                    start_idx = max(0, end_idx - length)
                    actual_len = end_idx - start_idx
                    if actual_len <= 0: return
                    
                    t = np.linspace(0.0, 1.0, actual_len)
                    # Fade out: (1-t)^2
                    curve = (1.0 - t) ** 2
                    arr[start_idx:end_idx] *= curve

                # 1. Global Start Fade In
                if n_samples > 0:
                    apply_fade_in(audio, 0, fade_in_len)
                
                # 2. Global End Fade Out
                if n_samples > 0:
                    apply_fade_out(audio, n_samples, fade_out_len)
                    
                # 3. Silence Intervals
                if hasattr(self, 'silence_intervals') and self.silence_intervals:
                    for (t0, t1) in self.silence_intervals:
                        i0 = int(t0 * self.fs)
                        i1 = int(t1 * self.fs)
                        
                        # Clamp
                        i0 = max(0, min(n_samples, i0))
                        i1 = max(0, min(n_samples, i1))
                        
                        # Zero out
                        audio[i0:i1] = 0.0
                        
                        # Fade out preceding region (using fade_out_len)
                        if i0 > 0:
                            apply_fade_out(audio, i0, fade_out_len)
                            
                        # Fade in succeeding region (using fade_in_len)
                        if i1 < n_samples:
                            apply_fade_in(audio, i1, fade_in_len)
                                
                self.synthesized_audio = audio
                
            except Exception as e:
                print(f"Spectral Filtering Error: {e}")
                # Don't fail synthesis if filtering fails, just warn?
                # Or maybe user wants to know.
                pass

            # Update Visualizer waveform on homepage
            try:
                duration = len(self.synthesized_audio) / self.fs
                times = np.linspace(0, duration, len(self.synthesized_audio))
                step = max(1, len(self.synthesized_audio) // 10000)
                self.visualizer.wave_plot.plot(times[::step], self.synthesized_audio[::step], clear=True, pen='c')
            except Exception:
                pass
            
            self.status_label.setText("Synthesis complete.")
        except Exception as e:
            QMessageBox.critical(self, "Synthesis Error", str(e))
            self.status_label.setText("Synthesis failed.")

    def export_audio(self):
        if self.synthesized_audio is None:
            QMessageBox.warning(self, "提示", "暂无合成音频。请先进行合成。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出音频", "", "WAV Files (*.wav)")
        if not path:
            return
        try:
            audio = self.synthesized_audio.astype(np.float32)
            sf.write(path, audio, self.fs)
            self.status_label.setText(f"已导出: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def on_energy_threshold_changed(self, v):
        self.energy_threshold = float(v)
        self.update_energy_locks()

    def update_energy_locks(self):
        if hasattr(self, 'original_audio') and self.original_audio is not None:
            y = self.original_audio.astype(float)
        elif hasattr(self, 'sound'):
            vals = self.sound.values
            if vals.ndim > 1:
                vals = vals[0]
            y = vals.astype(float)
        else:
            return
            
        fs = int(self.sound.sampling_frequency)
        frameshift_ms = 5
        win_ms = 25
        nwin = max(1, int(round(fs * win_ms / 1000.0)))
        step = max(1, int(round(fs * frameshift_ms / 1000.0)))
        n = len(y)
        m = 1 + max(0, (n - nwin) // step)
        rms = np.zeros(m, dtype=float)
        for i in range(m):
            s = i * step
            e = min(n, s + nwin)
            seg = y[s:e]
            if len(seg) == 0:
                rms[i] = 0.0
            else:
                rms[i] = float(np.sqrt(np.mean(seg ** 2)))
        max_rms = float(np.max(rms)) if np.max(rms) > 0 else 1.0
        rms_norm = rms / max_rms
        thr = float(getattr(self, 'energy_threshold', 0.02))
        low = rms_norm < thr
        if np.sum(low) > 0.95 * m:
            low[:] = False
        regions = []
        st = None
        for i in range(m):
            if low[i] and st is None:
                st = i
            elif (not low[i]) and (st is not None):
                t0 = st * frameshift_ms / 1000.0
                t1 = i * frameshift_ms / 1000.0
                regions.append((t0, t1))
                st = None
        if st is not None:
            t0 = st * frameshift_ms / 1000.0
            t1 = self.duration
            regions.append((t0, t1))
        self.locked_regions = regions
        disp = y.copy()
        for (t0, t1) in regions:
            s = int(round(t0 * fs))
            e = int(round(t1 * fs))
            s = max(0, min(n, s))
            e = max(0, min(n, e))
            disp[s:e] = 0.0
        # User reported waveform display issue. 
        # load_audio plots the full waveform. This method (update_energy_locks) was overwriting it with 'disp' (masked waveform).
        # We should NOT overwrite the main waveform view just to show locks, as it confuses the user (looks like empty audio).
        # The locks are visualized in the editors anyway.
        # try:
        #     duration = len(disp) / fs
        #     times = np.linspace(0, duration, len(disp))
        #     step_plot = max(1, len(disp) // 10000)
        #     self.visualizer.wave_plot.plot(times[::step_plot], disp[::step_plot], clear=True, pen='c')
        # except Exception:
        #     pass
        self.apply_locks_to_params()
        try:
            for ed in getattr(self, 'param_editors', []):
                ed.locked_regions = list(self.locked_regions)
                ed.update_plot()

        except Exception:
            pass

    def apply_locks_to_params(self):
        if not hasattr(self, 'locked_regions'):
            return
        for key, p in self.params.items():
            pts = list(p.points)
            filt = []
            for (x, y) in pts:
                ok = True
                for (t0, t1) in self.locked_regions:
                    if t0 <= x <= t1:
                        ok = False
                        break
                if ok:
                    filt.append((x, y))
            for (t0, t1) in self.locked_regions:
                filt.append((t0, p.default_value))
                filt.append((t1, p.default_value))
            filt.sort(key=lambda q: q[0])
            p.points = filt

    def play_audio(self):
        if self.synthesized_audio is not None:
            # Normalize to [-1, 1] for sounddevice float playback
            audio = self.synthesized_audio
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.99  # Leave a little headroom
            sd.play(audio, self.fs)
        else:
            QMessageBox.warning(self, "Warning", "No synthesized audio.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
