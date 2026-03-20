# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
import threading
from scipy.io import wavfile
from scipy import signal
from scipy.signal import windows
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, 
    QGridLayout, QMessageBox, QSizePolicy, QSpacerItem, QCheckBox, 
    QSlider, QDialog, QFormLayout, QDialogButtonBox, QProgressDialog,
    QFileDialog, QToolBar
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QDesktopServices, QIcon

import sounddevice as sd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from phonetic_toolbox.services.egg_service import EGGAnalysisService
from phonetic_toolbox.core.signals.filters import apply_highpass_filter, apply_lowpass_filter
from phonetic_toolbox.models.config import EGGConfig, DEFAULT_ROI_START, DEFAULT_ROI_DURATION
from phonetic_toolbox.models.egg_models import EGGAnalysisResult
from phonetic_toolbox.gui.workers.egg_workers import LoadWorker, EventsWorker, F0Worker
from phonetic_toolbox.gui.dialogs.egg_batch_dialog import BatchProcessingDialog

# --- Constants ---
DEFAULT_PEAK_PROMINENCE = 0.01
DEFAULT_VALLEY_PROMINENCE = 0.01
DEFAULT_ZOOM_WINDOW_MS = 50.0
DEFAULT_HIGHPASS_CUTOFF = 25
DEFAULT_LOWPASS_CUTOFF = 1000
DEFAULT_SPEC_WINDOW_MS = 20.0
DEFAULT_SPEC_VMIN = -70.0
DEFAULT_SPEC_VMAX = -10.0

# Set global font to Times New Roman and SimSun (for Chinese)
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False # Ensure minus sign displays correctly

DARK_PLOT_THEME = {
    'axes.edgecolor': '#AAAAAA',
    'axes.labelcolor': '#DDDDDD',
    'xtick.color': '#DDDDDD',
    'ytick.color': '#DDDDDD',
    'grid.color': '#555555',
    'figure.facecolor': 'none', # Transparent to match window
    'axes.facecolor': '#2E2E2E',
    'text.color': '#DDDDDD',
    'line.color': 'lightgray',
    'title.color': 'lightgray'
}

LIGHT_PLOT_THEME = {
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'grid.color': '#CCCCCC',
    'figure.facecolor': 'none', # Transparent to match window
    'axes.facecolor': '#FFFFFF',
    'text.color': '#000000',
    'line.color': 'black',
    'title.color': 'black'
}

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setStyleSheet("background-color:transparent;") # Ensure widget is transparent
        self.is_dark = True # Default
        self.apply_theme(self.is_dark)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    def apply_theme(self, is_dark: bool):
        self.is_dark = is_dark
        theme = DARK_PLOT_THEME if is_dark else LIGHT_PLOT_THEME
        self.fig.set_facecolor(theme['figure.facecolor'])
        self.axes.set_facecolor(theme['axes.facecolor'])
        self.axes.tick_params(axis='x', colors=theme['xtick.color'])
        self.axes.tick_params(axis='y', colors=theme['ytick.color'])
        self.axes.xaxis.label.set_color(theme['axes.labelcolor'])
        self.axes.yaxis.label.set_color(theme['axes.labelcolor'])
        self.axes.title.set_color(theme['title.color'])
        for spine in self.axes.spines.values():
            spine.set_color(theme['axes.edgecolor'])
        self.draw()

class InverseFilteringResultDialog(QDialog):
    def __init__(self, audio, filtered, egg, fs, start_s, end_s, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inverse Filtering Results")
        self.resize(1200, 800)
        self.setModal(False)
        
        layout = QVBoxLayout(self)
        canvas = MplCanvas(self, width=12, height=8)
        layout.addWidget(canvas)
        
        fig = canvas.fig
        fig.clear()
        
        # Using the logic from original plot_inverse_filtering_results
        plt.rcParams.update({'font.size': 10})
        common_style = {'color': 'lightgray'}
        freq_limit = 5000
        db_floor = -80
        epsilon = 1e-12
        min_fft_len = 44100

        fig.suptitle(f"Inverse Filtering Results ({start_s:.2f}s - {end_s:.2f}s)", color='lightgray')

        # --- 1. Top Left: Audio Spectra Overlay ---
        ax1 = fig.add_subplot(2, 2, 1)
        plot_success_ax1 = False
        N_orig = len(audio) if audio is not None else 0
        
        if N_orig > 0:
            audio_to_fft = audio
            N_fft_orig = N_orig
            if N_orig < min_fft_len:
                pad_width = min_fft_len - N_orig
                audio_to_fft = np.pad(audio_to_fft, (0, pad_width), mode='constant', constant_values=0)
                N_fft_orig = min_fft_len
            
            win_orig = windows.hamming(N_fft_orig, sym=False)
            audio_windowed = audio_to_fft * win_orig
            yf = fft(audio_windowed)
            xf = fftfreq(N_fft_orig, 1 / fs)
            mask = (xf >= 0) & (xf <= freq_limit)
            yf_masked = yf[mask]
            xf_masked = xf[mask]
            
            if len(yf_masked) > 0:
                magnitude_db = 20 * np.log10(np.abs(yf_masked) + epsilon)
                max_mag_db = np.nanmax(magnitude_db) if np.any(np.isfinite(magnitude_db)) else 0
                magnitude_db = np.maximum(magnitude_db, max_mag_db + db_floor)
                ax1.plot(xf_masked, magnitude_db, color='cyan', label='Original Audio', alpha=0.8)
                plot_success_ax1 = True

        N_filt = len(filtered) if filtered is not None else 0
        if N_filt > 0:
            filtered_to_fft = filtered
            N_fft_filt = N_filt
            if N_filt < min_fft_len:
                pad_width = min_fft_len - N_filt
                filtered_to_fft = np.pad(filtered_to_fft, (0, pad_width), mode='constant', constant_values=0)
                N_fft_filt = min_fft_len
            
            win_filt = windows.hamming(N_fft_filt, sym=False)
            filtered_windowed = filtered_to_fft * win_filt
            yf_filt = fft(filtered_windowed)
            xf_filt = fftfreq(N_fft_filt, 1 / fs)
            mask_filt = (xf_filt >= 0) & (xf_filt <= freq_limit)
            yf_filt_masked = yf_filt[mask_filt]
            xf_filt_masked = xf_filt[mask_filt]
            
            if len(yf_filt_masked) > 0:
                magnitude_filt_db = 20 * np.log10(np.abs(yf_filt_masked) + epsilon)
                max_mag_filt_db = np.nanmax(magnitude_filt_db) if np.any(np.isfinite(magnitude_filt_db)) else 0
                magnitude_filt_db = np.maximum(magnitude_filt_db, max_mag_filt_db + db_floor)
                ax1.plot(xf_filt_masked, magnitude_filt_db, color='lime', label='Filtered (Est. Source Deriv.)', alpha=0.8)
                plot_success_ax1 = True

        if plot_success_ax1:
            ax1.set_title("Audio Spectra Overlay (Zero-padded & Windowed)", **common_style)
            # ax1.set_xlabel("Frequency (Hz)", **common_style)
            # ax1.set_ylabel("Magnitude (dB)", **common_style)
            ax1.tick_params(axis='both', colors='lightgray')
            ax1.grid(True, linestyle=':', alpha=0.4, color='gray')
            ax1.set_xlim(0, freq_limit)
            ax1.legend(fontsize='small', facecolor='#444444', edgecolor='gray', labelcolor='lightgray')
        else:
            ax1.text(0.5, 0.5, 'No Audio Data', ha='center', va='center', transform=ax1.transAxes, color='gray')

        # --- 2. Top Right: Audio Waveform Overlay ---
        ax2 = fig.add_subplot(2, 2, 2)
        plot_success_ax2 = False
        if N_orig > 0 and N_filt > 0:
            min_len = min(N_orig, N_filt)
            center_sample = min_len // 2
            window_samples = int(0.050 * fs)
            start_idx = max(0, center_sample - window_samples // 2)
            end_idx = min(min_len, center_sample + window_samples // 2)

            if end_idx > start_idx:
                audio_zoom = audio[start_idx:end_idx]
                filtered_zoom = filtered[start_idx:end_idx]
                time_zoom_ms = (np.arange(start_idx, end_idx) - center_sample) / fs * 1000.0

                color_orig = 'cyan'
                line1 = ax2.plot(time_zoom_ms, audio_zoom, color=color_orig, label='Original Audio', alpha=0.9)
                # ax2.set_xlabel("Time relative to center (ms)", **common_style)
                # ax2.set_ylabel("Original Amp", color=color_orig)
                ax2.tick_params(axis='y', labelcolor=color_orig, colors=color_orig)
                ax2.tick_params(axis='x', colors='lightgray')
                ax2.grid(True, linestyle=':', alpha=0.4, color='gray')
                ax2.set_xlim(time_zoom_ms[0], time_zoom_ms[-1])

                ax2_twin = ax2.twinx()
                color_filt = 'lime'
                line2 = ax2_twin.plot(time_zoom_ms, filtered_zoom, color=color_filt, label='Filtered', alpha=0.9)
                # ax2_twin.set_ylabel("Filtered Amp", color=color_filt)
                ax2_twin.tick_params(axis='y', labelcolor=color_filt, colors=color_filt)
                ax2_twin.spines['right'].set_color(color_filt)
                ax2_twin.spines['left'].set_color(color_orig)

                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, loc='upper right', fontsize='small', facecolor='#444444', edgecolor='gray', labelcolor='lightgray')
                ax2.set_title("Audio Waveform Overlay (Center +/- 50ms)", **common_style)
                plot_success_ax2 = True

        if not plot_success_ax2:
            ax2.text(0.5, 0.5, 'Waveform unavailable', ha='center', va='center', transform=ax2.transAxes, color='gray')

        # --- 3. Bottom Left: EGG Spectrum ---
        ax3 = fig.add_subplot(2, 2, 3)
        N_egg = len(egg) if egg is not None else 0
        if N_egg > 0:
            egg_to_fft = egg
            N_fft_egg = N_egg
            if N_egg < min_fft_len:
                pad_width = min_fft_len - N_egg
                egg_to_fft = np.pad(egg_to_fft, (0, pad_width), mode='constant', constant_values=0)
                N_fft_egg = min_fft_len
            
            win_egg = windows.hamming(N_fft_egg, sym=False)
            egg_windowed = egg_to_fft * win_egg
            yf_egg = fft(egg_windowed)
            xf_egg = fftfreq(N_fft_egg, 1 / fs)
            mask_egg = (xf_egg >= 0) & (xf_egg <= freq_limit)
            yf_egg_masked = yf_egg[mask_egg]
            xf_egg_masked = xf_egg[mask_egg]
            
            if len(yf_egg_masked) > 0:
                magnitude_egg_db = 20 * np.log10(np.abs(yf_egg_masked) + epsilon)
                max_mag_egg_db = np.nanmax(magnitude_egg_db) if np.any(np.isfinite(magnitude_egg_db)) else 0
                magnitude_egg_db = np.maximum(magnitude_egg_db, max_mag_egg_db + db_floor)
                
                ax3.plot(xf_egg_masked, magnitude_egg_db, color='magenta')
                ax3.set_title("EGG Spectrum (Zero-padded)", **common_style)
                # ax3.set_xlabel("Frequency (Hz)", **common_style)
                # ax3.set_ylabel("Magnitude (dB)", **common_style)
                ax3.tick_params(axis='both', colors='lightgray')
                ax3.grid(True, linestyle=':', alpha=0.4, color='gray')
                ax3.set_xlim(0, freq_limit)
            else:
                ax3.text(0.5, 0.5, 'No EGG Freq Data', ha='center', va='center', transform=ax3.transAxes, color='gray')
        else:
            ax3.text(0.5, 0.5, 'No EGG Data', ha='center', va='center', transform=ax3.transAxes, color='gray')

        # --- 4. Bottom Right: EGG Waveform ---
        ax4 = fig.add_subplot(2, 2, 4)
        if N_egg > 0:
            center_sample_egg = N_egg // 2
            window_samples_egg = int(0.050 * fs)
            start_idx_egg = max(0, center_sample_egg - window_samples_egg // 2)
            end_idx_egg = min(N_egg, center_sample_egg + window_samples_egg // 2)
            
            if end_idx_egg > start_idx_egg:
                egg_zoom = egg[start_idx_egg:end_idx_egg]
                time_zoom_ms_egg = (np.arange(start_idx_egg, end_idx_egg) - center_sample_egg) / fs * 1000.0
                ax4.plot(time_zoom_ms_egg, egg_zoom, color='wheat')
                ax4.set_title("EGG Waveform (Center +/- 50ms)", **common_style)
                # ax4.set_xlabel("Time relative to center (ms)", **common_style)
                # ax4.set_ylabel("Amplitude", **common_style)
                ax4.tick_params(axis='both', colors='lightgray')
                ax4.grid(True, linestyle=':', alpha=0.4, color='gray')
                ax4.set_xlim(time_zoom_ms_egg[0], time_zoom_ms_egg[-1])
            else:
                ax4.text(0.5, 0.5, 'ROI too short', ha='center', va='center', transform=ax4.transAxes, color='gray')
        else:
            ax4.text(0.5, 0.5, 'No EGG Data', ha='center', va='center', transform=ax4.transAxes, color='gray')

        # Apply dark theme to all axes backgrounds
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#2E2E2E')
            for spine in ax.spines.values():
                spine.set_color('#AAAAAA')

        try:
            fig.set_layout_engine('constrained')
        except:
            fig.tight_layout()
            
        canvas.draw()

import matplotlib.cm as cm
import matplotlib.colors as mcolors

class EGGWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.service = EGGAnalysisService()
        self.config = EGGConfig()
        self.result = None
        
        # State
        self.current_filepath = None
        self.channels_flipped = False
        self.show_f0 = False
        self.f0_corrected = False
        self.show_glottal_movement = False
        self.show_filtered_egg = True
        
        self.current_roi_start = DEFAULT_ROI_START
        self.current_roi_duration = DEFAULT_ROI_DURATION
        self.zoom_window_ms = DEFAULT_ZOOM_WINDOW_MS
        self.is_dark = True
        
        self.last_clicked_time = None
        self.timeline_roi_patch = None
        self.glottal_event_lines = []
        self.spec_colorbar = None
        self.f0_contour_line = None
        self.spec_ax_f0 = None
        self.spec_vline = None
        self.cq_vline = None
        self._spec_dragging = False
        self._spec_drag_anchor_x = None
        self._spec_drag_anchor_start = None
        self._spec_press_x = None
        self._spec_press_time = None
        self._spec_drag_moved = False
        
        self.timeline_window_s = 60.0
        self.timeline_offset_s = 0.0
        
        # Workers
        self._active_thread = None
        self._active_worker = None
        self._active_cancel_event = None
        self._active_progress = None

        self.init_ui()
        self._setup_initial_plots()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Toolbar (Simulating Menu) ---
        toolbar = QHBoxLayout()
        btn_open = QPushButton("打开 WAV 文件")
        btn_open.clicked.connect(self.load_wav_file)
        toolbar.addWidget(btn_open)
        
        self.btn_swap = QPushButton("交换左右声道")
        self.btn_swap.setCheckable(True)
        self.btn_swap.clicked.connect(self.toggle_channel_flip)
        toolbar.addWidget(self.btn_swap)
        
        self.batch_button = QPushButton("批量处理")
        self.batch_button.clicked.connect(self.show_batch_dialog)
        self.batch_button.setEnabled(True)
        toolbar.addWidget(self.batch_button)
        
        self.help_button = QPushButton("帮助")
        self.help_button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.help_button.clicked.connect(self.show_help_dialog)
        toolbar.addWidget(self.help_button)
        
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # --- Top Grid for Plots ---
        top_grid_widget = QWidget()
        top_grid_layout = QGridLayout(top_grid_widget)
        top_grid_layout.setSpacing(15)
        top_grid_layout.setVerticalSpacing(20)
        main_layout.addWidget(top_grid_widget, stretch=2)

        # Top-Left Plot (CQ/SQ)
        self.cq_canvas = MplCanvas(self)
        self.cq_ax = self.cq_canvas.axes
        self.cq_ax_sq = self.cq_ax.twinx()
        top_grid_layout.addWidget(self.cq_canvas, 0, 0)

        # Top-Right Plot (Audio Zoom)
        self.audio_zoom_canvas = MplCanvas(self)
        self.audio_zoom_ax = self.audio_zoom_canvas.axes
        top_grid_layout.addWidget(self.audio_zoom_canvas, 0, 1)

        # Bottom-Left Plot (Spectrogram Container)
        self.spec_container = QWidget()
        self.spec_layout = QVBoxLayout(self.spec_container)
        self.spec_layout.setContentsMargins(0, 0, 0, 0)
        self.spec_layout.setSpacing(0)

        # Colorbar Canvas (Horizontal, top of Spectrogram)
        self.colorbar_canvas = MplCanvas(self, width=5, height=0.3, dpi=80)
        self.colorbar_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.colorbar_canvas.setFixedHeight(20) # Thinner
        self.spec_layout.addWidget(self.colorbar_canvas)

        self.spec_canvas = MplCanvas(self)
        self.spec_ax = self.spec_canvas.axes
        self.spec_layout.addWidget(self.spec_canvas)
        
        top_grid_layout.addWidget(self.spec_container, 1, 0)
        
        # Connect interactive events for left plots (Spec & CQ/SQ)
        for canvas in [self.cq_canvas, self.spec_canvas]:
            canvas.mpl_connect('scroll_event', self.on_left_scroll)
            canvas.mpl_connect('button_press_event', self.on_left_press)
            canvas.mpl_connect('motion_notify_event', self.on_left_drag)
            canvas.mpl_connect('button_release_event', self.on_left_release)

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
        self.egg_display_toggle_button.clicked.connect(self.toggle_egg_display_mode)
        egg_toolbar_layout.addWidget(self.egg_display_toggle_button)

        egg_toolbar_layout.addWidget(QLabel("高通频率(Hz):"))
        self.highpass_slider = QSlider(Qt.Orientation.Horizontal)
        self.highpass_slider.setMinimum(1)
        self.highpass_slider.setMaximum(50)
        self.highpass_slider.setSingleStep(1)
        self.highpass_slider.setValue(int(self.config.highpass_cutoff))
        self.highpass_slider.valueChanged.connect(self.handle_highpass_slider_change)
        egg_toolbar_layout.addWidget(self.highpass_slider)
        self.highpass_label = QLabel(f"{int(self.config.highpass_cutoff)}")
        egg_toolbar_layout.addWidget(self.highpass_label)

        self.egg_zoom_layout.addWidget(egg_toolbar)

        self.egg_zoom_canvas = MplCanvas(self)
        self.egg_zoom_ax = self.egg_zoom_canvas.axes
        self.egg_zoom_canvas.mpl_connect('button_press_event', self.on_zoom_plot_click)
        self.egg_zoom_layout.addWidget(self.egg_zoom_canvas)
        top_grid_layout.addWidget(self.egg_zoom_container, 1, 1)
        
        # Connect interactive events for zoom/pan
        for canvas in [self.audio_zoom_canvas, self.egg_zoom_canvas]:
            canvas.mpl_connect('scroll_event', self.on_zoom_scroll)
            canvas.mpl_connect('button_press_event', self.on_zoom_press)
            canvas.mpl_connect('motion_notify_event', self.on_zoom_drag)
            canvas.mpl_connect('button_release_event', self.on_zoom_release)
            
        self._zoom_drag_active = False
        self._zoom_drag_last_x = None

        # --- Bottom Widget for Controls and Timeline ---
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setSpacing(10)
        main_layout.addWidget(bottom_widget, stretch=1)

        # --- Controls Layout ---
        controls_container = QWidget()
        controls_layout = QGridLayout(controls_container)
        controls_layout.setSpacing(10)
        bottom_layout.addWidget(controls_container)

        # Row 0
        controls_layout.addWidget(QLabel("开始 (s):"), 0, 0)
        self.start_time_input = QLineEdit(f"{self.current_roi_start:.2f}")
        self.start_time_input.setFixedWidth(60)
        controls_layout.addWidget(self.start_time_input, 0, 1)
        
        controls_layout.addWidget(QLabel("时长 (s):"), 0, 2)
        self.duration_input = QLineEdit(f"{self.current_roi_duration:.2f}")
        self.duration_input.setFixedWidth(60)
        controls_layout.addWidget(self.duration_input, 0, 3)
        
        controls_layout.addWidget(QLabel("缩放 (ms):"), 0, 4)
        self.zoom_duration_input = QLineEdit(str(int(self.zoom_window_ms)))
        self.zoom_duration_input.setFixedWidth(50)
        controls_layout.addWidget(self.zoom_duration_input, 0, 5)
        
        update_roi_button = QPushButton("更新视图")
        update_roi_button.clicked.connect(self.update_roi_plots)
        controls_layout.addWidget(update_roi_button, 0, 6)
        
        self.play_button = QPushButton("播放音频")
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button, 0, 7)
        
        self.stop_button = QPushButton("停止播放")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button, 0, 8)

        self.save_analysis_button = QPushButton("保存分析")
        self.save_analysis_button.clicked.connect(self.save_analysis)
        self.save_analysis_button.setEnabled(False)
        controls_layout.addWidget(self.save_analysis_button, 0, 9)

        self.detect_glottal_button = QPushButton("声门移动")
        self.detect_glottal_button.setCheckable(True)
        self.detect_glottal_button.setEnabled(False)
        self.detect_glottal_button.setToolTip("该功能正在开发中")
        self.detect_glottal_button.toggled.connect(self.toggle_glottal_detection)
        controls_layout.addWidget(self.detect_glottal_button, 0, 10)

        self.inverse_filter_button = QPushButton("逆滤波")
        self.inverse_filter_button.clicked.connect(self.run_inverse_filtering)
        self.inverse_filter_button.setEnabled(False)
        controls_layout.addWidget(self.inverse_filter_button, 0, 11)

        controls_layout.addWidget(QLabel("IF阶数:"), 0, 12)
        self.if_order_input = QLineEdit("")
        self.if_order_input.setFixedWidth(45)
        controls_layout.addWidget(self.if_order_input, 0, 13)

        # Row 1
        controls_layout.addWidget(QLabel("峰值门限:"), 1, 0)
        self.prominence_input = QLineEdit(f"{self.config.peak_prominence:.3f}")
        self.prominence_input.setFixedWidth(60)
        self.prominence_input.editingFinished.connect(self.handle_peak_prominence_change)
        controls_layout.addWidget(self.prominence_input, 1, 1)

        self.auto_prom_checkbox = QCheckBox("自动")
        self.auto_prom_checkbox.setChecked(True)
        self.auto_prom_checkbox.toggled.connect(self.toggle_auto_prominence)
        controls_layout.addWidget(self.auto_prom_checkbox, 1, 2)

        controls_layout.addWidget(QLabel("谷值门限:"), 1, 3)
        self.valley_prominence_input = QLineEdit(f"{self.config.valley_prominence:.3f}")
        self.valley_prominence_input.setFixedWidth(60)
        self.valley_prominence_input.editingFinished.connect(self.handle_valley_prominence_change)
        controls_layout.addWidget(self.valley_prominence_input, 1, 4)

        controls_layout.addWidget(QLabel("语谱窗长 (ms):"), 1, 5)
        self.spec_window_input = QLineEdit(str(int(self.config.spec_window_ms)))
        self.spec_window_input.setFixedWidth(50)
        self.spec_window_input.editingFinished.connect(self.handle_spec_window_change)
        controls_layout.addWidget(self.spec_window_input, 1, 6)

        self.show_f0_checkbox = QCheckBox("Praat F0")
        self.show_f0_checkbox.setChecked(False)
        self.show_f0_checkbox.toggled.connect(self.toggle_f0_visibility)
        controls_layout.addWidget(self.show_f0_checkbox, 1, 7)

        self.correct_f0_checkbox = QCheckBox("校正 F0 (GCI)")
        self.correct_f0_checkbox.toggled.connect(self.toggle_f0_correction)
        controls_layout.addWidget(self.correct_f0_checkbox, 1, 8)

        self.gci_method_button = QPushButton("GCI：斜率")
        self.gci_method_button.clicked.connect(self.toggle_gci_method)
        controls_layout.addWidget(self.gci_method_button, 1, 9)

        self.goi_method_button = QPushButton("GOI：尺度")
        self.goi_method_button.clicked.connect(self.toggle_goi_method)
        # Set default to scale as requested
        self.config.goi_method = "scale"
        controls_layout.addWidget(self.goi_method_button, 1, 10)

        controls_layout.addWidget(QLabel("语谱最小dB:"), 1, 11)
        self.spec_vmin_input = QLineEdit(f"{self.config.spec_vmin:.1f}")
        self.spec_vmin_input.setFixedWidth(50)
        controls_layout.addWidget(self.spec_vmin_input, 1, 12)
        
        controls_layout.addWidget(QLabel("语谱最大dB:"), 1, 13)
        self.spec_vmax_input = QLineEdit(f"{self.config.spec_vmax:.1f}")
        self.spec_vmax_input.setFixedWidth(50)
        controls_layout.addWidget(self.spec_vmax_input, 1, 14)

        controls_layout.setColumnStretch(15, 1)

        # --- Timeline ---
        self.timeline_canvas = MplCanvas(self, height=1.5, dpi=80)
        self.timeline_ax = self.timeline_canvas.axes
        self.timeline_canvas.mpl_connect('button_press_event', self.on_timeline_click)
        bottom_layout.addWidget(self.timeline_canvas)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setVisible(False)
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_change)
        bottom_layout.addWidget(self.timeline_slider)

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        theme = DARK_PLOT_THEME if is_dark else LIGHT_PLOT_THEME
        
        for canvas in [self.cq_canvas, self.spec_canvas, self.audio_zoom_canvas, self.egg_zoom_canvas, self.timeline_canvas, self.colorbar_canvas]:
            canvas.apply_theme(is_dark)
            
        # Manually update twin axes
        for ax in [self.cq_ax_sq, self.spec_ax_f0]:
            if ax:
                ax.spines['top'].set_color(theme['axes.edgecolor'])
                ax.spines['bottom'].set_color(theme['axes.edgecolor'])
                # Do NOT set facecolor for twin axes as it overlays the main axis
                # ax.set_facecolor(theme['axes.facecolor']) 
                
        # Update specific elements that need manual recoloring
        self._setup_initial_plots()
        self.update_roi_plots()
        if self.last_clicked_time:
            self.update_zoom_plots(self.last_clicked_time)
        self.plot_timeline()
        
        # Force draw all
        for canvas in [self.cq_canvas, self.spec_canvas, self.audio_zoom_canvas, self.egg_zoom_canvas, self.timeline_canvas, self.colorbar_canvas]:
            canvas.draw()

    def _setup_initial_plots(self):
        theme = DARK_PLOT_THEME if self.is_dark else LIGHT_PLOT_THEME
        common_style = {'color': theme['title.color']}
        axis_color = theme['text.color']
        grid_color = theme['grid.color']
        
        cq_color = 'cyan' if self.is_dark else 'blue'
        sq_color = 'yellow' if self.is_dark else '#D4AF37' # Gold for light mode

        self.cq_ax.set_title("EGG CQ & SQ (Peak-Min Method)", **common_style)
        # self.cq_ax.set_xlabel("Time (s)", **common_style) # Removed
        self.cq_ax.tick_params(axis='x', colors=axis_color)
        self.cq_ax.tick_params(axis='y', colors=cq_color, labelcolor=cq_color)
        self.cq_ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
        self.cq_ax.set_facecolor(theme['axes.facecolor']) # Explicitly set facecolor
        self.cq_ax.set_ylim(0, 1)
        for spine in self.cq_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])

        self.cq_ax_sq.tick_params(axis='y', colors=sq_color, labelcolor=sq_color)
        self.cq_ax_sq.set_ylim(-1.1, 1.1)
        self.cq_ax_sq.spines['right'].set_color(sq_color)
        self.cq_ax_sq.spines['left'].set_color(cq_color)
        self.cq_ax_sq.spines['top'].set_color(theme['axes.edgecolor'])
        self.cq_ax_sq.spines['bottom'].set_color(theme['axes.edgecolor'])
        self.cq_ax_sq.grid(False)

        self.spec_ax.set_title("Spectrogram", **common_style)
        # self.spec_ax.set_xlabel("Time (s)", **common_style) # Removed
        # self.spec_ax.set_ylabel("Frequency (Hz)", **common_style) # Removed
        self.spec_ax.tick_params(axis='both', colors=axis_color)
        self.spec_ax.set_facecolor(theme['axes.facecolor']) # Explicitly set facecolor
        self.spec_ax.set_ylim(0, 5000)
        for spine in self.spec_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])

        if hasattr(self, 'spec_ax_f0') and self.spec_ax_f0:
             try: self.spec_ax_f0.remove()
             except Exception: pass
        self.spec_ax_f0 = self.spec_ax.twinx()
        self.spec_ax_f0.tick_params(axis='y', colors=axis_color)
        self.spec_ax_f0.set_ylim(50, 500)
        self.spec_ax_f0.grid(False)
        self.spec_ax_f0.spines['right'].set_color(axis_color)
        self.spec_ax_f0.spines['left'].set_visible(False)
        self.spec_ax_f0.spines['top'].set_visible(False)
        self.spec_ax_f0.spines['bottom'].set_visible(False)
        self.spec_ax_f0.set_yticks([])

        self.audio_zoom_ax.set_title(f"Audio (+/- {self.zoom_window_ms/2:.0f}ms)", **common_style)
        # self.audio_zoom_ax.set_xlabel("Time relative to click (ms)", **common_style) # Removed
        self.audio_zoom_ax.tick_params(axis='both', colors=axis_color)
        self.audio_zoom_ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
        self.audio_zoom_ax.set_facecolor(theme['axes.facecolor']) # Explicitly set facecolor
        self.audio_zoom_ax.set_xlim(-self.zoom_window_ms/2, self.zoom_window_ms/2)
        for spine in self.audio_zoom_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])

        self.egg_zoom_ax.set_title(f"EGG (+/- {self.zoom_window_ms/2:.0f}ms)", **common_style)
        # self.egg_zoom_ax.set_xlabel("Time relative to click (ms)", **common_style) # Removed
        self.egg_zoom_ax.tick_params(axis='both', colors=axis_color)
        self.egg_zoom_ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
        self.egg_zoom_ax.set_facecolor(theme['axes.facecolor']) # Explicitly set facecolor
        self.egg_zoom_ax.set_xlim(-self.zoom_window_ms/2, self.zoom_window_ms/2)
        for spine in self.egg_zoom_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])

        self.timeline_ax.set_title("Timeline Overview", **common_style)
        # self.timeline_ax.set_xlabel("Time (s)", **common_style) # Removed
        self.timeline_ax.set_yticks([])
        self.timeline_ax.tick_params(axis='x', colors=axis_color)
        self.timeline_ax.set_facecolor(theme['axes.facecolor']) # Explicitly set facecolor
        self.timeline_ax.set_xlim(0, 10)
        for spine in self.timeline_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])
        
        # Colorbar initial style
        self.colorbar_canvas.fig.set_facecolor(theme['figure.facecolor'])
        self.colorbar_canvas.axes.set_visible(False)

        # Apply manual margins for alignment
        # Left column: CQ/SQ and Spectrogram
        # We use fixed margins to ensure the plot areas align vertically, 
        # regardless of tick label widths.
        left_margin = 0.08
        right_margin = 0.92
        
        self.cq_canvas.fig.subplots_adjust(left=left_margin, right=right_margin, top=0.9, bottom=0.1)
        self.spec_canvas.fig.subplots_adjust(left=left_margin, right=right_margin, top=0.98, bottom=0.1)
        # Colorbar aligns with the plot
        self.colorbar_canvas.fig.subplots_adjust(left=left_margin, right=right_margin, top=1, bottom=0)

        # Right column: Audio Zoom and EGG Zoom
        self.audio_zoom_canvas.fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15)
        self.egg_zoom_canvas.fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15)
        
        self.timeline_canvas.fig.tight_layout()

        for canvas in [self.cq_canvas, self.spec_canvas, self.audio_zoom_canvas, self.egg_zoom_canvas, self.timeline_canvas, self.colorbar_canvas]:
            canvas.draw()


    # --- Actions ---
    def load_wav_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "打开 WAV 文件", "", "WAV 文件 (*.wav)")
        if not filepath: return
        self.channels_flipped = False
        self.btn_swap.setChecked(False)
        self._load_data_from_path(filepath)

    def toggle_channel_flip(self, checked):
        if self.current_filepath is None:
            self.btn_swap.setChecked(self.channels_flipped)
            return
        self.channels_flipped = checked
        self._load_data_from_path(self.current_filepath)

    def _load_data_from_path(self, filepath):
        self._cancel_active_task()
        self._reset_app_state()
        self.current_filepath = filepath
        
        # Update config from UI
        self.config.highpass_cutoff = float(self.highpass_slider.value())
        self.config.auto_prominence = self.auto_prom_checkbox.isChecked()
        try:
            self.config.valley_prominence = float(self.valley_prominence_input.text())
        except ValueError:
            self.config.valley_prominence = DEFAULT_VALLEY_PROMINENCE

        cancel_event = threading.Event()
        self._active_cancel_event = cancel_event
        self._active_progress = QProgressDialog("正在加载并分析…", "取消", 0, 100, self)
        self._active_progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._active_progress.canceled.connect(self._cancel_active_task)
        self._active_progress.show()

        worker = LoadWorker(self.service, filepath, self.config, self.channels_flipped, cancel_event)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_task_progress)
        worker.finished.connect(self._on_load_finished)
        worker.error.connect(self._on_task_error)
        worker.canceled.connect(self._on_task_canceled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        
        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    def _on_load_finished(self, result):
        self._finish_active_task()
        self.result = result
        
        self.current_roi_start = DEFAULT_ROI_START
        self.current_roi_duration = min(DEFAULT_ROI_DURATION, result.file_duration)
        self.start_time_input.setText(f"{self.current_roi_start:.2f}")
        self.duration_input.setText(f"{self.current_roi_duration:.2f}")
        
        # Enable controls
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.inverse_filter_button.setEnabled(True)
        self.save_analysis_button.setEnabled(True)
        
        # Set default IF order to 50 as requested
        self.if_order_input.setText("50")
        
        self.plot_timeline()
        self.update_roi_plots()

    def _reset_app_state(self):
        sd.stop()
        self.result = None
        self.glottal_event_lines = []
        self._setup_initial_plots()
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.inverse_filter_button.setEnabled(False)
        self.save_analysis_button.setEnabled(False)

    def _cancel_active_task(self):
        if self._active_cancel_event:
            self._active_cancel_event.set()

    def _finish_active_task(self):
        if self._active_progress:
            self._active_progress.close()
        self._active_progress = None
        self._active_cancel_event = None

    def _on_task_progress(self, value, text):
        if self._active_progress:
            self._active_progress.setLabelText(text)
            self._active_progress.setValue(value)

    def _on_task_error(self, msg):
        self._finish_active_task()
        QMessageBox.critical(self, "Error", msg)

    def _on_task_canceled(self):
        self._finish_active_task()

    # --- Plotting ---
    def _get_downsampling_step(self, n_samples):
        if n_samples < 5000:
            return 1
        step = 1
        current_len = n_samples
        while current_len > 10000:
            step *= 2
            current_len //= 2
        return step

    def plot_timeline(self):
        if self.result is None: return
        self.timeline_ax.cla()
        self.timeline_roi_patch = None
        
        # New downsampling logic
        full_sig = self.result.egg_signal_processed
        if full_sig is None: return
        
        step = self._get_downsampling_step(len(full_sig))
        sig_ds = full_sig[::step]
        
        # Reconstruct time vector
        total_duration = self.result.file_duration
        if len(sig_ds) > 0:
            time_vec = np.linspace(0, total_duration, len(sig_ds))
        else:
            time_vec = []
            
        theme = DARK_PLOT_THEME if self.is_dark else LIGHT_PLOT_THEME
        line_color = 'black' if not self.is_dark else 'gray'
        
        self.timeline_ax.plot(time_vec, sig_ds, lw=0.7, color=line_color)
        
        title_color = theme['title.color']
        tick_color = theme['text.color']
        
        self.timeline_ax.set_title("Timeline Overview", color=title_color)
        # self.timeline_ax.set_xlabel("Time (s)", color=title_color) # Removed
        self.timeline_ax.set_yticks([])
        self.timeline_ax.tick_params(axis='x', colors=tick_color)
        self.timeline_ax.set_facecolor(theme['axes.facecolor']) # Re-apply facecolor
        for spine in self.timeline_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])
        
        # Handle timeline window (max 60s)
        if self.result.file_duration > self.timeline_window_s:
            self.timeline_slider.setVisible(True)
            max_val = int(self.result.file_duration - self.timeline_window_s)
            self.timeline_slider.setMaximum(max_val)
            
            start_t = self.timeline_offset_s
            end_t = min(self.result.file_duration, start_t + self.timeline_window_s)
            self.timeline_ax.set_xlim(start_t, end_t)
        else:
            self.timeline_slider.setVisible(False)
            self.timeline_ax.set_xlim(0, self.result.file_duration)
            
        self.update_timeline_roi_visual()
        self.timeline_canvas.draw()

    def update_timeline_roi_visual(self):
        if self.timeline_roi_patch:
            try: self.timeline_roi_patch.remove()
            except ValueError: pass
            self.timeline_roi_patch = None
            
        if self.result and self.result.file_duration > 0:
            roi_start = self.current_roi_start
            roi_end = min(roi_start + self.current_roi_duration, self.result.file_duration)
            ylim = self.timeline_ax.get_ylim()
            if ylim[1] > ylim[0]:
                self.timeline_roi_patch = Rectangle((roi_start, ylim[0]), roi_end - roi_start, ylim[1] - ylim[0],
                                                    color='teal', alpha=0.3, zorder=-1)
                self.timeline_ax.add_patch(self.timeline_roi_patch)

    def update_roi_plots(self):
        if self.result is None: return
        
        # Parse inputs
        try:
            start_s = float(self.start_time_input.text())
            duration_s = float(self.duration_input.text())
            self.config.spec_vmin = float(self.spec_vmin_input.text())
            self.config.spec_vmax = float(self.spec_vmax_input.text())
        except ValueError:
            return

        self.current_roi_start = start_s
        self.current_roi_duration = duration_s
        end_s = start_s + duration_s
        
        # Update timeline ROI
        self.update_timeline_roi_visual()
        self.timeline_canvas.draw()

        theme = DARK_PLOT_THEME if self.is_dark else LIGHT_PLOT_THEME
        title_color = theme['title.color']
        tick_color = theme['text.color']
        cq_color = 'cyan' if self.is_dark else 'blue'
        sq_color = 'yellow' if self.is_dark else '#D4AF37'
        grid_color = theme['grid.color']

        # Calculate CQ/SQ for segment (re-run local analysis)
        # Use !show_filtered_egg to determine if we use raw signal
        use_raw = not self.show_filtered_egg
        cq_t, cq_v, sq_v = self.service.calculate_cq_sq_segment(self.result, start_s, end_s, self.config, use_raw_signal=use_raw)
        
        # Plot CQ/SQ
        self.cq_ax.cla()
        self.cq_ax_sq.cla()
        
        if cq_t is not None and len(cq_t) > 0:
            l1, = self.cq_ax.plot(cq_t, cq_v, label='CQ', color=cq_color, marker='.', linestyle='', markersize=5)
            l2, = self.cq_ax_sq.plot(cq_t, sq_v, label='SQ', color=sq_color, marker='x', linestyle='', markersize=5)
            
            # Combine legends from twin axes
            lines = [l1, l2]
            labels = [l.get_label() for l in lines]
            self.cq_ax.legend(lines, labels, loc='upper right', fontsize='small', facecolor='#444444', edgecolor='gray', labelcolor='lightgray')
            
        self.cq_ax.set_xlim(start_s, end_s)
        self.cq_ax.set_ylim(0, 1)
        self.cq_ax.set_title("EGG CQ & SQ", color=title_color)
        self.cq_ax.tick_params(axis='x', colors=tick_color)
        self.cq_ax.tick_params(axis='y', colors=cq_color)
        self.cq_ax.set_facecolor(theme['axes.facecolor'])
        self.cq_ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
        for spine in self.cq_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])

        self.cq_ax_sq.tick_params(axis='y', colors=sq_color)
        self.cq_ax_sq.set_ylim(-1.1, 1.1)
        self.cq_ax_sq.spines['right'].set_color(sq_color)
        self.cq_ax_sq.spines['left'].set_color(cq_color)
        self.cq_ax_sq.spines['top'].set_color(theme['axes.edgecolor'])
        self.cq_ax_sq.spines['bottom'].set_color(theme['axes.edgecolor'])
        self.cq_ax_sq.grid(False)
        self.cq_canvas.draw()

        # Plot Spectrogram
        self.spec_ax.cla()
        if self.spec_ax_f0: self.spec_ax_f0.cla()
        
        fs = self.result.fs
        s_idx = int(start_s * fs)
        e_idx = int(end_s * fs)
        if e_idx > s_idx and e_idx <= len(self.result.audio_signal):
            audio_seg = self.result.audio_signal[s_idx:e_idx]
            
            # Update separate Colorbar Canvas
            # Use dynamic NFFT based on spec_window_ms
            nfft = int(fs * self.config.spec_window_ms / 1000.0)
            noverlap = int(nfft * 0.75) # 75% overlap for smoother view
            
            Pxx, freqs, bins, im = self.spec_ax.specgram(
                audio_seg, NFFT=nfft, Fs=fs, noverlap=noverlap,
                cmap='jet', vmin=self.config.spec_vmin, vmax=self.config.spec_vmax
            )
            im.set_extent([start_s, end_s, freqs[0], freqs[-1]])
            self.spec_ax.set_ylim(0, 5000)
            
            # Update separate Colorbar Canvas
            self.colorbar_canvas.fig.clear()
            # Use manual positioning to match the plot below (left=0.08, right=0.92 -> width=0.84)
            self.colorbar_canvas.axes = self.colorbar_canvas.fig.add_axes([0.08, 0.05, 0.84, 0.9])
            
            # Create a ScalarMappable for the colorbar, independent of the spectrogram image
            # This avoids the "Adding colorbar to a different Figure" warning
            norm = mcolors.Normalize(vmin=self.config.spec_vmin, vmax=self.config.spec_vmax)
            sm = cm.ScalarMappable(norm=norm, cmap='jet')
            sm.set_array([]) # Dummy array
            
            cb = self.colorbar_canvas.fig.colorbar(sm, cax=self.colorbar_canvas.axes, orientation='horizontal')
            cb.set_ticks([]) # Hide ticks
            cb.outline.set_visible(False)
            self.colorbar_canvas.draw()

        self.spec_ax.set_xlim(start_s, end_s)
        self.spec_ax.set_title("Spectrogram", color=title_color)
        self.spec_ax.tick_params(axis='both', colors=tick_color)
        self.spec_ax.set_facecolor(theme['axes.facecolor'])
        for spine in self.spec_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])
        
        # Plot F0
        self._update_f0_contour()
        
        # Plot Glottal Movement
        if self.show_glottal_movement and self.result.glottal_movement_events:
            for t, m_type in self.result.glottal_movement_events:
                if start_s <= t <= end_s:
                    color = sq_color if m_type == "Rise" else '#FF69B4'
                    self.spec_ax.axvline(t, color=color, linestyle='--', alpha=0.8)
        
        # Redraw vertical line if exists
        if self.last_clicked_time is not None:
             if start_s <= self.last_clicked_time <= end_s:
                 if self.spec_vline: 
                     try: self.spec_vline.remove()
                     except: pass
                 self.spec_vline = self.spec_ax.axvline(self.last_clicked_time, color='red')
                 
                 if self.cq_vline:
                     try: self.cq_vline.remove()
                     except: pass
                 self.cq_vline = self.cq_ax.axvline(self.last_clicked_time, color='red')

        self.spec_canvas.draw()

    def _update_f0_contour(self):
        if not self.spec_ax_f0:
            self.spec_ax_f0 = self.spec_ax.twinx()
            
        theme = DARK_PLOT_THEME if self.is_dark else LIGHT_PLOT_THEME
        tick_color = theme['text.color']
        
        self.spec_ax_f0.cla()
        self.spec_ax_f0.tick_params(axis='y', colors=tick_color)
        self.spec_ax_f0.grid(False)
        self.spec_ax_f0.set_xlim(self.spec_ax.get_xlim())
        self.spec_ax_f0.spines['right'].set_color(tick_color)
        self.spec_ax_f0.spines['left'].set_visible(False)
        self.spec_ax_f0.spines['top'].set_visible(False)
        self.spec_ax_f0.spines['bottom'].set_visible(False)
        
        all_vals = []
        start_s, end_s = self.spec_ax.get_xlim()
        
        # 1. Praat F0 (White dots)
        if self.show_f0:
            times = self.result.audio_f0_times
            values = self.result.audio_f0_values
            if times is not None and len(times) > 0:
                mask = (times >= start_s) & (times <= end_s)
                if np.any(mask):
                    self.spec_ax_f0.plot(times[mask], values[mask], color='white', marker='.', markersize=2, linestyle='None', label='Praat F0')
                    all_vals.extend(values[mask])
        
        # 2. Corrected F0 (Red dots)
        if self.f0_corrected:
            times = self.result.gci_f0_times
            values = self.result.gci_f0_values
            if times is not None and len(times) > 0:
                mask = (times >= start_s) & (times <= end_s)
                if np.any(mask):
                    self.spec_ax_f0.plot(times[mask], values[mask], color='red', marker='.', markersize=2, linestyle='None', label='Corrected F0')
                    all_vals.extend(values[mask])
            
        # Auto scale
        if len(all_vals) > 0:
            all_vals = np.array(all_vals)
            valid = all_vals[~np.isnan(all_vals)]
            if len(valid) > 0:
                vmin, vmax = np.min(valid), np.max(valid)
                pad = (vmax - vmin) * 0.1 + 10
                self.spec_ax_f0.set_ylim(max(0, vmin - pad), vmax + pad)
            else:
                self.spec_ax_f0.set_ylim(50, 500)
        else:
            self.spec_ax_f0.set_ylim(50, 500)

        show_f0_axis = self.show_f0 or self.f0_corrected
        self.spec_ax_f0.spines['right'].set_visible(show_f0_axis)
        if show_f0_axis:
            self.spec_ax_f0.yaxis.set_major_locator(mticker.AutoLocator())
            self.spec_ax_f0.tick_params(axis='y', colors=tick_color, right=True, labelright=True)
        else:
            self.spec_ax_f0.tick_params(axis='y', right=False, labelright=False)
            self.spec_ax_f0.set_yticks([])

    def on_timeline_slider_change(self, value):
        self.timeline_offset_s = float(value)
        self.plot_timeline()

    def on_timeline_click(self, event):
        if event.inaxes == self.timeline_ax and event.xdata is not None:
            self.start_time_input.setText(f"{event.xdata:.2f}")
            self.update_roi_plots()

    def on_left_plot_click(self, event):
        # Deprecated by on_left_press
        pass

    def on_left_scroll(self, event):
        if event.inaxes not in [self.cq_ax, self.spec_ax, self.spec_ax_f0]:
            return
        if event.step == 0: return
        
        factor = 1.1 if event.step < 0 else 0.9
        
        current_start = self.current_roi_start
        current_duration = self.current_roi_duration
        current_center = current_start + current_duration / 2.0
        
        new_duration = current_duration * factor
        # Limit zoom
        new_duration = max(0.01, min(self.result.file_duration, new_duration))
        
        new_start = current_center - new_duration / 2.0
        new_start = max(0, min(self.result.file_duration - new_duration, new_start))
        
        self.start_time_input.setText(f"{new_start:.3f}")
        self.duration_input.setText(f"{new_duration:.3f}")
        
        self.update_roi_plots()

    def on_left_press(self, event):
        if event.inaxes in [self.cq_ax, self.spec_ax, self.spec_ax_f0] and event.button == 1:
            self._spec_dragging = True
            self._spec_drag_anchor_x = event.xdata
            self._spec_drag_anchor_start = self.current_roi_start
            self._spec_press_x = event.x
            self._spec_press_time = event.xdata
            self._spec_drag_moved = False

    def on_left_release(self, event):
        if self._spec_dragging and not self._spec_drag_moved and event is not None and event.xdata is not None:
            self.last_clicked_time = event.xdata
            self.update_zoom_plots(event.xdata)
            if self.spec_vline:
                self.spec_vline.set_xdata([event.xdata])
            if self.cq_vline:
                self.cq_vline.set_xdata([event.xdata])
            self.spec_canvas.draw_idle()
            self.cq_canvas.draw_idle()
        self._spec_dragging = False
        self._spec_drag_anchor_x = None
        self._spec_drag_anchor_start = None
        self._spec_press_x = None
        self._spec_press_time = None
        self._spec_drag_moved = False

    def on_left_drag(self, event):
        if not self._spec_dragging or event.inaxes not in [self.cq_ax, self.spec_ax, self.spec_ax_f0]:
            return
        if event.x is None or self._spec_press_x is None or self._spec_drag_anchor_start is None:
            return

        if event.x is not None and self._spec_press_x is not None and abs(event.x - self._spec_press_x) > 2:
            self._spec_drag_moved = True
        if not self._spec_drag_moved:
            return

        axis_width_px = event.inaxes.bbox.width if event.inaxes is not None else 0.0
        if axis_width_px <= 0:
            return

        seconds_per_pixel = self.current_roi_duration / axis_width_px
        dx_pixels = event.x - self._spec_press_x
        new_start = self._spec_drag_anchor_start - dx_pixels * seconds_per_pixel
        new_start = max(0, min(self.result.file_duration - self.current_roi_duration, new_start))
        if abs(new_start - self.current_roi_start) < 1e-4:
            return
        self.start_time_input.setText(f"{new_start:.3f}")
        self.update_roi_plots()

    def update_zoom_plots(self, center_time):
        if self.result is None: return
        
        try:
            zoom_ms = float(self.zoom_duration_input.text())
        except ValueError:
            zoom_ms = 50.0
            
        half_win = zoom_ms / 2000.0
        start_s = center_time - half_win
        end_s = center_time + half_win
        
        fs = self.result.fs
        s_idx = int(start_s * fs)
        e_idx = int(end_s * fs)
        
        theme = DARK_PLOT_THEME if self.is_dark else LIGHT_PLOT_THEME
        title_color = theme['title.color']
        grid_color = theme['grid.color']
        tick_color = theme['text.color']
        line_color = 'black' if not self.is_dark else 'cyan' # Audio color
        egg_line_color = 'black' if not self.is_dark else 'wheat' # EGG color
        
        # Audio
        self.audio_zoom_ax.cla()
        if e_idx > s_idx:
            s_idx_clamped = max(0, s_idx)
            e_idx_clamped = min(len(self.result.audio_signal), e_idx)
            audio_seg = self.result.audio_signal[s_idx_clamped:e_idx_clamped]
            
            # Downsample
            step = self._get_downsampling_step(len(audio_seg))
            audio_seg_ds = audio_seg[::step]
            
            time_axis = (np.arange(s_idx_clamped, e_idx_clamped) / fs - center_time) * 1000
            time_axis_ds = time_axis[::step]
            
            self.audio_zoom_ax.plot(time_axis_ds, audio_seg_ds, color=line_color)
            
        self.audio_zoom_ax.set_title(f"Audio (+/- {zoom_ms/2:.0f}ms)", color=title_color)
        self.audio_zoom_ax.set_xlim(-zoom_ms/2, zoom_ms/2)
        self.audio_zoom_ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
        self.audio_zoom_ax.tick_params(axis='both', colors=tick_color)
        self.audio_zoom_ax.set_facecolor(theme['axes.facecolor'])
        for spine in self.audio_zoom_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])
        self.audio_zoom_canvas.draw()
        
        # EGG
        self.egg_zoom_ax.cla()
        if e_idx > s_idx:
            s_idx_clamped = max(0, s_idx)
            e_idx_clamped = min(len(self.result.egg_signal_processed), e_idx)
            
            # Logic for signal display and analysis
            use_raw = not self.show_filtered_egg
            
            # For display, we want to show the signal corresponding to the mode
            # But if we are in 'filtered' mode, we want to show the signal processed with CURRENT slider value
            # The result.egg_signal_processed is static (processed at load).
            # To support slider changes, we should re-filter the raw signal for display too.
            
            # Get raw segment
            full_raw = self.result.egg_signal_raw
            raw_seg = full_raw[s_idx_clamped:e_idx_clamped]
            
            if use_raw:
                egg_seg_to_plot = raw_seg
            else:
                # Apply filters locally for display
                # Note: local filtering might have edge effects, but for zoom view it's acceptable or we need padding
                # Let's use a bit larger segment for filtering then crop
                pad = int(fs * 0.1) # 100ms padding
                p_s = max(0, s_idx_clamped - pad)
                p_e = min(len(full_raw), e_idx_clamped + pad)
                
                seg_padded = full_raw[p_s:p_e]
                seg_detrend = signal.detrend(seg_padded)
                seg_hp = apply_highpass_filter(seg_detrend, cutoff_freq=self.config.highpass_cutoff, fs=fs)
                seg_lp = apply_lowpass_filter(seg_hp, cutoff_freq=self.config.lowpass_cutoff, fs=fs)
                
                # Crop back
                crop_start = s_idx_clamped - p_s
                crop_end = crop_start + (e_idx_clamped - s_idx_clamped)
                egg_seg_to_plot = seg_lp[crop_start:crop_end]

            # Downsample for plot
            step = self._get_downsampling_step(len(egg_seg_to_plot))
            egg_seg_ds = egg_seg_to_plot[::step]
            
            time_axis = (np.arange(s_idx_clamped, e_idx_clamped) / fs - center_time) * 1000
            time_axis_ds = time_axis[::step]
            
            self.egg_zoom_ax.plot(time_axis_ds, egg_seg_ds, color=egg_line_color)
            
            # Plot events
            # Use local analysis for instant feedback, passing use_raw
            gci, goi, _ = self.service.get_events_segment(self.result, start_s, end_s, self.config, use_raw_signal=use_raw)
            
            # Helper for adding legend only once
            gci_label_added = False
            goi_label_added = False
            
            for t in gci:
                label = 'GCI' if not gci_label_added else None
                self.egg_zoom_ax.axvline((t - center_time)*1000, color='lime', label=label)
                self.audio_zoom_ax.axvline((t - center_time)*1000, color='lime')
                gci_label_added = True
            
            for t in goi:
                label = 'GOI' if not goi_label_added else None
                self.egg_zoom_ax.axvline((t - center_time)*1000, color='lime', linestyle='--', alpha=0.5, label=label)
                self.audio_zoom_ax.axvline((t - center_time)*1000, color='lime', linestyle='--', alpha=0.5)
                goi_label_added = True
                
            if gci_label_added or goi_label_added:
                self.egg_zoom_ax.legend(loc='upper right', fontsize='small', facecolor='#444444', edgecolor='gray', labelcolor='lightgray')
                
        self.egg_zoom_ax.set_title(f"EGG (+/- {zoom_ms/2:.0f}ms)", color=title_color)
        self.egg_zoom_ax.set_xlim(-zoom_ms/2, zoom_ms/2)
        self.egg_zoom_ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
        self.egg_zoom_ax.tick_params(axis='both', colors=tick_color)
        self.egg_zoom_ax.set_facecolor(theme['axes.facecolor'])
        for spine in self.egg_zoom_ax.spines.values(): spine.set_color(theme['axes.edgecolor'])
        self.egg_zoom_canvas.draw()
        self.audio_zoom_canvas.draw()

    def on_zoom_plot_click(self, event):
        # Deprecated by on_zoom_press
        pass
        
    # --- Interactive Events ---
    def on_zoom_scroll(self, event):
        if event.inaxes not in [self.audio_zoom_canvas.axes, self.egg_zoom_canvas.axes]:
            return
        if event.step == 0: return
        
        factor = 1.1 if event.step < 0 else 0.9
        new_zoom = self.zoom_window_ms * factor
        new_zoom = max(5.0, min(new_zoom, 5000.0))
        
        self.zoom_window_ms = new_zoom
        self.zoom_duration_input.setText(str(int(new_zoom)))
        
        if self.last_clicked_time:
             self.update_zoom_plots(self.last_clicked_time)

    def on_zoom_press(self, event):
        if event.inaxes in [self.audio_zoom_canvas.axes, self.egg_zoom_canvas.axes] and event.button == 1:
            self._zoom_drag_active = True
            self._zoom_drag_last_x = event.xdata

    def on_zoom_release(self, event):
        self._zoom_drag_active = False
        self._zoom_drag_last_x = None

    def on_zoom_drag(self, event):
        if not self._zoom_drag_active or event.inaxes not in [self.audio_zoom_canvas.axes, self.egg_zoom_canvas.axes]:
            return
        if event.xdata is None or self._zoom_drag_last_x is None:
            return
            
        dx = event.xdata - self._zoom_drag_last_x
        # xdata is in ms relative to center. dx > 0 means mouse moved right (pixels mapped to data).
        # To move view left (earlier time), center_time should decrease.
        dt_s = dx / 1000.0
        
        if self.last_clicked_time:
            self.last_clicked_time -= dt_s
            self.last_clicked_time = max(0, min(self.result.file_duration, self.last_clicked_time))
            self.update_zoom_plots(self.last_clicked_time)
            
            # Update vlines on main plots
            if self.spec_vline: 
                self.spec_vline.set_xdata([self.last_clicked_time])
            if self.cq_vline:
                self.cq_vline.set_xdata([self.last_clicked_time])
            self.spec_canvas.draw()
            self.cq_canvas.draw()

    # --- Button Handlers ---
    def toggle_egg_display_mode(self):
        self.show_filtered_egg = not self.show_filtered_egg
        self.egg_display_toggle_button.setText("显示：滤波波形" if self.show_filtered_egg else "显示：原始波形")
        self.update_roi_plots()
        if self.last_clicked_time:
            self.update_zoom_plots(self.last_clicked_time)

    def handle_highpass_slider_change(self, value):
        self.highpass_label.setText(str(value))
        self.config.highpass_cutoff = float(value)
        # Update plots immediately
        if self.show_filtered_egg:
             self.update_roi_plots()
             if self.last_clicked_time:
                 self.update_zoom_plots(self.last_clicked_time)

    def play_audio(self):
        if self.result is None: return
        start_s = self.current_roi_start
        end_s = start_s + self.current_roi_duration
        fs = self.result.fs
        s_idx = int(start_s * fs)
        e_idx = int(end_s * fs)
        sd.play(self.result.audio_signal[s_idx:e_idx], fs)

    def stop_audio(self):
        sd.stop()

    def toggle_f0_visibility(self, checked):
        self.show_f0 = checked
        if checked and self.result and self.result.audio_f0_values is None:
            self._calculate_f0()
        self.update_roi_plots()

    def _calculate_f0(self):
        # Calculate F0 using Praat
        if self.result is None: return
        
        progress = QProgressDialog("计算基频 (Praat)...", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.show()
        
        try:
            self.service.calculate_praat_f0(self.result)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"F0 calculation failed: {e}")
        finally:
            progress.close()

    def toggle_f0_correction(self, checked):
        self.f0_corrected = checked
        self._update_f0_contour()
        self.spec_canvas.draw()

    def toggle_gci_method(self):
        self.config.gci_method = "scale" if self.config.gci_method == "slope" else "slope"
        self.gci_method_button.setText(f"GCI：{'尺度' if self.config.gci_method == 'scale' else '斜率'}")
        self._trigger_reanalysis()
        self.update_roi_plots()
        if self.last_clicked_time:
            self.update_zoom_plots(self.last_clicked_time)

    def toggle_goi_method(self):
        self.config.goi_method = "scale" if self.config.goi_method == "slope" else "slope"
        self.goi_method_button.setText(f"GOI：{'尺度' if self.config.goi_method == 'scale' else '斜率'}")
        self._trigger_reanalysis()
        self.update_roi_plots()
        if self.last_clicked_time:
            self.update_zoom_plots(self.last_clicked_time)

    def toggle_auto_prominence(self, checked):
        self.config.auto_prominence = checked
        self.prominence_input.setEnabled(not checked)
        self._trigger_reanalysis()
        self.update_roi_plots()
        if self.last_clicked_time:
            self.update_zoom_plots(self.last_clicked_time)

    def handle_peak_prominence_change(self):
        try:
            self.config.peak_prominence = float(self.prominence_input.text())
            # Don't auto-uncheck, assume user knows what they are doing or checkbox handles it.
            # But user said "uncheck auto, change value, no effect".
            # The checkbox state is already handled by toggle_auto_prominence.
            # If user types in box, it means they want to use this value.
            # But if auto is checked, config.auto_prominence is True, so this value is ignored by core.
            # So we should probably uncheck auto if user edits this.
            self.auto_prom_checkbox.setChecked(False) 
            self._trigger_reanalysis()
            self.update_roi_plots()
            if self.last_clicked_time:
                self.update_zoom_plots(self.last_clicked_time)
        except ValueError: pass

    def handle_valley_prominence_change(self):
        try:
            self.config.valley_prominence = float(self.valley_prominence_input.text())
            self._trigger_reanalysis()
            self.update_roi_plots()
            if self.last_clicked_time:
                self.update_zoom_plots(self.last_clicked_time)
        except ValueError: pass

    def handle_spec_window_change(self):
        try:
            val = float(self.spec_window_input.text())
            self.config.spec_window_ms = val
            self.update_roi_plots()
        except ValueError:
            pass

    def _trigger_reanalysis(self):
        if self.result is None: return
        
        self._cancel_active_task()
        
        # Run EventsWorker
        cancel_event = threading.Event()
        self._active_cancel_event = cancel_event
        worker = EventsWorker(self.service, self.result, self.config, cancel_event)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_events_finished)
        worker.finished.connect(thread.quit)
        thread.start()
        
        self._active_thread = thread
        self._active_worker = worker

    def _on_events_finished(self, result):
        self.result = result
        self.update_roi_plots()
        if self.last_clicked_time:
            self.update_zoom_plots(self.last_clicked_time)

    def toggle_glottal_detection(self, checked):
        self.show_glottal_movement = checked
        if checked and self.result:
            self.service.detect_glottal_movement(self.result)
        self.update_roi_plots()

    def run_inverse_filtering(self):
        if self.result is None: return
        # Simply call the standalone plot function for now, or implement a dialog
        # Re-using the logic from main_app.py which calls apply_simplified_cp_inverse_filtering
        # and then plot_inverse_filtering_results
        
        # Basic implementation:
        start_s = self.current_roi_start
        end_s = start_s + self.current_roi_duration
        fs = self.result.fs
        
        # Get relative GCI
        gci = np.array(self.result.gci_times)
        gci_roi = gci[(gci >= start_s) & (gci < end_s)]
        gci_rel = gci_roi - start_s
        
        s_idx = int(start_s * fs)
        e_idx = int(end_s * fs)
        audio_seg = self.result.audio_signal[s_idx:e_idx]
        egg_seg = self.result.egg_signal_processed[s_idx:e_idx]
        
        filtered = self.service.apply_simplified_cp_inverse_filtering(
            audio_seg, fs, gci_rel, 
            lp_order=int(self.if_order_input.text()) if self.if_order_input.text() else None
        )
        
        if filtered is not None:
            dlg = InverseFilteringResultDialog(audio_seg, filtered, egg_seg, fs, start_s, end_s, self)
            dlg.exec()
        else:
            QMessageBox.warning(self, "Failed", "Inverse filtering failed.")

    def save_analysis(self):
        if self.result is None: return
        if self.current_filepath is None:
            QMessageBox.warning(self, "Error", "No file loaded.")
            return

        try:
            start_s = self.current_roi_start
            duration_s = self.current_roi_duration
            end_s = start_s + duration_s
            
            output_dir = os.path.dirname(self.current_filepath)
            base_name_full = os.path.basename(self.current_filepath)
            base_name = os.path.splitext(base_name_full)[0]
            start_str = f"{start_s:.2f}s".replace('.', '_')
            end_str = f"{end_s:.2f}s".replace('.', '_')
            base_name_ts = f"{base_name}_{start_str}_{end_str}"
            
            # Save CSV
            csv_path = os.path.join(output_dir, f"{base_name_ts}_DATA.csv")
            self._save_csv_data(start_s, end_s, csv_path)
            
            # Save Plots
            plot_spec_path = os.path.join(output_dir, f"{base_name_ts}_SPEC_F0.png")
            plot_cq_path = os.path.join(output_dir, f"{base_name_ts}_CQ_SQ.png")
            plot_wave_path = os.path.join(output_dir, f"{base_name_ts}_WAVEFORMS.png")
            self._save_plots(start_s, end_s, plot_spec_path, plot_cq_path, plot_wave_path)
            
            QMessageBox.information(self, "成功", f"分析结果已保存至:\n{output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _save_csv_data(self, start_s, end_s, csv_path):
        # 1. Calculate CQ/SQ for ROI
        use_raw = not self.show_filtered_egg
        cq_t, cq_v, sq_v = self.service.calculate_cq_sq_segment(self.result, start_s, end_s, self.config, use_raw_signal=use_raw)
        
        cq_sq_df = pd.DataFrame()
        if cq_t is not None and len(cq_t) > 0:
            cq_sq_df = pd.DataFrame({
                'Time_CQ_SQ': cq_t,
                'CQ': cq_v,
                'SQ': sq_v
            }).set_index('Time_CQ_SQ')
            
        # 2. Get F0 Data (Praat)
        f0_praat_df = pd.DataFrame()
        if self.result.audio_f0_times is not None:
            f0_times = self.result.audio_f0_times
            f0_values = self.result.audio_f0_values
            mask = (f0_times >= start_s) & (f0_times <= end_s)
            if np.any(mask):
                f0_praat_df = pd.DataFrame({
                    'Time_F0_Praat': f0_times[mask],
                    'F0_Praat (Hz)': f0_values[mask]
                }).set_index('Time_F0_Praat')

        # 3. Get F0 Data (GCI)
        f0_gci_df = pd.DataFrame()
        if self.result.gci_f0_times is not None:
            gci_times = self.result.gci_f0_times
            gci_values = self.result.gci_f0_values
            mask = (gci_times >= start_s) & (gci_times <= end_s)
            if np.any(mask):
                f0_gci_df = pd.DataFrame({
                    'Time_F0_GCI': gci_times[mask],
                    'F0_GCI (Hz)': gci_values[mask]
                }).set_index('Time_F0_GCI')
                
        # 4. Glottal Movement
        glottal_df = pd.DataFrame()
        if self.result.glottal_movement_events:
            g_times = [x[0] for x in self.result.glottal_movement_events if start_s <= x[0] <= end_s]
            g_types = [x[1] for x in self.result.glottal_movement_events if start_s <= x[0] <= end_s]
            if g_times:
                glottal_df = pd.DataFrame({
                    'Time_Glottal': g_times,
                    'Glottal_Movement': g_types
                }).set_index('Time_Glottal')
                
        # Combine
        df = cq_sq_df.join(f0_praat_df, how='outer').join(f0_gci_df, how='outer').join(glottal_df, how='outer')
        df.sort_index(inplace=True)
        df.to_csv(csv_path, na_rep='NaN', index_label="Time (s)")

    def _save_plots(self, start_s, end_s, spec_path, cq_path, wave_path):
        LIGHT_STYLE = {
            'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
            'xtick.color': 'black', 'ytick.color': 'black',
            'grid.color': '#DDDDDD', 'grid.linestyle': ':',
            'figure.facecolor': 'white', 'axes.facecolor': 'white',
            'savefig.facecolor': 'white', 'text.color': 'black',
            'lines.color': 'black', 'patch.edgecolor': 'black',
            'font.family': ['Times New Roman', 'SimSun'],
            'axes.unicode_minus': False
        }
        
        with plt.style.context(LIGHT_STYLE):
            # --- 1. Save CQ/SQ Plot ---
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1_sq = ax1.twinx()
            ax1.set_title(f"EGG CQ & SQ ({start_s:.2f}s - {end_s:.2f}s)")
            
            use_raw = not self.show_filtered_egg
            cq_t, cq_v, sq_v = self.service.calculate_cq_sq_segment(self.result, start_s, end_s, self.config, use_raw_signal=use_raw)
            
            if cq_t is not None and len(cq_t) > 0:
                l1, = ax1.plot(cq_t, cq_v, label='CQ', color='blue', marker='.', linestyle='', markersize=5)
                l2, = ax1_sq.plot(cq_t, sq_v, label='SQ', color='green', marker='x', linestyle='', markersize=5)
                lines = [l1, l2]
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right')
            
            # ax1.set_xlabel("Time (s)")
            # ax1.set_ylabel("Contact Quotient (CQ)", color='blue')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='y', colors='blue', labelcolor='blue')
            
            # ax1_sq.set_ylabel("Speed Quotient (SQ)", color='green')
            ax1_sq.set_ylim(-1.1, 1.1)
            ax1_sq.tick_params(axis='y', colors='green', labelcolor='green')
            
            ax1.set_xlim(start_s, end_s)
            ax1.grid(True)
            fig1.tight_layout()
            fig1.savefig(cq_path, dpi=150)
            plt.close(fig1)
            
            # --- 2. Save Spectrogram Plot ---
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            try:
                fig2.set_layout_engine('constrained')
            except:
                fig2.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)

            ax2_f0 = ax2.twinx()
            ax2.set_title(f"Spectrogram ({start_s:.2f}s - {end_s:.2f}s)")
            
            fs = self.result.fs
            s_idx = int(start_s * fs)
            e_idx = int(end_s * fs)
            if e_idx > s_idx:
                audio_seg = self.result.audio_signal[s_idx:e_idx]
                nfft = int(fs * self.config.spec_window_ms / 1000.0)
                noverlap = int(nfft * 0.75)
                
                Pxx, freqs, bins, im = ax2.specgram(
                    audio_seg, NFFT=nfft, Fs=fs, noverlap=noverlap,
                    cmap='gray_r', vmin=self.config.spec_vmin, vmax=self.config.spec_vmax
                )
                im.set_extent([start_s, end_s, freqs[0], freqs[-1]])
                fig2.colorbar(im, ax=ax2, label='Magnitude (dB)')
            
            # ax2.set_xlabel("Time (s)")
            # ax2.set_ylabel("Frequency (Hz)")
            ax2.set_ylim(0, 5000)
            
            # Plot F0
            # 1. Praat F0
            if self.show_f0 and self.result.audio_f0_times is not None:
                f0_times = self.result.audio_f0_times
                f0_values = self.result.audio_f0_values
                mask = (f0_times >= start_s) & (f0_times <= end_s)
                if np.any(mask):
                    ax2_f0.plot(f0_times[mask], f0_values[mask], color='black', marker='.', markersize=2, linestyle='None')
            
            # 2. Corrected F0
            if self.f0_corrected and self.result.gci_f0_times is not None:
                f0_times = self.result.gci_f0_times
                f0_values = self.result.gci_f0_values
                mask = (f0_times >= start_s) & (f0_times <= end_s)
                if np.any(mask):
                    ax2_f0.plot(f0_times[mask], f0_values[mask], color='red', marker='.', markersize=2, linestyle='None')
            
            # ax2_f0.set_ylabel("F0 (Hz)")
            ax2_f0.set_ylim(50, 500)
            ax2.set_xlim(start_s, end_s)
            
            fig2.savefig(spec_path, dpi=150)
            plt.close(fig2)
            
            # --- 3. Save Waveforms Plot ---
            fig3, (ax_audio, ax_egg) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig3.suptitle(f"Waveforms ({start_s:.2f}s - {end_s:.2f}s)")
            
            if e_idx > s_idx:
                time_vec = np.linspace(start_s, end_s, len(audio_seg))
                ax_audio.plot(time_vec, audio_seg, color='black', lw=0.5)
                
                egg_full = self.result.egg_signal_raw if use_raw else self.result.egg_signal_processed
                egg_seg = egg_full[s_idx:e_idx]
                
                # If we are using raw for display but saving, user might want "what they see".
                # But for saving plot, let's stick to what we used for CQ/SQ (use_raw logic).
                # But actually, if use_raw is False, we should filter egg_seg locally if we want it to match
                # exactly the "Filtered" view which might depend on slider.
                # However, calculate_cq_sq_segment handles filtering internally if use_raw is False.
                # Here we just want to plot.
                
                if not use_raw:
                    # Apply filters locally to match view
                    seg_detrend = signal.detrend(egg_seg)
                    seg_hp = apply_highpass_filter(seg_detrend, cutoff_freq=self.config.highpass_cutoff, fs=fs)
                    seg_lp = apply_lowpass_filter(seg_hp, cutoff_freq=self.config.lowpass_cutoff, fs=fs)
                    egg_to_plot = seg_lp
                else:
                    egg_to_plot = egg_seg
                
                ax_egg.plot(time_vec, egg_to_plot, color='black', lw=0.5)
                
            # ax_audio.set_ylabel("Audio Amplitude")
            ax_audio.grid(True)
            
            # ax_egg.set_ylabel("EGG Amplitude")
            # ax_egg.set_xlabel("Time (s)")
            ax_egg.grid(True)
            ax_egg.set_xlim(start_s, end_s)
            
            fig3.tight_layout()
            fig3.savefig(wave_path, dpi=150)
            plt.close(fig3)

    def show_batch_dialog(self):
        dlg = BatchProcessingDialog(self.service, self.config, self)
        dlg.exec()

    def show_help_dialog(self):
        help_file = Path(r"d:\PhoneticToolbox\PhoneticToolbox_v2\Phonetic_Export\index.html")
        if not help_file.exists():
            QMessageBox.warning(self, "帮助", f"未找到帮助文件：{help_file}")
            return
        url = QUrl.fromLocalFile(str(help_file))
        url.setFragment("s1764306596730")
        opened = QDesktopServices.openUrl(url)
        if not opened:
            QMessageBox.warning(self, "帮助", "帮助页面打开失败。")
