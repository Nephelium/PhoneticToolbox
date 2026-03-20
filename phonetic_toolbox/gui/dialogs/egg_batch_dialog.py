from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QProgressBar, QFileDialog, QMessageBox, QTextEdit,
                             QGroupBox, QCheckBox, QComboBox, QFormLayout, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import os
import glob
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy import signal
from phonetic_toolbox.models.config import EGGConfig
from phonetic_toolbox.core.signals.filters import apply_highpass_filter, apply_lowpass_filter
from phonetic_toolbox.core.egg.analysis import calculate_cq_sq

# Set font for batch plots too
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def save_batch_plots(result, config, output_dir, base_name, f0_corrected=True, show_f0=True):
    """
    Generate and save 3 plots for the full file duration.
    """
    start_s = 0.0
    end_s = result.file_duration
    fs = result.fs
    
    # 1. Plot Paths
    plot_spec_path = os.path.join(output_dir, f"{base_name}_SPEC_F0.png")
    plot_cq_path = os.path.join(output_dir, f"{base_name}_CQ_SQ.png")
    plot_wave_path = os.path.join(output_dir, f"{base_name}_WAVEFORMS.png")
    
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
        # --- 1. CQ/SQ ---
        try:
            fig1 = Figure(figsize=(12, 6))
            canvas1 = FigureCanvasAgg(fig1)
            ax1 = fig1.add_subplot(111)
            ax1_sq = ax1.twinx()
            ax1.set_title(f"EGG CQ & SQ ({base_name})")
            
            # Recalculate global CQ/SQ if needed, or use result's if available
            # BatchWorker ensures these are in result
            gci = result.gci_times
            goi = result.goi_times
            peaks = result.peak_times
            
            cq_t, cq_v, sq_v = calculate_cq_sq(gci, goi, peaks)
            
            if cq_t is not None and len(cq_t) > 0:
                # Plot with some downsampling if too dense? Matplotlib handles it but slow.
                # For batch, full data is fine.
                l1, = ax1.plot(cq_t, cq_v, label='CQ', color='blue', marker='.', linestyle='', markersize=2)
                l2, = ax1_sq.plot(cq_t, sq_v, label='SQ', color='green', marker='x', linestyle='', markersize=2)
                lines = [l1, l2]
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right')
            
            # ax1.set_xlabel("Time (s)") # Removed
            # ax1.set_ylabel("Contact Quotient (CQ)", color='blue')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='y', colors='blue', labelcolor='blue')
            
            # ax1_sq.set_ylabel("Speed Quotient (SQ)", color='green')
            ax1_sq.set_ylim(-1.1, 1.1)
            ax1_sq.tick_params(axis='y', colors='green', labelcolor='green')
            
            ax1.set_xlim(start_s, end_s)
            ax1.grid(True)
            fig1.tight_layout()
            fig1.savefig(plot_cq_path, dpi=100)
            # plt.close(fig1) # Not needed for Figure object
        except Exception as e:
            print(f"Error saving CQ plot for {base_name}: {e}")

        # --- 2. Spectrogram ---
        try:
            fig2 = Figure(figsize=(12, 6))
            canvas2 = FigureCanvasAgg(fig2)
            ax2 = fig2.add_subplot(111)
            
            try: fig2.set_layout_engine('constrained')
            except: pass
            
            ax2_f0 = ax2.twinx()
            ax2.set_title(f"Spectrogram ({base_name})")
            
            # Downsample audio for spectrogram if too long to save memory/time?
            # matplotlib specgram handles 1D array.
            
            nfft = int(fs * config.spec_window_ms / 1000.0)
            noverlap = int(nfft * 0.75)
            
            Pxx, freqs, bins, im = ax2.specgram(
                result.audio_signal, NFFT=nfft, Fs=fs, noverlap=noverlap,
                cmap='gray_r', vmin=config.spec_vmin, vmax=config.spec_vmax
            )
            fig2.colorbar(im, ax=ax2, label='Magnitude (dB)')
            
            # ax2.set_xlabel("Time (s)")
            # ax2.set_ylabel("Frequency (Hz)")
            ax2.set_ylim(0, 5000)
            ax2.set_xlim(start_s, end_s)
            
            # Plot F0
            # 1. Praat F0
            if show_f0 and result.audio_f0_times is not None:
                ax2_f0.plot(result.audio_f0_times, result.audio_f0_values, color='black', marker='.', markersize=2, linestyle='None', label='Praat F0')
            
            # 2. Corrected F0
            if f0_corrected and result.gci_f0_times is not None:
                ax2_f0.plot(result.gci_f0_times, result.gci_f0_values, color='red', marker='.', markersize=2, linestyle='None', label='Corrected F0')
            
            # ax2_f0.set_ylabel("F0 (Hz)")
            ax2_f0.set_ylim(50, 500)
            
            fig2.savefig(plot_spec_path, dpi=100)
        except Exception as e:
            print(f"Error saving Spec plot for {base_name}: {e}")

        # --- 3. Waveforms ---
        try:
            fig3 = Figure(figsize=(12, 6))
            canvas3 = FigureCanvasAgg(fig3)
            # subplots(2, 1) equivalent
            ax_audio = fig3.add_subplot(2, 1, 1)
            ax_egg = fig3.add_subplot(2, 1, 2, sharex=ax_audio)
            
            fig3.suptitle(f"Waveforms ({base_name})")
            
            # Downsample for waveform plot to avoid massive PDF/PNG
            ds_target = 50000
            step = max(1, len(result.time_vector) // ds_target)
            
            t_ds = result.time_vector[::step]
            a_ds = result.audio_signal[::step]
            e_ds = result.egg_signal_processed[::step]
            
            ax_audio.plot(t_ds, a_ds, color='black', lw=0.5)
            # ax_audio.set_ylabel("Audio")
            ax_audio.grid(True)
            
            ax_egg.plot(t_ds, e_ds, color='black', lw=0.5)
            # ax_egg.set_ylabel("EGG")
            # ax_egg.set_xlabel("Time (s)")
            ax_egg.grid(True)
            ax_egg.set_xlim(start_s, end_s)
            
            fig3.tight_layout()
            fig3.savefig(plot_wave_path, dpi=100)
        except Exception as e:
            print(f"Error saving Wave plot for {base_name}: {e}")

class BatchWorker(QObject):
    progress = pyqtSignal(int, int, str) # current, total, filename
    finished = pyqtSignal()
    log = pyqtSignal(str)
    
    def __init__(self, service, input_dir, output_dir, config, params):
        super().__init__()
        self.service = service
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.params = params
        self._is_canceled = False
        
    def cancel(self):
        self._is_canceled = True
        
    def run(self):
        wav_files = glob.glob(os.path.join(self.input_dir, "*.wav"))
        total = len(wav_files)
        if total == 0:
            self.log.emit("未找到 WAV 文件")
            self.finished.emit()
            return
            
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except Exception as e:
                self.log.emit(f"无法创建输出目录: {e}")
                self.finished.emit()
                return
        
        # Unpack params
        silence_thresh = self.params.get('silence_threshold', 0.01)
        swap_channels = self.params.get('swap_channels', False)
        keep_f0 = self.params.get('keep_f0', True)
        keep_corr_f0 = self.params.get('keep_corr_f0', True)
        gen_images = self.params.get('generate_images', False)
        
        # Update config with filter/method params
        self.config.highpass_cutoff = self.params.get('highpass', 25.0)
        self.config.gci_method = self.params.get('gci_method', 'slope')
        self.config.goi_method = self.params.get('goi_method', 'scale')
        
        self.log.emit(f"开始批量处理 (总数: {total})")
        self.log.emit(f"配置: HP={self.config.highpass_cutoff}Hz, GCI={self.config.gci_method}, GOI={self.config.goi_method}")
        self.log.emit(f"选项: 静音阈值={silence_thresh}, 交换声道={swap_channels}, 生成图片={gen_images}")
            
        for i, wav_path in enumerate(wav_files):
            if self._is_canceled:
                self.log.emit("任务已取消")
                break
                
            filename = os.path.basename(wav_path)
            self.progress.emit(i + 1, total, filename)
            self.log.emit(f"正在处理: {filename}")
            
            try:
                # Load
                result = self.service.load_file(wav_path, self.config, flip_channels=swap_channels)
                
                # Analyze Events
                result = self.service.analyze_events(result, self.config)
                
                # Calculate F0 (Praat)
                # Always calculate if we need it for images or CSV
                if keep_f0 or gen_images:
                    self.service.calculate_praat_f0(result)
                
                # Prepare CSV Data
                # Re-calculate global CQ/SQ/F0 arrays mapped to time
                # We need a unified time axis for CSV. Usually GCI times are good, or frame-based.
                # User wants "CQ、SQ、基频、校正基频" in CSV.
                # CQ/SQ are event-based (GCI). Corrected F0 is event-based (GCI).
                # Praat F0 is frame-based.
                # To combine them, we can use GCI times as the reference for event-based metrics, 
                # and interpolate Praat F0 to GCI times.
                
                gci = result.gci_times
                goi = result.goi_times
                peaks = result.peak_times
                
                cq_t, cq_v, sq_v = calculate_cq_sq(gci, goi, peaks)
                
                if cq_t is None: cq_t = []
                if cq_v is None: cq_v = []
                if sq_v is None: sq_v = []
                
                # Build DataFrame
                df = pd.DataFrame({'Time (s)': cq_t, 'CQ': cq_v, 'SQ': sq_v})
                
                # Add Corrected F0 (GCI-based)
                if keep_corr_f0 and result.gci_f0_times is not None:
                    # Interpolate GCI F0 to CQ times (should be same/similar if based on GCI)
                    # Actually GCI F0 is defined at mid-interval. CQ is defined at GCI.
                    # Let's align everything to the 'Time (s)' (which is GCI times).
                    f0_interp = np.interp(cq_t, result.gci_f0_times, result.gci_f0_values, left=np.nan, right=np.nan)
                    df['F0_GCI (Hz)'] = f0_interp
                    
                # Add Praat F0
                if keep_f0 and result.audio_f0_times is not None:
                    f0_interp = np.interp(cq_t, result.audio_f0_times, result.audio_f0_values, left=np.nan, right=np.nan)
                    df['F0_Praat (Hz)'] = f0_interp
                    
                # Silence Thresholding
                # Calculate intensity/envelope of audio
                # Simple RMS in window? Or just absolute value?
                # "Values below threshold set to NaN".
                # Let's use smoothed envelope.
                audio_abs = np.abs(result.audio_signal)
                # Smooth over ~20ms
                win_len = int(0.02 * result.fs)
                if win_len > 0:
                    kernel = np.ones(win_len) / win_len
                    envelope = np.convolve(audio_abs, kernel, mode='same')
                else:
                    envelope = audio_abs
                    
                # Interpolate envelope to CQ times
                # result.time_vector is for audio
                env_at_cq = np.interp(cq_t, result.time_vector, envelope)
                
                # Apply threshold
                silence_mask = env_at_cq < silence_thresh
                
                cols_to_mask = ['CQ', 'SQ']
                if 'F0_GCI (Hz)' in df.columns: cols_to_mask.append('F0_GCI (Hz)')
                if 'F0_Praat (Hz)' in df.columns: cols_to_mask.append('F0_Praat (Hz)')
                
                df.loc[silence_mask, cols_to_mask] = np.nan
                
                # Save CSV
                base_name = os.path.splitext(filename)[0]
                csv_path = os.path.join(self.output_dir, base_name + ".csv")
                df.to_csv(csv_path, index=False, float_format='%.6f', na_rep='NaN')
                
                self.log.emit(f"  CSV Saved: {csv_path}")
                
                # Generate Images
                if gen_images:
                    self.log.emit(f"  Generating images...")
                    save_batch_plots(result, self.config, self.output_dir, base_name, 
                                     f0_corrected=keep_corr_f0, show_f0=keep_f0)
                
            except Exception as e:
                self.log.emit(f"  失败: {str(e)}")
                traceback.print_exc()
                
        self.finished.emit()

class BatchProcessingDialog(QDialog):
    def __init__(self, service, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EGG 批量处理")
        self.resize(700, 550)
        self.service = service
        self.config = config
        
        layout = QVBoxLayout(self)
        
        # Input Dir
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("输入目录:"))
        self.input_edit = QLineEdit()
        h1.addWidget(self.input_edit)
        btn_in = QPushButton("浏览...")
        btn_in.clicked.connect(self.browse_input)
        h1.addWidget(btn_in)
        layout.addLayout(h1)
        
        # Output Dir
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("输出目录:"))
        self.output_edit = QLineEdit()
        h2.addWidget(self.output_edit)
        btn_out = QPushButton("浏览...")
        btn_out.clicked.connect(self.browse_output)
        h2.addWidget(btn_out)
        layout.addLayout(h2)
        
        # --- Parameters Group ---
        grp = QGroupBox("处理选项")
        gl = QGridLayout() # Use GridLayout for alignment
        
        # Row 1: Silence, Highpass, Swap
        gl.addWidget(QLabel("静音阈值:"), 0, 0)
        self.silence_edit = QLineEdit("0.01")
        gl.addWidget(self.silence_edit, 0, 1)
        
        gl.addWidget(QLabel("高通滤波(Hz):"), 0, 2)
        self.highpass_edit = QLineEdit("25")
        gl.addWidget(self.highpass_edit, 0, 3)
        
        self.swap_chk = QCheckBox("交换左右声道")
        gl.addWidget(self.swap_chk, 0, 4)
        
        # Row 2: Methods
        gl.addWidget(QLabel("GCI 方法:"), 1, 0)
        self.gci_combo = QComboBox()
        self.gci_combo.addItems(["Slope (斜率)", "Scale (尺度)"])
        gl.addWidget(self.gci_combo, 1, 1)
        
        gl.addWidget(QLabel("GOI 方法:"), 1, 2)
        self.goi_combo = QComboBox()
        self.goi_combo.addItems(["Scale (尺度)", "Slope (斜率)"]) # Default Scale
        gl.addWidget(self.goi_combo, 1, 3)
        
        # Row 3: Output Options
        self.keep_f0_chk = QCheckBox("保留基频 (Praat)")
        self.keep_f0_chk.setChecked(True)
        gl.addWidget(self.keep_f0_chk, 2, 0, 1, 2)
        
        self.keep_corr_f0_chk = QCheckBox("保留校正基频 (GCI)")
        self.keep_corr_f0_chk.setChecked(True)
        gl.addWidget(self.keep_corr_f0_chk, 2, 2, 1, 2)
        
        self.gen_images_chk = QCheckBox("生成图片")
        gl.addWidget(self.gen_images_chk, 2, 4)
        
        grp.setLayout(gl)
        layout.addWidget(grp)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Buttons
        h3 = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.clicked.connect(self.start_processing)
        h3.addWidget(self.btn_start)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setEnabled(False)
        h3.addWidget(self.btn_cancel)
        
        layout.addLayout(h3)
        
        self.worker = None
        self.thread = None
        
    def browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if d: self.input_edit.setText(d)
        
    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d: self.output_edit.setText(d)
        
    def start_processing(self):
        in_dir = self.input_edit.text()
        out_dir = self.output_edit.text()
        
        if not in_dir or not out_dir:
            QMessageBox.warning(self, "警告", "请选择输入和输出目录")
            return
            
        try:
            silence_val = float(self.silence_edit.text())
            highpass_val = float(self.highpass_edit.text())
        except ValueError:
            QMessageBox.warning(self, "错误", "阈值和频率必须为数字")
            return
            
        params = {
            'silence_threshold': silence_val,
            'highpass': highpass_val,
            'swap_channels': self.swap_chk.isChecked(),
            'gci_method': 'slope' if 'Slope' in self.gci_combo.currentText() else 'scale',
            'goi_method': 'slope' if 'Slope' in self.goi_combo.currentText() else 'scale',
            'keep_f0': self.keep_f0_chk.isChecked(),
            'keep_corr_f0': self.keep_corr_f0_chk.isChecked(),
            'generate_images': self.gen_images_chk.isChecked()
        }
            
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log_text.clear()
        
        self.worker = BatchWorker(self.service, in_dir, out_dir, self.config, params)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log_msg)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()
        
    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.log_msg("正在取消...")
            
    def on_progress(self, current, total, filename):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
    def log_msg(self, msg):
        self.log_text.append(msg)
        
    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log_msg("处理完成")
