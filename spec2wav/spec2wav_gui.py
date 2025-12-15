import sys
import traceback
import tempfile
import os

# Print debug info for frozen app
if getattr(sys, 'frozen', False):
    print(f"[Spec2Wav] Running in frozen mode")
    print(f"[Spec2Wav] sys._MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}")
    print(f"[Spec2Wav] sys.path: {sys.path[:5]}...")  # Print first 5 paths

try:
    import cv2
    print(f"[Spec2Wav] cv2 imported successfully: {cv2.__version__}")
except ImportError as e:
    print(f"[Spec2Wav] ERROR: Failed to import cv2: {e}")
    raise

try:
    import numpy as np
    print(f"[Spec2Wav] numpy imported successfully: {np.__version__}")
except ImportError as e:
    print(f"[Spec2Wav] ERROR: Failed to import numpy: {e}")
    raise

try:
    import soundfile as sf
    print(f"[Spec2Wav] soundfile imported successfully")
except ImportError as e:
    print(f"[Spec2Wav] ERROR: Failed to import soundfile: {e}")
    raise

try:
    from scipy import signal
    print(f"[Spec2Wav] scipy.signal imported successfully")
except ImportError as e:
    print(f"[Spec2Wav] ERROR: Failed to import scipy.signal: {e}")
    raise

# Use numpy.fft for better compatibility (scipy.fft may not exist in older versions)
try:
    from numpy.fft import fft, ifft
    print(f"[Spec2Wav] numpy.fft imported successfully")
except ImportError as e:
    print(f"[Spec2Wav] ERROR: Failed to import numpy.fft: {e}")
    raise

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QMessageBox, QDialog, 
                             QFormLayout, QLineEdit, QDialogButtonBox, QSizePolicy, QFileDialog)
from PyQt6.QtCore import Qt, QPoint, QUrl, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QGuiApplication, QCursor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

print(f"[Spec2Wav] All imports successful")


# ============================================================================
# Pure NumPy/SciPy implementation of Griffin-Lim algorithm
# This avoids the numba/llvmlite dependency that causes DLL loading issues
# when packaged with PyInstaller.
# ============================================================================

def _stft(y, n_fft, hop_length, win_length, window):
    """
    Short-time Fourier Transform using pure NumPy/SciPy.
    
    Parameters:
        y: Input signal
        n_fft: FFT size
        hop_length: Number of samples between frames
        win_length: Window length
        window: Window function array
        
    Returns:
        Complex STFT matrix
    """
    try:
        # Safety checks
        if n_fft < 2:
            n_fft = 2
        if hop_length < 1:
            hop_length = 1
        if len(y) == 0:
            return np.zeros((n_fft // 2 + 1, 1), dtype=np.complex128)
        
        # Pad signal
        pad_length = n_fft // 2
        y_padded = np.pad(y, (pad_length, pad_length), mode='reflect')
        
        # Calculate number of frames
        n_frames = max(1, 1 + (len(y_padded) - n_fft) // hop_length)
        
        # Create output matrix
        stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
        
        # Pad window to n_fft if needed
        if len(window) < n_fft:
            window_padded = np.zeros(n_fft)
            start = (n_fft - len(window)) // 2
            window_padded[start:start + len(window)] = window
            window = window_padded
        elif len(window) > n_fft:
            window = window[:n_fft]
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            if end > len(y_padded):
                break
            frame = y_padded[start:end] * window
            spectrum = fft(frame)
            stft_matrix[:, i] = spectrum[:n_fft // 2 + 1]
        
        return stft_matrix
    except Exception as e:
        print(f"[_stft] Error: {e}")
        traceback.print_exc()
        raise


def _istft(stft_matrix, hop_length, win_length, n_fft, window, length=None):
    """
    Inverse Short-time Fourier Transform using pure NumPy/SciPy.
    
    Parameters:
        stft_matrix: Complex STFT matrix
        hop_length: Number of samples between frames
        win_length: Window length
        n_fft: FFT size
        window: Window function array
        length: Expected output length (optional)
        
    Returns:
        Reconstructed signal
    """
    try:
        n_frames = stft_matrix.shape[1]
        
        # Safety check for n_fft
        if n_fft < 2:
            n_fft = 2
        
        # Reconstruct full spectrum (conjugate symmetric)
        full_spectrum = np.zeros((n_fft, n_frames), dtype=np.complex128)
        
        # Safely copy the STFT matrix
        n_bins = min(n_fft // 2 + 1, stft_matrix.shape[0])
        full_spectrum[:n_bins, :] = stft_matrix[:n_bins, :]
        
        # Fill conjugate symmetric part
        if n_fft > 2:
            conj_start = n_fft // 2 + 1
            conj_end = n_fft
            src_start = n_bins - 2
            src_end = 0
            if src_start > src_end:
                full_spectrum[conj_start:conj_end, :] = np.conj(stft_matrix[src_start:src_end:-1, :])
        
        # Pad window to n_fft if needed
        if len(window) < n_fft:
            window_padded = np.zeros(n_fft)
            start = (n_fft - len(window)) // 2
            window_padded[start:start + len(window)] = window
            window = window_padded
        elif len(window) > n_fft:
            window = window[:n_fft]
        
        # Calculate output length
        expected_length = n_fft + hop_length * (n_frames - 1)
        y = np.zeros(expected_length)
        window_sum = np.zeros(expected_length)
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            if end > expected_length:
                break
            frame = np.real(ifft(full_spectrum[:, i]))
            y[start:end] += frame * window
            window_sum[start:end] += window ** 2
        
        # Normalize by window sum (avoid division by zero)
        window_sum = np.maximum(window_sum, 1e-8)
        y = y / window_sum
        
        # Remove padding safely
        pad_length = n_fft // 2
        if pad_length > 0 and len(y) > 2 * pad_length:
            y = y[pad_length:-pad_length]
        
        if length is not None:
            if len(y) > length:
                y = y[:length]
            elif len(y) < length:
                y = np.pad(y, (0, length - len(y)))
        
        return y
    except Exception as e:
        print(f"[_istft] Error: {e}")
        traceback.print_exc()
        raise


def griffinlim_numpy(S, n_iter=32, hop_length=512, win_length=None, n_fft=2048):
    """
    Griffin-Lim algorithm for phase reconstruction using pure NumPy/SciPy.
    
    This implementation avoids the numba/llvmlite dependency that causes
    DLL loading issues when packaged with PyInstaller.
    
    Parameters:
        S: Magnitude spectrogram (n_fft/2+1, n_frames)
        n_iter: Number of iterations
        hop_length: Number of samples between frames
        win_length: Window length (defaults to n_fft)
        n_fft: FFT size
        
    Returns:
        Reconstructed audio signal
    """
    try:
        print(f"[griffinlim_numpy] Starting with S.shape={S.shape}, n_fft={n_fft}, hop_length={hop_length}")
        sys.stdout.flush()
        
        # Safety checks
        if S.size == 0:
            print("[griffinlim_numpy] Warning: Empty spectrogram")
            return np.zeros(1)
        
        if n_fft < 2:
            n_fft = 2
        if hop_length < 1:
            hop_length = 1
        
        if win_length is None:
            win_length = n_fft
        
        # Ensure win_length <= n_fft and > 0
        win_length = max(1, min(win_length, n_fft))
        
        # Create Hann window
        window = signal.windows.hann(win_length, sym=False)
        
        # Initialize with random phase
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = S * angles
        
        # Estimate output length
        n_frames = S.shape[1]
        length = max(1, hop_length * (n_frames - 1) + n_fft - 2 * (n_fft // 2))
        
        print(f"[griffinlim_numpy] n_frames={n_frames}, estimated length={length}")
        sys.stdout.flush()
        
        # Griffin-Lim iterations
        for iter_idx in range(n_iter):
            # Inverse STFT
            y = _istft(S_complex, hop_length, win_length, n_fft, window, length)
            
            # Forward STFT
            S_rebuilt = _stft(y, n_fft, hop_length, win_length, window)
            
            # Update phase while keeping magnitude
            # Handle shape mismatch
            if S_rebuilt.shape != S.shape:
                # Resize S_rebuilt to match S
                min_rows = min(S_rebuilt.shape[0], S.shape[0])
                min_cols = min(S_rebuilt.shape[1], S.shape[1])
                angles = np.exp(1j * np.angle(S_rebuilt[:min_rows, :min_cols]))
                S_complex = np.zeros_like(S, dtype=np.complex128)
                S_complex[:min_rows, :min_cols] = S[:min_rows, :min_cols] * angles
            else:
                angles = np.exp(1j * np.angle(S_rebuilt))
                S_complex = S * angles
        
        # Final inverse STFT
        y = _istft(S_complex, hop_length, win_length, n_fft, window, length)
        
        print(f"[griffinlim_numpy] Completed, output length={len(y)}")
        sys.stdout.flush()
        
        return y
    except Exception as e:
        print(f"[griffinlim_numpy] Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise


def resample_audio(y, orig_sr, target_sr):
    """
    Resample audio using linear interpolation.
    
    This implementation uses pure NumPy to avoid scipy.signal.resample
    which can cause crashes in PyInstaller frozen apps due to FFT/BLAS issues.
    
    Parameters:
        y: Input audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio signal
    """
    print(f"[resample_audio] Starting resample from {orig_sr} to {target_sr}, input length={len(y)}")
    sys.stdout.flush()
    
    if orig_sr == target_sr:
        return y
    
    # Calculate new length
    duration = len(y) / orig_sr
    new_length = int(duration * target_sr)
    
    print(f"[resample_audio] Calculated new_length={new_length}")
    sys.stdout.flush()
    
    # Use linear interpolation (pure NumPy, no scipy dependency)
    # This is safer for PyInstaller frozen apps
    old_indices = np.arange(len(y))
    new_indices = np.linspace(0, len(y) - 1, new_length)
    y_resampled = np.interp(new_indices, old_indices, y)
    
    print(f"[resample_audio] Resample completed, output length={len(y_resampled)}")
    sys.stdout.flush()
    
    return y_resampled


def amplitude_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Convert amplitude spectrogram to dB-scaled spectrogram.
    
    Parameters:
        S: Input amplitude spectrogram
        ref: Reference value for dB conversion
        amin: Minimum amplitude threshold
        top_db: Maximum dB range
        
    Returns:
        dB-scaled spectrogram
    """
    S = np.asarray(S)
    
    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = np.abs(ref)
    
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    
    return log_spec


# ============================================================================
# Spectrogram processing functions
# ============================================================================

def process_spectrogram_image(img_gray, time_start, time_end, freq_start, freq_end, min_dB=-30.0, max_dB=0.0):
    """
    Process the grayscale image into a linear spectrogram matrix.
    """
    img_height, img_width = img_gray.shape

    # Flip image (vertical flip) because usually frequency increases upwards, but image origin is top-left
    img_flipped = np.flipud(img_gray)

    # Map grayscale to dB
    # Black (0) -> max_dB, White (255) -> min_dB
    log_spectrogram = max_dB - (img_flipped / 255.0) * (max_dB - min_dB) 

    # dB to Linear
    linear_spectrogram = 10 ** (log_spectrogram / 10.0) * 10 

    # Calculate parameters
    duration = time_end - time_start
    # Sampling rate default to 44100 as requested
    sr = 44100
    
    # Calculate hop_length
    if img_width > 0:
        # We need to map the image width (pixels) to audio samples.
        # Total samples = duration * sr
        # hop_length = Total samples / img_width
        total_samples = duration * sr
        hop_length = int(total_samples / img_width)
    else:
        hop_length = 512 # Fallback
    
    # Ensure hop_length is at least 1
    hop_length = max(1, hop_length)

    # n_fft matches height
    # The height of the image corresponds to the frequency range (freq_end - freq_start)
    # In STFT, n_fft/2 + 1 bins cover 0 to sr/2 Hz.
    # However, our image might only cover a specific frequency range.
    # But Griffin-Lim typically assumes the full spectrogram from 0 to Nyquist.
    # If the image represents 0 to freq_end, and we force sr=44100, we might need to pad the spectrogram if freq_end < sr/2.
    
    # For simplicity and to match user request "Spectrogram display matches original range":
    # We will assume the image represents the full frequency range UP TO freq_end.
    # And we will resample/resize the spectrogram to match the target sr=44100 requirements if needed,
    # OR simpler: we construct the spectrogram such that it corresponds to the image, 
    # and then resample the AUDIO to 44100 at the end?
    # No, Griffin-Lim needs a consistent STFT frame.
    
    # Let's adjust logic:
    # 1. Image height H -> Freq range [freq_start, freq_end]
    # 2. We want output audio at 44100 Hz. Nyquist = 22050 Hz.
    # 3. We need to construct a full spectrogram (0 to 22050 Hz) for Griffin-Lim to work standardly with 44100Hz.
    #    OR we use the calculated SR = 2 * freq_end for synthesis, and THEN resample audio to 44100.
    #    The latter is much safer to preserve the spectral structure drawn by the user.
    
    # Strategy: Generate at natural sampling rate (2 * freq_end), then resample to 44100.
    native_sr = int(2 * freq_end)
    
    if img_width > 0:
        hop_length = int(duration / img_width * native_sr)
    else:
        hop_length = 512
    hop_length = max(1, hop_length)
    
    n_fft = 2 * (img_height - 1)

    return linear_spectrogram, log_spectrogram, hop_length, n_fft, native_sr


def spectrogram_to_audio(spectrogram, hop_length, window_length, n_fft, sr=22050, n_iter=32):
    """
    Griffin-Lim reconstruction using pure NumPy/SciPy implementation.
    
    This avoids the numba/llvmlite dependency that causes DLL loading issues
    when packaged with PyInstaller.
    """
    # Ensure window_length is not greater than n_fft
    if window_length > n_fft:
        window_length = n_fft
    
    # Use our pure NumPy/SciPy Griffin-Lim implementation
    audio = griffinlim_numpy(spectrogram, n_iter=n_iter, hop_length=hop_length, 
                             win_length=window_length, n_fft=n_fft)
    return audio

# Note: AudioWorker class removed - using synchronous processing instead
# to avoid QThread issues in PyInstaller frozen apps

class SelectionOverlay(QWidget):
    selection_confirmed = pyqtSignal(list, object) # points, pixmap

    def __init__(self, pixmap, geometry):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        
        # Set geometry to match the screen's logical geometry
        self.setGeometry(geometry)
        
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        self.original_pixmap = pixmap
        self.points = []
        
        # Prepare display pixmap
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format.Format_ARGB32)
        
        # Create a darkened version for background
        dark_overlay = QImage(image.size(), QImage.Format.Format_ARGB32)
        dark_overlay.fill(QColor(0, 0, 0, 100))
        
        painter = QPainter(image)
        painter.drawImage(0, 0, dark_overlay)
        painter.end()
        
        self.display_pixmap = QPixmap.fromImage(image)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Draw the pixmap scaled to the widget size (logical size)
        painter.drawPixmap(self.rect(), self.display_pixmap)
        
        # Draw points and lines
        painter.setPen(QPen(Qt.GlobalColor.red, 5))
        for p in self.points:
            painter.drawPoint(p)
            
        if len(self.points) > 1:
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i+1])
            if len(self.points) == 4:
                painter.drawLine(self.points[-1], self.points[0])

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Capture logical coordinates
            self.points.append(event.pos())
            self.update()
            
            if len(self.points) == 4:
                # Give a small delay to show the last line
                QTimer.singleShot(200, self.finish_selection)
        elif event.button() == Qt.MouseButton.RightButton:
            # Cancel/Reset
            self.points = []
            self.update()

    def finish_selection(self):
        # We need to map logical coordinates (event.pos) back to the physical image coordinates
        # Calculate scale factor
        scale_x = self.original_pixmap.width() / self.width()
        scale_y = self.original_pixmap.height() / self.height()
        
        scaled_points = []
        for p in self.points:
            scaled_points.append(QPoint(int(p.x() * scale_x), int(p.y() * scale_y)))
            
        self.selection_confirmed.emit(scaled_points, self.original_pixmap)
        self.close()

class ParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置参数")
        layout = QFormLayout(self)
        
        self.time_start = QLineEdit("0")
        self.time_end = QLineEdit("5.0")
        self.freq_start = QLineEdit("0")
        self.freq_end = QLineEdit("5000")
        
        layout.addRow("起始时间 (s):", self.time_start)
        layout.addRow("结束时间 (s):", self.time_end)
        layout.addRow("起始频率 (Hz):", self.freq_start)
        layout.addRow("结束频率 (Hz):", self.freq_end)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        try:
            return (float(self.time_start.text()), float(self.time_end.text()), 
                    float(self.freq_start.text()), float(self.freq_end.text()))
        except ValueError:
            return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            print("[Spec2Wav] Initializing MainWindow...")
            self.setWindowTitle("语谱图转音频工具 (Spec2Wav)")
            self.resize(1000, 600)
            
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.main_layout = QVBoxLayout(self.central_widget)
            self._init_ui()
            print("[Spec2Wav] MainWindow initialized successfully")
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[Spec2Wav] Error initializing MainWindow: {e}\n{error_msg}")
            raise
    
    def _init_ui(self):
        
        """Initialize UI components."""
        # Image Display Area
        self.image_layout = QHBoxLayout()
        
        # Left Image (Selected Spectrogram)
        self.left_label = QLabel("请选择语谱图")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        self.left_label.setScaledContents(True)
        self.left_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        # Right Image (Generated Spectrogram)
        self.right_label = QLabel("生成的语谱图")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        self.right_label.setScaledContents(True)
        self.right_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        self.image_layout.addWidget(self.left_label)
        self.image_layout.addWidget(self.right_label)
        
        self.main_layout.addLayout(self.image_layout, stretch=1)
        
        # Buttons Area
        self.button_layout = QHBoxLayout()
        
        self.btn_select = QPushButton("选择语谱图")
        self.btn_process = QPushButton("语谱图转音频")
        self.btn_play = QPushButton("播放音频")
        self.btn_export = QPushButton("导出音频")
        
        self.button_layout.addWidget(self.btn_select)
        self.button_layout.addWidget(self.btn_process)
        self.button_layout.addWidget(self.btn_play)
        self.button_layout.addWidget(self.btn_export)
        
        self.main_layout.addLayout(self.button_layout)
        
        # Connections
        self.btn_select.clicked.connect(self.start_selection)
        self.btn_process.clicked.connect(self.process_audio)
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_export.clicked.connect(self.export_audio)
        
        # State
        self.current_img_gray = None
        self.generated_audio = None
        self.sr = 22050
        self.params = None
        self.overlay = None  # Keep reference to overlay
        
        # Audio Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.temp_audio_file = None

    def start_selection(self):
        # Minimize/Hide main window
        self.hide()
        
        # Capture screen after a short delay to allow window to hide
        QTimer.singleShot(300, self.capture_and_show_overlay)

    def capture_and_show_overlay(self):
        screen = QGuiApplication.primaryScreen()
        if not screen:
            self.show()
            QMessageBox.critical(self, "错误", "无法获取屏幕")
            return
            
        # Get screen geometry and device pixel ratio
        geometry = screen.geometry()
        device_pixel_ratio = screen.devicePixelRatio()
        
        # Grab window captures the screen at physical resolution
        screenshot = screen.grabWindow(0)
        
        # If device_pixel_ratio > 1 (e.g. 125%, 150% scaling), the screenshot size
        # will be larger than the logical geometry size.
        # We need to ensure the overlay matches the logical geometry for mouse events to line up,
        # but displays the high-res screenshot correctly.
        
        self.overlay = SelectionOverlay(screenshot, geometry)
        self.overlay.selection_confirmed.connect(self.handle_selection)
        self.overlay.show()
        
        QMessageBox.information(self.overlay, "提示", "请依次点击语谱图的四个顶点 (左上 -> 右上 -> 右下 -> 左下 顺序最佳，或任意顺序)")

    def handle_selection(self, points, full_pixmap):
        try:
            self.show()
            if len(points) != 4:
                return

            print(f"[Spec2Wav] Processing selection with {len(points)} points")
            
            # Convert points to numpy array
            pts = np.array([(p.x(), p.y()) for p in points], dtype="float32")
            
            # Sort points to Top-Left, Top-Right, Bottom-Right, Bottom-Left
            # Simple sorting based on sum and diff
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = pts[np.argmin(s)] # TL
            rect[2] = pts[np.argmax(s)] # BR
            rect[1] = pts[np.argmin(diff)] # TR
            rect[3] = pts[np.argmax(diff)] # BL
            
            (tl, tr, br, bl) = rect
            
            # Compute width of new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            # Compute height of new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            print(f"[Spec2Wav] Computed dimensions: {maxWidth}x{maxHeight}")
            
            # Destination points
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # Perspective Transform
            M = cv2.getPerspectiveTransform(rect, dst)
            
            # Convert QPixmap to QImage then to numpy array for cv2
            # Note: QPixmap -> QImage -> bits -> numpy
            qimg = full_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
            width = qimg.width()
            height = qimg.height()
            
            ptr = qimg.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            
            # Warp
            warped = cv2.warpPerspective(arr, M, (maxWidth, maxHeight))
            
            # Convert to grayscale
            self.current_img_gray = cv2.cvtColor(warped, cv2.COLOR_RGBA2GRAY)
            
            # Display in Left Label
            h, w = self.current_img_gray.shape
            bytes_per_line = w
            # Ensure contiguous and use tobytes for PyQt6 compatibility
            if not self.current_img_gray.flags['C_CONTIGUOUS']:
                self.current_img_gray = np.ascontiguousarray(self.current_img_gray)
                
            qimg_gray = QImage(self.current_img_gray.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            self.left_label.setPixmap(QPixmap.fromImage(qimg_gray))
            print(f"[Spec2Wav] Selection processed successfully, image shape: {self.current_img_gray.shape}")
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[Spec2Wav] Error in handle_selection: {e}\n{error_msg}")
            QMessageBox.critical(self, "错误", f"处理选区时出错: {e}")
        
        # Ask for parameters
        # self.ask_parameters() # Don't ask immediately

    def ask_parameters(self):
        dialog = ParameterDialog(self)
        if dialog.exec():
            self.params = dialog.get_values()
        else:
            self.params = None

    def process_audio(self):
        try:
            if self.current_img_gray is None:
                QMessageBox.warning(self, "警告", "请先选择语谱图")
                return
            
            # Always ask for parameters when processing starts
            self.ask_parameters()
            if not self.params:
                return

            time_start, time_end, freq_start, freq_end = self.params
            print(f"[Spec2Wav] Processing with params: time={time_start}-{time_end}s, freq={freq_start}-{freq_end}Hz")
            sys.stdout.flush()
            
            # Disable buttons and show processing message
            self.btn_process.setEnabled(False)
            self.btn_process.setText("处理中...")
            QApplication.processEvents()  # Force UI update
            
            # Process synchronously (avoid QThread issues in frozen app)
            linear_spec, log_spec, hop_length, n_fft, sr = process_spectrogram_image(
                self.current_img_gray, time_start, time_end, freq_start, freq_end
            )
            print(f"[Spec2Wav] Spectrogram processed: shape={linear_spec.shape}, hop_length={hop_length}, n_fft={n_fft}, sr={sr}")
            sys.stdout.flush()
            QApplication.processEvents()
            
            # Do the audio processing synchronously
            self._process_audio_sync(linear_spec, hop_length, n_fft, sr)
            
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[Spec2Wav] Error in process_audio: {e}\n{error_msg}")
            sys.stdout.flush()
            self.btn_process.setEnabled(True)
            self.btn_process.setText("语谱图转音频")
            QMessageBox.critical(self, "错误", f"处理失败: {e}")
    
    def _process_audio_sync(self, linear_spec, hop_length, n_fft, sr):
        """Synchronous audio processing to avoid QThread issues in frozen apps."""
        try:
            print(f"[Spec2Wav] Starting sync processing with sr={sr}, n_fft={n_fft}, hop_length={hop_length}")
            sys.stdout.flush()
            
            window_length = 128
            n_iter = 32
            
            # Generate audio at native sampling rate
            native_audio = spectrogram_to_audio(linear_spec, hop_length, window_length, n_fft, sr, n_iter)
            print(f"[Spec2Wav] Generated native audio with length={len(native_audio)}")
            print(f"[Spec2Wav] native_audio dtype={native_audio.dtype}, min={np.min(native_audio):.4f}, max={np.max(native_audio):.4f}")
            sys.stdout.flush()
            QApplication.processEvents()
            
            # Resample to 44100 Hz
            target_sr = 44100
            print(f"[Spec2Wav] About to resample from {sr} to {target_sr}...")
            sys.stdout.flush()
            
            if sr != target_sr:
                audio = resample_audio(native_audio, orig_sr=sr, target_sr=target_sr)
                print(f"[Spec2Wav] Resampled audio from {sr} to {target_sr}, new length={len(audio)}")
            else:
                audio = native_audio
                print(f"[Spec2Wav] No resampling needed, sr already {sr}")
            sys.stdout.flush()
            QApplication.processEvents()
            
            # Compute reconstructed spectrogram for display
            print(f"[Spec2Wav] Computing reconstructed spectrogram...")
            sys.stdout.flush()
            
            window = signal.windows.hann(min(window_length, n_fft), sym=False)
            print(f"[Spec2Wav] Created Hann window with length={len(window)}")
            sys.stdout.flush()
            
            reconstructed_spec = np.abs(_stft(native_audio, n_fft, hop_length, window_length, window))
            print(f"[Spec2Wav] Computed reconstructed spectrogram with shape={reconstructed_spec.shape}")
            sys.stdout.flush()
            
            # Convert to dB for display
            print(f"[Spec2Wav] Converting to dB...")
            sys.stdout.flush()
            reconstructed_db = amplitude_to_db(reconstructed_spec, ref=np.max)
            print(f"[Spec2Wav] Converted to dB, shape={reconstructed_db.shape}")
            sys.stdout.flush()
            
            # Store results
            self.generated_audio = audio
            self.sr = target_sr
            
            # Update UI
            self.btn_process.setEnabled(True)
            self.btn_process.setText("语谱图转音频")
            
            # Display reconstructed spectrogram
            self._display_reconstructed_spectrogram(reconstructed_db)
            
            print("[Spec2Wav] Processing completed successfully")
            sys.stdout.flush()
            QMessageBox.information(self, "成功", "音频生成完毕！")
            
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[Spec2Wav] Error in _process_audio_sync: {e}\n{error_msg}")
            sys.stdout.flush()
            self.btn_process.setEnabled(True)
            self.btn_process.setText("语谱图转音频")
            QMessageBox.critical(self, "错误", f"音频生成失败: {e}")
    
    def _display_reconstructed_spectrogram(self, reconstructed_db):
        """Display the reconstructed spectrogram in the right label."""
        try:
            max_val = np.max(reconstructed_db)
            min_val = np.min(reconstructed_db)
            
            if max_val == min_val:
                norm_spec = np.zeros_like(reconstructed_db, dtype=np.uint8)
            else:
                norm_spec = 255 * (1 - (reconstructed_db - min_val) / (max_val - min_val))
                norm_spec = norm_spec.astype(np.uint8)
            
            # Flip back for display (low freq at bottom)
            norm_spec_flipped = np.ascontiguousarray(np.flipud(norm_spec))
            
            h_r, w_r = norm_spec_flipped.shape
            qimg_res = QImage(norm_spec_flipped.tobytes(), w_r, h_r, w_r, QImage.Format.Format_Grayscale8)
            self.right_label.setPixmap(QPixmap.fromImage(qimg_res))
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[Spec2Wav] Error displaying spectrogram: {e}")
            print(error_msg)

    def play_audio(self):
        if self.generated_audio is None:
            QMessageBox.warning(self, "警告", "没有生成的音频")
            return
            
        # Save to temp file to play
        try:
            # Create temp file
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, self.generated_audio, self.sr)
            
            self.temp_audio_file = path
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"播放失败: {e}")

    def export_audio(self):
        if self.generated_audio is None:
            QMessageBox.warning(self, "警告", "没有生成的音频")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "导出音频", "generated_audio.wav", "WAV Files (*.wav)")
        if file_path:
            try:
                sf.write(file_path, self.generated_audio, self.sr)
                QMessageBox.information(self, "成功", f"音频已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def closeEvent(self, event):
        # Cleanup temp file
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
