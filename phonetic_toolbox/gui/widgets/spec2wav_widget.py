import sys
import traceback
import tempfile
import os
import numpy as np
import cv2
import soundfile as sf
from typing import Optional, Tuple, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QDialog, QFormLayout, QLineEdit, QDialogButtonBox, 
    QSizePolicy, QFileDialog, QApplication
)
from PyQt6.QtCore import Qt, QPoint, QUrl, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QGuiApplication, QCursor, QAction, QDesktopServices
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from ...services.spec2wav_service import Spec2WavService
from ...models.spec2wav_models import Spec2WavConfig, Spec2WavResult
from ...core.spec2wav.common import amplitude_to_db

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
    def __init__(self, parent=None, dark_mode=True):
        super().__init__(parent)
        self.setWindowTitle("设置参数")
        layout = QFormLayout(self)
        
        self.time_start = QLineEdit("0")
        self.time_end = QLineEdit("5.0")
        self.freq_start = QLineEdit("0")
        self.freq_end = QLineEdit("5000")
        self.win_length = QLineEdit("10.0")
        
        layout.addRow("起始时间 (s):", self.time_start)
        layout.addRow("结束时间 (s):", self.time_end)
        layout.addRow("起始频率 (Hz):", self.freq_start)
        layout.addRow("结束频率 (Hz):", self.freq_end)
        layout.addRow("窗宽 (ms):", self.win_length)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        # Apply simple styling based on mode
        # Note: Global styles should handle this if QDialog is used correctly
        
    def get_values(self):
        try:
            return (float(self.time_start.text()), float(self.time_end.text()), 
                    float(self.freq_start.text()), float(self.freq_end.text()),
                    float(self.win_length.text()))
        except ValueError:
            return None

class Spec2WavWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.service = Spec2WavService()
        self.setWindowTitle("语谱图转音频工具 (Spec2Wav)")
        self.resize(1000, 400)
        
        self.main_layout = QVBoxLayout(self)
        self._init_ui()
        
        # State
        self.current_img_gray = None
        self.generated_audio = None
        self.sr = 22050
        self.params = None
        self.overlay = None
        self.is_dark = True
        
        # Audio Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.temp_audio_file = None

    def _init_ui(self):
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
        self.btn_help = QPushButton("帮助")
        
        self.button_layout.addWidget(self.btn_select)
        self.button_layout.addWidget(self.btn_process)
        self.button_layout.addWidget(self.btn_play)
        self.button_layout.addWidget(self.btn_export)
        self.button_layout.addWidget(self.btn_help)
        
        self.main_layout.addLayout(self.button_layout)
        
        # Connections
        self.btn_select.clicked.connect(self.start_selection)
        self.btn_process.clicked.connect(self.process_audio)
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_export.clicked.connect(self.export_audio)
        self.btn_help.clicked.connect(self._open_help_doc)

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        if self.is_dark:
            panel_bg = "#000000"
            panel_fg = "#ffffff"
            border = "#666666"
        else:
            panel_bg = "#ffffff"
            panel_fg = "#111111"
            border = "#bfbfbf"
        panel_style = (
            f"border: 1px solid {border}; "
            f"background-color: {panel_bg}; "
            f"color: {panel_fg};"
        )
        self.left_label.setStyleSheet(panel_style)
        self.right_label.setStyleSheet(panel_style)

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
            
        # Get screen geometry
        geometry = screen.geometry()
        
        # Grab window captures the screen at physical resolution
        screenshot = screen.grabWindow(0)
        
        self.overlay = SelectionOverlay(screenshot, geometry)
        self.overlay.selection_confirmed.connect(self.handle_selection)
        self.overlay.show()
        
        QMessageBox.information(self.overlay, "提示", "请依次点击语谱图的四个顶点 (左上 -> 右上 -> 右下 -> 左下 顺序最佳，或任意顺序)")

    def handle_selection(self, points, full_pixmap):
        try:
            self.show()
            if len(points) != 4:
                return

            # Convert points to numpy array
            pts = np.array([(p.x(), p.y()) for p in points], dtype="float32")
            
            # Sort points
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
            if not self.current_img_gray.flags['C_CONTIGUOUS']:
                self.current_img_gray = np.ascontiguousarray(self.current_img_gray)
                
            qimg_gray = QImage(self.current_img_gray.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            self.left_label.setPixmap(QPixmap.fromImage(qimg_gray))
            
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[Spec2Wav] Error in handle_selection: {e}\n{error_msg}")
            QMessageBox.critical(self, "错误", f"处理选区时出错: {e}")
        
    def ask_parameters(self):
        dialog = ParameterDialog(self, self.is_dark)
        if dialog.exec():
            self.params = dialog.get_values()
        else:
            self.params = None

    def process_audio(self):
        try:
            if self.current_img_gray is None:
                QMessageBox.warning(self, "警告", "请先选择语谱图")
                return
            
            self.ask_parameters()
            if not self.params:
                return

            time_start, time_end, freq_start, freq_end, win_length_ms = self.params
            
            # Disable buttons
            self.btn_process.setEnabled(False)
            self.btn_process.setText("处理中...")
            QApplication.processEvents()
            
            # Create Config
            config = Spec2WavConfig(
                image_data=self.current_img_gray,
                time_start=time_start,
                time_end=time_end,
                freq_start=freq_start,
                freq_end=freq_end,
                win_length_ms=win_length_ms,
                target_sr=44100 # Default target
            )
            
            # Call Service
            result = self.service.convert(config)
            
            # Store results
            self.generated_audio = result.audio
            self.sr = result.sr
            
            # Update UI
            self.btn_process.setEnabled(True)
            self.btn_process.setText("语谱图转音频")
            
            # Display reconstructed spectrogram
            if result.reconstructed_spectrogram_db is not None:
                self._display_reconstructed_spectrogram(result.reconstructed_spectrogram_db)
            
            QMessageBox.information(self, "成功", "音频生成完毕！")
            
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[Spec2Wav] Error in process_audio: {e}\n{error_msg}")
            self.btn_process.setEnabled(True)
            self.btn_process.setText("语谱图转音频")
            QMessageBox.critical(self, "错误", f"处理失败: {e}")

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
            print(f"[Spec2Wav] Error displaying spectrogram: {e}")

    def play_audio(self):
        if self.generated_audio is None:
            QMessageBox.warning(self, "警告", "没有生成的音频")
            return
            
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
                self.service.save_audio(self.generated_audio, self.sr, file_path)
                QMessageBox.information(self, "成功", f"音频已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _open_help_doc(self):
        help_file = r"d:\PhoneticToolbox\PhoneticToolbox_v2\Phonetic_Export\index.html"
        if not os.path.exists(help_file):
            QMessageBox.warning(self, "帮助", f"未找到帮助文件：{help_file}")
            return
        url = QUrl.fromLocalFile(help_file)
        url.setFragment("s1765795948853")
        opened = QDesktopServices.openUrl(url)
        if not opened:
            QMessageBox.warning(self, "帮助", "帮助页面打开失败。")

    def closeEvent(self, event):
        # Cleanup temp file
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except:
                pass
        event.accept()
