from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import wave

import cv2
import matplotlib
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QIcon, QImage, QPixmap
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMessageBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressDialog,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

from phonetic_toolbox.core.lip.metrics import lip_extract
from phonetic_toolbox.utils import get_resource_path

matplotlib.use("QtAgg")

def resolve_ffmpeg_executable() -> str | None:
    bundled = Path(get_resource_path(r"tools\ffmpeg.exe"))
    if bundled.exists():
        return str(bundled)
    external = shutil.which("ffmpeg")
    if external:
        return external
    return None

def resolve_ffprobe_executable(ffmpeg_exe: str | None = None) -> str | None:
    candidates: list[str] = []
    if ffmpeg_exe:
        ffmpeg_path = Path(ffmpeg_exe)
        if ffmpeg_path.name.lower() == "ffmpeg.exe":
            candidates.append(str(ffmpeg_path.with_name("ffprobe.exe")))
        elif ffmpeg_path.name.lower() == "ffmpeg":
            candidates.append(str(ffmpeg_path.with_name("ffprobe")))
    external = shutil.which("ffprobe")
    if external:
        candidates.append(external)
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return str(p)
    return None

def _probe_stream_start_offset_seconds(video_path: str, ffprobe: str) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=start_time",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return 0.0
    if out.returncode != 0:
        return 0.0
    text = (out.stdout or "").strip().splitlines()
    if not text:
        return 0.0
    try:
        val = float(text[0].strip())
    except Exception:
        return 0.0
    if not np.isfinite(val) or val <= 0:
        return 0.0
    return float(val)

def _probe_first_audio_packet_pts_seconds(video_path: str, ffprobe: str) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,best_effort_timestamp_time,dts_time",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "-read_intervals",
        "%+#1",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return 0.0
    if out.returncode != 0:
        return 0.0
    vals: list[float] = []
    for line in (out.stdout or "").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            v = float(s)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return 0.0
    v = min(vals)
    if v <= 0:
        return 0.0
    return float(v)

def probe_audio_stream_start_offset(video_path: str, ffmpeg_exe: str | None) -> float:
    ffprobe = resolve_ffprobe_executable(ffmpeg_exe)
    if not ffprobe:
        return 0.0
    stream_offset = _probe_stream_start_offset_seconds(video_path, ffprobe)
    packet_offset = _probe_first_audio_packet_pts_seconds(video_path, ffprobe)
    return float(max(stream_offset, packet_offset))

def estimate_lip_audio_offset_seconds(
    audio_samples: np.ndarray,
    sample_rate: int,
    lip_times: np.ndarray,
    lip_open: np.ndarray,
    search_seconds: float = 2.0,
) -> float:
    if audio_samples.size < 8 or lip_times.size < 8 or lip_open.size < 8 or sample_rate <= 0:
        return 0.0
    audio = np.abs(audio_samples.astype(np.float64))
    win = max(1, int(round(0.01 * sample_rate)))
    env = np.convolve(audio, np.ones(win, dtype=np.float64) / float(win), mode="same")
    audio_t = np.arange(len(env), dtype=np.float64) / float(sample_rate)

    lip_t = np.array(lip_times, dtype=np.float64)
    lip_y = np.array(lip_open, dtype=np.float64)
    n = min(lip_t.size, lip_y.size)
    lip_t = lip_t[:n]
    lip_y = lip_y[:n]
    if n < 8:
        return 0.0

    t0 = max(float(audio_t[0]), float(lip_t[0]) - search_seconds)
    t1 = min(float(audio_t[-1]), float(lip_t[-1]) + search_seconds)
    if t1 - t0 <= 1.0:
        return 0.0
    grid_fs = 200.0
    grid = np.arange(t0, t1, 1.0 / grid_fs, dtype=np.float64)
    if grid.size < 100:
        return 0.0

    audio_grid = np.interp(grid, audio_t, env)
    audio_grid = (audio_grid - np.mean(audio_grid)) / (np.std(audio_grid) + 1e-12)

    best_offset = 0.0
    best_score = -1e12
    for off in np.arange(-search_seconds, search_seconds + 0.0001, 0.005):
        shifted = np.interp(grid, lip_t + off, lip_y, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted)
        if np.count_nonzero(valid) < 100:
            continue
        lv = shifted[valid]
        lv = (lv - np.mean(lv)) / (np.std(lv) + 1e-12)
        av = audio_grid[valid]
        score = float(np.mean(av * lv))
        if score > best_score:
            best_score = score
            best_offset = float(off)
    return best_offset

class LipGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("唇形提取")
        self.setMinimumSize(1000, 720)
        self.is_dark = True

        self._state_lock = threading.Lock()
        self.is_recording = False
        self.save_directory = str(self._default_save_directory())
        self._recording_start_ts: float | None = None
        self._recording_start_epoch: float | None = None
        self._audio_stream: sd.InputStream | None = None
        self._live_audio_stream: sd.InputStream | None = None
        self._live_audio_stream_active = False
        self._video_capture: cv2.VideoCapture | None = None
        self._face_mesh = None
        self._mesh_connections: tuple[tuple[int, int], ...] = ()
        self._fps_frame_counter = 0
        self._fps_last_ts = time.perf_counter()

        self._frame_times: list[float] = []
        self._frame_times_abs: list[float] = []
        self._landmarks_full: list[np.ndarray] = []
        self._audio_chunks: list[np.ndarray] = []
        self._audio_chunk_times: list[float] = []
        self._audio_chunk_times_abs: list[float] = []
        self._audio_first_chunk_epoch: float | None = None
        self._audio_adc_to_wall_offset: float | None = None
        self._lip_first_frame_epoch: float | None = None
        self.is_raw_recording = False
        self._raw_video_writer: cv2.VideoWriter | None = None
        self._raw_audio_stream: sd.InputStream | None = None
        self._raw_audio_chunks: list[np.ndarray] = []
        self._raw_recording_session_dir: Path | None = None
        self._raw_recording_video_temp_path: Path | None = None
        self._raw_recording_audio_path: Path | None = None
        self._raw_recording_output_path: Path | None = None
        self._raw_recording_sample_rate: int = 44100
        self._metrics: dict[str, list[float]] = {
            "area": [],
            "face_width": [],
            "face_height": [],
            "height_px": [],
            "outer_width_px": [],
            "inner_width_px": [],
            "total_width_px": [],
            "open_px": [],
            "length": [],
            "height": [],
            "outer_width": [],
            "inner_width": [],
            "total_width": [],
            "open": [],
            "circularity": [],
        }

        self._build_ui()
        self._setup_video_pipeline()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        control_grid = QGridLayout()
        control_grid.setHorizontalSpacing(8)
        control_grid.setVerticalSpacing(8)

        self.path_btn = QPushButton("设置保存路径")
        self.path_btn.clicked.connect(self._select_save_directory)
        self.path_edit = QLineEdit(self.save_directory)
        self.path_edit.setReadOnly(True)

        self.start_btn = QPushButton("实时参数计算")
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn = QPushButton("停止计算并保存")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        self.raw_start_btn = QPushButton("开始视频录制")
        self.raw_start_btn.clicked.connect(self.start_raw_recording)
        self.raw_stop_btn = QPushButton("停止录制并保存")
        self.raw_stop_btn.clicked.connect(self.stop_raw_recording)
        self.raw_stop_btn.setEnabled(False)
        self.upload_video_btn = QPushButton("上传视频并识别")
        self.upload_video_btn.clicked.connect(self.upload_video)
        self.replay_btn = QPushButton("唇形动画回放")
        self.replay_btn.clicked.connect(self._open_animation_player)
        self.help_btn = QPushButton("帮助")
        self.help_btn.clicked.connect(self._open_help)
        self.help_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")

        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row0 = QHBoxLayout()
        row0.addWidget(self.path_btn)
        row0.addWidget(self.path_edit, 1)
        control_grid.addLayout(row0, 0, 0)

        row1 = QHBoxLayout()
        row1.addWidget(self.start_btn)
        row1.addWidget(self.stop_btn)
        row1.addWidget(self.raw_start_btn)
        row1.addWidget(self.raw_stop_btn)
        row1.addWidget(self.upload_video_btn)
        row1.addWidget(self.replay_btn)
        row1.addWidget(self.help_btn)
        control_grid.addLayout(row1, 1, 0)

        row2 = QHBoxLayout()
        row2.addWidget(self.status_label, 1)
        row2.addWidget(self.fps_label)
        control_grid.addLayout(row2, 2, 0)

        root.addLayout(control_grid)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid #666; background-color: #111;")
        root.addWidget(self.video_label, 1)

    def _setup_video_pipeline(self) -> None:
        self._video_capture = cv2.VideoCapture(0)
        self._setup_live_audio_stream()
        try:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mesh_connections = tuple(
                set(mp.solutions.face_mesh.FACEMESH_TESSELATION)
                | set(mp.solutions.face_mesh.FACEMESH_CONTOURS)
            )
        except Exception:
            self._face_mesh = None
            self._mesh_connections = ()
            self.status_label.setText("FaceMesh 初始化失败")

        self._video_timer = QTimer(self)
        self._video_timer.setInterval(30)
        self._video_timer.timeout.connect(self._on_video_tick)
        self._video_timer.start()

    def _setup_live_audio_stream(self) -> None:
        if self._live_audio_stream is not None:
            return

        def _live_audio_callback(indata, _frames, time_info, _status):
            if not self.is_recording:
                return
            with self._state_lock:
                start_ts = self._recording_start_ts
                start_epoch = self._recording_start_epoch
            if start_ts is None:
                return
            chunk_wall_now = time.time()
            adc_time = None
            if hasattr(time_info, "inputBufferAdcTime"):
                try:
                    adc_time = float(getattr(time_info, "inputBufferAdcTime"))
                except Exception:
                    adc_time = None
            elif isinstance(time_info, dict):
                try:
                    adc_time = float(time_info.get("inputBufferAdcTime"))
                except Exception:
                    adc_time = None

            if adc_time is not None and np.isfinite(adc_time):
                if self._audio_adc_to_wall_offset is None:
                    self._audio_adc_to_wall_offset = chunk_wall_now - adc_time
                chunk_abs = self._audio_adc_to_wall_offset + adc_time
            else:
                chunk_abs = chunk_wall_now

            self._audio_chunks.append(indata.copy())
            if start_epoch is not None:
                self._audio_chunk_times.append(chunk_abs - start_epoch)
            else:
                self._audio_chunk_times.append(time.perf_counter() - start_ts)
            self._audio_chunk_times_abs.append(chunk_abs)
            if self._audio_first_chunk_epoch is None:
                self._audio_first_chunk_epoch = chunk_abs
            if start_epoch is None:
                self._recording_start_epoch = chunk_abs

        try:
            stream = sd.InputStream(
                samplerate=44100,
                channels=1,
                dtype="int16",
                blocksize=1024,
                callback=_live_audio_callback,
            )
            stream.start()
            self._live_audio_stream = stream
            self._live_audio_stream_active = True
        except Exception:
            self._live_audio_stream = None
            self._live_audio_stream_active = False

    def set_theme(self, is_dark: bool) -> None:
        self.is_dark = is_dark
        if is_dark:
            self.video_label.setStyleSheet("border: 1px solid #666; background-color: #111;")
        else:
            self.video_label.setStyleSheet("border: 1px solid #888; background-color: #f5f5f5;")

    def _select_save_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择保存目录", self.save_directory)
        if selected:
            self.save_directory = selected
            self.path_edit.setText(selected)
            self.status_label.setText("保存路径已更新")

    @staticmethod
    def _default_save_directory() -> Path:
        downloads = Path.home() / "Downloads"
        if downloads.exists() and downloads.is_dir():
            return downloads
        return Path.home()

    def _on_video_tick(self) -> None:
        if self._video_capture is None or not self._video_capture.isOpened():
            self.status_label.setText("摄像头未就绪")
            return

        ok, frame = self._video_capture.read()
        frame_abs = time.time()
        if not ok:
            self.status_label.setText("读取摄像头失败")
            return
        raw_frame = frame.copy()

        h, w, _ = frame.shape
        full_landmarks: np.ndarray | None = None
        metrics_frame: dict[str, float] | None = None

        if self._face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._face_mesh.process(rgb)
            if result.multi_face_landmarks:
                face_landmarks = result.multi_face_landmarks[0]
                full_landmarks = np.array(
                    [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
                    dtype=np.float32,
                )
                metrics_frame = lip_extract(full_landmarks)
                self._draw_overlay(frame, full_landmarks)

        now = time.perf_counter()
        with self._state_lock:
            recording = self.is_recording
            start_ts = self._recording_start_ts
            start_epoch = self._recording_start_epoch
        if recording and start_ts is not None:
            if start_epoch is not None:
                rel_time = frame_abs - start_epoch
            else:
                rel_time = now - start_ts
            if self._lip_first_frame_epoch is None:
                self._lip_first_frame_epoch = frame_abs
            self._frame_times.append(rel_time)
            self._frame_times_abs.append(frame_abs)
            if full_landmarks is None:
                self._landmarks_full.append(np.full((478, 2), np.nan, dtype=np.float32))
            else:
                self._landmarks_full.append(full_landmarks.astype(np.float32))
            if metrics_frame is None:
                for key in self._metrics:
                    self._metrics[key].append(float("nan"))
            else:
                for key in self._metrics:
                    self._metrics[key].append(float(metrics_frame[key]))

        self._fps_frame_counter += 1
        if now - self._fps_last_ts >= 1.0:
            fps = self._fps_frame_counter / (now - self._fps_last_ts)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._fps_frame_counter = 0
            self._fps_last_ts = now

        self._draw_metrics_overlay(
            frame=frame,
            metrics=metrics_frame,
            relative_time=(now - start_ts) if start_ts is not None else None,
        )

        if self.is_raw_recording and self._raw_video_writer is not None:
            self._raw_video_writer.write(raw_frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb_frame.data,
            rgb_frame.shape[1],
            rgb_frame.shape[0],
            rgb_frame.strides[0],
            QImage.Format.Format_RGB888,
        )
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _draw_overlay(self, frame: np.ndarray, all_points: np.ndarray) -> None:
        color = (0, 255, 0)
        h, w, _ = frame.shape
        for start_idx, end_idx in self._mesh_connections:
            if start_idx >= len(all_points) or end_idx >= len(all_points):
                continue
            p1 = (int(all_points[start_idx][0]), int(all_points[start_idx][1]))
            p2 = (int(all_points[end_idx][0]), int(all_points[end_idx][1]))
            if self._point_in_bounds(p1, w, h) and self._point_in_bounds(p2, w, h):
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        for point in all_points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1, cv2.LINE_AA)

    def _draw_metrics_overlay(
        self,
        frame: np.ndarray,
        metrics: dict[str, float] | None,
        relative_time: float | None,
    ) -> None:
        color = (0, 255, 0)
        if metrics is None:
            lines = ["No face detected"]
        else:
            lines = [
                (
                    "A:{:.4f} H:{:.3f} W:{:.3f} O:{:.3f}".format(
                        metrics["area"],
                        metrics["height"],
                        metrics["outer_width"],
                        metrics["open"],
                    )
                ),
                (
                    "IW:{:.3f} TW:{:.3f} C:{:.3f}".format(
                        metrics["inner_width"],
                        metrics["total_width"],
                        metrics["circularity"],
                    )
                ),
                (
                    "FW:{:.1f}px FH:{:.1f}px".format(
                        metrics["face_width"],
                        metrics["face_height"],
                    )
                ),
            ]
        if relative_time is not None:
            lines.append(f"t={relative_time:.2f}s")

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        x0, y0 = 8, 18
        line_h = 16
        max_width = 0
        for text in lines:
            text_w, _ = cv2.getTextSize(text, font, scale, thickness)[0]
            max_width = max(max_width, text_w)
        panel_h = line_h * len(lines) + 8
        cv2.rectangle(frame, (x0 - 4, y0 - 14), (x0 + max_width + 6, y0 - 14 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0 - 4, y0 - 14), (x0 + max_width + 6, y0 - 14 + panel_h), color, 1)
        for idx, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x0, y0 + idx * line_h),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _open_help(self) -> None:
        help_path = get_resource_path(r"Phonetic_Export\index.html")
        if not help_path or not Path(help_path).exists():
            self.status_label.setText("帮助文件不存在")
            return
        url = QUrl.fromLocalFile(help_path)
        url.setFragment("s1765795202719")
        opened = QDesktopServices.openUrl(url)
        if not opened:
            self.status_label.setText("帮助页面打开失败")

    def _open_animation_player(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择唇形数据文件夹", self.save_directory)
        if not folder:
            return
        bundle = self._load_recording_bundle(Path(folder))
        if bundle is None:
            QMessageBox.warning(
                self,
                "唇形动画回放",
                "未找到可用数据，请选择包含 audio_recording.pkl 与 audio_recording.wav 的文件夹。",
            )
            return
        dialog = LipAnimationDialog(bundle=bundle, mesh_connections=self._mesh_connections, parent=self)
        dialog.setWindowIcon(self.windowIcon())
        dialog.exec()

    def upload_video(self) -> None:
        with self._state_lock:
            if self.is_recording:
                QMessageBox.warning(self, "上传视频", "录制进行中，请先停止录制。")
                return
        live_timer_was_active = self._video_timer.isActive()
        if live_timer_was_active:
            self._video_timer.stop()
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            self.save_directory,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.m4v)",
        )
        if not video_path:
            if live_timer_was_active:
                self._video_timer.start()
            return
        ffmpeg = resolve_ffmpeg_executable()
        if not ffmpeg:
            QMessageBox.warning(self, "上传视频", "未检测到 ffmpeg，无法保留视频音频。")
            if live_timer_was_active:
                self._video_timer.start()
            return
        audio_stream_start_offset = probe_audio_stream_start_offset(video_path, ffmpeg)

        output_root = Path(self.save_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        session_name = f"lip_tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}_video"
        session_dir = output_root / session_name
        suffix = 1
        while session_dir.exists():
            session_dir = output_root / f"{session_name}_{suffix}"
            suffix += 1
        session_dir.mkdir(parents=True, exist_ok=True)

        wav_path = session_dir / "audio_recording.wav"
        audio_extract_cmd = [
            ffmpeg,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "1",
            str(wav_path),
        ]
        extract_result = subprocess.run(audio_extract_cmd, capture_output=True, text=True)
        if extract_result.returncode != 0:
            shutil.rmtree(session_dir, ignore_errors=True)
            QMessageBox.warning(self, "上传视频", f"提取音频失败：\n{extract_result.stderr[:400]}")
            if live_timer_was_active:
                self._video_timer.start()
            return

        if audio_stream_start_offset > 1e-9:
            try:
                with wave.open(str(wav_path), "rb") as wav_in:
                    channels = wav_in.getnchannels()
                    sampwidth = wav_in.getsampwidth()
                    sample_rate = wav_in.getframerate()
                    raw = wav_in.readframes(wav_in.getnframes())
                if channels == 1 and sampwidth == 2:
                    audio_samples = np.frombuffer(raw, dtype=np.int16)
                    silence_count = int(round(audio_stream_start_offset * sample_rate))
                    if silence_count > 0:
                        padded = np.concatenate(
                            [np.zeros(silence_count, dtype=np.int16), audio_samples]
                        )
                        with wave.open(str(wav_path), "wb") as wav_out:
                            wav_out.setnchannels(1)
                            wav_out.setsampwidth(2)
                            wav_out.setframerate(sample_rate)
                            wav_out.writeframes(padded.tobytes())
            except Exception:
                pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            shutil.rmtree(session_dir, ignore_errors=True)
            QMessageBox.warning(self, "上传视频", "无法读取视频文件。")
            if live_timer_was_active:
                self._video_timer.start()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        has_total = total_frames > 0

        progress = QProgressDialog("正在识别视频帧...", "取消", 0, max(total_frames, 1), self)
        progress.setWindowTitle("上传视频并识别")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        if not has_total:
            progress.setRange(0, 0)
        progress.show()

        metrics: dict[str, list[float]] = {
            "area": [],
            "face_width": [],
            "face_height": [],
            "height_px": [],
            "outer_width_px": [],
            "inner_width_px": [],
            "total_width_px": [],
            "open_px": [],
            "length": [],
            "height": [],
            "outer_width": [],
            "inner_width": [],
            "total_width": [],
            "open": [],
            "circularity": [],
        }
        frame_times: list[float] = []
        frame_times_abs: list[float] = []
        landmarks_full: list[np.ndarray] = []
        start_epoch = 0.0
        frame_index = 0
        canceled = False
        last_valid_landmarks: np.ndarray | None = None
        last_valid_metrics: dict[str, float] | None = None
        detected_frame_count = 0
        filled_frame_count = 0
        nan_frame_count = 0

        try:
            while True:
                if progress.wasCanceled():
                    canceled = True
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                rel_time = frame_index / fps
                abs_time = start_epoch + rel_time
                frame_times.append(rel_time)
                frame_times_abs.append(abs_time)

                h, w, _ = frame.shape
                full_landmarks: np.ndarray | None = None
                metrics_frame: dict[str, float] | None = None
                if self._face_mesh is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self._face_mesh.process(rgb)
                    if result.multi_face_landmarks:
                        face_landmarks = result.multi_face_landmarks[0]
                        full_landmarks = np.array(
                            [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
                            dtype=np.float32,
                        )
                        metrics_frame = lip_extract(full_landmarks)
                        last_valid_landmarks = full_landmarks.astype(np.float32)
                        last_valid_metrics = {k: float(metrics_frame[k]) for k in metrics}
                        detected_frame_count += 1

                if full_landmarks is None:
                    if last_valid_landmarks is not None:
                        landmarks_full.append(last_valid_landmarks.copy())
                        filled_frame_count += 1
                    else:
                        landmarks_full.append(np.full((478, 2), np.nan, dtype=np.float32))
                        nan_frame_count += 1
                else:
                    landmarks_full.append(full_landmarks.astype(np.float32))

                if metrics_frame is None:
                    if last_valid_metrics is not None:
                        for key in metrics:
                            metrics[key].append(last_valid_metrics[key])
                    else:
                        for key in metrics:
                            metrics[key].append(float("nan"))
                else:
                    for key in metrics:
                        metrics[key].append(float(metrics_frame[key]))

                frame_index += 1
                if has_total:
                    progress.setValue(frame_index)
                    progress.setLabelText(f"正在识别视频帧... {frame_index}/{total_frames}")
                else:
                    progress.setLabelText(f"正在识别视频帧... {frame_index}")
                QApplication.processEvents()
        finally:
            cap.release()
            progress.close()
            if live_timer_was_active:
                self._video_timer.start()

        if canceled or frame_index == 0:
            shutil.rmtree(session_dir, ignore_errors=True)
            if canceled:
                self.status_label.setText("已取消视频识别")
            else:
                QMessageBox.warning(self, "上传视频", "视频中未读取到有效帧。")
            return

        if np.isnan(np.array(metrics["open"], dtype=np.float64)).all():
            shutil.rmtree(session_dir, ignore_errors=True)
            QMessageBox.warning(self, "上传视频", "整段视频未检测到有效人脸，无法生成唇形参数。")
            return

        def _fill_leading_nans(values: list[float]) -> list[float]:
            arr = np.array(values, dtype=np.float64)
            valid = np.where(np.isfinite(arr))[0]
            if valid.size == 0:
                return values
            first = int(valid[0])
            if first > 0:
                arr[:first] = arr[first]
            return arr.tolist()

        for key in metrics:
            metrics[key] = _fill_leading_nans(metrics[key])

        if landmarks_full:
            first_valid_idx = None
            for idx, lm in enumerate(landmarks_full):
                if np.isfinite(lm).all():
                    first_valid_idx = idx
                    break
            if first_valid_idx is not None and first_valid_idx > 0:
                for idx in range(first_valid_idx):
                    landmarks_full[idx] = landmarks_full[first_valid_idx].copy()

        sample_rate = 44100
        chunk_size = 1024
        audio_samples_for_offset = np.array([], dtype=np.float64)
        with wave.open(str(wav_path), "rb") as wav_in:
            sample_rate = wav_in.getframerate()
            total_samples = wav_in.getnframes()
            raw_audio_bytes = wav_in.readframes(total_samples)
            if wav_in.getnchannels() == 1 and wav_in.getsampwidth() == 2 and total_samples > 0:
                audio_samples_for_offset = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float64) / 32768.0
        chunk_count = int(np.ceil(total_samples / chunk_size)) if total_samples > 0 else 0
        audio_frame_timestamps = [
            start_epoch + (chunk_idx * chunk_size / sample_rate)
            for chunk_idx in range(chunk_count)
        ]
        timestamps_payload = {
            "start_time": start_epoch,
            "frame_timestamps": audio_frame_timestamps,
            "sample_rate": sample_rate,
            "chunk_size": chunk_size,
        }
        with open(session_dir / "audio_recording_timestamps.pkl", "wb") as handle:
            pickle.dump(timestamps_payload, handle)

        auto_lip_offset = 0.0
        try:
            auto_lip_offset = estimate_lip_audio_offset_seconds(
                audio_samples=audio_samples_for_offset,
                sample_rate=sample_rate,
                lip_times=np.array(frame_times, dtype=np.float64),
                lip_open=np.array(metrics["open"], dtype=np.float64),
                search_seconds=2.0,
            )
        except Exception:
            auto_lip_offset = 0.0
        auto_lip_offset = float(np.clip(auto_lip_offset, -2.0, 2.0))

        duration = frame_times[-1] if frame_times else None
        data_payload = {
            "absolute_timestamps": frame_times_abs,
            "relative_times": frame_times,
            "area": metrics["area"],
            "face_width": metrics["face_width"],
            "face_height": metrics["face_height"],
            "height_px": metrics["height_px"],
            "outer_width_px": metrics["outer_width_px"],
            "inner_width_px": metrics["inner_width_px"],
            "total_width_px": metrics["total_width_px"],
            "open_px": metrics["open_px"],
            "length": metrics["length"],
            "height": metrics["height"],
            "outer_width": metrics["outer_width"],
            "inner_width": metrics["inner_width"],
            "total_width": metrics["total_width"],
            "open": metrics["open"],
            "circularity": metrics["circularity"],
            "landmarks": landmarks_full,
            "metadata": {
                "recording_start_time": frame_times_abs[0] if frame_times_abs else None,
                "lip_first_frame_time": frame_times_abs[0] if frame_times_abs else None,
                "audio_first_frame_time": start_epoch,
                "lip_manual_offset": auto_lip_offset,
                "recording_duration": duration,
                "fps": fps,
                "source": "uploaded_video",
                "video_time_base_mode": "frame_index_over_fps",
                "audio_stream_start_offset": audio_stream_start_offset,
                "auto_lip_offset_estimate": auto_lip_offset,
                "time_alignment_mode": "anchored_audio_start",
                "video_total_frames": frame_index,
                "video_detected_frames": detected_frame_count,
                "video_filled_frames": filled_frame_count,
                "video_nan_frames": nan_frame_count,
                "created_at": datetime.now().isoformat(),
            },
        }
        with open(session_dir / "audio_recording.pkl", "wb") as handle:
            pickle.dump(data_payload, handle)

        valid_ratio = (detected_frame_count / frame_index * 100.0) if frame_index > 0 else 0.0
        self.status_label.setText(
            f"视频识别完成: 总帧{frame_index} 检测{detected_frame_count} 补全{filled_frame_count}"
        )
        QMessageBox.information(
            self,
            "上传视频",
            (
                f"唇形与音频数据已保存到:\n{session_dir}\n\n"
                f"总帧数: {frame_index}\n"
                f"检测成功帧: {detected_frame_count} ({valid_ratio:.1f}%)\n"
                f"补全帧: {filled_frame_count}\n"
                f"无可补全帧: {nan_frame_count}\n"
                f"自动唇形offset: {auto_lip_offset:+.3f}s"
            ),
        )

    def start_raw_recording(self) -> None:
        if self.is_raw_recording:
            return
        if self._video_capture is None or not self._video_capture.isOpened():
            QMessageBox.warning(self, "开始录制", "摄像头未就绪。")
            return

        output_root = Path(self.save_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        session_name = f"raw_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = output_root / session_name
        suffix = 1
        while session_dir.exists():
            session_dir = output_root / f"{session_name}_{suffix}"
            suffix += 1
        session_dir.mkdir(parents=True, exist_ok=True)

        width = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._video_capture.get(cv2.CAP_PROP_FPS))
        if width <= 0:
            width = 640
        if height <= 0:
            height = 480
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        video_temp_path = session_dir / "raw_video_only.mp4"
        writer = cv2.VideoWriter(
            str(video_temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            QMessageBox.warning(self, "开始录制", "视频写入器初始化失败。")
            shutil.rmtree(session_dir, ignore_errors=True)
            return

        self._raw_audio_chunks = []
        sample_rate = self._raw_recording_sample_rate

        def _raw_audio_callback(indata, _frames, _time_info, _status):
            if not self.is_raw_recording:
                return
            self._raw_audio_chunks.append(indata.copy())

        if self._live_audio_stream is not None and self._live_audio_stream_active:
            try:
                self._live_audio_stream.stop()
            except Exception:
                pass
            self._live_audio_stream_active = False

        try:
            audio_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=1024,
                callback=_raw_audio_callback,
            )
            audio_stream.start()
        except Exception as exc:
            writer.release()
            shutil.rmtree(session_dir, ignore_errors=True)
            QMessageBox.warning(self, "开始录制", f"音频流初始化失败:\n{exc}")
            return

        self.is_raw_recording = True
        self._raw_video_writer = writer
        self._raw_audio_stream = audio_stream
        self._raw_recording_session_dir = session_dir
        self._raw_recording_video_temp_path = video_temp_path
        self._raw_recording_audio_path = session_dir / "raw_audio.wav"
        self._raw_recording_output_path = session_dir / "raw_recording.mp4"
        self.raw_start_btn.setEnabled(False)
        self.raw_stop_btn.setEnabled(True)
        self.status_label.setText("正在录制原始视频...")

    def stop_raw_recording(self) -> None:
        if not self.is_raw_recording:
            return
        self.is_raw_recording = False

        if self._raw_audio_stream is not None:
            try:
                self._raw_audio_stream.stop()
                self._raw_audio_stream.close()
            finally:
                self._raw_audio_stream = None

        if self._raw_video_writer is not None:
            self._raw_video_writer.release()
            self._raw_video_writer = None

        session_dir = self._raw_recording_session_dir
        video_temp_path = self._raw_recording_video_temp_path
        audio_path = self._raw_recording_audio_path
        output_path = self._raw_recording_output_path
        self.raw_start_btn.setEnabled(True)
        self.raw_stop_btn.setEnabled(False)

        if session_dir is None or video_temp_path is None or audio_path is None or output_path is None:
            self.status_label.setText("录制结束")
            return

        audio_array = (
            np.concatenate(self._raw_audio_chunks, axis=0).reshape(-1).astype(np.int16)
            if self._raw_audio_chunks
            else np.array([], dtype=np.int16)
        )
        with wave.open(str(audio_path), "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(self._raw_recording_sample_rate)
            wav_out.writeframes(audio_array.tobytes())

        ffmpeg = resolve_ffmpeg_executable()
        merged_ok = False
        if ffmpeg and video_temp_path.exists() and audio_path.exists():
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(video_temp_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            merged_ok = result.returncode == 0 and output_path.exists()

        if merged_ok:
            self.status_label.setText(f"录制完成: {output_path}")
            QMessageBox.information(self, "停止录制", f"原始录制已保存:\n{output_path}")
        else:
            self.status_label.setText(f"录制完成(未封装): {session_dir}")
            QMessageBox.warning(
                self,
                "停止录制",
                (
                    "已保存原始视频和音频，但未能合成 MP4。\n"
                    f"目录: {session_dir}"
                ),
            )

        self._raw_audio_chunks = []
        self._raw_recording_session_dir = None
        self._raw_recording_video_temp_path = None
        self._raw_recording_audio_path = None
        self._raw_recording_output_path = None

        if self._live_audio_stream is not None and not self._live_audio_stream_active:
            try:
                self._live_audio_stream.start()
                self._live_audio_stream_active = True
            except Exception:
                self._live_audio_stream_active = False

    @staticmethod
    def _load_recording_bundle(folder: Path) -> dict | None:
        data_path = folder / "audio_recording.pkl"
        wav_path = folder / "audio_recording.wav"
        if not data_path.exists() or not wav_path.exists():
            return None
        try:
            with open(data_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        landmarks = payload.get("landmarks")
        rel_times = payload.get("relative_times")
        if landmarks is None or rel_times is None:
            return None
        landmarks_np: list[np.ndarray] = []
        for lm in landmarks:
            arr = np.array(lm, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            landmarks_np.append(arr)
        if not landmarks_np:
            return None
        times = np.array(rel_times, dtype=np.float64)
        if times.size == 0:
            return None
        if len(landmarks_np) != times.size:
            n = min(len(landmarks_np), int(times.size))
            landmarks_np = landmarks_np[:n]
            times = times[:n]
        return {
            "folder": folder,
            "wav_path": wav_path,
            "times": times,
            "landmarks": landmarks_np,
        }

    @staticmethod
    def _point_in_bounds(point: tuple[int, int], width: int, height: int) -> bool:
        return 0 <= point[0] < width and 0 <= point[1] < height

    def start_recording(self) -> None:
        with self._state_lock:
            if self.is_recording:
                return
            self.is_recording = True
            self._recording_start_ts = time.perf_counter()
            self._recording_start_epoch = time.time()

        self._frame_times.clear()
        self._frame_times_abs.clear()
        self._landmarks_full.clear()
        self._audio_chunks.clear()
        self._audio_chunk_times.clear()
        self._audio_chunk_times_abs.clear()
        self._audio_first_chunk_epoch = None
        self._audio_adc_to_wall_offset = None
        self._lip_first_frame_epoch = None
        for key in self._metrics:
            self._metrics[key].clear()
        if self._live_audio_stream is None or not self._live_audio_stream_active:
            self._setup_live_audio_stream()

        self.path_btn.setEnabled(False)
        self.start_btn.setText("参数计算中…")
        self.start_btn.setStyleSheet("background-color: #d35454; color: white; font-weight: bold;")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("实时计算中")

    def stop_recording(self) -> None:
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False

        if self._audio_chunks:
            audio_array = np.concatenate(self._audio_chunks, axis=0).reshape(-1).astype(np.int16)
        else:
            audio_array = np.array([], dtype=np.int16)

        lip_offset_seconds = 0.0
        if audio_array.size > 0 and len(self._frame_times) > 2 and len(self._metrics["open"]) > 2:
            offset_dialog = LipOffsetAdjustDialog(
                audio_array=audio_array,
                sample_rate=44100,
                lip_times=np.array(self._frame_times, dtype=np.float64),
                lip_metrics={k: np.array(v, dtype=np.float64) for k, v in self._metrics.items()},
                is_dark=self.is_dark,
                parent=self,
            )
            offset_dialog.setWindowIcon(self.windowIcon())
            result = offset_dialog.exec()
            if result == QDialog.DialogCode.Rejected:
                self.path_btn.setEnabled(True)
                self.start_btn.setText("实时参数计算")
                self.start_btn.setStyleSheet("")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status_label.setText("已取消保存")
                return
            lip_offset_seconds = offset_dialog.selected_offset_seconds

        saved_path = self._save_recording_files(
            audio_array=audio_array,
            lip_offset_seconds=lip_offset_seconds,
        )

        self.path_btn.setEnabled(True)
        self.start_btn.setText("实时参数计算")
        self.start_btn.setStyleSheet("")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"已保存: {saved_path}")

    def _save_recording_files(
        self,
        audio_array: np.ndarray | None = None,
        lip_offset_seconds: float = 0.0,
    ) -> str:
        output_root = Path(self.save_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        session_dir = output_root / f"lip_tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(parents=True, exist_ok=True)

        if audio_array is None:
            if self._audio_chunks:
                audio_array = np.concatenate(self._audio_chunks, axis=0).reshape(-1).astype(np.int16)
            else:
                audio_array = np.array([], dtype=np.int16)

        sample_rate = 44100
        chunk_size = 1024
        audio_start_epoch = self._recording_start_epoch
        if audio_start_epoch is None:
            audio_start_epoch = self._audio_first_chunk_epoch
        if audio_start_epoch is None:
            audio_start_epoch = time.time()

        wav_path = session_dir / "audio_recording.wav"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())

        start_epoch = audio_start_epoch
        if self._audio_chunk_times_abs:
            audio_frame_timestamps = list(self._audio_chunk_times_abs)
            start_epoch = audio_frame_timestamps[0]
        else:
            audio_frame_timestamps = [start_epoch + rel_t for rel_t in self._audio_chunk_times]
        timestamps_payload = {
            "start_time": start_epoch,
            "frame_timestamps": audio_frame_timestamps,
            "sample_rate": sample_rate,
            "chunk_size": chunk_size,
        }
        timestamps_path = session_dir / "audio_recording_timestamps.pkl"
        with open(timestamps_path, "wb") as handle:
            pickle.dump(timestamps_payload, handle)

        if self._landmarks_full:
            landmarks_arr = np.stack(self._landmarks_full).astype(np.float32)
            landmarks_arr = self._interpolate_short_landmark_gaps(landmarks_arr, max_gap=2)
            landmarks_data: list[np.ndarray] = [frame_landmarks.copy() for frame_landmarks in landmarks_arr]
        else:
            landmarks_data = []

        metrics_local: dict[str, list[float]] = {}
        for key in self._metrics:
            vals = np.array(self._metrics[key], dtype=np.float64)
            vals = self._interpolate_short_numeric_gaps(vals, max_gap=2)
            metrics_local[key] = vals.tolist()

        frame_times_abs = np.array(self._frame_times_abs, dtype=np.float64)
        if frame_times_abs.size > 0:
            frame_time_anchor = float(audio_start_epoch)
            frame_times = (frame_times_abs - frame_time_anchor).astype(np.float64)
            if frame_times.size > 0 and frame_times[0] < 0:
                frame_times = frame_times - frame_times[0]
        else:
            frame_times = np.array(self._frame_times, dtype=np.float64)

        duration = float(frame_times[-1]) if frame_times.size > 0 else None
        data_payload = {
            "absolute_timestamps": list(frame_times + float(audio_start_epoch)),
            "relative_times": list(frame_times),
            "area": metrics_local["area"],
            "face_width": metrics_local["face_width"],
            "face_height": metrics_local["face_height"],
            "height_px": metrics_local["height_px"],
            "outer_width_px": metrics_local["outer_width_px"],
            "inner_width_px": metrics_local["inner_width_px"],
            "total_width_px": metrics_local["total_width_px"],
            "open_px": metrics_local["open_px"],
            "length": metrics_local["length"],
            "height": metrics_local["height"],
            "outer_width": metrics_local["outer_width"],
            "inner_width": metrics_local["inner_width"],
            "total_width": metrics_local["total_width"],
            "open": metrics_local["open"],
            "circularity": metrics_local["circularity"],
            "landmarks": landmarks_data,
            "metadata": {
                "recording_start_time": float(audio_start_epoch),
                "lip_first_frame_time": float(audio_start_epoch) if frame_times.size > 0 else None,
                "audio_first_frame_time": self._audio_first_chunk_epoch,
                "lip_manual_offset": float(lip_offset_seconds),
                "time_alignment_mode": "anchored_audio_start",
                "recording_duration": duration,
                "fps": (len(frame_times) / duration) if duration and duration > 0 else None,
                "created_at": datetime.now().isoformat(),
            },
        }
        data_path = session_dir / "audio_recording.pkl"
        with open(data_path, "wb") as handle:
            pickle.dump(data_payload, handle)

        return str(session_dir)

    @staticmethod
    def _interpolate_short_numeric_gaps(arr: np.ndarray, max_gap: int = 2) -> np.ndarray:
        x = np.array(arr, dtype=np.float64)
        n = len(x)
        i = 0
        while i < n:
            if np.isfinite(x[i]):
                i += 1
                continue
            s = i
            while i < n and not np.isfinite(x[i]):
                i += 1
            e = i - 1
            gap_len = e - s + 1
            left = s - 1
            right = i
            if gap_len <= max_gap and left >= 0 and right < n and np.isfinite(x[left]) and np.isfinite(x[right]):
                for k in range(gap_len):
                    alpha = (k + 1) / (gap_len + 1)
                    x[s + k] = (1.0 - alpha) * x[left] + alpha * x[right]
        return x

    @staticmethod
    def _interpolate_short_landmark_gaps(arr: np.ndarray, max_gap: int = 2) -> np.ndarray:
        x = np.array(arr, dtype=np.float32)
        valid = np.isfinite(x).all(axis=(1, 2))
        n = x.shape[0]
        i = 0
        while i < n:
            if valid[i]:
                i += 1
                continue
            s = i
            while i < n and not valid[i]:
                i += 1
            e = i - 1
            gap_len = e - s + 1
            left = s - 1
            right = i
            if gap_len <= max_gap and left >= 0 and right < n and valid[left] and valid[right]:
                for k in range(gap_len):
                    alpha = (k + 1) / (gap_len + 1)
                    x[s + k] = (1.0 - alpha) * x[left] + alpha * x[right]
        return x

    def closeEvent(self, event: QCloseEvent) -> None:
        with self._state_lock:
            recording = self.is_recording
        if recording:
            self.stop_recording()
        if self.is_raw_recording:
            self.stop_raw_recording()

        if self._video_timer.isActive():
            self._video_timer.stop()
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._live_audio_stream is not None:
            try:
                self._live_audio_stream.stop()
                self._live_audio_stream.close()
            finally:
                self._live_audio_stream = None
                self._live_audio_stream_active = False
        super().closeEvent(event)


class LipOffsetAdjustDialog(QDialog):
    def __init__(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        lip_times: np.ndarray,
        lip_metrics: dict[str, np.ndarray],
        is_dark: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("唇形/音频偏移校正")
        self.resize(1180, 820)
        self._is_dark = bool(is_dark)

        self._audio = np.array(audio_array, dtype=np.float64)
        if self._audio.size > 0:
            self._audio /= 32768.0
        self._sample_rate = int(sample_rate)
        raw_times = np.array(lip_times, dtype=np.float64)
        self._lip_times, keep_idx = self._clean_time_axis(raw_times)
        self._lip_metrics = {}
        for key, values in lip_metrics.items():
            arr = np.array(values, dtype=np.float64)
            if arr.size == 0 or self._lip_times.size == 0:
                self._lip_metrics[key] = np.array([], dtype=np.float64)
                continue
            n = min(arr.size, raw_times.size)
            if n <= 0:
                self._lip_metrics[key] = np.array([], dtype=np.float64)
                continue
            arr = arr[:n]
            local_times = raw_times[:n]
            local_keep = keep_idx[keep_idx < n]
            if local_keep.size == 0:
                self._lip_metrics[key] = np.array([], dtype=np.float64)
            else:
                order = np.argsort(local_times[local_keep])
                self._lip_metrics[key] = arr[local_keep][order]
        self._metric_key = "open" if "open" in self._lip_metrics else next(iter(self._lip_metrics.keys()))
        self._estimated_offset = self._estimate_offset_seconds()
        self.selected_offset_seconds = 0.0
        self._pan_active = False
        self._press_x = None

        self._build_ui()
        self._apply_initial_view()
        self._redraw()

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.lbl_est = QLabel(f"估算 offset: {self._estimated_offset:+.3f} s（搜索范围 ±2.0 s）")
        root.addWidget(self.lbl_est)
        self.lbl_limit = QLabel("提示：当前页面只显示前60秒的音频，用于唇形和音频信号的同步矫正。")
        root.addWidget(self.lbl_limit)

        self.figure = Figure()
        self.canvas = Canvas(self.figure)
        self.ax_audio = self.figure.add_subplot(211)
        self.ax_lip = self.figure.add_subplot(212, sharex=self.ax_audio)
        root.addWidget(self.canvas, 1)

        ctl = QHBoxLayout()
        self.param_combo = QComboBox()
        metric_labels = {
            "open": "唇开度",
            "area": "唇面积",
            "outer_width": "唇宽度",
            "inner_width": "内唇宽",
            "total_width": "总宽",
            "height": "唇高度",
            "circularity": "唇圆度",
        }
        for key in self._lip_metrics.keys():
            self.param_combo.addItem(metric_labels.get(key, key), key)
        idx = self.param_combo.findData(self._metric_key)
        if idx >= 0:
            self.param_combo.setCurrentIndex(idx)

        self.manual_check = QCheckBox("手动设置 offset")
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-2.0, 2.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setSuffix(" s")
        self.offset_spin.setValue(self._estimated_offset)
        self.offset_spin.setEnabled(False)

        self.btn_apply = QPushButton("应用并保存")
        self.btn_no = QPushButton("不应用直接保存")
        self.btn_cancel = QPushButton("取消")

        ctl.addWidget(QLabel("唇形参数"))
        ctl.addWidget(self.param_combo)
        ctl.addSpacing(12)
        ctl.addWidget(self.manual_check)
        ctl.addWidget(self.offset_spin)
        ctl.addStretch()
        ctl.addWidget(self.btn_apply)
        ctl.addWidget(self.btn_no)
        ctl.addWidget(self.btn_cancel)
        root.addLayout(ctl)

        self.param_combo.currentIndexChanged.connect(self._on_param_changed)
        self.manual_check.toggled.connect(self._on_manual_toggled)
        self.offset_spin.valueChanged.connect(self._redraw)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_no.clicked.connect(self._on_skip)
        self.btn_cancel.clicked.connect(self.reject)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._apply_theme()

    def _apply_theme(self):
        if self._is_dark:
            self.setStyleSheet("QDialog { background-color: #17191c; color: #f0f0f0; }")
            self.figure.set_facecolor("#17191c")
        else:
            self.setStyleSheet("")
            self.figure.set_facecolor("#ffffff")

    @staticmethod
    def _clean_time_axis(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if times.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
        finite_idx = np.where(np.isfinite(times))[0]
        if finite_idx.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
        finite_times = times[finite_idx]
        order = np.argsort(finite_times)
        sorted_idx = finite_idx[order]
        sorted_times = times[sorted_idx]
        keep = np.ones(sorted_times.size, dtype=bool)
        if sorted_times.size > 1:
            keep[1:] = np.diff(sorted_times) > 1e-9
        return sorted_times[keep], sorted_idx[keep]

    def _apply_initial_view(self):
        audio_dur = len(self._audio) / self._sample_rate if self._sample_rate > 0 else 0.0
        lip_dur = float(self._lip_times[-1]) if self._lip_times.size > 0 else 0.0
        total = max(audio_dur, lip_dur)
        right = total if total > 0 else 1.0
        self.ax_audio.set_xlim(0.0, right)

    def _on_param_changed(self):
        key = self.param_combo.currentData()
        if isinstance(key, str):
            self._metric_key = key
        self._redraw()

    def _on_manual_toggled(self, checked: bool):
        self.offset_spin.setEnabled(checked)
        if not checked:
            self.offset_spin.setValue(self._estimated_offset)
        self._redraw()

    def _current_offset(self) -> float:
        if self.manual_check.isChecked():
            return float(self.offset_spin.value())
        return float(self._estimated_offset)

    def _on_apply(self):
        self.selected_offset_seconds = self._current_offset()
        self.accept()

    def _on_skip(self):
        self.selected_offset_seconds = 0.0
        self.accept()

    def _on_scroll(self, event):
        if event.inaxes not in (self.ax_audio, self.ax_lip):
            return
        xlim = self.ax_audio.get_xlim()
        center = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) * 0.5
        width = xlim[1] - xlim[0]
        factor = 0.8 if event.button == "up" else 1.25
        new_w = max(0.02, width * factor)
        left = center - (center - xlim[0]) * (new_w / width)
        right = left + new_w
        left = max(0.0, left)
        self.ax_audio.set_xlim(left, right)
        self._redraw()

    def _on_press(self, event):
        if event.inaxes not in (self.ax_audio, self.ax_lip):
            return
        if event.button != 1:
            return
        self._pan_active = True
        self._press_x = event.xdata

    def _on_release(self, _event):
        self._pan_active = False
        self._press_x = None

    def _on_motion(self, event):
        if not self._pan_active or self._press_x is None:
            return
        if event.inaxes not in (self.ax_audio, self.ax_lip):
            return
        if event.xdata is None:
            return
        dx = event.xdata - self._press_x
        x0, x1 = self.ax_audio.get_xlim()
        width = x1 - x0
        new_x0 = max(0.0, x0 - dx)
        new_x1 = new_x0 + width
        self.ax_audio.set_xlim(new_x0, new_x1)
        self._press_x = event.xdata
        self._redraw()

    @staticmethod
    def _limit_visible_audio_points(x: np.ndarray, y: np.ndarray, x0: float, x1: float) -> tuple[np.ndarray, np.ndarray]:
        mask = (x >= x0) & (x <= x1)
        if not np.any(mask):
            return np.array([]), np.array([])
        xv = x[mask]
        yv = y[mask]
        if len(xv) > 1:
            xv = xv[::2]
            yv = yv[::2]
        while len(xv) > 9999:
            xv = xv[::2]
            yv = yv[::2]
        return xv, yv

    def _audio_envelope(self) -> tuple[np.ndarray, np.ndarray]:
        if self._audio.size == 0:
            return np.array([]), np.array([])
        y = np.abs(self._audio)
        win = max(1, int(round(0.01 * self._sample_rate)))
        kernel = np.ones(win, dtype=np.float64) / float(win)
        env = np.convolve(y, kernel, mode="same")
        t = np.arange(len(env), dtype=np.float64) / float(self._sample_rate)
        return t, env

    def _estimate_offset_seconds(self) -> float:
        if self._audio.size < 8 or self._lip_times.size < 8 or "open" not in self._lip_metrics:
            return 0.0
        audio_t, audio_env = self._audio_envelope()
        lip_y = np.array(self._lip_metrics["open"], dtype=np.float64)
        if lip_y.size != self._lip_times.size:
            n = min(lip_y.size, self._lip_times.size)
            lip_y = lip_y[:n]
            lip_t = self._lip_times[:n]
        else:
            lip_t = self._lip_times
        if lip_t.size < 8:
            return 0.0

        audio_mask = audio_t <= 60.0
        if np.any(audio_mask):
            audio_t = audio_t[audio_mask]
            audio_env = audio_env[audio_mask]
        lip_mask = lip_t <= 60.0
        if np.any(lip_mask):
            lip_t = lip_t[lip_mask]
            lip_y = lip_y[lip_mask]
        if audio_t.size < 8 or lip_t.size < 8:
            return 0.0

        t0 = max(float(audio_t[0]), float(lip_t[0]) - 2.0)
        t1 = min(float(audio_t[-1]), float(lip_t[-1]) + 2.0)
        if t1 - t0 <= 1.0:
            return 0.0
        grid_fs = 200.0
        grid = np.arange(t0, t1, 1.0 / grid_fs, dtype=np.float64)
        if grid.size < 100:
            return 0.0
        audio_grid = np.interp(grid, audio_t, audio_env)
        audio_grid = (audio_grid - np.mean(audio_grid)) / (np.std(audio_grid) + 1e-12)

        best_offset = 0.0
        best_score = -1e12
        for off in np.arange(-2.0, 2.0001, 0.002):
            shifted = np.interp(grid, lip_t + off, lip_y, left=np.nan, right=np.nan)
            valid = np.isfinite(shifted)
            if np.count_nonzero(valid) < 100:
                continue
            lv = shifted[valid]
            lv = (lv - np.mean(lv)) / (np.std(lv) + 1e-12)
            av = audio_grid[valid]
            score = float(np.mean(av * lv))
            if score > best_score:
                best_score = score
                best_offset = float(off)
        return best_offset

    def _redraw(self):
        x0, x1 = self.ax_audio.get_xlim()
        self.ax_audio.cla()
        self.ax_lip.cla()
        if self._is_dark:
            bg = "#17191c"
            fg = "#e5e5e5"
            grid = 0.18
        else:
            bg = "#ffffff"
            fg = "#222222"
            grid = 0.2
        self.ax_audio.set_facecolor(bg)
        self.ax_lip.set_facecolor(bg)
        self.ax_audio.tick_params(colors=fg)
        self.ax_lip.tick_params(colors=fg)
        for spine in self.ax_audio.spines.values():
            spine.set_color(fg)
        for spine in self.ax_lip.spines.values():
            spine.set_color(fg)
        audio_t = np.arange(self._audio.size, dtype=np.float64) / float(self._sample_rate) if self._sample_rate > 0 else np.array([])
        if audio_t.size > 0:
            xa, ya = self._limit_visible_audio_points(audio_t, self._audio, x0, x1)
            self.ax_audio.plot(xa, ya, color="#00b3ff", linewidth=1.0)
        self.ax_audio.set_ylabel("Amplitude", color=fg)

        key = self._metric_key
        lip_y = np.array(self._lip_metrics.get(key, []), dtype=np.float64)
        lip_t = self._lip_times
        if lip_y.size != lip_t.size:
            n = min(lip_y.size, lip_t.size)
            lip_y = lip_y[:n]
            lip_t = lip_t[:n]
        if lip_t.size > 0:
            mask0 = (lip_t >= x0) & (lip_t <= x1)
            xo = lip_t[mask0]
            yo = lip_y[mask0]
            self.ax_lip.plot(xo, yo, color="#bababa", linewidth=1.0, alpha=0.55, label="原始")

            off = self._current_offset()
            shifted_t = lip_t + off
            mask1 = (shifted_t >= x0) & (shifted_t <= x1)
            xs = shifted_t[mask1]
            ys = lip_y[mask1]
            self.ax_lip.plot(xs, ys, color="#1f77b4", linewidth=1.4, label=f"对齐后 ({off:+.3f}s)")
            self.ax_lip.legend(loc="upper left")
        self.ax_lip.set_xlabel("Time (s)", color=fg)
        self.ax_lip.set_ylabel(self.param_combo.currentText(), color=fg)
        self.ax_lip.grid(True, alpha=grid)
        self.ax_audio.grid(True, alpha=grid)
        self.ax_audio.set_xlim(x0, x1)
        self.figure.set_facecolor(bg)
        self.canvas.draw_idle()


class LipAnimationDialog(QDialog):
    def __init__(self, bundle: dict, mesh_connections: tuple[tuple[int, int], ...], parent=None):
        super().__init__(parent)
        self.setWindowTitle("唇形动画回放")
        self.resize(1100, 760)

        self._mesh_connections = mesh_connections
        self._wav_path: Path = bundle["wav_path"]
        self._times: np.ndarray = bundle["times"]
        self._landmarks: list[np.ndarray] = bundle["landmarks"]
        self._duration_s = float(self._times[-1]) if self._times.size > 0 else 0.0
        if self._times.size >= 2:
            dt = np.diff(self._times)
            dt = dt[dt > 1e-6]
            self._fps = float(1.0 / np.median(dt)) if dt.size > 0 else 30.0
        else:
            self._fps = 30.0

        self._audio_output = QAudioOutput()
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        self._player.setSource(QUrl.fromLocalFile(str(self._wav_path)))
        self._player.positionChanged.connect(self._on_player_position_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(30)
        self._sync_timer.timeout.connect(self._sync_frame_from_player)

        self._render_w, self._render_h = self._infer_render_size()
        self._current_index = 0

        self._build_ui()
        self._show_frame(0)

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid #555; background: black;")
        root.addWidget(self.video_label, 1)

        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, len(self._landmarks) - 1))
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.lbl_time = QLabel("0.00s / 0.00s")
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.lbl_time)
        root.addLayout(slider_row)

        range_row = QHBoxLayout()
        self.spin_start = QDoubleSpinBox()
        self.spin_end = QDoubleSpinBox()
        for spin in (self.spin_start, self.spin_end):
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setRange(0.0, max(0.0, self._duration_s))
            spin.setSuffix(" s")
        self.spin_start.setValue(0.0)
        self.spin_end.setValue(max(0.0, self._duration_s))
        self.spin_start.valueChanged.connect(self._normalize_range)
        self.spin_end.valueChanged.connect(self._normalize_range)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["高清", "标准", "小体积"])

        range_row.addWidget(QLabel("起点"))
        range_row.addWidget(self.spin_start)
        range_row.addWidget(QLabel("终点"))
        range_row.addWidget(self.spin_end)
        range_row.addWidget(QLabel("视频质量"))
        range_row.addWidget(self.quality_combo)
        range_row.addStretch()
        root.addLayout(range_row)

        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_save_video = QPushButton("保存为视频")
        self.btn_save_gif = QPushButton("保存为GIF")
        self.btn_close = QPushButton("关闭")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_save_video.clicked.connect(self._save_video)
        self.btn_save_gif.clicked.connect(self._save_gif)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_save_video)
        btn_row.addWidget(self.btn_save_gif)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    def _infer_render_size(self) -> tuple[int, int]:
        max_x = 640.0
        max_y = 480.0
        for lm in self._landmarks:
            valid = lm[np.isfinite(lm).all(axis=1)]
            if valid.size == 0:
                continue
            max_x = max(max_x, float(np.max(valid[:, 0])) + 20.0)
            max_y = max(max_y, float(np.max(valid[:, 1])) + 20.0)
        w = int(max(320, min(1920, round(max_x))))
        h = int(max(240, min(1080, round(max_y))))
        return w, h

    def _normalize_range(self):
        if self.spin_start.value() > self.spin_end.value():
            self.spin_end.setValue(self.spin_start.value())

    def _time_to_index(self, t: float) -> int:
        pos = int(np.searchsorted(self._times, t))
        if pos <= 0:
            return 0
        if pos >= len(self._times):
            return len(self._times) - 1
        left = pos - 1
        right = pos
        if abs(float(self._times[right]) - t) < abs(t - float(self._times[left])):
            return right
        return left

    def _render_frame(self, idx: int) -> np.ndarray:
        pts = self._landmarks[idx]
        return self._render_points(pts)

    def _render_points(self, pts: np.ndarray) -> np.ndarray:
        canvas = np.zeros((self._render_h, self._render_w, 3), dtype=np.uint8)
        color = (0, 255, 0)
        for s, e in self._mesh_connections:
            if s >= len(pts) or e >= len(pts):
                continue
            p1 = pts[s]
            p2 = pts[e]
            if not np.isfinite(p1).all() or not np.isfinite(p2).all():
                continue
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            if 0 <= x1 < self._render_w and 0 <= y1 < self._render_h and 0 <= x2 < self._render_w and 0 <= y2 < self._render_h:
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        for p in pts:
            if not np.isfinite(p).all():
                continue
            x, y = int(p[0]), int(p[1])
            if 0 <= x < self._render_w and 0 <= y < self._render_h:
                cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)
        return self._crop_to_face_region(canvas, pts)

    def _crop_to_face_region(self, frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
        valid = pts[np.isfinite(pts).all(axis=1)]
        if valid.shape[0] < 3:
            return frame

        min_x = float(np.min(valid[:, 0]))
        max_x = float(np.max(valid[:, 0]))
        min_y = float(np.min(valid[:, 1]))
        max_y = float(np.max(valid[:, 1]))
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        side = int(max(bbox_w, bbox_h) * 1.35)
        side = max(220, min(side, 720))

        cx = int(round((min_x + max_x) * 0.5))
        cy = int(round((min_y + max_y) * 0.5))
        half = side // 2
        x0 = cx - half
        y0 = cy - half
        x1 = x0 + side
        y1 = y0 + side

        out = np.zeros((side, side, 3), dtype=np.uint8)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(frame.shape[1], x1)
        src_y1 = min(frame.shape[0], y1)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return out

        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
        return out

    def _show_frame(self, idx: int):
        if idx < 0 or idx >= len(self._landmarks):
            return
        self._current_index = idx
        frame = self._render_frame(idx)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        cur_t = float(self._times[idx])
        self.lbl_time.setText(f"{cur_t:.2f}s / {self._duration_s:.2f}s")
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

    def _on_slider_changed(self, idx: int):
        if idx < 0 or idx >= len(self._landmarks):
            return
        self._show_frame(idx)
        self._player.setPosition(int(float(self._times[idx]) * 1000.0))

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self._sync_timer.stop()
            return
        start_t = max(self.spin_start.value(), min(self.spin_end.value(), float(self._times[self._current_index])))
        self._player.setPosition(int(start_t * 1000.0))
        self._player.play()
        self._sync_timer.start()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("暂停")
        else:
            self.btn_play.setText("播放")

    def _on_player_position_changed(self, position_ms: int):
        t = position_ms / 1000.0
        idx = self._time_to_index(t)
        self._show_frame(idx)
        if t >= self.spin_end.value():
            self._player.pause()
            self._sync_timer.stop()
            self.btn_play.setText("播放")

    def _sync_frame_from_player(self):
        t = self._player.position() / 1000.0
        idx = self._time_to_index(t)
        self._show_frame(idx)

    def _selected_indices(self) -> list[int]:
        start_t = self.spin_start.value()
        end_t = self.spin_end.value()
        if end_t < start_t:
            start_t, end_t = end_t, start_t
        return [i for i, t in enumerate(self._times) if start_t <= float(t) <= end_t]

    def _quality_profile(self) -> tuple[int, int]:
        quality = self.quality_combo.currentText()
        if quality == "高清":
            return 1080, 20
        if quality == "标准":
            return 720, 24
        return 540, 28

    def _resampled_landmarks(self, indices: list[int], fps: float) -> tuple[np.ndarray, np.ndarray]:
        selected_times = self._times[indices]
        start_t = float(selected_times[0])
        end_t = float(selected_times[-1])
        if end_t <= start_t:
            end_t = start_t + (1.0 / fps)
        target_times = np.arange(start_t, end_t + (0.5 / fps), 1.0 / fps, dtype=np.float64)
        src_landmarks = np.stack([self._landmarks[i] for i in indices], axis=0).astype(np.float32)
        point_n = src_landmarks.shape[1]
        out = np.full((target_times.size, point_n, 2), np.nan, dtype=np.float32)
        for p in range(point_n):
            for axis in range(2):
                y = src_landmarks[:, p, axis]
                valid = np.isfinite(y)
                if np.count_nonzero(valid) < 2:
                    continue
                out[:, p, axis] = np.interp(target_times, selected_times[valid], y[valid]).astype(np.float32)
        return target_times, out

    def _save_video(self):
        indices = self._selected_indices()
        if len(indices) < 2:
            QMessageBox.warning(self, "保存视频", "请选择至少包含两帧的时间区间。")
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "保存视频", "lip_animation.mp4", "MP4 Files (*.mp4)")
        if not output_path:
            return
        ffmpeg = resolve_ffmpeg_executable()
        if not ffmpeg:
            QMessageBox.warning(self, "保存视频", "未检测到 ffmpeg，无法导出带音频视频。")
            return

        target_h, crf = self._quality_profile()
        fps = max(10.0, min(60.0, self._fps))
        export_times, export_landmarks = self._resampled_landmarks(indices, fps)
        first_frame = self._render_frame(indices[0])
        h0, w0 = first_frame.shape[:2]
        scale = target_h / float(h0)
        target_w = max(2, int(round(w0 * scale / 2.0) * 2))
        target_size = (target_w, int(target_h))

        with tempfile.TemporaryDirectory(prefix="lip_anim_") as td:
            tmp_dir = Path(td)
            silent_video = tmp_dir / "silent.mp4"
            segment_wav = tmp_dir / "segment.wav"

            writer = cv2.VideoWriter(
                str(silent_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                target_size,
            )
            for pts in export_landmarks:
                frame = self._render_points(pts)
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                writer.write(frame)
            writer.release()

            start_t = float(export_times[0])
            end_t = float(export_times[-1] + (1.0 / fps))
            with wave.open(str(self._wav_path), "rb") as wav_in:
                sr = wav_in.getframerate()
                channels = wav_in.getnchannels()
                sw = wav_in.getsampwidth()
                start_i = max(0, int(start_t * sr))
                end_i = max(start_i, int(end_t * sr))
                wav_in.setpos(start_i)
                audio_bytes = wav_in.readframes(end_i - start_i)
            with wave.open(str(segment_wav), "wb") as wav_out:
                wav_out.setnchannels(channels)
                wav_out.setsampwidth(sw)
                wav_out.setframerate(sr)
                wav_out.writeframes(audio_bytes)

            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(silent_video),
                "-i",
                str(segment_wav),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                QMessageBox.warning(self, "保存视频", f"导出失败:\n{result.stderr[:400]}")
                return
        QMessageBox.information(self, "保存视频", f"导出完成:\n{output_path}")

    def _save_gif(self):
        indices = self._selected_indices()
        if len(indices) < 2:
            QMessageBox.warning(self, "保存GIF", "请选择至少包含两帧的时间区间。")
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "保存GIF", "lip_animation.gif", "GIF Files (*.gif)")
        if not output_path:
            return
        try:
            from PIL import Image
        except Exception:
            QMessageBox.warning(self, "保存GIF", "未安装 Pillow，无法导出 GIF。")
            return

        fps = max(8.0, min(20.0, self._fps))
        _, export_landmarks = self._resampled_landmarks(indices, fps)
        images: list[Image.Image] = []
        for pts in export_landmarks:
            frame = self._render_points(pts)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        if not images:
            QMessageBox.warning(self, "保存GIF", "没有可导出的帧。")
            return
        duration_ms = int(round(1000.0 / fps))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=duration_ms,
            loop=0,
        )
        QMessageBox.information(self, "保存GIF", f"导出完成:\n{output_path}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._sync_timer.stop()
        self._player.stop()
        super().closeEvent(event)


def launch_lip_gui(is_dark: bool = True) -> LipGUI:
    window = LipGUI()
    window.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
    window.set_theme(is_dark)
    window.show()
    return window
