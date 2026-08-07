from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import sys
import threading
import time
import wave
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox


def _prepare_import_path() -> None:
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(
        os.path.join(current_file, "..", "..", "..", "..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_prepare_import_path()

from phonetic_toolbox.core.lip.metrics import extract_lip_metrics


RATE = 44100
CHANNELS = 1
CHUNK = 1024

_WINDOW_NAME = "MediaPipe Lip Tracking"
_METRIC_KEYS = (
    "area",
    "face_width",
    "face_height",
    "height_px",
    "outer_width_px",
    "inner_width_px",
    "total_width_px",
    "open_px",
    "length",
    "height",
    "outer_width",
    "inner_width",
    "total_width",
    "open",
    "circularity",
)


class AudioRecorder:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.stream: sd.InputStream | None = None
        self.frames: list[bytes] = []
        self.frame_timestamps: list[float] = []
        self.start_time: float | None = None
        self.is_recording = False
        self.lock = threading.Lock()

    def start_recording(self) -> None:
        self.frames = []
        self.frame_timestamps = []
        self.start_time = time.time()
        self.is_recording = True

        def _callback(indata, _frames, _time_info, _status):
            if not self.is_recording:
                return
            with self.lock:
                self.frames.append(indata.copy().tobytes())
                self.frame_timestamps.append(time.time())

        self.stream = sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK,
            callback=_callback,
        )
        self.stream.start()

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        self.is_recording = False

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        with wave.open(self.output_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.frames))

        timestamps_file = os.path.splitext(self.output_file)[0] + "_timestamps.pkl"
        with open(timestamps_file, "wb") as handle:
            pickle.dump(
                {
                    "start_time": self.start_time,
                    "frame_timestamps": self.frame_timestamps,
                    "sample_rate": RATE,
                    "chunk_size": CHUNK,
                },
                handle,
            )


def _select_or_create_directory(load_existing: bool) -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    if load_existing:
        selected = filedialog.askdirectory(title="Select directory containing data")
        root.destroy()
        return selected or None

    parent_dir = filedialog.askdirectory(title="请选择保存数据的文件夹 (Select Save Folder)")
    root.destroy()
    if not parent_dir:
        return None

    base_dir_name = f"lip_tracking_data_{datetime.now().strftime('%Y%m%d')}"
    save_dir = os.path.join(parent_dir, base_dir_name)
    suffix = 1
    while os.path.exists(save_dir):
        save_dir = os.path.join(parent_dir, f"{base_dir_name}_{suffix}")
        suffix += 1
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _save_data(directory: str, metrics: dict[str, list]) -> str:
    data = {
        "absolute_timestamps": metrics["absolute_timestamps"],
        "relative_times": metrics["relative_times"],
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
        "landmarks": metrics["landmarks"],
        "metadata": {
            "recording_start_time": (
                metrics["absolute_timestamps"][0]
                if metrics["absolute_timestamps"]
                else None
            ),
            "recording_duration": (
                metrics["relative_times"][-1] if metrics["relative_times"] else None
            ),
            "fps": _safe_fps(
                frame_count=len(metrics["absolute_timestamps"]),
                duration=(
                    metrics["relative_times"][-1]
                    if metrics["relative_times"]
                    else None
                ),
            ),
            "created_at": datetime.now().isoformat(),
        },
    }
    data_file = os.path.join(directory, "audio_recording.pkl")
    with open(data_file, "wb") as handle:
        pickle.dump(data, handle)
    return data_file


def _load_data(directory: str) -> dict | None:
    data_file = os.path.join(directory, "audio_recording.pkl")
    if not os.path.exists(data_file):
        return None
    with open(data_file, "rb") as handle:
        return pickle.load(handle)


def _safe_fps(frame_count: int, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    return frame_count / duration


def _plot_metrics(directory: str, data: dict) -> None:
    relative_times = data.get("relative_times", [])
    plt.figure(figsize=(15, 9))
    plt.subplot(3, 2, 1)
    plt.plot(relative_times, data.get("area", []))
    plt.title("Lip Area Ratio")
    plt.xlabel("Time (s)")
    plt.ylabel("Area Ratio")
    plt.subplot(3, 2, 2)
    plt.plot(relative_times, data.get("height", []))
    plt.title("Normalized Lip Height")
    plt.xlabel("Time (s)")
    plt.ylabel("Height")
    plt.subplot(3, 2, 3)
    plt.plot(relative_times, data.get("outer_width", []))
    plt.title("Normalized Outer Lip Width")
    plt.xlabel("Time (s)")
    plt.ylabel("Width")
    plt.subplot(3, 2, 4)
    plt.plot(relative_times, data.get("open", []))
    plt.title("Normalized Lip Openness")
    plt.xlabel("Time (s)")
    plt.ylabel("Opening")
    plt.subplot(3, 2, 5)
    plt.plot(relative_times, data.get("total_width", []))
    plt.title("Normalized Inner + Outer Lip Width")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Width")
    plt.subplot(3, 2, 6)
    plt.plot(relative_times, data.get("circularity", []))
    plt.title("Lip Circularity")
    plt.xlabel("Time (s)")
    plt.ylabel("Circularity")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "lip_metrics_plot.png"))
    plt.close()

    plt.figure(figsize=(15, 9))
    plt.subplot(3, 2, 1)
    plt.plot(relative_times, data.get("face_width", []))
    plt.title("Face Width (pixels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixels")
    plt.subplot(3, 2, 2)
    plt.plot(relative_times, data.get("height_px", []))
    plt.title("Lip Height (pixels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixels")
    plt.subplot(3, 2, 3)
    plt.plot(relative_times, data.get("outer_width_px", []))
    plt.title("Outer Lip Width (pixels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixels")
    plt.subplot(3, 2, 4)
    plt.plot(relative_times, data.get("open_px", []))
    plt.title("Lip Openness (pixels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixels")
    plt.subplot(3, 2, 5)
    plt.plot(relative_times, data.get("total_width_px", []))
    plt.title("Inner + Outer Lip Width (pixels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixels")
    plt.subplot(3, 2, 6)
    plt.plot(relative_times, data.get("face_height", []))
    plt.title("Face Height (pixels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixels")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "raw_metrics_plot.png"))
    plt.close()


def _run_load_mode() -> int:
    load_dir = _select_or_create_directory(load_existing=True)
    if not load_dir:
        return 1
    data = _load_data(load_dir)
    if data is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("唇形提取", f"未找到数据文件: {load_dir}\\audio_recording.pkl")
        root.destroy()
        return 1
    _plot_metrics(load_dir, data)
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("唇形提取", f"已加载并绘图:\n{load_dir}")
    root.destroy()
    return 0


def _empty_metrics() -> dict[str, list]:
    metrics: dict[str, list] = {
        "absolute_timestamps": [],
        "relative_times": [],
        "landmarks": [],
    }
    for key in _METRIC_KEYS:
        metrics[key] = []
    return metrics


def _create_face_mesh(refine_landmarks: bool):
    try:
        import mediapipe as mp
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("唇形提取", f"缺少 mediapipe 或加载失败:\n{exc}")
        root.destroy()
        return None
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _ask_record_options() -> tuple[str, bool] | None:
    """弹出录制模式选择对话框。

    返回 (mode, refine_landmarks)，mode 为 "realtime" 或 "offline"；
    用户直接关闭对话框时返回 None。
    """
    root = tk.Tk()
    root.title("唇形录制 - 模式选择")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    result: dict[str, object] = {}
    refine_var = tk.BooleanVar(value=True)

    tk.Label(root, text="请选择录制模式：", font=("", 11, "bold")).pack(
        padx=16, pady=(14, 6)
    )
    tk.Label(
        root,
        text=(
            "实时推理：边录边识别，点位帧率受推理速度限制\n"
            "离线推理：先完整录制视频与音频，结束后逐帧识别，处理完自动删除视频\n"
            "（离线模式适用于被试不同意留存视频、且要求满帧率点位的场合）"
        ),
        justify=tk.LEFT,
    ).pack(padx=16, pady=(0, 8))

    def _choose(mode: str) -> None:
        result["mode"] = mode
        result["refine"] = bool(refine_var.get())
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=4)
    tk.Button(
        btn_frame,
        text="实时推理录制",
        width=18,
        command=lambda: _choose("realtime"),
    ).grid(row=0, column=0, padx=6)
    tk.Button(
        btn_frame,
        text="离线推理录制",
        width=18,
        command=lambda: _choose("offline"),
    ).grid(row=0, column=1, padx=6)
    tk.Checkbutton(
        root,
        text="虹膜精修 refine_landmarks（对唇形指标无影响，关闭可略微提速）",
        variable=refine_var,
    ).pack(padx=16, pady=(6, 12), anchor="w")

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    if "mode" not in result:
        return None
    return str(result["mode"]), bool(result["refine"])


def _fill_leading_gaps(metrics: dict[str, list]) -> None:
    """用首个有效帧回填开头的 NaN 帧（人脸尚未进入画面时的前导帧）。"""
    landmarks = metrics["landmarks"]
    first_valid = None
    for idx, lm in enumerate(landmarks):
        if np.isfinite(np.asarray(lm, dtype=float)).all():
            first_valid = idx
            break
    if first_valid is None or first_valid == 0:
        return
    for idx in range(first_valid):
        landmarks[idx] = np.array(landmarks[first_valid], dtype=float).copy()
    for key in _METRIC_KEYS:
        values = metrics[key]
        first_value = values[first_valid]
        for idx in range(first_valid):
            values[idx] = first_value


def _record_new_data(save_dir: str) -> int:
    options = _ask_record_options()
    if options is None:
        return 0
    mode, refine_landmarks = options
    if mode == "offline":
        return _record_offline(save_dir, refine_landmarks)
    return _record_realtime(save_dir, refine_landmarks)


def _record_realtime(save_dir: str, refine_landmarks: bool) -> int:
    face_mesh = _create_face_mesh(refine_landmarks)
    if face_mesh is None:
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("唇形提取", "无法打开摄像头。")
        root.destroy()
        face_mesh.close()
        return 1

    audio_file = os.path.join(save_dir, "audio_recording.wav")
    recorder = AudioRecorder(audio_file)
    metrics = _empty_metrics()

    recording = False
    start_time: float | None = None
    cv2.namedWindow(_WINDOW_NAME)

    exit_code = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                exit_code = 1
                break

            status = (
                "RECORDING [realtime]"
                if recording
                else "STANDBY [realtime] - Press r to Record, q to Quit"
            )
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if recording else (255, 255, 255),
                2,
            )

            if recording:
                absolute_time = time.time()
                if start_time is None:
                    start_time = absolute_time
                relative_time = absolute_time - start_time
                metrics["absolute_timestamps"].append(absolute_time)
                metrics["relative_times"].append(relative_time)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb_frame)
                if result.multi_face_landmarks:
                    first_face = result.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    landmarks = np.array(
                        [(lm.x * w, lm.y * h) for lm in first_face.landmark],
                        dtype=float,
                    )
                    one_frame = extract_lip_metrics(landmarks)
                    for key in _METRIC_KEYS:
                        metrics[key].append(one_frame[key])
                    metrics["landmarks"].append(landmarks)

            cv2.imshow(_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                recording = not recording
                if recording:
                    start_time = None
                    recorder.start_recording()
                else:
                    recorder.stop_recording()
            elif key == ord("q"):
                break
    finally:
        if recorder.is_recording:
            recorder.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

    _save_data(save_dir, metrics)
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("唇形提取", f"数据已保存到:\n{save_dir}")
    root.destroy()
    return exit_code


def _record_offline(save_dir: str, refine_landmarks: bool) -> int:
    """先完整录制视频+音频，结束后逐帧离线识别，最后删除视频。"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("唇形提取", "无法打开摄像头。")
        root.destroy()
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    audio_file = os.path.join(save_dir, "audio_recording.wav")
    recorder = AudioRecorder(audio_file)
    video_temp = os.path.join(save_dir, "recorded_video_temp.mp4")

    recording = False
    writer: cv2.VideoWriter | None = None
    frame_timestamps: list[float] = []
    exit_code = 0
    cv2.namedWindow(_WINDOW_NAME)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                exit_code = 1
                break

            if recording:
                frame_timestamps.append(time.time())
                writer.write(frame)
                status = "RECORDING [offline] - r: stop & process, q: quit"
                color = (0, 0, 255)
            else:
                status = "STANDBY [offline] - r: record, q: quit"
                color = (255, 255, 255)
            cv2.putText(
                frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            cv2.imshow(_WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if not recording:
                    new_writer = cv2.VideoWriter(
                        video_temp,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not new_writer.isOpened():
                        new_writer.release()
                        root = tk.Tk()
                        root.withdraw()
                        messagebox.showerror("唇形提取", "视频写入器初始化失败。")
                        root.destroy()
                        continue
                    writer = new_writer
                    frame_timestamps = []
                    recorder.start_recording()
                    recording = True
                else:
                    recording = False
                    recorder.stop_recording()
                    writer.release()
                    writer = None
                    break
            elif key == ord("q"):
                break
    finally:
        if recorder.is_recording:
            recorder.stop_recording()
        if writer is not None:
            writer.release()
        cap.release()

    metrics: dict[str, list] | None = None
    try:
        if frame_timestamps and os.path.exists(video_temp):
            metrics = _process_video_offline(
                video_path=video_temp,
                frame_timestamps=frame_timestamps,
                refine_landmarks=refine_landmarks,
            )
        root = tk.Tk()
        root.withdraw()
        if metrics is None:
            messagebox.showinfo("唇形提取", "未生成数据（未录制或处理已取消）。")
        else:
            _save_data(save_dir, metrics)
            messagebox.showinfo(
                "唇形提取", f"数据已保存到:\n{save_dir}\n\n临时视频已删除。"
            )
        root.destroy()
    finally:
        cv2.destroyAllWindows()
        try:
            if os.path.exists(video_temp):
                os.remove(video_temp)
        except Exception:
            pass
    return exit_code


def _process_video_offline(
    video_path: str,
    frame_timestamps: list[float],
    refine_landmarks: bool,
) -> dict[str, list] | None:
    """对已录制的视频逐帧运行 FaceMesh，返回 metrics 字典。

    时间轴使用录制时记录的每帧墙钟时间戳，不受写盘速率影响。
    未检测到人脸的帧用最近一次有效结果回填；按 q/ESC 可取消（返回 None）。
    """
    face_mesh = _create_face_mesh(refine_landmarks)
    if face_mesh is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        face_mesh.close()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_landmarks = 478 if refine_landmarks else 468
    metrics = _empty_metrics()
    t0 = frame_timestamps[0]
    last_landmarks: np.ndarray | None = None
    last_values: dict[str, float] | None = None
    canceled = False
    frame_index = 0
    start_perf = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index < len(frame_timestamps):
                abs_ts = frame_timestamps[frame_index]
            else:
                abs_ts = frame_timestamps[-1] + (
                    frame_index - len(frame_timestamps) + 1
                ) / 30.0

            h, w, _ = frame.shape
            landmarks: np.ndarray | None = None
            values: dict[str, float] | None = None
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                first_face = result.multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x * w, lm.y * h) for lm in first_face.landmark],
                    dtype=float,
                )
                values = extract_lip_metrics(landmarks)
                last_landmarks = landmarks
                last_values = values
            elif last_landmarks is not None:
                landmarks = last_landmarks.copy()
                values = last_values

            metrics["absolute_timestamps"].append(abs_ts)
            metrics["relative_times"].append(abs_ts - t0)
            if landmarks is None or values is None:
                metrics["landmarks"].append(np.full((n_landmarks, 2), np.nan))
                for key in _METRIC_KEYS:
                    metrics[key].append(float("nan"))
            else:
                metrics["landmarks"].append(landmarks)
                for key in _METRIC_KEYS:
                    metrics[key].append(float(values[key]))

            frame_index += 1
            if frame_index % 5 == 0:
                elapsed = time.perf_counter() - start_perf
                speed = frame_index / max(elapsed, 1e-6)
                progress = (
                    f"Offline processing: {frame_index}/"
                    f"{total if total > 0 else '?'}  ({speed:.1f} fps)  q: cancel"
                )
                cv2.putText(
                    frame,
                    progress,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(_WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    canceled = True
                    break
    finally:
        cap.release()
        face_mesh.close()

    if canceled or frame_index == 0:
        return None

    _fill_leading_gaps(metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Lip feature tracking")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    if args.load:
        return _run_load_mode()

    save_dir = _select_or_create_directory(load_existing=False)
    if not save_dir:
        return 1
    return _record_new_data(save_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
