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


def _record_new_data(save_dir: str) -> int:
    try:
        import mediapipe as mp
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("唇形提取", f"缺少 mediapipe 或加载失败:\n{exc}")
        root.destroy()
        return 1

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

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
    metrics: dict[str, list] = {
        "absolute_timestamps": [],
        "relative_times": [],
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
        "landmarks": [],
    }

    recording = False
    start_time: float | None = None
    cv2.namedWindow("MediaPipe Lip Tracking")

    exit_code = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                exit_code = 1
                break

            status = "RECORDING" if recording else "STANDBY - Press r to Record, q to Quit"
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
                    for key in (
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
                    ):
                        metrics[key].append(one_frame[key])
                    metrics["landmarks"].append(landmarks)

            cv2.imshow("MediaPipe Lip Tracking", frame)
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
