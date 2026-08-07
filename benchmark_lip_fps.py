# -*- coding: utf-8 -*-
"""唇形录制性能基准测试（多分辨率扫描）。

对每个候选分辨率（480p/720p/1080p，以摄像头实际支持为准）测量：
  A. 纯相机采集速率（预览速率上限）
  B. 采集 + VideoWriter 写盘速率（离线模式的采集速率）
  C. FaceMesh 单帧推理耗时（refine_landmarks=True/False），
     用人脸测试图缩放到对应分辨率测量
  D. 推算：实时推理模式的录制数据帧率、离线模式处理耗时

用法： python benchmark_lip_fps.py
"""
from __future__ import annotations

import os
import tempfile
import time
import urllib.request

import cv2
import numpy as np

FACE_IMAGE_URL = (
    "https://gitee.com/mirrors_opencv/opencv/raw/master/samples/data/lena.jpg"
)
CANDIDATE_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
N_CAPTURE = 150          # 采集测速帧数
N_INFER = 100            # 推理测速迭代次数
WARMUP = 10


def fetch_face_image() -> np.ndarray | None:
    cache = os.path.join(tempfile.gettempdir(), "pt_benchmark_lena.jpg")
    if not os.path.exists(cache):
        try:
            req = urllib.request.Request(
                FACE_IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            data = urllib.request.urlopen(req, timeout=10).read()
            with open(cache, "wb") as fh:
                fh.write(data)
        except Exception as exc:
            print(f"[warn] 人脸测试图下载失败: {exc}")
            return None
    return cv2.imread(cache)


def measure_capture(cap, n: int, writer: cv2.VideoWriter | None = None) -> tuple[float, int]:
    """返回 (每帧平均耗时秒, 实际帧数)。"""
    for _ in range(WARMUP):
        cap.read()
    t0 = time.perf_counter()
    got = 0
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if writer is not None:
            writer.write(frame)
        got += 1
    dt = time.perf_counter() - t0
    return dt / max(got, 1), got


def measure_inference(face_mesh, frame_bgr: np.ndarray, n: int) -> tuple[float, int]:
    """返回 (单帧平均耗时秒, 检测到的点数)。无脸时点数为 0。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    for _ in range(WARMUP):
        face_mesh.process(rgb)
    t0 = time.perf_counter()
    n_points = 0
    for _ in range(n):
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            n_points = len(result.multi_face_landmarks[0].landmark)
    dt = time.perf_counter() - t0
    return dt / n, n_points


def open_camera_at(width: int, height: int):
    """请求指定分辨率，返回 (cap, 实际宽, 实际高, 声明fps)；不支持则 cap 为 None。"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, 0, 0, 0.0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if (w, h) != (width, height):
        cap.release()
        return None, w, h, fps
    return cap, w, h, fps


def main() -> None:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh

    print("=" * 72)
    print("唇形录制基准测试（多分辨率）")
    print("=" * 72)

    face_img = fetch_face_image()
    if face_img is None:
        print("[error] 无人脸测试图，无法进行推理测速。")
        return

    summary: list[dict] = []

    for req_w, req_h in CANDIDATE_RESOLUTIONS:
        cap, w, h, cam_fps = open_camera_at(req_w, req_h)
        if cap is None:
            print(f"\n### {req_w}x{req_h}: 摄像头不支持，跳过")
            continue

        print(f"\n### {w}x{h} @ 声明 {cam_fps:.1f} fps")

        # A: 纯采集
        t_read, got = measure_capture(cap, N_CAPTURE)
        print(f"[A] 纯采集:        {t_read*1000:6.1f} ms/帧 -> {1/t_read:5.1f} fps (n={got})")

        # B: 采集 + 写盘
        tmp_video = os.path.join(
            tempfile.gettempdir(), f"pt_benchmark_capture_{w}x{h}.mp4"
        )
        writer = cv2.VideoWriter(
            tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), cam_fps, (w, h)
        )
        t_write, got = measure_capture(cap, N_CAPTURE, writer=writer)
        writer.release()
        os.remove(tmp_video)
        print(f"[B] 采集+写盘:     {t_write*1000:6.1f} ms/帧 -> {1/t_write:5.1f} fps (n={got})")
        cap.release()

        # C: 推理耗时（人脸图缩放到该分辨率）
        face_frame = cv2.resize(face_img, (w, h))
        row: dict = {"res": f"{w}x{h}", "cam_fps": cam_fps,
                     "t_read": t_read, "t_write": t_write}
        for refine in (True, False):
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=refine,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            t_inf, npts = measure_inference(face_mesh, face_frame, N_INFER)
            face_mesh.close()
            face_str = f"检测到人脸({npts}点)" if npts else "未检测到人脸!"
            print(
                f"[C] 推理 refine={str(refine):5s}: {t_inf*1000:6.1f} ms/帧 "
                f"-> 推理上限 {1/t_inf:5.1f} fps | {face_str}"
            )
            row[f"t_inf_{refine}"] = t_inf

        # D: 推算
        for refine in (True, False):
            t_inf = row[f"t_inf_{refine}"]
            realtime_fps = 1.0 / (t_read + t_inf)
            offline_sec = cam_fps * 60 * t_inf
            print(
                f"[D] refine={str(refine):5s}: 实时模式录制数据率 ≈ {realtime_fps:5.1f} 帧/秒; "
                f"离线模式 1 分钟素材处理 ≈ {offline_sec:4.0f} 秒"
            )
        summary.append(row)

    # 汇总表
    if summary:
        print("\n" + "=" * 72)
        print("汇总（人脸帧，refine=True / False）:")
        print(f"{'分辨率':10s} {'采集fps':>8s} {'采集+写盘':>10s} "
              f"{'推理ms(T)':>10s} {'推理ms(F)':>10s} "
              f"{'实时率(T)':>9s} {'实时率(F)':>9s} {'离线1分钟(T)':>12s}")
        for row in summary:
            rt_t = 1.0 / (row["t_read"] + row["t_inf_True"])
            rt_f = 1.0 / (row["t_read"] + row["t_inf_False"])
            off_t = row["cam_fps"] * 60 * row["t_inf_True"]
            print(
                f"{row['res']:10s} {1/row['t_read']:8.1f} {1/row['t_write']:10.1f} "
                f"{row['t_inf_True']*1000:10.1f} {row['t_inf_False']*1000:10.1f} "
                f"{rt_t:9.1f} {rt_f:9.1f} {off_t:11.0f}s"
            )
    print(
        "\n说明: 实时模式数据率 = 1/(取帧+推理)；离线模式数据率 = 采集+写盘速率，"
        "代价是录制结束后追加离线处理时间。"
    )


if __name__ == "__main__":
    main()
