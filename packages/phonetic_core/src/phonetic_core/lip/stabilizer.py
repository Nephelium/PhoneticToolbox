from __future__ import annotations

import math
import numpy as np

# Source: V2 lip_gui.py LandmarkStabilizer, unchanged numerical body.
class LandmarkStabilizer:
    """面部特征点低通滤波器（速度下限 One Euro + 强运动直通门控）。

    特征点为 2D 像素坐标 (N, 2)。设计目标：滑条对滤波强度有绝对控制权
    （1Hz=强防抖 … 摄像头帧率=几乎原始），同时嘴唇等自然快速运动不拖慢。

    1. 逐点自适应截止频率（One Euro）：截止频率 = 滑条基准频率 +
       beta × 该点速度。速度用相邻『原始』帧差分估计（不含滤波滞后），
       并减去随面部尺寸自适应的噪声速度下限——静止时纯抖动不会把截止
       频率抬回去，滑条设多少就是多少。
    2. 速度场邻居平均：逐点速度先与网格邻居做一轮平均再计算截止频率，
       相邻点平滑强度接近，不破坏嘴唇等局部几何形状。
    3. 强运动直通门控：新测量与当前估计的位移明显超过噪声门限（约为面
       部对角线的 1%~2.4%）时判定为真实运动，增益直接提升到门控值、立
       即跟随，嘴唇快速运动在任何档位下都几乎零延迟；静止时位移小于门
       限，增益完全由滑条截止频率决定。
    """

    def __init__(
        self,
        min_cutoff_hz: float = 15.0,
        beta: float = 0.08,
        d_cutoff_hz: float = 1.0,
        neighbor_indices: list[np.ndarray] | None = None,
    ) -> None:
        self.min_cutoff = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff_hz)
        self._neighbors = neighbor_indices
        self.reset()

    def set_neighbors(self, neighbor_indices: list[np.ndarray] | None) -> None:
        self._neighbors = neighbor_indices

    def reset(self) -> None:
        self._x_hat: np.ndarray | None = None
        self._dx_hat: np.ndarray | None = None
        self._x_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
        return dt / (tau + dt)

    def _neighbor_average(self, speeds: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """速度场与网格邻居做一轮平均，保持相邻点平滑强度一致。"""
        if self._neighbors is None or len(self._neighbors) != speeds.shape[0]:
            return speeds
        accum = speeds.astype(np.float64).copy()
        counts = np.ones(speeds.shape[0], dtype=np.float64)
        valid_f = valid.astype(np.float64)
        for i, nbrs in enumerate(self._neighbors):
            if nbrs.size == 0:
                continue
            accum[i] += float(np.dot(speeds[nbrs], valid_f[nbrs]))
            counts[i] += float(valid_f[nbrs].sum())
        return (accum / counts).astype(np.float32)

    def filter(self, points: np.ndarray, timestamp: float) -> np.ndarray:
        """对一帧特征点做滤波。points 形状 (N, 2)，允许含 NaN 行。"""
        x = np.asarray(points, dtype=np.float32)
        valid = np.isfinite(x).all(axis=1)
        if not np.any(valid):
            return x

        if (
            self._x_hat is None
            or self._t_prev is None
            or self._dx_hat is None
            or self._x_prev is None
        ):
            self._x_hat = x.copy()
            self._x_prev = x.copy()
            self._dx_hat = np.zeros_like(x)
            self._t_prev = float(timestamp)
            return x.copy()

        dt = float(timestamp) - self._t_prev
        if dt <= 0.0:
            return self._x_hat.copy()
        dt = min(dt, 0.25)

        # 个别点之前缺失（NaN）而本帧恢复时，直接用当前观测重新初始化
        finite_hat = np.isfinite(self._x_hat).all(axis=1)
        reinit = valid & ~finite_hat
        if np.any(reinit):
            self._x_hat[reinit] = x[reinit]
            self._x_prev[reinit] = x[reinit]
            self._dx_hat[reinit] = 0.0

        # 1) 速度估计：相邻原始帧差分 + 一阶平滑，减去噪声速度下限
        a_d = self._alpha(self.d_cutoff, dt)
        prev_valid = np.isfinite(self._x_prev).all(axis=1)
        both = valid & prev_valid
        dx = np.zeros_like(x)
        dx[both] = (x[both] - self._x_prev[both]) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat
        dx_hat[~valid] = 0.0

        speeds = np.linalg.norm(dx_hat, axis=1)
        speeds = self._neighbor_average(speeds, valid)

        anchor_rows = valid & np.isfinite(self._x_hat).all(axis=1)
        valid_pts = self._x_hat[anchor_rows]
        span = valid_pts.max(axis=0) - valid_pts.min(axis=0)
        face_scale = float(np.linalg.norm(span))
        speeds = np.maximum(speeds - 0.11 * face_scale, 0.0)

        cutoffs = self.min_cutoff + self.beta * speeds
        taus = 1.0 / (2.0 * math.pi * np.maximum(cutoffs, 1e-6))
        alphas = (dt / (taus + dt)).astype(np.float32)

        # 2) 强运动直通门控：位移明显超过噪声门限时直接跟随
        res = x - self._x_hat
        mags = np.linalg.norm(res, axis=1)
        mags[~np.isfinite(mags)] = 0.0
        n0 = 0.010 * face_scale
        n1 = 0.024 * face_scale
        gate = np.clip((mags - n0) / max(n1 - n0, 1e-6), 0.0, 1.0)
        alpha_eff = np.maximum(alphas, gate).astype(np.float32)

        x_new = self._x_hat.copy()
        a_col = alpha_eff[valid, None]
        x_new[valid] = self._x_hat[valid] + a_col * res[valid]

        self._x_hat = x_new
        self._dx_hat = dx_hat
        self._x_prev = x.copy()
        self._t_prev = float(timestamp)
        return x_new
