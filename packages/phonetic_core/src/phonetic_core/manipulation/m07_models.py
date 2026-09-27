# M07: inherited V2 models; source IDs SRC-ZAIWA, REF-ZAIWA.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


class F0Backend(str, Enum):
    PARSELMOUTH = "parselmouth"
    REAPER = "reaper"


class F0AlignmentMode(str, Enum):
    NORMALIZED = "normalize"
    ONSET = "onset"


class ContinuumType(IntEnum):
    PHONATION_ONLY = 1
    F0_ONLY = 2
    F0_AND_PHONATION = 3

    @property
    def display_name(self) -> str:
        return {
            self.PHONATION_ONLY: "仅改变发声类型",
            self.F0_ONLY: "仅改变F0",
            self.F0_AND_PHONATION: "同时改变F0和发声类型",
        }[self]


@dataclass(frozen=True)
class PhonationAnalysisConfig:
    target_sample_rate: int = 11025
    f0_backend: F0Backend = F0Backend.PARSELMOUTH
    min_f0_hz: float = 50.0
    max_f0_hz: float = 300.0
    f0_frame_interval_ms: float = 1.0
    frame_length: int = 128
    frame_shift: int = 32
    lpc_order: int = 20
    preemphasis: float = 0.98
    window_name: str = "hamming"
    negative_peak_threshold: float = -0.005
    pulse_inner_periods: float = 0.5
    pulse_outer_periods: float = 1.5
    trim_silence: bool = True
    silence_threshold_db: float = -45.0
    silence_padding_ms: float = 8.0
    voiced_margin_ms: float = 30.0

    def validate(self) -> None:
        if self.target_sample_rate <= 0:
            raise ValueError("目标采样率必须大于 0。")
        if self.min_f0_hz <= 0 or self.max_f0_hz <= self.min_f0_hz:
            raise ValueError("最低 F0 必须大于 0 且小于最高 F0。")
        if self.f0_frame_interval_ms <= 0:
            raise ValueError("F0 帧距必须大于 0。")
        if self.frame_length < 2:
            raise ValueError("窗长必须至少为 2 点。")
        if self.frame_shift <= 0 or self.frame_shift >= self.frame_length:
            raise ValueError("帧移必须大于 0 且小于窗长。")
        if self.lpc_order < 1 or self.lpc_order >= self.frame_length:
            raise ValueError("LPC 阶数必须大于 0 且小于窗长。")
        if not 0.0 <= self.preemphasis < 1.0:
            raise ValueError("预加重必须位于 [0, 1) 范围内。")
        if self.window_name not in {"hamming", "hann", "blackman", "rectangular"}:
            raise ValueError(f"不支持的窗函数：{self.window_name}")
        if self.pulse_inner_periods <= 0:
            raise ValueError("脉冲内窗必须大于 0。")
        if self.pulse_outer_periods <= self.pulse_inner_periods:
            raise ValueError("脉冲外窗必须大于脉冲内窗。")
        if self.silence_padding_ms < 0 or self.voiced_margin_ms < 0:
            raise ValueError("静音保留和有声边界不能为负数。")


@dataclass(frozen=True)
class PhonationGenerationConfig:
    step_count: int = 9
    energy_match: bool = True
    normalize_to_source: bool = True
    output_peak_limit: float = 0.98

    def validate(self) -> None:
        if self.step_count < 2:
            raise ValueError("合成步数必须至少为 2。")
        if not 0.0 < self.output_peak_limit <= 1.0:
            raise ValueError("峰值限制必须位于 (0, 1] 范围内。")


@dataclass
class PhonationAnalysisResult:
    sample_rate: int
    signal: np.ndarray
    start_sample: int
    end_sample: int
    f0_hz: np.ndarray
    lpc_coefficients: np.ndarray
    residual: np.ndarray
    pulses: np.ndarray
    config: PhonationAnalysisConfig

    def copy_with_f0(self, f0_hz: np.ndarray) -> "PhonationAnalysisResult":
        values = np.asarray(f0_hz, dtype=np.float64)
        if values.shape != self.f0_hz.shape:
            raise ValueError("编辑后的 F0 轨迹长度与分析结果不一致。")
        return PhonationAnalysisResult(
            sample_rate=self.sample_rate,
            signal=self.signal,
            start_sample=self.start_sample,
            end_sample=self.end_sample,
            f0_hz=values.copy(),
            lpc_coefficients=self.lpc_coefficients,
            residual=self.residual,
            pulses=self.pulses,
            config=self.config,
        )


@dataclass(frozen=True)
class PhonationContinuumResult:
    sample_rate: int
    audio_steps: np.ndarray
    continuum_type: ContinuumType


