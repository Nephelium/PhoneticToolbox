# M07: V2 numerical implementation, import-only migration. SRC-ZAIWA / REF-ZAIWA.
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import signal

from .m07_models import (
    ContinuumType,
    F0AlignmentMode,
    PhonationAnalysisResult,
)


CancelCallback = Callable[[], bool] | None


def _raise_if_cancelled(cancel_requested: CancelCallback) -> None:
    if cancel_requested is not None and cancel_requested():
        raise InterruptedError("任务已取消。")


def trim_edge_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -45.0,
    pad_ms: float = 8.0,
) -> np.ndarray:
    """Remove leading/trailing low-RMS frames while retaining a small pad."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    frame_length = max(1, int(round(sample_rate * 0.01)))
    frame_shift = max(1, int(round(sample_rate * 0.005)))
    if len(x) < frame_length:
        return x.copy()

    starts = np.arange(0, len(x) - frame_length + 1, frame_shift, dtype=int)
    rms = np.asarray(
        [np.sqrt(np.mean(x[start : start + frame_length] ** 2)) for start in starts],
        dtype=np.float64,
    )
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    if peak_rms <= 1e-12:
        return x.copy()
    threshold = peak_rms * (10.0 ** (threshold_db / 20.0))
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return x.copy()

    padding = int(round(sample_rate * pad_ms / 1000.0))
    start_sample = max(0, int(starts[active[0]]) - padding)
    end_sample = min(
        len(x),
        int(starts[active[-1]]) + frame_length + padding,
    )
    if end_sample <= start_sample:
        return x.copy()
    return x[start_sample:end_sample].copy()


def analysis_window(frame_length: int, window_name: str = "hamming") -> np.ndarray:
    name = window_name.lower()
    if name == "hann":
        return np.hanning(frame_length)
    if name == "blackman":
        return np.blackman(frame_length)
    if name == "rectangular":
        return np.ones(frame_length, dtype=np.float64)
    if name != "hamming":
        raise ValueError(f"不支持的窗函数：{window_name}")
    return np.hamming(frame_length)


def enframe(audio: np.ndarray, frame_length: int, frame_shift: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    if len(x) <= frame_length:
        frame_count = 1
    else:
        frame_count = 1 + (len(x) - frame_length + frame_shift - 1) // frame_shift
    frames = np.zeros((frame_count, frame_length), dtype=np.float64)
    for index in range(frame_count):
        start = index * frame_shift
        chunk = x[start : start + frame_length]
        frames[index, : len(chunk)] = chunk
    return frames


def lpc_coefficients(frame: np.ndarray, order: int) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    autocorrelation = np.correlate(values, values, mode="full")
    autocorrelation = autocorrelation[len(values) - 1 : len(values) + order]
    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[0] = 1.0
    if autocorrelation.size < order + 1 or autocorrelation[0] <= 1e-12:
        return coefficients

    error = float(autocorrelation[0])
    for index in range(1, order + 1):
        accumulator = autocorrelation[index] + np.dot(
            coefficients[1:index],
            autocorrelation[index - 1 : 0 : -1],
        )
        reflection = float(-accumulator / max(error, 1e-12))
        reflection = float(np.clip(reflection, -0.999999, 0.999999))
        previous = coefficients.copy()
        coefficients[1:index] = (
            previous[1:index] + reflection * previous[index - 1 : 0 : -1]
        )
        coefficients[index] = reflection
        error *= 1.0 - reflection * reflection
        if error <= 1e-12:
            break
    return coefficients


def build_lpc_residual(
    audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    frame_length: int,
    frame_shift: int,
    order: int,
    preemphasis: float = 0.98,
    window_name: str = "hamming",
    cancel_requested: CancelCallback = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        raise ValueError("音频为空，无法进行 LPC 分析。")
    start_sample = int(np.clip(start_sample, 0, len(x) - 1))
    end_sample = int(np.clip(end_sample, start_sample, len(x) - 1))
    segment = x[start_sample : end_sample + 1]
    frames = enframe(segment, frame_length, frame_shift)
    window = analysis_window(frame_length, window_name)
    coefficients = np.zeros((len(frames), order + 1), dtype=np.float64)
    residual_frames = np.zeros_like(frames)
    preemphasis_filter = np.array([1.0, -float(preemphasis)])

    for index, frame in enumerate(frames):
        _raise_if_cancelled(cancel_requested)
        lpc = lpc_coefficients(
            signal.lfilter(preemphasis_filter, [1.0], frame) * window,
            order,
        )
        coefficients[index] = lpc
        residual_frames[index] = signal.lfilter(lpc, [1.0], frame)

    residual = x.copy()
    residual[start_sample : end_sample + 1] = 0.0
    for index, frame in enumerate(residual_frames):
        start = start_sample + index * frame_shift
        end = min(start + frame_length, end_sample + 1)
        residual[start:end] += frame[: end - start] * window[: end - start]
    return coefficients, residual


def voiced_bounds(
    f0_hz: np.ndarray,
    sample_rate: int,
    sample_count: int,
    margin_ms: float = 30.0,
) -> tuple[int, int]:
    voiced = np.flatnonzero(np.asarray(f0_hz) > 0)
    if voiced.size == 0:
        raise ValueError("没有检测到有效 F0，请调整 F0 范围或更换算法。")
    start = max(0, int(round((voiced[0] - margin_ms) / 1000.0 * sample_rate)))
    end = min(
        sample_count - 1,
        int(round((voiced[-1] + margin_ms) / 1000.0 * sample_rate)),
    )
    return start, end


def fill_unvoiced_inside(
    f0_hz: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    out = np.asarray(f0_hz, dtype=np.float64).copy()
    start_ms = int(round(start_sample / sample_rate * 1000.0))
    end_ms = int(round(end_sample / sample_rate * 1000.0))
    inside = np.arange(max(0, start_ms), min(len(out), end_ms + 1))
    if inside.size == 0:
        return out
    valid_inside = inside[np.isfinite(out[inside]) & (out[inside] > 0)]
    if valid_inside.size == 0:
        return out
    midpoint = (start_ms + end_ms) / 2.0
    missing = ~np.isfinite(out[inside]) | (out[inside] <= 0)
    for index in inside[missing]:
        out[index] = out[valid_inside[0]] if index < midpoint else out[valid_inside[-1]]
    return out


def f0_at_sample(f0_hz: np.ndarray, sample_rate: int, sample: int) -> float:
    values = np.asarray(f0_hz, dtype=np.float64)
    if values.size == 0:
        return 120.0
    index = int(round(sample / sample_rate * 1000.0))
    index = int(np.clip(index, 0, len(values) - 1))
    value = float(values[index])
    return value if np.isfinite(value) and value > 0 else 120.0


def detect_pulses(
    residual: np.ndarray,
    f0_hz: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    negative_peak_threshold: float = -0.005,
    pulse_inner_periods: float = 0.5,
    pulse_outer_periods: float = 1.5,
    cancel_requested: CancelCallback = None,
) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    center = int(round((start_sample + end_sample) / 2))
    period = max(1, int(round(sample_rate / f0_at_sample(f0_hz, sample_rate, center))))
    left = max(start_sample, center - period)
    right = min(end_sample, center + period)
    if right <= left:
        raise ValueError("有效分析区间过短，无法检测声门脉冲。")
    first_pulse = left + int(np.argmin(values[left:right]))
    pulses = [first_pulse]

    current = first_pulse
    while current > start_sample and values[current] < negative_peak_threshold:
        _raise_if_cancelled(cancel_requested)
        current_f0 = f0_at_sample(f0_hz, sample_rate, current)
        left = max(
            start_sample,
            int(round(current - pulse_outer_periods * sample_rate / current_f0)),
        )
        right = max(
            left + 1,
            min(
                end_sample,
                int(round(current - pulse_inner_periods * sample_rate / current_f0)),
            ),
        )
        next_pulse = left + int(np.argmin(values[left:right]))
        if next_pulse >= current:
            break
        pulses.insert(0, next_pulse)
        current = next_pulse

    current = first_pulse
    while current < end_sample and values[current] < negative_peak_threshold:
        _raise_if_cancelled(cancel_requested)
        current_f0 = f0_at_sample(f0_hz, sample_rate, current)
        left = min(
            end_sample - 1,
            max(
                start_sample,
                int(round(current + pulse_inner_periods * sample_rate / current_f0)),
            ),
        )
        right = min(
            end_sample,
            int(round(current + pulse_outer_periods * sample_rate / current_f0)),
        )
        if right <= left:
            break
        next_pulse = left + int(np.argmin(values[left:right]))
        if next_pulse <= current:
            break
        pulses.append(next_pulse)
        current = next_pulse

    pulse_array = np.asarray(sorted(set(pulses)), dtype=int)
    pulse_array = pulse_array[
        (pulse_array > start_sample) & (pulse_array < end_sample)
    ]
    if pulse_array.size < 3:
        minimum_distance = max(10, int(sample_rate / 300.0))
        fallback, _ = signal.find_peaks(
            -values[start_sample:end_sample],
            distance=minimum_distance,
        )
        pulse_array = fallback + start_sample
    if pulse_array.size < 2:
        raise ValueError("检测到的声门脉冲不足，请调整 F0 或脉冲检测参数。")
    return pulse_array.astype(int)


def voiced_range(f0_hz: np.ndarray) -> tuple[int, int] | None:
    values = np.asarray(f0_hz, dtype=np.float64)
    voiced = np.flatnonzero(np.isfinite(values) & (values > 0))
    if voiced.size == 0:
        return None
    return int(voiced[0]), int(voiced[-1])


def build_f0_control_axis(
    source_f0: np.ndarray,
    target_f0: np.ndarray,
    point_count: int,
    mode: F0AlignmentMode,
) -> np.ndarray:
    count = max(2, int(point_count))
    if mode == F0AlignmentMode.NORMALIZED:
        return np.linspace(0.0, 100.0, count)
    durations = []
    for track in (source_f0, target_f0):
        bounds = voiced_range(track)
        durations.append(0 if bounds is None else bounds[1] - bounds[0])
    return np.linspace(0.0, float(max(*durations, 1)), count)


def sample_f0_on_axis(
    f0_hz: np.ndarray,
    axis_values: np.ndarray,
    mode: F0AlignmentMode,
) -> np.ndarray:
    bounds = voiced_range(f0_hz)
    output = np.zeros(len(axis_values), dtype=np.float64)
    if bounds is None:
        return output
    start, end = bounds
    segment = np.asarray(f0_hz[start : end + 1], dtype=np.float64)
    valid = np.isfinite(segment) & (segment > 0)
    if not np.any(valid):
        return output
    source_axis = (
        np.linspace(0.0, 100.0, len(segment))
        if mode == F0AlignmentMode.NORMALIZED
        else np.arange(len(segment), dtype=np.float64)
    )
    valid_axis = source_axis[valid]
    inside = (axis_values >= valid_axis[0]) & (axis_values <= valid_axis[-1])
    output[inside] = np.interp(
        axis_values[inside],
        valid_axis,
        segment[valid],
    )
    return output


def sample_f0_control_points(
    f0_hz: np.ndarray,
    point_count: int,
    mode: F0AlignmentMode,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = voiced_range(f0_hz)
    if mode == F0AlignmentMode.NORMALIZED:
        axis = np.linspace(0.0, 100.0, max(2, int(point_count)))
    else:
        duration = 0 if bounds is None else bounds[1] - bounds[0]
        axis = np.linspace(0.0, float(max(duration, 1)), max(2, int(point_count)))
    return axis, sample_f0_on_axis(f0_hz, axis, mode)


def interpolate_f0_control_points(
    axis_values: np.ndarray,
    values_hz: np.ndarray,
    original_f0: np.ndarray,
    mode: F0AlignmentMode,
) -> np.ndarray:
    output = np.zeros(len(original_f0), dtype=np.float64)
    bounds = voiced_range(original_f0)
    if bounds is None:
        return output
    values = np.asarray(values_hz, dtype=np.float64)
    axis = np.asarray(axis_values, dtype=np.float64)
    valid = np.isfinite(axis) & np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return output
    start, end = bounds
    segment_length = end - start + 1
    grid = (
        np.linspace(0.0, 100.0, segment_length)
        if mode == F0AlignmentMode.NORMALIZED
        else np.arange(segment_length, dtype=np.float64)
    )
    valid_axis = axis[valid]
    valid_values = values[valid]
    order = np.argsort(valid_axis)
    valid_axis = valid_axis[order]
    valid_values = valid_values[order]
    unique_axis, unique_indices = np.unique(valid_axis, return_index=True)
    unique_values = valid_values[unique_indices]
    interpolated = np.interp(grid, unique_axis, unique_values)
    interpolated[grid < unique_axis[0]] = 0.0
    interpolated[grid > unique_axis[-1]] = 0.0
    output[start : end + 1] = interpolated
    return output


def interpolate_target_f0(
    source: PhonationAnalysisResult,
    target: PhonationAnalysisResult,
    step_count: int,
) -> np.ndarray:
    target_on_source = source.f0_hz.copy()
    source_start_ms = int(round(source.start_sample / source.sample_rate * 1000.0))
    source_end_ms = int(round(source.end_sample / source.sample_rate * 1000.0))
    target_start_ms = int(round(target.start_sample / target.sample_rate * 1000.0))
    target_end_ms = int(round(target.end_sample / target.sample_rate * 1000.0))
    source_length = max(1, source_end_ms - source_start_ms + 1)
    target_segment = target.f0_hz[
        max(0, target_start_ms) : min(len(target.f0_hz), target_end_ms + 1)
    ]
    target_segment = target_segment[
        np.isfinite(target_segment) & (target_segment > 0)
    ]
    if target_segment.size == 0:
        raise ValueError("目标音频没有可用于合成的有效 F0。")
    target_resampled = signal.resample(target_segment, source_length)
    available = min(source_length, len(target_on_source) - source_start_ms)
    if available > 0:
        target_on_source[source_start_ms : source_start_ms + available] = (
            target_resampled[:available]
        )
    f0_steps = np.zeros((len(source.f0_hz), step_count), dtype=np.float64)
    for step in range(step_count):
        alpha = step / (step_count - 1)
        f0_steps[:, step] = source.f0_hz + (target_on_source - source.f0_hz) * alpha
    return f0_steps


def resample_to_length(values: np.ndarray, length: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(source) == length:
        return source.copy()
    if source.size == 0:
        return np.zeros(length, dtype=np.float64)
    return signal.resample(source, length)


def make_residual_continuum(
    source: PhonationAnalysisResult,
    target: PhonationAnalysisResult,
    continuum_type: ContinuumType,
    step_count: int,
    energy_match: bool,
    cancel_requested: CancelCallback = None,
) -> np.ndarray:
    if step_count < 2:
        raise ValueError("合成步数必须至少为 2。")
    source_pulses = np.asarray(source.pulses, dtype=int)
    target_pulses = np.asarray(target.pulses, dtype=int)
    if source_pulses.size < 2 or target_pulses.size < 2:
        raise ValueError("检测到的声门脉冲不足，无法合成连续统。")

    residual_steps = np.zeros((len(source.residual), step_count), dtype=np.float64)
    f0_steps = interpolate_target_f0(source, target, step_count)
    source_percent = (source_pulses - source.start_sample) / max(
        1, source.end_sample - source.start_sample + 1
    )
    target_percent = (target_pulses - target.start_sample) / max(
        1, target.end_sample - target.start_sample + 1
    )

    if continuum_type == ContinuumType.PHONATION_ONLY:
        target_like = source.residual.copy()
        for index in range(len(source_pulses) - 1):
            _raise_if_cancelled(cancel_requested)
            target_index = int(
                np.argmin(np.abs(target_percent[:-1] - source_percent[index]))
            )
            source_period = source.residual[
                source_pulses[index] + 1 : source_pulses[index + 1] + 1
            ]
            target_period = target.residual[
                target_pulses[target_index] + 1 : target_pulses[target_index + 1] + 1
            ]
            target_period = resample_to_length(target_period, len(source_period))
            if energy_match and np.max(np.abs(target_period)) > 0:
                target_period = target_period / np.max(np.abs(target_period)) * (
                    np.max(np.abs(source_period)) or 1.0
                )
            target_like[
                source_pulses[index] + 1 : source_pulses[index + 1] + 1
            ] = target_period
        for step in range(step_count):
            alpha = step / (step_count - 1)
            residual_steps[:, step] = (
                source.residual + (target_like - source.residual) * alpha
            )
        return residual_steps

    for step in range(step_count):
        _raise_if_cancelled(cancel_requested)
        residual_steps[:, step] = source.residual
        residual_steps[source_pulses[0] : source_pulses[-1], step] = 0.0
        sample = int(source_pulses[0])
        while sample <= source_pulses[-1]:
            _raise_if_cancelled(cancel_requested)
            millisecond = int(round(sample / source.sample_rate * 1000.0))
            millisecond = int(np.clip(millisecond, 0, f0_steps.shape[0] - 1))
            period = max(
                1,
                int(round(source.sample_rate / max(f0_steps[millisecond, step], 1.0))),
            )
            if continuum_type == ContinuumType.F0_ONLY:
                source_index = int(
                    np.argmin(np.abs(source_pulses[:-1] - sample))
                )
                source_period = source.residual[
                    source_pulses[source_index] + 1 : source_pulses[source_index + 1] + 1
                ]
                output_period = resample_to_length(source_period, period)
            elif continuum_type == ContinuumType.F0_AND_PHONATION:
                percent = (sample - source.start_sample) / max(
                    1, source.end_sample - source.start_sample + 1
                )
                source_index = int(
                    np.argmin(np.abs(source_percent[:-1] - percent))
                )
                target_index = int(
                    np.argmin(np.abs(target_percent[:-1] - percent))
                )
                source_period = resample_to_length(
                    source.residual[
                        source_pulses[source_index] + 1 : source_pulses[source_index + 1] + 1
                    ],
                    period,
                )
                target_period = resample_to_length(
                    target.residual[
                        target_pulses[target_index] + 1 : target_pulses[target_index + 1] + 1
                    ],
                    period,
                )
                if energy_match and np.max(np.abs(target_period)) > 0:
                    target_period = target_period / np.max(np.abs(target_period)) * (
                        np.max(np.abs(source_period)) or 1.0
                    )
                alpha = step / (step_count - 1)
                output_period = source_period + (target_period - source_period) * alpha
            else:
                raise ValueError(f"不支持的连续统类型：{continuum_type}")
            end = min(sample + 1 + len(output_period), len(source.residual))
            residual_steps[sample + 1 : end, step] += output_period[: end - sample - 1]
            sample += period
    return residual_steps


def synthesize_from_residual(
    source: PhonationAnalysisResult,
    residual_steps: np.ndarray,
    normalize_to_source: bool = True,
    output_peak_limit: float = 0.98,
    cancel_requested: CancelCallback = None,
) -> np.ndarray:
    values = np.asarray(residual_steps, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(source.signal):
        raise ValueError("残差信号连续统的形状与源音频不一致。")
    output = values.copy()
    output[source.start_sample : source.end_sample + 1, :] = 0.0
    window = analysis_window(source.config.frame_length, source.config.window_name)

    for step in range(values.shape[1]):
        _raise_if_cancelled(cancel_requested)
        segment = values[source.start_sample : source.end_sample + 1, step]
        frames = enframe(
            segment,
            source.config.frame_length,
            source.config.frame_shift,
        )
        for index, frame in enumerate(frames):
            _raise_if_cancelled(cancel_requested)
            if index >= len(source.lpc_coefficients):
                break
            start = source.start_sample + index * source.config.frame_shift
            end = min(start + source.config.frame_length, source.end_sample + 1)
            synthesized = signal.lfilter(
                [1.0],
                source.lpc_coefficients[index],
                frame,
            )
            output[start:end, step] += (
                synthesized[: end - start] * window[: end - start]
            )
        if normalize_to_source:
            source_level = float(np.mean(np.abs(source.signal)))
            output_level = float(np.mean(np.abs(output[:, step])))
            if output_level > 1e-12:
                output[:, step] *= source_level / output_level
        peak = float(np.max(np.abs(output[:, step])))
        if output_peak_limit > 0 and peak > output_peak_limit:
            output[:, step] *= output_peak_limit / peak
    if not np.isfinite(output).all():
        raise ValueError("重合成产生了非有限数值，请调整 LPC 或脉冲检测参数。")
    return output
