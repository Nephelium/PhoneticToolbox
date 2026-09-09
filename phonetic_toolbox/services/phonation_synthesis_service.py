from __future__ import annotations

import csv
import json
import math
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import numpy as np
from scipy import signal
from scipy.io import wavfile

from phonetic_toolbox.core.acoustic.f0_praat import compute_praat_f0_track
from phonetic_toolbox.core.acoustic.f0_reaper import compute_reaper_f0
from phonetic_toolbox.core.manipulation.phonation_synthesis import (
    build_lpc_residual,
    detect_pulses,
    fill_unvoiced_inside,
    make_residual_continuum,
    synthesize_from_residual,
    trim_edge_silence,
    voiced_bounds,
)
from phonetic_toolbox.models.phonation_synthesis_models import (
    ContinuumType,
    F0Backend,
    PhonationAnalysisConfig,
    PhonationAnalysisResult,
    PhonationContinuumResult,
    PhonationExportResult,
    PhonationGenerationConfig,
)
from phonetic_toolbox.services.io.wav import read_wav_float_mono, write_wav


ProgressCallback = Callable[[int, str], None] | None
F0Estimator = Callable[[np.ndarray, int, PhonationAnalysisConfig], np.ndarray]


class PhonationSynthesisService:
    """Orchestrate WAV IO, F0 extraction, LPC analysis and continuum export."""

    def __init__(self, f0_estimator: F0Estimator | None = None) -> None:
        self._f0_estimator = f0_estimator or self._estimate_f0

    @staticmethod
    def _report(progress: ProgressCallback, value: int, message: str) -> None:
        if progress is not None:
            progress(int(value), message)

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("任务已取消。")

    @staticmethod
    def track_to_millisecond_grid(
        times_sec: np.ndarray,
        values_hz: np.ndarray,
        duration_sec: float,
    ) -> np.ndarray:
        grid_times = np.arange(int(math.ceil(duration_sec * 1000.0)) + 1) / 1000.0
        output = np.zeros_like(grid_times)
        times = np.asarray(times_sec, dtype=np.float64)
        values = np.asarray(values_hz, dtype=np.float64)
        if times.shape != values.shape:
            raise ValueError("F0 时间轴和值的长度不一致。")
        valid = np.isfinite(times) & np.isfinite(values) & (values > 0)
        if not np.any(valid):
            return output
        valid_times = times[valid]
        valid_values = values[valid]
        order = np.argsort(valid_times)
        valid_times = valid_times[order]
        valid_values = valid_values[order]
        output = np.interp(
            grid_times,
            valid_times,
            valid_values,
            left=0.0,
            right=0.0,
        )
        output[grid_times < valid_times[0]] = 0.0
        output[grid_times > valid_times[-1]] = 0.0
        return output

    def _estimate_f0(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: PhonationAnalysisConfig,
    ) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="phonation_f0_") as directory:
            wav_path = Path(directory) / "input.wav"
            wavfile.write(wav_path, sample_rate, np.asarray(audio, dtype=np.float32))
            if config.f0_backend == F0Backend.PARSELMOUTH:
                track = compute_praat_f0_track(
                    wav_path=wav_path,
                    frameshift_ms=config.f0_frame_interval_ms,
                    min_f0=config.min_f0_hz,
                    max_f0=config.max_f0_hz,
                    method="ac",
                )
                times = track.times
                values = track.values
            elif config.f0_backend == F0Backend.REAPER:
                result = compute_reaper_f0(
                    wav_path=wav_path,
                    frame_interval_sec=config.f0_frame_interval_ms / 1000.0,
                    min_f0=config.min_f0_hz,
                    max_f0=config.max_f0_hz,
                )
                times = np.asarray(result.get("rTimes", []), dtype=np.float64)
                values = np.asarray(result.get("rF0", []), dtype=np.float64)
                voiced = np.asarray(result.get("rVoiced", []), dtype=np.float64)
                if voiced.shape == values.shape:
                    values = np.where(voiced > 0, values, np.nan)
            else:
                raise ValueError(f"不支持的 F0 算法：{config.f0_backend}")
        return self.track_to_millisecond_grid(
            times,
            values,
            duration_sec=len(audio) / sample_rate,
        )

    @staticmethod
    def _resample_audio(
        audio: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        values = np.asarray(audio, dtype=np.float64)
        if source_rate == target_rate:
            return values.copy()
        divisor = math.gcd(int(source_rate), int(target_rate))
        return signal.resample_poly(
            values,
            target_rate // divisor,
            source_rate // divisor,
        ).astype(np.float64)

    def analyze_file(
        self,
        wav_path: str | Path,
        config: PhonationAnalysisConfig,
        *,
        progress: ProgressCallback = None,
        cancel_event: threading.Event | None = None,
    ) -> PhonationAnalysisResult:
        config.validate()
        path = Path(wav_path)
        if not path.is_file():
            raise FileNotFoundError(f"WAV 文件不存在：{path}")

        self._check_cancel(cancel_event)
        self._report(progress, 5, f"正在读取 {path.name}...")
        source_rate, audio = read_wav_float_mono(path)
        audio = self._resample_audio(audio, source_rate, config.target_sample_rate)
        if audio.size == 0:
            raise ValueError(f"WAV 文件为空：{path}")
        if config.trim_silence:
            audio = trim_edge_silence(
                audio,
                config.target_sample_rate,
                config.silence_threshold_db,
                config.silence_padding_ms,
            )
        self._check_cancel(cancel_event)

        self._report(progress, 20, f"正在提取 {path.name} 的初始 F0...")
        initial_f0 = self._f0_estimator(audio, config.target_sample_rate, config)
        start_sample, end_sample = voiced_bounds(
            initial_f0,
            config.target_sample_rate,
            len(audio),
            config.voiced_margin_ms,
        )

        self._check_cancel(cancel_event)
        self._report(progress, 45, f"正在分析 {path.name} 的 LPC 残差...")
        coefficients, residual = build_lpc_residual(
            audio,
            start_sample,
            end_sample,
            config.frame_length,
            config.frame_shift,
            config.lpc_order,
            config.preemphasis,
            config.window_name,
            cancel_requested=(cancel_event.is_set if cancel_event is not None else None),
        )

        self._check_cancel(cancel_event)
        self._report(progress, 70, f"正在提取 {path.name} 的残差 F0...")
        residual_f0 = self._f0_estimator(residual, config.target_sample_rate, config)
        residual_f0 = fill_unvoiced_inside(
            residual_f0,
            config.target_sample_rate,
            start_sample,
            end_sample,
        )

        self._check_cancel(cancel_event)
        self._report(progress, 88, f"正在检测 {path.name} 的声门脉冲...")
        pulses = detect_pulses(
            residual,
            residual_f0,
            config.target_sample_rate,
            start_sample,
            end_sample,
            config.negative_peak_threshold,
            config.pulse_inner_periods,
            config.pulse_outer_periods,
            cancel_requested=(cancel_event.is_set if cancel_event is not None else None),
        )
        self._report(progress, 100, f"{path.name} 分析完成。")
        return PhonationAnalysisResult(
            sample_rate=config.target_sample_rate,
            signal=audio,
            start_sample=start_sample,
            end_sample=end_sample,
            f0_hz=residual_f0,
            lpc_coefficients=coefficients,
            residual=residual,
            pulses=pulses,
            config=config,
            source_path=path.resolve(),
        )

    def analyze_file_pair(
        self,
        source_path: str | Path,
        target_path: str | Path,
        config: PhonationAnalysisConfig,
        *,
        progress: ProgressCallback = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[PhonationAnalysisResult, PhonationAnalysisResult]:
        def source_progress(value: int, message: str) -> None:
            self._report(progress, int(value * 0.48), message)

        def target_progress(value: int, message: str) -> None:
            self._report(progress, 50 + int(value * 0.48), message)

        source = self.analyze_file(
            source_path,
            config,
            progress=source_progress,
            cancel_event=cancel_event,
        )
        self._check_cancel(cancel_event)
        target = self.analyze_file(
            target_path,
            config,
            progress=target_progress,
            cancel_event=cancel_event,
        )
        self._report(progress, 100, "源音频和目标音频分析完成。")
        return source, target

    def generate_continuum(
        self,
        source: PhonationAnalysisResult,
        target: PhonationAnalysisResult,
        continuum_type: ContinuumType,
        generation: PhonationGenerationConfig,
        *,
        progress: ProgressCallback = None,
        cancel_event: threading.Event | None = None,
    ) -> PhonationContinuumResult:
        generation.validate()
        self._check_cancel(cancel_event)
        self._report(progress, 10, f"正在生成{continuum_type.display_name}的残差连续统...")
        cancel_callback = cancel_event.is_set if cancel_event is not None else None
        residual_steps = make_residual_continuum(
            source,
            target,
            continuum_type,
            generation.step_count,
            generation.energy_match,
            cancel_requested=cancel_callback,
        )
        self._check_cancel(cancel_event)
        self._report(progress, 55, "正在通过源声道滤波器重合成...")
        audio_steps = synthesize_from_residual(
            source,
            residual_steps,
            generation.normalize_to_source,
            generation.output_peak_limit,
            cancel_requested=cancel_callback,
        )
        self._report(progress, 90, "连续统合成完成，准备写入文件...")
        return PhonationContinuumResult(
            sample_rate=source.sample_rate,
            audio_steps=audio_steps,
            continuum_type=continuum_type,
        )

    @staticmethod
    def _to_int16(audio: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.asarray(audio, dtype=np.float64))
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > 1.0:
            values = values / peak
        values = np.clip(values, -1.0, 1.0)
        return np.round(values * 32767.0).astype(np.int16)

    @staticmethod
    def _write_manifest(directory: Path, payload: dict) -> None:
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "manifest.json")

    def _new_batch(self, output_root: str | Path) -> Path:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime("batch_%Y%m%d_%H%M%S_")
        directory = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        self._write_manifest(directory, {"status": "incomplete", "outputs": []})
        return directory

    def _complete_batch(
        self, directory: Path, exports, source=None, target=None, generation=None,
    ) -> None:
        payload = {"status": "complete", "outputs": [
            {
                "type": item.continuum_type.name,
                "reverse_direction": item.reverse_direction,
                "files": [str(path.relative_to(directory)) for path in (*item.step_files, item.combined_file)],
            }
            for item in exports
        ]}
        if source is not None and target is not None:
            for role, analysis in (("source", source), ("target", target)):
                payload[role] = {
                    "path": str(analysis.source_path) if analysis.source_path else None,
                    "sample_rate": analysis.sample_rate,
                    "start_sample": analysis.start_sample,
                    "end_sample": analysis.end_sample,
                    "analysis_config": asdict(analysis.config),
                }
            payload["generation_config"] = asdict(generation)
            payload["f0_file"] = self.save_f0_csv(directory, source.f0_hz, target.f0_hz).name
        self._write_manifest(directory, payload)

    def export_continuum(
        self,
        result: PhonationContinuumResult,
        output_directory: str | Path,
        *,
        reverse_direction: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> PhonationExportResult:
        """Export to a new batch; existing stimuli are never overwritten."""
        self._check_cancel(cancel_event)
        directory = self._new_batch(output_directory)
        exported = self._write_continuum(
            result, directory, reverse_direction=reverse_direction, cancel_event=cancel_event,
        )
        self._check_cancel(cancel_event)
        self._complete_batch(directory, (exported,))
        return exported

    def _write_continuum(
        self,
        result: PhonationContinuumResult,
        output_directory: str | Path,
        *,
        reverse_direction: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> PhonationExportResult:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        step_files: list[Path] = []
        for index in range(result.audio_steps.shape[1]):
            self._check_cancel(cancel_event)
            path = directory / f"step{index + 1:02d}.wav"
            write_wav(
                path,
                result.sample_rate,
                self._to_int16(result.audio_steps[:, index]),
            )
            step_files.append(path)
        combined_file = directory / "combined_steps.wav"
        self._check_cancel(cancel_event)
        write_wav(
            combined_file,
            result.sample_rate,
            self._to_int16(result.audio_steps.T.reshape(-1)),
        )
        return PhonationExportResult(
            output_directory=directory,
            step_files=tuple(step_files),
            combined_file=combined_file,
            continuum_type=result.continuum_type,
            reverse_direction=reverse_direction,
        )

    def generate_selected(
        self,
        source: PhonationAnalysisResult,
        target: PhonationAnalysisResult,
        continuum_type: ContinuumType,
        generation: PhonationGenerationConfig,
        output_root: str | Path,
        *,
        progress: ProgressCallback = None,
        cancel_event: threading.Event | None = None,
    ) -> PhonationExportResult:
        result = self.generate_continuum(
            source,
            target,
            continuum_type,
            generation,
            progress=progress,
            cancel_event=cancel_event,
        )
        self._check_cancel(cancel_event)
        batch = self._new_batch(output_root)
        exported = self._write_continuum(
            result,
            batch / continuum_type.display_name,
            cancel_event=cancel_event,
        )
        self._check_cancel(cancel_event)
        self._complete_batch(batch, (exported,), source, target, generation)
        self._report(progress, 100, f"已生成：{exported.output_directory}")
        return exported

    def generate_all(
        self,
        source: PhonationAnalysisResult,
        target: PhonationAnalysisResult,
        generation: PhonationGenerationConfig,
        output_root: str | Path,
        *,
        progress: ProgressCallback = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[PhonationExportResult, ...]:
        generation.validate()
        self._check_cancel(cancel_event)
        root = self._new_batch(output_root)
        designs = [
            (source, target, False, "源到目标", ContinuumType.F0_ONLY),
            (source, target, False, "源到目标", ContinuumType.PHONATION_ONLY),
            (source, target, False, "源到目标", ContinuumType.F0_AND_PHONATION),
            (target, source, True, "目标到源", ContinuumType.F0_ONLY),
            (target, source, True, "目标到源", ContinuumType.PHONATION_ONLY),
            (target, source, True, "目标到源", ContinuumType.F0_AND_PHONATION),
        ]
        exports: list[PhonationExportResult] = []
        for index, (from_analysis, to_analysis, reverse, direction, kind) in enumerate(designs):
            self._check_cancel(cancel_event)
            base = int(index / len(designs) * 100)
            span = max(1, int(100 / len(designs)))

            def item_progress(value: int, message: str, *, _base=base, _span=span) -> None:
                self._report(progress, _base + int(value / 100 * _span), message)

            result = self.generate_continuum(
                from_analysis,
                to_analysis,
                kind,
                generation,
                progress=item_progress,
                cancel_event=cancel_event,
            )
            exported = self._write_continuum(
                result,
                root / f"{direction}_{kind.display_name}",
                reverse_direction=reverse,
                cancel_event=cancel_event,
            )
            exports.append(exported)
        self._check_cancel(cancel_event)
        self._complete_batch(root, exports, source, target, generation)
        self._report(progress, 100, f"全部连续统已生成到：{root}")
        return tuple(exports)

    def save_f0_csv(
        self,
        output_root: str | Path,
        source_f0: np.ndarray,
        target_f0: np.ndarray,
    ) -> Path:
        directory = Path(output_root)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / "edited_f0.csv"
        source = np.asarray(source_f0, dtype=np.float64)
        target = np.asarray(target_f0, dtype=np.float64)
        row_count = max(len(source), len(target))
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("time_ms", "source_f0", "target_f0"))
            for index in range(row_count):
                writer.writerow(
                    (
                        index,
                        float(source[index]) if index < len(source) else 0.0,
                        float(target[index]) if index < len(target) else 0.0,
                    )
                )
        return output_path
