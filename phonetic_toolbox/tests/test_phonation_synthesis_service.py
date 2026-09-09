from __future__ import annotations

from pathlib import Path
import json
import threading

import numpy as np
import pytest
from scipy.io import wavfile

from phonetic_toolbox.models.phonation_synthesis_models import (
    ContinuumType,
    PhonationAnalysisConfig,
    PhonationContinuumResult,
    PhonationGenerationConfig,
)
from phonetic_toolbox.services.phonation_synthesis_service import (
    PhonationSynthesisService,
)


def _write_sine(path: Path, *, sample_rate: int = 8000, duration: float = 0.5) -> None:
    time = np.arange(int(sample_rate * duration), dtype=float) / sample_rate
    audio = 0.45 * np.sin(2.0 * np.pi * 120.0 * time)
    wavfile.write(path, sample_rate, (audio * 32767).astype(np.int16))


def _fake_f0(audio: np.ndarray, sample_rate: int, config: PhonationAnalysisConfig) -> np.ndarray:
    values = np.zeros(int(np.ceil(len(audio) / sample_rate * 1000.0)) + 1)
    values[20:-20] = 120.0
    return values


def test_track_to_millisecond_grid_does_not_extrapolate():
    grid = PhonationSynthesisService.track_to_millisecond_grid(
        times_sec=np.array([0.1, 0.2, 0.3]),
        values_hz=np.array([100.0, 120.0, 140.0]),
        duration_sec=0.4,
    )

    assert np.all(grid[:100] == 0.0)
    assert grid[100] == pytest.approx(100.0)
    assert grid[250] == pytest.approx(130.0)
    assert np.all(grid[301:] == 0.0)


def test_analyze_file_uses_models_and_returns_finite_result(tmp_path: Path):
    wav_path = tmp_path / "source.wav"
    _write_sine(wav_path)
    service = PhonationSynthesisService(f0_estimator=_fake_f0)
    config = PhonationAnalysisConfig(
        target_sample_rate=4000,
        frame_length=64,
        frame_shift=16,
        lpc_order=12,
        trim_silence=False,
    )

    result = service.analyze_file(wav_path, config)

    assert result.sample_rate == 4000
    assert result.signal.shape == result.residual.shape
    assert result.lpc_coefficients.shape[1] == 13
    assert result.pulses.size >= 2
    assert np.isfinite(result.residual).all()


def test_analyze_file_rejects_missing_input(tmp_path: Path):
    service = PhonationSynthesisService(f0_estimator=_fake_f0)

    with pytest.raises(FileNotFoundError, match="WAV"):
        service.analyze_file(tmp_path / "missing.wav", PhonationAnalysisConfig())


def test_save_f0_csv_has_stable_columns(tmp_path: Path):
    service = PhonationSynthesisService(f0_estimator=_fake_f0)
    output = service.save_f0_csv(
        tmp_path,
        source_f0=np.array([0.0, 100.0, 110.0]),
        target_f0=np.array([90.0, 95.0]),
    )

    assert output.name == "edited_f0.csv"
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "time_ms,source_f0,target_f0"
    assert len(lines) == 4


def test_export_continuum_writes_steps_and_combined_file(tmp_path: Path):
    service = PhonationSynthesisService(f0_estimator=_fake_f0)
    result = PhonationContinuumResult(
        sample_rate=8000,
        audio_steps=np.column_stack(
            [np.linspace(-0.2, 0.2, 100), np.linspace(-0.1, 0.1, 100)]
        ),
        continuum_type=ContinuumType.F0_ONLY,
    )

    exported = service.export_continuum(result, tmp_path / "仅改变F0")

    assert [path.name for path in exported.step_files] == ["step01.wav", "step02.wav"]
    assert exported.combined_file.name == "combined_steps.wav"
    assert all(path.exists() for path in exported.step_files)
    assert exported.combined_file.exists()


def test_generate_all_uses_six_direction_and_type_directories(tmp_path: Path, monkeypatch):
    service = PhonationSynthesisService(f0_estimator=_fake_f0)
    source_path = tmp_path / "source.wav"
    target_path = tmp_path / "target.wav"
    _write_sine(source_path)
    _write_sine(target_path)
    config = PhonationAnalysisConfig(
        target_sample_rate=4000,
        frame_length=64,
        frame_shift=16,
        lpc_order=12,
        trim_silence=False,
    )
    source, target = service.analyze_file_pair(source_path, target_path, config)

    def fake_generate(source_result, target_result, continuum_type, generation, **_kwargs):
        del source_result, target_result, generation
        return PhonationContinuumResult(
            sample_rate=4000,
            audio_steps=np.zeros((80, 2), dtype=float),
            continuum_type=continuum_type,
        )

    monkeypatch.setattr(service, "generate_continuum", fake_generate)
    exports = service.generate_all(
        source,
        target,
        PhonationGenerationConfig(step_count=2),
        tmp_path / "output",
    )

    assert len(exports) == 6
    names = {item.output_directory.name for item in exports}
    assert names == {
        "源到目标_仅改变F0",
        "源到目标_仅改变发声类型",
        "源到目标_同时改变F0和发声类型",
        "目标到源_仅改变F0",
        "目标到源_仅改变发声类型",
        "目标到源_同时改变F0和发声类型",
    }
    batch = exports[0].output_directory.parent
    assert all(item.output_directory.parent == batch for item in exports)
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["source"]["path"] == str(source_path.resolve())
    assert manifest["target"]["path"] == str(target_path.resolve())
    assert manifest["generation_config"]["step_count"] == 2
    assert len(manifest["outputs"]) == 6
    assert (batch / manifest["f0_file"]).exists()


def test_repeated_exports_preserve_old_stimuli_and_do_not_mix_steps(tmp_path):
    service = PhonationSynthesisService()
    first = PhonationContinuumResult(11025, np.full((100, 9), 0.1), ContinuumType.F0_ONLY)
    second = PhonationContinuumResult(11025, np.full((100, 3), 0.2), ContinuumType.F0_ONLY)
    previous = service.export_continuum(first, tmp_path)
    originals = {p: p.read_bytes() for p in (*previous.step_files, previous.combined_file)}
    current = service.export_continuum(second, tmp_path)
    assert current.output_directory != previous.output_directory
    assert len(list(current.output_directory.glob("step*.wav"))) == 3
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    manifest = json.loads((current.output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["outputs"][0]["files"]) == 4


def test_cancelled_export_is_marked_incomplete(tmp_path, monkeypatch):
    from phonetic_toolbox.services import phonation_synthesis_service as module
    service = PhonationSynthesisService()
    result = PhonationContinuumResult(11025, np.full((100, 3), 0.2), ContinuumType.F0_ONLY)
    cancel = threading.Event()
    write = module.write_wav

    def cancel_after_first_write(*args):
        write(*args)
        cancel.set()

    monkeypatch.setattr(module, "write_wav", cancel_after_first_write)
    with pytest.raises(InterruptedError):
        service.export_continuum(result, tmp_path, cancel_event=cancel)
    batch, = tmp_path.iterdir()
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert not (batch / "combined_steps.wav").exists()
