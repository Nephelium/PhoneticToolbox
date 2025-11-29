import numpy as np
from pathlib import Path

from PhoneticToolbox.services.praat_service import (
    _harmonic_mag_db,
    compute_harmonics_H1H2H4,
    compute_CPP,
    compute_HNR,
    compute_SHR,
    compute_harmonic_at_fixed_freq,
    read_wav_mono_float,
    compute_shrp_f0,
)


def synth_signal(fs: int, duration_s: float, f0: float) -> np.ndarray:
    t = np.arange(int(fs * duration_s)) / fs
    # 合成含 H1/H2/H4 的信号：幅度分别 1.0, 0.5, 0.25
    y = (
        1.0 * np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    return y.astype(float)


def test_harmonics_basic():
    fs = 16000
    f0 = 100.0
    y = synth_signal(fs, 0.5, f0)
    frameshift_ms = 10
    Nperiods = 3
    # 构造 F0 序列（50 帧，全部有声）
    nframes = int(0.5 * 1000 / frameshift_ms)
    F0 = np.full(nframes, f0, dtype=float)
    result = compute_harmonics_H1H2H4(y, fs, frameshift_ms, F0, Nperiods)
    H1 = result["H1"]
    H2 = result["H2"]
    H4 = result["H4"]
    # H1/H2/H4 均应非 NaN 且量级合理
    assert np.isnan(H1).sum() < nframes // 5
    assert np.isnan(H2).sum() < nframes // 5
    assert np.isnan(H4).sum() < nframes // 5
    # H1 应大于 H2，大于 H4
    m1 = np.nanmedian(H1)
    m2 = np.nanmedian(H2)
    m4 = np.nanmedian(H4)
    assert m1 > m2
    assert m2 > m4


def test_cpp_hnr_shapes():
    fs = 16000
    f0 = 120.0
    y = synth_signal(fs, 0.5, f0)
    frameshift_ms = 10
    nframes = int(0.5 * 1000 / frameshift_ms)
    F0 = np.full(nframes, f0, dtype=float)
    cpp = compute_CPP(y, fs, frameshift_ms, F0, N_periods=5)
    assert cpp.shape[0] == nframes
    hnr = compute_HNR(y, fs, frameshift_ms, F0, N_periods=5)
    # 存在四个频带键（MATLAB 命名）
    assert all(k in hnr for k in ["HNR05", "HNR15", "HNR25", "HNR35"])


def test_shr_range_and_mask():
    fs = 16000
    f0 = 120.0
    y = synth_signal(fs, 0.4, f0)
    frameshift_ms = 10
    nframes = int(0.4 * 1000 / frameshift_ms)
    F0 = np.full(nframes, f0, dtype=float)
    shr = compute_SHR(y, fs, frameshift_ms, F0, minf0=40.0, maxf0=500.0)
    assert shr.shape[0] == nframes
    valid = shr[~np.isnan(shr)]
    assert valid.size > 0
    assert np.all(valid >= 0.0) and np.all(valid <= 1.0)


def test_h5k_computation_across_frames():
    fs = 16000
    f0 = 120.0
    # 合成包含 5kHz 分量的信号，确保 H5K 可检测到有意义的幅度
    t = np.arange(int(fs * 0.5)) / fs
    y = (
        0.8 * np.sin(2 * np.pi * f0 * t)
        + 0.2 * np.sin(2 * np.pi * 5000.0 * t)
    ).astype(float)
    frameshift_ms = 10
    nframes = int(0.5 * 1000 / frameshift_ms)
    F0 = np.full(nframes, f0, dtype=float)
    Nperiods = 3
    h5k = compute_harmonic_at_fixed_freq(y, fs, frameshift_ms, F0, Nperiods, 5000.0)
    assert h5k.shape[0] == nframes
    # 至少一半帧应为非 NaN（避免只在最后一帧有值）
    assert np.count_nonzero(~np.isnan(h5k)) >= nframes // 2


def test_read_wav_mono_float_normalizes_and_mono(tmp_path: Path):
    fs = 16000
    t = np.arange(fs // 10) / fs
    mono = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    stereo = np.stack([mono, 0.2 * mono], axis=1)
    from scipy.io import wavfile
    p1 = tmp_path / "test_int16.wav"
    wavfile.write(str(p1), fs, np.int16(np.clip(mono, -1.0, 1.0) * 32767))
    p2 = tmp_path / "test_float32_stereo.wav"
    wavfile.write(str(p2), fs, stereo.astype(np.float32))
    fs1, y1 = read_wav_mono_float(p1)
    fs2, y2 = read_wav_mono_float(p2)
    assert fs1 == fs and fs2 == fs
    assert y1.ndim == 1 and y2.ndim == 1
    assert np.max(np.abs(y1)) <= 1.0 + 1e-6
    assert np.max(np.abs(y2)) <= 1.0 + 1e-6


def test_compute_shrp_f0_shape(tmp_path: Path):
    fs = 16000
    t = np.arange(fs // 2) / fs
    y = (0.7 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    from scipy.io import wavfile
    p = tmp_path / "test_shrp.wav"
    wavfile.write(str(p), fs, np.int16(np.clip(y, -1.0, 1.0) * 32767))
    res = compute_shrp_f0(p, frameshift_ms=10, min_f0=40.0, max_f0=500.0, shr_threshold=0.4)
    f0 = res.get("shrF0")
    assert isinstance(f0, np.ndarray)
    assert f0.shape[0] == int(round(y.size / fs * 1000.0 / 10.0))
