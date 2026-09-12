"""Bounded, device-independent analysis of the samples sent to the output stream."""
import threading
import numpy as np
from scipy.signal import resample_poly


class OutputHistory:
    def __init__(self, sample_rate, seconds=12):
        self.sample_rate = sample_rate
        self.data = np.zeros(round(sample_rate * seconds), dtype=np.float32)
        self.lock = threading.Lock()
        self.count = 0
        self.generation = 0

    def reset(self):
        with self.lock:
            self.count = 0
            self.generation += 1

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float32).ravel()
        with self.lock:
            start = self.count
            self.count += len(samples)
            if len(samples) > len(self.data):
                start += len(samples) - len(self.data)
                samples = samples[-len(self.data):]
            offset = start % len(self.data)
            first = min(len(samples), len(self.data) - offset)
            self.data[offset:offset+first] = samples[:first]
            self.data[:len(samples)-first] = samples[first:]

    def snapshot(self, seconds):
        with self.lock:
            n = min(self.count, round(seconds*self.sample_rate), len(self.data))
            indices = np.arange(self.count-n, self.count) % len(self.data)
            return self.data[indices].copy(), self.count/self.sample_rate, self.generation


def analyze_output(samples, sample_rate, *, seconds=3., window_ms=20., hop_ms=10., end=0., generation=0):
    """Hann-window amplitude spectrum, dBFS; zero padding does not add resolution.

    Waveform is a min/max envelope (preserves short peaks). Spectrum uses a
    low-pass decimation to 16 kHz, then centers each window at its real time.
    """
    settings = np.asarray([seconds, window_ms, hop_ms], dtype=float)
    if not np.isfinite(settings).all() or not 1 <= seconds <= 12 or not 5 <= window_ms <= 80 or not 5 <= hop_ms <= 40:
        raise ValueError('分析范围：音频窗 1–12 秒，分析窗 5–80 ms，步长 5–40 ms')
    samples = np.asarray(samples, dtype=np.float32)[-round(seconds*sample_rate):]
    if not np.isfinite(samples).all():
        raise ValueError('Non-finite output samples')
    duration = len(samples)/sample_rate
    bins = min(700, len(samples))
    edges = np.linspace(0, len(samples), bins+1, dtype=int)
    waveform = [[float(samples[a:b].min()), float(samples[a:b].max())] for a,b in zip(edges[:-1], edges[1:])]
    rate = 16000
    if sample_rate != 48000:
        raise ValueError('Output monitor requires 48 kHz output')
    y = resample_poly(samples, 1, 3) if len(samples) else np.empty(0)
    size = round(rate*window_ms/1000)
    # Limit the view to 320 time columns; expose the effective step to the user.
    hop = max(round(rate*hop_ms/1000), int(np.ceil(seconds*rate/320)))
    nfft = 1 << (size-1).bit_length()
    win = np.hanning(size)
    if len(y) >= size:
        frames = np.lib.stride_tricks.sliding_window_view(y, size)[::hop]
        mag = np.abs(np.fft.rfft(frames*win, n=nfft, axis=1))*2/win.sum()
        freqs = np.fft.rfftfreq(nfft, 1/rate)
        # Max-pool frequency bins when needed, retaining harmonic peaks.
        spectrum = mag[:, freqs <= 6000]
        group = max(1, int(np.ceil(spectrum.shape[1]/256)))
        spectrum = np.maximum.reduceat(spectrum, np.arange(0,spectrum.shape[1],group), axis=1)
        db = np.clip(20*np.log10(np.maximum(spectrum, 1e-6)), -90, 0)
        pixels = np.rint((db+90)/90*255).astype(np.uint8)
        spectral = pixels.tolist()
        frequency_step = rate/nfft*group
    else:
        spectral = []
        frequency_step = rate/nfft
    return {'generation':generation, 'end':end, 'duration':duration, 'seconds':seconds,
            'waveform':waveform, 'spectrogram':spectral,
            'frame_offset':size/(2*rate), 'hop_seconds':hop/rate,
            'frequency_step':frequency_step, 'window_ms':window_ms,
            'bin_hz':rate/nfft, 'resolution_hz':1000/window_ms, 'db_range':90}
