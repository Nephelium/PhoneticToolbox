"""V2 GUI/batch numerical export rules, separated from rendering and file I/O.

Source PENDING-EGG: egg_widget._save_plots and egg_batch_dialog.BatchWorker.
Sample-aligned/1 corrects the waveform endpoint, Praat grid and CSV ROI boundary.
The legacy repeated local filtering and batch CSV-only silence mask are retained.
"""
import numpy as np
from scipy import signal
from .filters import apply_highpass_filter, apply_lowpass_filter
from .metrics import calculate_cq_sq
from .errors import EggError


def batch_columns(result, *, keep_praat=True, keep_gci=True, silence_threshold=.01):
    times, cq, sq = calculate_cq_sq(result.gci_times, result.goi_times, result.peak_times)
    times = np.asarray(times if times is not None else [], dtype=float)
    columns = {'CQ': np.asarray(cq if cq is not None else [], dtype=float).copy(),
               'SQ': np.asarray(sq if sq is not None else [], dtype=float).copy()}
    for keep, key, t, v in [(keep_gci, 'F0_GCI (Hz)', result.gci_f0_times, result.gci_f0_values),
                           (keep_praat, 'F0_Praat (Hz)', result.audio_f0_times, result.audio_f0_values)]:
        if keep:
            columns[key] = np.interp(times, t, v, left=np.nan, right=np.nan) if t is not None and len(t) else np.full(len(times), np.nan)
    # Same 20-ms mean absolute amplitude, not RMS. Short arrays are rejected
    # explicitly rather than letting np.convolve('same') change frame count.
    window = int(.02 * result.fs)
    if window > len(result.audio_signal): raise EggError('invalid_audio')
    envelope = np.abs(result.audio_signal)
    if window > 0: envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
    mask = np.interp(times, result.time_vector, envelope) < silence_threshold
    for values in columns.values(): values[mask] = np.nan
    return times, columns, mask


def waveform_series(result, config, start_sample, end_sample, *, raw=False, batch=False):
    if not 0 <= start_sample < end_sample <= len(result.time_vector): raise EggError('invalid_roi')
    times = result.time_vector[start_sample:end_sample]
    audio = result.audio_signal[start_sample:end_sample]
    values = (result.egg_signal_raw if raw else result.egg_signal_processed)[start_sample:end_sample]
    if not raw and not batch:
        values = apply_lowpass_filter(apply_highpass_filter(signal.detrend(values),
            cutoff_freq=config.highpass_cutoff, fs=result.fs), cutoff_freq=config.lowpass_cutoff, fs=result.fs)
    step = max(1, len(times)//50000) if batch else 1
    return times[::step], audio[::step], values[::step]


def spectral_series(audio, fs, window_ms):
    # Exactly the V2 Matplotlib specgram default PSD calculation. Backend only
    # renders this array; it must not substitute the M01 Praat spectrogram.
    from matplotlib.mlab import specgram
    nfft = int(fs * window_ms / 1000)
    if nfft < 2 or len(audio) == 0: raise EggError('invalid_roi')
    overlap=int(nfft*.75); hop=nfft-overlap
    if len(audio)<=nfft: return specgram(audio,NFFT=nfft,Fs=fs,noverlap=overlap)
    count=1+(len(audio)-nfft)//hop
    # FFT windows are independent. Limit temporary complex matrices without
    # changing windows, overlap, PSD normalization or time/frequency grids.
    power=np.empty((nfft//2+1,count),dtype=np.float64)
    frequencies=None
    for first in range(0,count,256):
        last=min(count,first+256)
        chunk,frequencies,_=specgram(audio[first*hop:(last-1)*hop+nfft],NFFT=nfft,Fs=fs,noverlap=overlap)
        power[:,first:last]=chunk
    bins=np.arange(nfft/2,len(audio)-nfft/2+1,hop)/fs
    return power,frequencies,bins
