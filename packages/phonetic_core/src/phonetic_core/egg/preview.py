"""V2 egg_widget.update_zoom_plots numerical display path (PENDING-EGG).

The display uses raw EGG with 100ms padding, while local events use 50ms.
Do not replace either with the CQ or export waveform filtering policy.
"""
import numpy as np
from scipy import signal
from .filters import apply_highpass_filter, apply_lowpass_filter
from .errors import roi, cutoffs, EggError


def micro_waveforms(result, config, center, width_ms=50., *, raw=False):
    if not np.isfinite(width_ms) or not 10 <= width_ms <= 200:
        raise EggError('invalid_roi')
    start, end = center-width_ms/2000, center+width_ms/2000
    roi(max(0., start), end); cutoffs(config, result.fs)
    first = max(0, int(start*result.fs))
    last = min(len(result.time_vector), int(end*result.fs))
    if first >= last: raise EggError('invalid_roi')
    values = result.egg_signal_raw[first:last]
    if not raw:
        pad = int(result.fs*.1)
        left, right = max(0, first-pad), min(len(result.time_vector), last+pad)
        values = apply_lowpass_filter(apply_highpass_filter(
            signal.detrend(result.egg_signal_raw[left:right]),
            cutoff_freq=config.highpass_cutoff, fs=result.fs),
            cutoff_freq=config.lowpass_cutoff, fs=result.fs)[first-left:last-left]
    return result.time_vector[first:last], result.audio_signal[first:last], values


def inverse_comparison(audio, filtered, egg, fs):
    """V2 InverseFilteringResultDialog: pad first, then periodic Hamming/FFT.

    The legacy title said +/-50ms; its actual central window is 50ms total.
    Preserve samples and label that measured extent accurately in the UI.
    """
    from scipy.fft import fft, fftfreq
    from scipy.signal import windows
    size = max(len(audio), 44100)
    frequencies = fftfreq(size, 1/fs)
    take = (frequencies >= 0) & (frequencies <= 5000)
    spectra = []
    for values in (audio, filtered, egg):
        padded = np.pad(values, (0, size-len(values)), mode='constant')
        magnitude = 20*np.log10(np.abs(fft(padded*windows.hamming(size,sym=False))[take])+1e-12)
        spectra.append(np.maximum(magnitude,np.nanmax(magnitude)-80))
    center = len(audio)//2; half = int(.050*fs)//2
    first,last = max(0,center-half), min(len(audio),center+half)
    return frequencies[take], spectra, (np.arange(first,last)-center)/fs, [v[first:last] for v in (audio,filtered,egg)]
