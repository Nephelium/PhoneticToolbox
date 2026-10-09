"""M01-E user-requested read-only Praat spectrogram. source_id: SRC-PRAAT.

Real Sound.to_spectrogram analysis; bounded grayscale display, not parameter
export or calibrated SPL. Raw power and frequency arrays remain available to
independent tests; adapters remove them before serializing the display payload.
"""
import math
import numpy as np
import parselmouth


def grayscale_pixels(power, frequencies, frequency_step):
    """Shared Praat-style 50 dB range and 6 dB/oct display preemphasis."""
    if np.max(power)<=0:return np.full(power.shape,255,dtype=np.uint8)
    db=10*np.log10(np.maximum(power,np.finfo(float).tiny))
    db+=6*np.log2(np.maximum(frequencies,frequency_step/2)/1000)[:,None]
    return np.rint(255*np.clip((np.max(db)-db)/50,0,1)).astype(np.uint8)


def spectrogram_preview(audio,*,channel,start,end,width=1000):
    fs=audio.sample_rate_hz;data=audio.samples;duration=len(data)/fs
    channels=1 if data.ndim==1 else data.shape[1]
    if type(channel)!=int or not 0<=channel<channels:raise ValueError('Invalid channel')
    if not all(math.isfinite(v) for v in (start,end)) or not 0<=start<end<=duration+1e-9 or end-start<.01:
        raise ValueError('Spectrogram needs a valid window of at least 10 ms')
    if type(width)!=int or not 100<=width<=1000:raise ValueError('Invalid display width')
    first=max(0,int((start-.01)*fs));last=min(len(data),math.ceil((end+.01)*fs))
    selected=data[first:last] if data.ndim==1 else data[first:last,channel]
    values=selected.astype(np.float64)
    if data.dtype==np.uint8:values=(values-128)/128
    elif data.dtype==np.int16:values/=32768
    elif data.dtype==np.int32:values/=2147483648
    sound=parselmouth.Sound(values,sampling_frequency=float(fs),start_time=first/fs)
    fmax=min(5000.,fs/2);step=max(.002,(end-start)/width)
    spectrum=sound.to_spectrogram(window_length=.005,maximum_frequency=fmax,time_step=step,
        frequency_step=max(20.,fmax/250),window_shape=parselmouth.SpectralAnalysisWindowShape.GAUSSIAN)
    times=spectrum.xs();keep=(times>=start)&(times<=end)
    power=spectrum.values[:,keep];times=times[keep];freq=spectrum.ys()
    if not times.size or not freq.size or power.shape[1]>1002 or power.shape[0]>252:
        raise ValueError('Spectrogram grid unavailable or exceeds display budget')
    if not np.isfinite(power).all() or np.any(power<0):
        raise ValueError('Spectrogram power is not finite and nonnegative')
    pixels=grayscale_pixels(power,freq,spectrum.dy)
    return dict(start=float(start),end=float(end),frequency_max=fmax,x1=float(times[0]),dx=float(spectrum.dx),
        y1=float(spectrum.y1),dy=float(spectrum.dy),width=int(power.shape[1]),height=int(power.shape[0]),
        pixels=pixels.tobytes(),backend='praat',parselmouth_version=parselmouth.__version__,praat_version=parselmouth.PRAAT_VERSION,
        window_length=.005,dynamic_range=50.,preemphasis=6.,time_step=step,_power=power,_frequencies=freq)
