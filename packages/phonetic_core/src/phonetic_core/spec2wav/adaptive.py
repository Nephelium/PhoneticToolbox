"""Calibrated image GL with quarter-window hops and Aoki pulse initialization.

Method: Aoki (2023), doi:10.1250/ast.44.186. Parabolic peak timing and
linear fractional-sample pulse deposition follow the local research adapter.
The 15 ms switch is a product policy, not a threshold asserted by the paper.
"""
import math
import numpy as np
from scipy.signal import find_peaks, resample_poly
from scipy.signal.windows import hann
from .reconstruction import window_pad

# Perraudin, Balazs & Søndergaard (2013), independently of Aoki initialization.
GL_MOMENTUM = 0.99


def auto_gain(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError('invalid_spectrogram_result')
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    gain = min(.08 / rms, .95 / peak) if rms > 1e-12 else 1.
    return audio * gain, dict(output_gain=gain, target_rms=.08,
                             peak_limit=.95, output_rms=rms*gain, output_peak=peak*gain)


def pulse_source(amplitude, times, sr, length):
    projection = np.asarray(amplitude, np.float64).sum(axis=0)
    peaks, _ = find_peaks(projection)
    peaks = peaks[projection[peaks] > max(1e-12, projection.max()*1e-6)]
    left, mid, right = projection[peaks-1], projection[peaks], projection[peaks+1]
    denominator = left-2*mid+right
    offset = np.divide(.5*(left-right), denominator, out=np.zeros_like(mid), where=np.abs(denominator)>1e-20)
    offset = np.clip(offset, -.5, .5)
    positions = (times[peaks]+offset*(times[peaks+1]-times[peaks-1])/2)*sr
    heights = mid-.25*(left-right)*offset
    indices = np.floor(positions).astype(int); fraction = positions-indices
    valid = (indices>=0)&(indices+1<length)
    source = np.zeros(length)
    np.add.at(source, indices[valid], heights[valid]*(1-fraction[valid]))
    np.add.at(source, indices[valid]+1, heights[valid]*fraction[valid])
    source /= max(float(np.abs(source).max()), 1e-12)
    return source, int(valid.sum())


def reconstruct_calibrated(gray, *, time_start=0., time_end=1., freq_start=0., freq_end=11025.,
                           min_db=-60., max_db=0., win_length_ms=10., n_iter=32,
                           target_sr=44100, seed=0, frequency_axis='linear', phase_initialization='auto'):
    gray = np.asarray(gray)
    if gray.ndim!=2 or gray.dtype!=np.uint8 or min(gray.shape)<2 or gray.size>1_000_000:
        raise ValueError('invalid_spectrogram_image')
    values=(time_start,time_end,freq_start,freq_end,min_db,max_db,win_length_ms)
    if not all(np.isfinite(v) for v in values) or not (0<=time_start<time_end<=time_start+30 and
            0<=freq_start<freq_end<=48000 and -160<=min_db<max_db<=20 and 0<win_length_ms<=160):
        raise ValueError('invalid_spectrogram_config')
    if frequency_axis not in ('linear','log') or (frequency_axis=='log' and freq_start<=0):
        raise ValueError('invalid_spectrogram_config')
    if type(n_iter)!=int or not 1<=n_iter<=256:
        raise ValueError('invalid_spectrogram_config')
    if phase_initialization not in ('auto','pulse','random'):
        raise ValueError('invalid_spectrogram_config')
    # Multiples of 4 kHz represent integer milliseconds and quarter hops exactly.
    sr=max(24000, int(math.ceil(2*freq_end/4000))*4000)
    milliseconds=max(1, math.floor(win_length_ms+.5))
    win=milliseconds*sr//1000; hop=win//4
    nfft=max(1024, 1 << (win-1).bit_length())
    length=round((time_end-time_start)*sr)
    if length<2 or nfft>16384:raise ValueError('spectrogram_budget')
    frames=length//hop+1
    if (nfft//2+1)*frames>4_000_000 or (nfft//2+1)*frames*n_iter>256_000_000:
        raise ValueError('spectrogram_budget')
    amplitude=10**((max_db-np.flipud(gray)/255*(max_db-min_db))/20.)
    frequencies=(np.geomspace if frequency_axis=='log' else np.linspace)(freq_start,freq_end,len(gray))
    times=np.linspace(0,time_end-time_start,gray.shape[1])
    fft_f=np.arange(nfft//2+1)*sr/nfft; fft_t=np.arange(frames)*hop/sr
    rows=np.stack([np.interp(fft_f,frequencies,c,left=0,right=0) for c in amplitude.T],axis=1)
    magnitude=np.stack([np.interp(fft_t,times,row) for row in rows]);magnitude[0]=0
    window=window_pad(hann(win,sym=False),nfft)
    def forward(y):
        padded=np.pad(y,(nfft//2,nfft//2),mode='reflect')
        view=np.lib.stride_tricks.sliding_window_view(padded,nfft)[::hop]
        return np.fft.rfft(view*window,axis=1).T
    out_length=nfft+hop*(frames-1)
    weight=np.zeros(out_length)
    for i in range(frames):weight[i*hop:i*hop+nfft]+=window**2
    def inverse(spec):
        chunks=np.fft.irfft(spec.T,n=nfft,axis=1)*window
        y=np.zeros(out_length)
        for i,chunk in enumerate(chunks):y[i*hop:i*hop+nfft]+=chunk
        y/=np.maximum(weight,1e-12)
        return np.pad(y[nfft//2:nfft//2+length],(0,max(0,nfft//2+length-len(y))))
    phase=np.exp(2j*np.pi*np.random.RandomState(seed).rand(*magnitude.shape))
    pulse=phase_initialization=='pulse' or (phase_initialization=='auto' and milliseconds<15); count=0
    if pulse:
        excitation,count=pulse_source(amplitude,times,sr,length)
        initial=forward(excitation); mask=np.abs(initial)>1e-12
        phase[mask]=initial[mask]/np.abs(initial[mask])
    old=np.zeros_like(phase)
    for _ in range(n_iter):
        rebuilt=forward(inverse(magnitude*phase))
        accelerated=rebuilt-(GL_MOMENTUM/(1+GL_MOMENTUM))*old
        phase=accelerated/np.maximum(np.abs(accelerated),1e-10);old=rebuilt
    native=inverse(magnitude*phase)
    output_sr=target_sr or math.ceil(2*freq_end)
    factor=math.gcd(sr,output_sr)
    audio=resample_poly(native,output_sr//factor,sr//factor) if sr!=output_sr else native
    audio=audio[:round((time_end-time_start)*output_sr)]
    audio,gain=auto_gain(audio)
    db=20*np.log10(np.maximum(np.abs(forward(native)),1e-10));db-=db.max()
    image=np.flipud(np.uint8(np.clip(-db/80*255,0,255)))
    return dict(audio=audio,sr=output_sr,image=image,metadata=dict(schema_version='m09-calibrated/1',
        sample_rate=output_sr,samples=len(audio),duration=len(audio)/output_sr,requested_duration=time_end-time_start,
        time_origin=time_start,native_sr=sr,n_fft=nfft,win_length=win,window_ms=milliseconds,hop_length=hop,
        hop_ms=milliseconds/4,n_iter=n_iter,seed=seed,phase_initialization='pulse' if pulse else 'random',
        pulse_count=count,pulse_threshold_ms=15,requested_phase_initialization=phase_initialization,
        threshold_basis='applied_integer_window_ms' if phase_initialization=='auto' else 'user_choice',momentum=GL_MOMENTUM,
        frequency_axis=frequency_axis,magnitude_mapping='20log10-relative-amplitude',
        phase_method='griffin-lim',pcm16_clipped_samples=0,**gain))


def comparison_image(audio,sr,metadata,config,shape):
    """Resample the measured output spectrum onto the input's physical axes."""
    height,width=shape
    native_sr=metadata['native_sr'];nfft=metadata['n_fft'];hop=metadata['hop_length']
    factor=math.gcd(sr,native_sr)
    native=resample_poly(audio,native_sr//factor,sr//factor) if sr!=native_sr else np.asarray(audio)
    window=window_pad(hann(metadata['win_length'],sym=not metadata['schema_version'].startswith('m09-calibrated')),nfft)
    padded=np.pad(native,(nfft//2,nfft//2),mode='reflect')
    frames=np.lib.stride_tricks.sliding_window_view(padded,nfft)[::hop]
    magnitude=np.abs(np.fft.rfft(frames*window,axis=1)).T
    frequencies=np.arange(nfft//2+1)*native_sr/nfft
    axis=np.geomspace if config.get('frequency_axis')=='log' else np.linspace
    target_f=axis(config['freq_start'],config['freq_end'],height)
    times=np.arange(magnitude.shape[1])*hop/native_sr
    target_t=np.linspace(0,config['time_end']-config['time_start'],width)
    rows=np.stack([np.interp(target_f,frequencies,col,left=0,right=0) for col in magnitude.T],axis=1)
    raster=np.stack([np.interp(target_t,times,row) for row in rows])
    db=20*np.log10(np.maximum(raster,1e-10));db-=db.max()
    return np.flipud(np.uint8(np.clip(-db/80*255,0,255)))
