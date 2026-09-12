"""Bounded array-only v2 reconstruction with task-local phase initialization."""
import numpy as np
from scipy.signal.windows import hann


def window_pad(window,n_fft):
    result=np.zeros(n_fft);start=(n_fft-len(window))//2
    result[start:start+len(window)]=window
    return result


def stft(y,n_fft,hop,window):
    padded=np.pad(y,(n_fft//2,n_fft//2),mode='reflect')
    count=1+(len(padded)-n_fft)//hop
    result=np.zeros((n_fft//2+1,count),dtype=np.complex128)
    window=window_pad(window,n_fft)
    for i in range(count):result[:,i]=np.fft.fft(padded[i*hop:i*hop+n_fft]*window)[:n_fft//2+1]
    return result


def istft(spectrum,n_fft,hop,window,length):
    frames=spectrum.shape[1];full=np.zeros((n_fft,frames),dtype=np.complex128)
    full[:n_fft//2+1]=spectrum
    if n_fft>2:full[n_fft//2+1:]=np.conj(spectrum[-2:0:-1])
    window=window_pad(window,n_fft)
    y=np.zeros(n_fft+hop*(frames-1));weight=np.zeros_like(y)
    for i in range(frames):
        y[i*hop:i*hop+n_fft]+=np.real(np.fft.ifft(full[:,i]))*window
        weight[i*hop:i*hop+n_fft]+=window**2
    y/=np.maximum(weight,1e-8)
    if len(y)>n_fft:y=y[n_fft//2:-n_fft//2]
    return np.pad(y[:length],(0,max(0,length-len(y))))


def reconstruct(gray,*,time_start=0.,time_end=1.,freq_start=0.,freq_end=11025.,min_db=-30.,max_db=0.,win_length_ms=10.,n_iter=32,target_sr=44100,seed=0):
    gray=np.asarray(gray)
    if gray.ndim!=2 or gray.dtype!=np.uint8 or min(gray.shape)<2 or gray.size>1_000_000:raise ValueError('invalid_spectrogram_image')
    values=(time_start,time_end,freq_start,freq_end,min_db,max_db,win_length_ms)
    if not all(np.isfinite(v) for v in values):raise ValueError('invalid_spectrogram_config')
    if not (0<=time_start<time_end and time_end-time_start<=30 and 0<=freq_start<freq_end<=48000 and -160<=min_db<max_db<=20 and 0<win_length_ms<=1000):raise ValueError('invalid_spectrogram_config')
    if type(n_iter)!=int or not 1<=n_iter<=128 or type(seed)!=int or not 0<=seed<=4294967295 or target_sr not in (0,8000,16000,22050,24000,32000,44100,48000,96000):raise ValueError('invalid_spectrogram_config')
    native_sr=int(2*freq_end)
    if native_sr<2:raise ValueError('invalid_spectrogram_config')
    magnitude=10**((max_db-np.flipud(gray)/255.*(max_db-min_db))/10.)*10
    # Explicit behavior correction: v2 ignored freq_start. Zero-start is exact v2.
    if freq_start:
        count=int(np.ceil((gray.shape[0]-1)*freq_end/(freq_end-freq_start)))+1
        if count>8193 or count*gray.shape[1]>1_000_000:raise ValueError('spectrogram_budget')
        grid=np.linspace(0,freq_end,count);source=np.linspace(freq_start,freq_end,gray.shape[0])
        magnitude=np.stack([np.interp(grid,source,column,left=0.) for column in magnitude.T],axis=1)
    n_fft=2*(len(magnitude)-1)
    if n_fft>16384 or magnitude.size*n_iter>32_000_000:raise ValueError('spectrogram_budget')
    hop=max(1,int((time_end-time_start)/gray.shape[1]*native_sr))
    length=hop*(gray.shape[1]-1)
    if length>3_000_000:raise ValueError('spectrogram_budget')
    win=max(1,min(int(win_length_ms/1000*native_sr),n_fft));window=hann(win,sym=False)
    spectrum=magnitude*np.exp(2j*np.pi*np.random.RandomState(seed).rand(*magnitude.shape))
    for _ in range(n_iter):
        y=istft(spectrum,n_fft,hop,window,length)
        spectrum=magnitude*np.exp(1j*np.angle(stft(y,n_fft,hop,window)))
    native=istft(spectrum,n_fft,hop,window,length)
    sr=target_sr or native_sr
    audio=native if sr==native_sr else np.interp(np.linspace(0,len(native)-1,int(len(native)/native_sr*sr)),np.arange(len(native)),native)
    if not len(audio) or not np.all(np.isfinite(audio)):raise ValueError('invalid_spectrogram_result')
    comparison=np.abs(stft(native,n_fft,hop,np.hanning(win)))
    db=10*np.log10(np.maximum(1e-10,comparison))-10*np.log10(max(1e-10,float(comparison.max())))
    db=np.maximum(db,db.max()-80)
    image=np.flipud(np.uint8(np.clip((db.max()-db)/max(1e-10,float(db.max()-db.min()))*255,0,255)))
    return dict(audio=audio,sr=sr,image=image,metadata=dict(schema_version='m09/1',seed=seed,native_sr=native_sr,
        sample_rate=sr,samples=len(audio),duration=len(audio)/sr,requested_duration=time_end-time_start,
        time_origin=time_start,hop_length=hop,n_fft=n_fft,win_length=win,n_iter=n_iter,
        frequency_mapping='legacy-zero-start' if freq_start==0 else 'band-interpolation-zero-fill-v1',
        magnitude_mapping='legacy-10log10-times10',resampling='linear-interpolation',
        pcm16_clipped_samples=int(np.count_nonzero((audio>=1)|(audio< -1))),
        warning='Magnitude-only approximate reconstruction; original phase cannot be recovered.'))
