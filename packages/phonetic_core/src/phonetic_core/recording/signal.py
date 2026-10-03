"""Recording meters and explicitly versioned conservative STFT attenuation.

ptb-spectral-subtraction/1: periodic Hann, N=1024, hop=256, mean noise periodogram,
gain=max(0.15, sqrt(max(P-strength*N,0)/max(P,1e-20))), nearest-edge
three-bin frequency smoothing, zero-padded STFT/iSTFT and exact frame crop.
Only selected microphone channels are transformed. This is not AU's method.
"""
import numpy as np
from scipy import signal, ndimage

NFFT, HOP = 1024, 256


def audio(value):
    x = np.asarray(value, dtype=np.float32)
    if x.ndim != 2 or not 1 <= x.shape[1] <= 8 or not np.isfinite(x).all():
        raise ValueError('音频须为有限的帧×通道数组（1–8 通道）')
    return x


def meter(value, gain_db=0.0):
    x = audio(value)
    if not np.isfinite(gain_db) or not -60 <= gain_db <= 24:
        raise ValueError('数字增益须为 −60 至 +24 dB')
    if not len(x):
        return {'peak': [0.0]*x.shape[1], 'rms': [0.0]*x.shape[1], 'dc': [0.0]*x.shape[1], 'raw_clip': [0]*x.shape[1], 'digital_clip': [0]*x.shape[1], 'near_full_scale': [0]*x.shape[1], 'correlation': None}
    absolute = np.abs(x.astype(np.float64))
    corr = None
    if x.shape[1] == 2 and len(x)>2 and np.all(np.std(x, axis=0)>1e-8):
        corr = float(np.corrcoef(x.T)[0,1])
    return {'peak': absolute.max(axis=0).tolist(), 'rms': np.sqrt(np.mean(x.astype(np.float64)**2,axis=0)).tolist(),
            'dc': np.mean(x.astype(np.float64),axis=0).tolist(), 'raw_clip': np.sum(absolute>=1.0,axis=0).tolist(),
            'digital_clip': np.sum(absolute*10**(gain_db/20)>1.0,axis=0).tolist(),
            'near_full_scale': np.sum(absolute>=0.98,axis=0).tolist(), 'correlation': corr}


class DisplaySpectrum:
    """Bounded, time-sampled display windows across an entire visible interval.

    Sequential input may span disk chunks or edit boundaries. Each display cell
    represents a periodic-Hann window at its centre, not an audio resampling or
    an average of every FFT in that cell. Padding is confined to view edges.
    """
    def __init__(self, frames, sample_rate, channel=0, width=640):
        if frames < 0 or sample_rate <= 0 or not 0 <= channel < 8:
            raise ValueError('语谱范围、采样率或通道无效')
        self.frames, self.sample_rate, self.channel = int(frames), sample_rate, channel
        columns = min(max(1, int(width)), 640, (self.frames+HOP-1)//HOP)
        self.edges = np.linspace(0, self.frames, columns+1)
        self.centres = np.floor((self.edges[:-1]+self.edges[1:])/2).astype(np.int64)
        self.starts = self.centres-NFFT//2
        self.windows = np.zeros((columns, NFFT), dtype=np.float32)
        self.offset = 0

    def add(self, value):
        x = audio(value)
        if self.channel >= x.shape[1]:
            raise ValueError('语谱通道不存在')
        stop = self.offset+len(x)
        if stop > self.frames:
            raise ValueError('语谱输入超出可见范围')
        first = np.searchsorted(self.starts+NFFT, self.offset, side='right')
        last = np.searchsorted(self.starts, stop, side='left')
        for i in range(first, last):
            a, b = max(self.offset, self.starts[i]), min(stop, self.starts[i]+NFFT)
            self.windows[i, a-self.starts[i]:b-self.starts[i]] = x[a-self.offset:b-self.offset, self.channel]
        self.offset = stop

    def result(self):
        if self.offset != self.frames:
            raise ValueError('语谱输入未覆盖完整可见范围')
        if not len(self.windows):
            return {'rows': [], 'frequencies': [], 'times': [], 'time_edges': [0.0]}
        window = signal.get_window('hann', NFFT)
        magnitude = np.abs(np.fft.rfft(self.windows*window, axis=1))/window.sum()
        frequencies = np.fft.rfftfreq(NFFT, 1/self.sample_rate)
        keep = frequencies <= min(5000, self.sample_rate/2)
        return {'rows': np.maximum(-100,20*np.log10(np.maximum(magnitude[:,keep],1e-10))).tolist(),
                'frequencies': frequencies[keep].tolist(), 'times': (self.centres/self.sample_rate).tolist(),
                'time_edges': (self.edges/self.sample_rate).tolist(), 'max_frequency': min(5000,self.sample_rate/2),
                'nfft': NFFT, 'time_sampled': self.frames > len(self.windows)*HOP}


def spectrum(value, sample_rate, channel=0):
    x = audio(value)
    display = DisplaySpectrum(len(x), sample_rate, channel, width=128)
    display.add(x)
    return display.result()


def noise_profile(value):
    x = np.asarray(value,dtype=np.float32)
    if x.ndim != 1 or len(x)<3*NFFT or not np.isfinite(x).all():
        raise ValueError('噪声样本至少需要 3072 帧且均为有限值')
    _,_,z = signal.stft(x,window='hann',nperseg=NFFT,noverlap=NFFT-HOP,boundary=None,padded=False)
    power = np.abs(z).astype(np.float64)**2
    if float(power.mean())<=1e-18:
        raise ValueError('噪声样本为静音，无法估计有效噪声谱')
    energies=power.sum(axis=0)
    spread=float(np.percentile(energies,90)/max(np.percentile(energies,10),1e-20))
    return power.mean(axis=1), {'algorithm':'ptb-spectral-subtraction/1','nfft':NFFT,'hop':HOP,'window':'periodic_hann','gain_floor':0.15,'energy_ratio_p90_p10':spread,'warning':'噪声段统计变化较大，可能含语音或非定常噪声，请试听确认' if spread>6 else ''}


def denoise(value, profile, channels, strength=1.0):
    x=audio(value)
    if not 0.1<=strength<=2.0 or not np.isfinite(strength):
        raise ValueError('降噪强度须为 0.1–2.0')
    p=np.asarray(profile,dtype=np.float64)
    if p.shape!=(NFFT//2+1,) or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('噪声谱无效')
    y=x.copy()
    for ch in channels:
        if not 0<=ch<x.shape[1]:raise ValueError('处理通道不存在')
        padded=np.pad(x[:,ch],(0,max(0,NFFT-len(x))))
        _,_,z=signal.stft(padded,window='hann',nperseg=NFFT,noverlap=NFFT-HOP,boundary='zeros',padded=True)
        power=np.abs(z).astype(np.float64)**2
        gains=np.sqrt(np.maximum(power-strength*p[:,None],0)/np.maximum(power,1e-20))
        gains=ndimage.uniform_filter1d(np.maximum(0.15,gains),size=3,axis=0,mode='nearest')
        _,clean=signal.istft(z*gains,window='hann',nperseg=NFFT,noverlap=NFFT-HOP,boundary=True)
        y[:,ch]=clean[:len(x)]
    if not np.isfinite(y).all():raise ValueError('降噪产生非有限样本')
    return y


def gain_audio(value, gain_db, channels):
    x=audio(value);meter(x,gain_db);out=x.copy()
    for ch in channels:
        if not 0<=ch<x.shape[1]:raise ValueError('处理通道不存在')
        out[:,ch]=x[:,ch]*10**(gain_db/20)
    if not np.isfinite(out).all():raise ValueError('数字增益产生非有限样本')
    return out
