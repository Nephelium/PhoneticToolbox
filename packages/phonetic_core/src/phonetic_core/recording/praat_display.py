"""M16 bounded Praat display, source_id: SRC-PRAAT; no audio transformation.

The same Gaussian 5 ms analysis and grayscale as the shared spectrogram are
used at bounded centre windows. Long views sample time, not audio. A window's
effective Gaussian support is 10 ms; two extra samples ensure Praat can fit it.
"""
import numpy as np
import parselmouth
from phonetic_core.spectrogram import grayscale_pixels
from .signal import audio, HOP


class PraatDisplaySpectrum:
    def __init__(self, frames, sample_rate, channel=0, width=640):
        if frames < 0 or not 1 <= sample_rate <= 96000 or not 0 <= channel < 8:
            raise ValueError('语谱范围、采样率或通道无效')
        self.frames,self.sample_rate,self.channel=int(frames),sample_rate,channel
        columns=min(max(1,int(width)),640,(self.frames+HOP-1)//HOP)
        self.edges=np.linspace(0,self.frames,columns+1)
        self.window_frames=int(np.ceil(.01*sample_rate))+2
        self.starts=np.floor((self.edges[:-1]+self.edges[1:])/2-self.window_frames/2).astype(np.int64)
        self.windows=np.zeros((columns,self.window_frames),dtype=np.float32)
        self.offset=0

    def add(self,value):
        x=audio(value)
        if self.channel>=x.shape[1]:raise ValueError('语谱通道不存在')
        stop=self.offset+len(x)
        if stop>self.frames:raise ValueError('语谱输入超出可见范围')
        first=np.searchsorted(self.starts+self.window_frames,self.offset,side='right')
        last=np.searchsorted(self.starts,stop,side='left')
        for i in range(first,last):
            a,b=max(self.offset,self.starts[i]),min(stop,self.starts[i]+self.window_frames)
            self.windows[i,a-self.starts[i]:b-self.starts[i]]=x[a-self.offset:b-self.offset,self.channel]
        self.offset=stop

    def result(self):
        if self.offset!=self.frames:raise ValueError('语谱输入未覆盖完整可见范围')
        if not len(self.windows):return {'rows':[],'pixels':[],'frequencies':[],'times':[],'time_edges':[0.]}
        fmax=min(5000,self.sample_rate/2)
        spectra=[]
        for window in self.windows:
            sound=parselmouth.Sound(window.astype(np.float64),sampling_frequency=float(self.sample_rate))
            spec=sound.to_spectrogram(window_length=.005,maximum_frequency=fmax,time_step=1.,
                frequency_step=max(20.,fmax/250),window_shape=parselmouth.SpectralAnalysisWindowShape.GAUSSIAN)
            spectra.append(spec.values[:,0])
        power=np.asarray(spectra).T
        if not np.isfinite(power).all() or np.any(power<0):raise ValueError('Praat 语谱功率无效')
        frequencies=spec.ys()
        pixels=grayscale_pixels(power,frequencies,spec.dy)
        return {'rows':(10*np.log10(np.maximum(power,np.finfo(float).tiny))).T.tolist(),
                'pixels':pixels.T.tolist(),'frequencies':frequencies.tolist(),
                'frequency_step':float(spec.dy),'times':((self.starts+self.window_frames/2)/self.sample_rate).tolist(),
                'time_edges':(self.edges/self.sample_rate).tolist(),'max_frequency':fmax,
                'time_sampled':self.frames>len(self.windows)*HOP,'backend':'praat',
                'parselmouth_version':parselmouth.__version__,'praat_version':parselmouth.PRAAT_VERSION,
                'window_length':.005,'dynamic_range':50.,'preemphasis':6.,'display_revision':'m16-praat-display/1'}


def spectrum(value,sample_rate,channel=0):
    x=audio(value)
    display=PraatDisplaySpectrum(len(x),sample_rate,channel,width=128)
    display.add(x)
    return display.result()
