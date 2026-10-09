"""M03 long-recording array ports, bounded sample caches and global references.

No file/process/GUI access. Overlap-crop filtering and per-block pitch paths
are versioned separately from the whole-recording compatibility algorithm.
"""
from collections import OrderedDict
from dataclasses import replace
import math
import numpy as np
from scipy import signal
from .model import EGGAnalysisResult
from .errors import EggError, check_cancel, cutoffs
from .events import find_gci_goi_peak_min_criterion
from ._legacy import LegacyCalculations

METHOD_VERSION = 'egg-bounded/2'
MAX_SECONDS = 1800
BLOCK_SECONDS = 20


def scan(read, frames, fs, cancel_event=None):
    peaks = np.zeros(2); sums = np.zeros(2); products = np.zeros(2)
    for first in range(0, frames, fs * BLOCK_SECONDS):
        check_cancel(cancel_event)
        last = min(frames, first + fs * BLOCK_SECONDS)
        values = np.asarray(read(first, last), dtype=np.float32)
        if values.shape != (last-first, 2) or not np.isfinite(values).all():
            raise EggError('invalid_signal')
        peaks = np.maximum(peaks, np.max(np.abs(values), axis=0))
        sums += values.sum(axis=0, dtype=np.float64)
        products += (values * (np.arange(first, last, dtype=float)/fs)[:, None]).sum(axis=0)
    mean = (frames-1)/(2*fs)
    variance = (frames*frames-1)/(12*fs*fs)
    slopes = (products/frames - mean*sums/frames)/max(variance, 1e-30)
    return dict(peaks=peaks, slopes=slopes, intercepts=sums/frames-slopes*mean)


class TimeView:
    def __init__(self, frames, fs): self.frames, self.fs = frames, fs
    def __len__(self): return self.frames
    def __getitem__(self, key):
        if isinstance(key, slice):
            a, b, step = key.indices(self.frames)
            return np.arange(a, b, step, dtype=float)/self.fs
        return (key if key >= 0 else self.frames+key)/self.fs


class SignalView:
    def __init__(self, context, channel, filtered=False):
        self.context, self.channel, self.filtered = context, channel, filtered
    def __len__(self): return self.context.frames
    def __getitem__(self, key):
        if not isinstance(key, slice):
            index=key if key>=0 else len(self)+key
            if not 0<=index<len(self):raise IndexError(index)
            return self[index:index+1][0]
        first, last, step = key.indices(len(self))
        if last-first > self.context.fs*62: return SliceView(self,first,last,step)
        if not self.filtered:
            data = self.context.read(first, last)[:, self.channel].astype(np.float32)
            peak = self.context.reference['peaks'][self.channel]
            if peak: data = data/peak*np.float32(.7)
            return data[::step]
        output = []
        size = self.context.fs*BLOCK_SECONDS
        for block in range(first//size, (last+size-1)//size):
            values = self.context.filtered_block(block, self.channel)
            origin = block*size
            output.append(values[max(0, first-origin):min(len(values), last-origin)])
        return (np.concatenate(output) if output else np.array([]))[::step]


class SliceView:
    """Lazy whole-export view. Only a bounded inner slice may materialize."""
    def __init__(self, source, first, last, step=1):
        self.source,self.first,self.last,self.step=source,first,last,step
    def __len__(self): return max(0,(self.last-self.first+self.step-1)//self.step)
    def __getitem__(self,key):
        if not isinstance(key,slice):return self.source[self.first+key*self.step]
        a,b,step=key.indices(len(self))
        return self.source[self.first+a*self.step:min(self.last,self.first+b*self.step):self.step*step]


class Context:
    def __init__(self, read, frames, fs, reference, config):
        self.read, self.frames, self.fs, self.reference, self.config = read, frames, fs, reference, config
        self.cache = OrderedDict()
        # IIR decay criterion. This is a crop approximation, not whole-file
        # filtfilt identity; low cutoff settings need more than a fixed halo.
        radius = 0.; self.filters=[]
        for cutoff, kind in ((config.highpass_cutoff, 'high'), (config.lowpass_cutoff, 'low')):
            _,poles,_ = signal.butter(4, cutoff/(fs/2), btype=kind,output='zpk')
            self.filters.append(signal.butter(4, cutoff/(fs/2), btype=kind,output='sos'))
            radius = max(radius, float(np.max(np.abs(poles))))
        if not 0 < radius < 1: raise EggError('filter_failed')
        self.padding = max(fs, int(math.ceil(math.log(1e-10)/math.log(radius))))
        # Bound pathological nearly singular settings explicitly.
        if self.padding > fs*30: raise EggError('filter_failed')

    def filtered_block(self, index, channel):
        key = (index, channel)
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        first = index*self.fs*BLOCK_SECONDS; last = min(self.frames, first+self.fs*BLOCK_SECONDS)
        left, right = max(0, first-self.padding), min(self.frames, last+self.padding)
        ref = self.reference; peak = max(ref['peaks'][channel], 1e-30)
        values = np.asarray(self.read(left, right)[:, channel], dtype=float)
        values -= ref['intercepts'][channel]+ref['slopes'][channel]*(np.arange(left, right)/self.fs)
        values *= .7/peak
        for coefficients in self.filters:values=signal.sosfiltfilt(coefficients,values)
        value = values[first-left:last-left].copy()
        self.cache[key] = value
        while len(self.cache) > 3: self.cache.popitem(last=False)
        return value


def prepare(read, frames, fs, reference, config, *, flip_channels=False):
    cutoffs(config, fs)
    context = Context(read, frames, fs, reference, config)
    egg, audio = (1, 0) if flip_channels else (0, 1)
    return EGGAnalysisResult(TimeView(frames, fs), SignalView(context, egg),
        SignalView(context, egg, True), SignalView(context, audio), fs=fs,
        file_duration=(frames-1)/fs, preprocessing_config=config, method_version=METHOD_VERSION)


def analyze_events(result, config, cancel_event=None, *, span=None):
    result = replace(result, analysis_config=config)
    events = [[], [], []]; fs = result.fs; frames = len(result.time_vector)
    begin,end=span if span is not None else (0,frames)
    for first in range(begin, end, fs*BLOCK_SECONDS):
        check_cancel(cancel_event)
        last = min(end, first+fs*BLOCK_SECONDS)
        left, right = max(0, first-fs), min(frames, last+fs)
        found = find_gci_goi_peak_min_criterion(result.egg_signal_processed[left:right], fs,
            criterion_level=config.criterion_level, peak_prominence=config.peak_prominence,
            valley_prominence=config.valley_prominence, use_local_prominence=config.auto_prominence,
            local_window_s=.2, local_hop_s=.1, min_auto_prom=config.min_auto_prominence,
            gci_method=config.gci_method, goi_method=config.goi_method, cancel_event=cancel_event)
        for target, values in zip(events, found):
            samples = np.asarray(values)*fs+left
            target.extend((samples[(samples >= first)&(samples < last)]/fs).tolist())
    result.gci_times, result.goi_times, result.peak_times = [np.unique(v).tolist() for v in events]
    LegacyCalculations()._calculate_gci_f0(result)
    return result
