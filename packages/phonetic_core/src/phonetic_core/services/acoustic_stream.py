"""M01 bounded/1. Array-reader port; no files, GUI, processes or database.

Local pitch paths and zero-phase filter edges are explicitly versioned, not
claimed to be bit-identical to an unbounded whole-recording invocation.
Sources: existing M01 and PENDING-EGG cores. Original cycle values are retained.
"""
import math
import numpy as np
from ..models.audio import AudioInput
from ..models.audio_bounds import MAX_SECONDS,validate_source
from ..models.associations import AcousticAssociations
from ..acoustic.energy import compute_energy
from ..acoustic.alignment import align_track_to_grid, smooth_preserving_gaps
from ..acoustic.lip import interpolate_lip
from ..egg.config import EGGConfig
from ..egg.filters import apply_highpass_filter, apply_lowpass_filter
from ..egg.events import find_gci_goi_peak_min_criterion
from ..egg.metrics import calculate_cq_sq
from ..egg._legacy import LegacyCalculations
from ..egg.model import EGGAnalysisResult
from .acoustic import analyze_audio

REVISION = 'acoustic-bounded/1'


def layout(frames, rate, config, block_seconds=20.):
    """All ownership is by global frame index, never floating-point time joins."""
    hop = config.frameshift_ms / 1000.
    count = int(math.floor(frames / rate / hop + 1e-8))
    if count > 2_000_000: raise ValueError('m01_frame_budget')
    step = max(1, int(block_seconds / hop))
    context = max(1., 2 * config.n_periods / config.min_f0,
                  2 * max(config.windowsize_ms,config.energy_window_ms) / 1000.,
                  max(config.smooth_win_size,config.lip_smooth_win_size) * hop + .2)
    padding = int(math.ceil(context / hop))
    for first in range(0, count, step):
        last = min(count, first + step)
        origin = max(0, first-padding)
        stop = min(frames, int(round((last+padding)*hop*rate)))
        yield first, last, origin, int(round(origin*hop*rate)), stop


def gap_interpolate(times, values, target, max_gap):
    """No extrapolation or bridges across removed/invalid cycles."""
    times, values, target = map(lambda a: np.asarray(a, dtype=float), (times, values, target))
    result = align_track_to_grid(times, values, target)
    if len(times) > 1:
        right = np.searchsorted(times, target, side='left')
        between = (right > 0) & (right < len(times))
        pos = np.flatnonzero(between)
        r = right[pos]
        gap = times[r] - times[r-1]
        # Exact samples remain valid even after a long gap.
        result[pos[(gap > max_gap) & (np.abs(target[pos]-times[r]) > 1e-10)]] = np.nan
    return result


def scan(read, frames, rate, channels, config, audio_channel, progress=lambda *a: None, check=lambda: None):
    peaks = np.zeros(channels)
    sums = np.zeros(channels); products = np.zeros(channels)
    energy_max = -np.inf
    hop = config.frameshift_ms / 1000.
    blocks = list(layout(frames, rate, config))
    for n,(first,last,origin,start,stop) in enumerate(blocks):
        check(); data = np.asarray(read(start,stop), dtype=float)
        if data.ndim != 2 or data.shape != (stop-start, channels) or not np.isfinite(data).all():
            raise ValueError('m01_invalid_source')
        left = int(round(first*hop*rate))-start
        right = min(len(data), int(round(last*hop*rate))-start)
        # Include the final fractional frame in global amplitude/detrend stats.
        if n == len(blocks)-1: right = len(data)
        owned = data[left:right]
        peaks = np.maximum(peaks,np.max(np.abs(owned),axis=0))
        sums += owned.sum(axis=0)
        products += (owned * (np.arange(start+left,start+right)/rate)[:,None]).sum(axis=0)
        y = data.mean(axis=1) if audio_channel is None else data[:,audio_channel]
        energy = compute_energy(y,rate,config.frameshift_ms,np.zeros(last-origin),config.energy_window_ms)
        finite = energy[first-origin:last-origin]
        if np.isfinite(finite).any(): energy_max=max(energy_max,float(np.nanmax(finite)))
        progress('scan',(n+1)/len(blocks))
    mean_t=(frames-1)/(2*rate)
    variance_t=(frames*frames-1)/(12*rate*rate)
    slopes=(products/frames-mean_t*sums/frames)/max(variance_t,1e-12)
    return dict(peaks=peaks,slopes=slopes,intercepts=sums/frames-slopes*mean_t,
                energy_max=energy_max if np.isfinite(energy_max) else 0.)


def egg_cycles(read, frames, rate, config, options, reference, progress=lambda *a:None, check=lambda:None):
    """Deduplicate events by global sample ownership, then compute cycles once."""
    ec = EGGConfig(**{k:options[k] for k in ('highpass_cutoff','lowpass_cutoff','gci_method','goi_method','auto_prominence','peak_prominence','valley_prominence')})
    egg_channel=options['egg_channel']; audio_channel=options['audio_channel']
    peak=max(reference['peaks'][egg_channel],1e-30)
    events=[[],[],[]]
    # Event analysis uses integer sample boundaries and a one-second halo.
    step=rate*20; pad=rate
    for first in range(0,frames,step):
        check(); last=min(frames,first+step); start=max(0,first-pad); stop=min(frames,last+pad)
        y=np.asarray(read(start,stop))[:,egg_channel]
        times=np.arange(start,stop)/rate
        y=(y-reference['intercepts'][egg_channel]-reference['slopes'][egg_channel]*times)*(.7/peak)
        y=apply_lowpass_filter(apply_highpass_filter(y,ec.highpass_cutoff,rate),ec.lowpass_cutoff,rate)
        found=find_gci_goi_peak_min_criterion(y,rate,peak_prominence=ec.peak_prominence,
            valley_prominence=ec.valley_prominence,gci_method=ec.gci_method,goi_method=ec.goi_method,
            criterion_level=ec.criterion_level,use_local_prominence=ec.auto_prominence,
            local_window_s=.2,local_hop_s=.1,min_auto_prom=ec.min_auto_prominence)
        for dest,values in zip(events,found):
            # Scale-method crossings can be sub-sample. Preserve that precision.
            samples=np.asarray(values,dtype=float)*rate+start
            dest.extend(samples[(samples>=first)&(samples<last)].tolist())
        progress('egg',last/frames)
    gci,goi,peaks=[np.unique(v).astype(float)/rate for v in events]
    if len(gci)<2:
        return {'Time_s':np.array([]),'GCI_s':np.array([]),'Next_GCI_s':np.array([]),'GOI_s':np.array([]),
                'CQ':np.array([]),'SQ':np.array([]),'gF0':np.array([]),'F0_Time_s':np.array([])}
    times,cq,sq=calculate_cq_sq(gci,goi,peaks)
    empty=np.array([])
    result=EGGAnalysisResult(empty,empty,empty,empty,gci_times=gci.tolist())
    LegacyCalculations()._calculate_gci_f0(result)
    mid=(gci[:-1]+gci[1:])/2
    f0=np.full(len(mid),np.nan)
    if result.gci_f0_times is not None:
        positions=np.searchsorted(mid,result.gci_f0_times)
        f0[positions]=result.gci_f0_values
    gi=np.searchsorted(goi,gci[:-1],side='right'); opening=np.full(len(mid),np.nan)
    valid=gi<len(goi); opening[valid]=goi[gi[valid]]
    opening[opening>=gci[1:]]=np.nan
    # Preserve M03 batch's global-normalized, 20 ms mean-absolute audio mask.
    mask=np.ones(len(mid),dtype=bool);f0_mask=mask.copy(); audio_peak=max(reference['peaks'][audio_channel],1e-30)
    half=int(.02*rate)//2
    for first in range(0,frames,step):
        check();last=min(frames,first+step);start=max(0,first-half-2);stop=min(frames,last+half+2)
        y=np.abs(read(start,stop)[:,audio_channel])*(.7/audio_peak)
        envelope=np.convolve(y,np.ones(max(1,int(.02*rate)))/max(1,int(.02*rate)),mode='same')
        selected=(times>=first/rate)&(times<last/rate)
        mask[selected]=np.interp(times[selected],np.arange(start,stop)/rate,envelope)<options['silence_threshold']
        fselected=(mid>=first/rate)&(mid<last/rate)
        f0_mask[fselected]=np.interp(mid[fselected],np.arange(start,stop)/rate,envelope)<options['silence_threshold']
    cq[mask]=np.nan;sq[mask]=np.nan;f0[f0_mask]=np.nan
    return dict(Time_s=times,GCI_s=gci[:-1],Next_GCI_s=gci[1:],GOI_s=opening,CQ=cq,SQ=sq,gF0=f0,F0_Time_s=mid)


def iter_parameters(read,frames,rate,channels,config,options,backends=None,associations=None,
                    progress=lambda *a:None,check=lambda:None):
    """Yields (kind, arrays, metadata) for bounded persistence outside the core."""
    validate_source(frames,rate,channels)
    associations=associations or AcousticAssociations()
    egg=options.get('egg')
    audio_channel=options.get('audio_channel')
    if channels==1: audio_channel=0;egg=None
    if audio_channel is not None and not 0<=audio_channel<channels:raise ValueError('m01_channel_invalid')
    if egg:
        if not 0<=egg['egg_channel']<channels or egg['egg_channel']==audio_channel:raise ValueError('m01_channel_invalid')
        if egg['lowpass_cutoff']>=rate/2:raise ValueError('m01_egg_filter_range')
        egg=dict(egg,audio_channel=audio_channel)
    reference=scan(read,frames,rate,channels,config,audio_channel,progress,check)
    cycles=egg_cycles(read,frames,rate,config,egg,reference,progress,check) if egg else None
    metadata=dict(computation_revision=REVISION,duration_s=frames/rate,sample_rate_hz=rate,channels=channels,
                  audio_channel=audio_channel,egg=egg,frame_origin='global-zero/1',context_policy='overlap-crop/1',
                  pitch_policy='per-block-path/1',reference_intensity_db=reference['energy_max'],block_seconds=20.,
                  egg_revision='egg-bounded/1' if egg else None)
    metadata['egg_skipped']='mono_source' if options.get('egg') and channels==1 else None
    metadata['egg_definitions']={'CQ':'contact_duration / cycle_duration, accepted 0.05 < CQ < 0.95',
        'SQ':'(decontacting_duration - contacting_duration) / contact_duration',
        'gF0':'1 / GCI_period in Hz, legacy global MAD filter, retained values below 100 Hz',
        'times':'CQ/SQ at GCI start, gF0 at consecutive-GCI midpoint',
        'derived':'acoustic samples with interpolated GCI F0; separate gF0 columns; no extra jitter/shimmer'} if egg else None
    if cycles is not None:yield 'egg_cycles',cycles,metadata
    hop=config.frameshift_ms/1000.
    blocks=list(layout(frames,rate,config))
    for number,(first,last,origin,start,stop) in enumerate(blocks):
        check(); data=read(start,stop)
        source=data if audio_channel is None else data[:,audio_channel]
        offset=origin*hop
        def gci_track(local_times):
            return gap_interpolate(cycles['F0_Time_s'],cycles['gF0'],local_times+offset,egg['max_gap_ms']/1000.)
        result=analyze_audio(AudioInput(source,rate),config,backends=backends,cancellation=check,
            extra_f0=gci_track if egg and egg['derived'] else None,
            silence_reference_db=reference['energy_max'],selective=True)
        frame=result.to_dataframe()
        indexes=np.arange(len(frame))+origin
        owned=(indexes>=first)&(indexes<last)
        frame=frame.loc[owned].copy();frame['Time_s']=indexes[owned]*hop
        values={k:frame[k].to_numpy() for k in frame.columns}
        target=values['Time_s']
        all_times=np.arange(origin,origin+len(result.time_axis))*hop
        if egg and egg['storage']=='aligned':
            # Include halo for smoothing so chunk edges do not restart its window.
            for name in ('CQ','SQ','gF0'):
                src_t=cycles['F0_Time_s'] if name=='gF0' else cycles['Time_s']
                track=gap_interpolate(src_t,cycles[name],all_times,egg['max_gap_ms']/1000.)
                width=max(1,int(round(egg['smooth_ms']/config.frameshift_ms)))
                values[name]=smooth_preserving_gaps(track,width)[owned]
        if cycles is not None and egg['storage']=='cycles':values.pop('gF0',None)
        if associations.lip is not None:
            lip=interpolate_lip(associations.lip,all_times,smooth_win=config.lip_smooth_win_size,companion_start=associations.lip_companion_start)
            for key,value in lip.items():
                if not config.selected_parameter_keys or key in config.selected_parameter_keys:values[key]=value[owned]
        for tier in associations.tiers:
            labels=np.full(len(target),'',dtype=object)
            for interval in tier.intervals:
                left=np.searchsorted(target,interval.xmin,side='left')
                right=np.searchsorted(target,interval.xmax,side='left')
                labels[left:right]=interval.text
            values['text_'+tier.name]=labels
        progress('analysis',(number+1)/len(blocks))
        yield 'params',values,{**metadata,'backends':result.backend_events,'block':number+1,'blocks':len(blocks)}
