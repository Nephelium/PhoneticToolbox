"""M06-R4 source-preserving analysis/resynthesis, separate from Klatt controls.

source_ids: SRC-PYWORLD, SRC-WORLD, REF-WORLD-2016, REF-D4C-2016, SRC-PRAAT.
No file/process/device operations. Native REAPER is an explicitly injected port.
"""
import numpy as np
from ..models.audio import AudioInput
from ..acoustic.alignment import align_track_to_grid
from ..acoustic.f0_praat import compute_praat_f0_track
from .klatt.api import validate

FRAME_MS = 10.
MAX_CELLS = 1_000_000


def world_library():
    try:
        import pyworld
    except ImportError as exc:
        raise ValueError('m06_world_unavailable') from exc
    if pyworld.__version__ != '0.3.5':
        raise ValueError('m06_world_version')
    return pyworld


def f0_track(audio, method, times, bounds, *, reaper=None):
    """The selected analysis backend; preserve NaN/UV and real timestamps."""
    low,high=bounds
    if method=='reaper':
        if reaper is None:raise ValueError('m06_reaper_unavailable')
        try:track=reaper(audio,FRAME_MS/1000.,low,high,hilbert=False,no_highpass=False)
        except Exception as exc:raise ValueError('m06_reaper_failed') from exc
        values=align_track_to_grid(track.times,track.values,times);actual=track.actual_backend
    elif method=='harvest':
        if not 16000<=audio.sample_rate_hz<=48000 or not 40<=low<high<=1000:
            raise ValueError('m06_world_input_range')
        pw=world_library()
        values,source_times=pw.harvest(np.ascontiguousarray(audio.scientific_mono()),audio.sample_rate_hz,
                                     f0_floor=low,f0_ceil=high,frame_period=FRAME_MS)
        values=align_track_to_grid(source_times,np.where(values>0,values,np.nan),times)
        actual='world_harvest'
    else:
        track=compute_praat_f0_track(audio,FRAME_MS,low,high,method=method.removeprefix('praat_'))
        values=align_track_to_grid(track.times,track.values,times);actual=method
    return np.where(np.isfinite(values)&(values>0),values,0.),actual


def _input(config, audio):
    c=validate(config);y=np.ascontiguousarray(audio.scientific_mono(),dtype=np.float64)
    fs=audio.sample_rate_hz;duration=len(y)/fs
    if not .1<=duration<=10 or len(y)>480000 or not .1<=c['duration']<=10:
        raise ValueError('m06_input_budget')
    if c['render']['method']=='world' and (not 16000<=fs<=48000 or not 40<=c['f0_range'][0]<c['f0_range'][1]<=1000):
        raise ValueError('m06_world_input_range')
    if c['render']['method']=='psola' and not 40<=c['f0_range'][0]<c['f0_range'][1]<=1000:
        raise ValueError('m06_psola_f0_range')
    return c,y,fs,duration


def extract_natural(config, audio, *, reaper=None):
    c,y,fs,duration=_input(config,audio)
    times=np.arange(int(np.floor(duration*1000/FRAME_MS))+1)*FRAME_MS/1000.
    f0,actual=f0_track(AudioInput(y,fs),c['f0_method'],times,c['f0_range'],reaper=reaper)
    ratio=duration/c['duration']
    for curve in c['curves'].values():
        curve['points']=[[t*ratio,v] for t,v in curve['points']]
    c['duration']=duration;c['sample_rate']=fs;c['silence']=[];c['boundaries']=[]
    c['f0_transform']=dict(preset=None,offset_hz=0.)
    voiced=f0>0
    fill=np.interp(times,times[voiced],f0[voiced]) if voiced.any() else np.full(times.shape,np.mean(c['f0_range']))
    points=[[float(t),float(v)] for t,v in zip(times,fill) if t<=duration]
    if points[-1][0]<duration:points.append([duration,points[-1][1]])
    c['curves']['F0']=dict(points=points,override=None)
    return validate(c),dict(extraction_revision='m06-natural-extract/1',actual_f0_backend=actual,
                           f0_time_axis_s=times.tolist(),measured_f0_hz=[float(v) if v>0 else None for v in f0],
                           voiced_mask=voiced.tolist())


def _target_pitch(c,times,source_f0,target_times,source_duration):
    mapped=target_times*source_duration/c['duration']
    measured=align_track_to_grid(times,np.where(source_f0>0,source_f0,np.nan),mapped)
    voiced=np.isfinite(measured)&(measured>0)
    if c['render']['pitch']=='original':pitch=np.where(voiced,measured,0.)
    else:
        curve=c['curves']['F0'];points=np.asarray(curve['points'])
        values=np.interp(target_times,points[:,0],points[:,1]) if curve['override'] is None else np.full(target_times.shape,curve['override'])
        pitch=np.where(voiced,values,0.)
    if np.any((pitch>0)&((pitch<40)|(pitch>1000))):raise ValueError('m06_resynthesis_pitch_range')
    return np.ascontiguousarray(pitch),mapped


def _remap_frames(values,times,mapped):
    index=np.interp(mapped,times,np.arange(len(times)))
    left=np.floor(index).astype(int);right=np.minimum(left+1,len(times)-1)
    weight=(index-left)[:,None]
    return np.ascontiguousarray(values[left]*(1-weight)+values[right]*weight)


def _restore_digital_silence(result, source, fs, ratio):
    """Restore interiors of >=40 ms exact-zero spans, never gate low speech.

    Praat subtracts the whole-signal mean before PSOLA, which can make a digital
    zero pause nonzero. Keep 10 ms native boundary transitions then ramp for
    5 ms. This operation is explicit in diagnostics and does not classify UV.
    """
    edges=np.diff(np.r_[False,source==0.,False].astype(int))
    spans=[]
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if b-a<round(.04*fs):continue
        lo=min(len(result),round((a+.01*fs)*ratio));hi=min(len(result),round((b-.01*fs)*ratio))
        ramp=min(round(.005*fs*ratio),max(0,(hi-lo)//2))
        if hi<=lo or ramp<1:continue
        result[lo:lo+ramp]*=np.linspace(1,0,ramp)
        result[lo+ramp:hi-ramp]=0.
        result[hi-ramp:hi]*=np.linspace(0,1,ramp)
        spans.append([a/fs,b/fs])
    return np.asarray(spans,dtype=float).reshape(-1,2)


def resynthesize(config, audio, *, reaper=None, cancelled=lambda:False):
    """Return output, auditable diagnostics, and numeric analysis arrays.

    Spectrum ratio warps frequency, AP ratio multiplies WORLD noise amplitude
    ratios in voiced frames only. Neither parameter is an AH/HNR estimate.
    """
    c,y,fs,duration=_input(config,audio);r=c['render'];method=r['method']
    if method not in ('world','psola'):raise ValueError('m06_method_action_mismatch')
    ratio=c['duration']/duration
    if not .5<=ratio<=2 or round(c['duration']*fs)>480000:raise ValueError('m06_resynthesis_duration')
    if cancelled():raise InterruptedError('m06_cancelled')
    times=np.arange(int(np.floor(duration*1000/FRAME_MS))+1)*FRAME_MS/1000.
    target_times=np.arange(int(np.floor(c['duration']*1000/FRAME_MS))+1)*FRAME_MS/1000.
    f0,actual=f0_track(AudioInput(y,fs),c['f0_method'],times,c['f0_range'],reaper=reaper)
    target_f0,mapped=_target_pitch(c,times,f0,target_times,duration)
    arrays=dict(source_times_s=times,source_f0_hz=f0,target_times_s=target_times,target_f0_hz=target_f0)
    info=dict(computation_revision='m06-'+method+'/1',method=method,actual_f0_backend=actual,
              frame_shift_ms=FRAME_MS,input_sample_rate_hz=fs,output_sample_rate_hz=fs,
              input_duration_s=duration,output_duration_s=c['duration'],duration_ratio=ratio,
              pitch_mode=r['pitch'],spectral_ratio=r['spectral_ratio'] if method=='world' else None,
              aperiodicity_ratio=r['aperiodicity_ratio'] if method=='world' else None,
              input_channels=1 if audio.samples.ndim==1 else audio.samples.shape[1],channel_policy='arithmetic_mean',
              voiced_frames=int((f0>0).sum()),total_frames=len(f0),
              amplitude_policy='no_normalization; attenuation_only_if_peak_exceeds_0.99',
              fades_applied=False,klatt_controls_applied=False)
    if method=='world':
        pw=world_library();size=pw.get_cheaptrick_fft_size(fs,f0_floor=float(c['f0_range'][0]))
        if max(len(times),len(target_times))*(size//2+1)>MAX_CELLS:raise ValueError('m06_world_matrix_budget')
        sp=pw.cheaptrick(y,f0,times,fs,f0_floor=float(c['f0_range'][0]),fft_size=size)
        if cancelled():raise InterruptedError('m06_cancelled')
        ap=pw.d4c(y,f0,times,fs,fft_size=size)
        arrays.update(spectral_envelope_power=sp,aperiodicity_amplitude_ratio=ap)
        out_sp=_remap_frames(sp,times,mapped);out_ap=_remap_frames(ap,times,mapped)
        if r['spectral_ratio']!=1:
            bins=np.arange(sp.shape[1],dtype=float)
            out_sp=np.ascontiguousarray([np.interp(bins/r['spectral_ratio'],bins,row,left=row[0],right=row[-1]) for row in out_sp])
        voiced=target_f0>0
        out_ap[voiced]=np.clip(out_ap[voiced]*r['aperiodicity_ratio'],1e-12,1.-1e-12)
        out_ap[~voiced]=1.-1e-12
        result=pw.synthesize(target_f0,out_sp,out_ap,fs,frame_period=FRAME_MS)
        info.update(pyworld_version=pw.__version__,fft_size=size,spectral_bins=sp.shape[1],
                    spectrum_method='CheapTrick',aperiodicity_method='D4C',d4c_threshold=.85,
                    random_seed_policy='WORLD internal generator; task seed unused')
    else:
        import parselmouth
        from parselmouth.praat import call
        if r['pitch']=='curve' and not np.any(f0>0):raise ValueError('m06_psola_no_voiced')
        sound=parselmouth.Sound(y,sampling_frequency=fs)
        manipulation=call(sound,'To Manipulation',FRAME_MS/1000.,*c['f0_range'])
        # Praat's source pulses are independently estimated by To Manipulation.
        # The selected F0 track sets target pitch, never claims REAPER pulses.
        tier=call('Create PitchTier','target',0.,duration)
        source_target_f0,_=_target_pitch(c,times,f0,times*ratio,duration)
        for t,v in zip(times,source_target_f0):
            if v>0 and t<=duration:call(tier,'Add point',float(t),float(v))
        if np.any(source_target_f0>0):call([manipulation,tier],'Replace pitch tier')
        duration_tier=call('Create DurationTier','duration',0.,duration)
        call(duration_tier,'Add point',0.,ratio)
        call([manipulation,duration_tier],'Replace duration tier')
        result=call(manipulation,'Get resynthesis (overlap-add)').values[0].copy()
        info.update(parselmouth_version=parselmouth.__version__,praat_version=parselmouth.PRAAT_VERSION,
                    pulse_backend='Praat To Manipulation',praat_removed_input_mean=float(np.mean(y)),
                    voicing_policy='Praat source pulses; selected F0 controls target PitchTier only')
    if cancelled():raise InterruptedError('m06_cancelled')
    n=round(c['duration']*fs)
    if not np.isfinite(result).all():raise ValueError('m06_invalid_output')
    # WORLD rounds up to one frame; Praat differs by <=1 sample. Record adjustment.
    info['raw_output_samples']=len(result);info['output_samples']=n
    if abs(len(result)-n)>max(round(fs*FRAME_MS/1000)+2,2):raise ValueError('m06_invalid_output_length')
    result=np.pad(result[:n],(0,max(0,n-len(result))))
    if method=='psola':
        arrays['digital_silence_intervals_s']=_restore_digital_silence(result,y,fs,ratio)
        info['digital_silence_restoration']='exact-zero spans >=40ms; 10ms margins and 5ms ramps'
    peak=float(np.max(np.abs(result)));gain=min(1.,.99/peak) if peak else 1.
    info.update(output_peak_before_attenuation=peak,output_gain=gain)
    return result*gain,info,arrays
