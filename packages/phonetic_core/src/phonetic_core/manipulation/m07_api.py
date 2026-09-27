"""M07 array-only orchestration. F0 extraction is an injected port.

The numeric sequence is inherited from V2; validation and ownership are separate.
source_ids: SRC-ZAIWA, REF-ZAIWA, SRC-PRAAT, SRC-REAPER.
"""
from dataclasses import asdict
import math
import numpy as np
from .m07_models import *
from .m07_legacy import (trim_edge_silence, voiced_bounds, build_lpc_residual,
    fill_unvoiced_inside, detect_pulses, make_residual_continuum, synthesize_from_residual,
    build_f0_control_axis, sample_f0_on_axis, interpolate_f0_control_points)
from .m07_grid import resample_audio, track_to_millisecond_grid

ALGORITHM_VERSION = 'm07-v2-lpc-residual/1'
MAX_SAMPLES = 110250

def validate_config(config):
    for key,value in asdict(config).items():
        if isinstance(value,(float,int)) and not math.isfinite(value):
            raise ValueError('m07_nonfinite_parameter')
    config.validate()
    if config.target_sample_rate != 11025:raise ValueError('m07_sample_rate')
    if not (32<=config.frame_length<=1024 and 8<=config.frame_shift<=512 and 4<=config.lpc_order<=60):raise ValueError('m07_frame_budget')
    if not (20<=config.min_f0_hz<=500 and 50<=config.max_f0_hz<=1000 and .5<=config.f0_frame_interval_ms<=20):raise ValueError('m07_f0_range')
    if not (-.2<=config.negative_peak_threshold<=0 and .1<=config.pulse_inner_periods<=1.2 and .6<=config.pulse_outer_periods<=3):raise ValueError('m07_pulse_range')
    if not (-80<=config.silence_threshold_db<=-10 and 0<=config.silence_padding_ms<=80 and 0<=config.voiced_margin_ms<=120):raise ValueError('m07_silence_range')
    if config.f0_backend not in (F0Backend.PARSELMOUTH,F0Backend.REAPER):raise ValueError('m07_backend_unavailable')

def analyze(audio, source_rate, config, estimator, stop=lambda:False):
    validate_config(config)
    values=np.asarray(audio,dtype=np.float64)
    if values.ndim!=1 or not values.size or not np.isfinite(values).all():raise ValueError('m07_invalid_audio')
    if not 1000<=source_rate<=192000 or len(values)>480000 or len(values)/source_rate>10:raise ValueError('m07_input_budget')
    def check():
        if stop():raise InterruptedError('m07_cancelled')
    check();values=resample_audio(values,source_rate,config.target_sample_rate)
    if config.trim_silence:values=trim_edge_silence(values,11025,config.silence_threshold_db,config.silence_padding_ms)
    check();initial=estimator(values,11025,config)
    if not np.any(np.asarray(initial)>0):raise ValueError('m07_no_voiced_region')
    start,end=voiced_bounds(initial,11025,len(values),config.voiced_margin_ms)
    check();coeff,residual=build_lpc_residual(values,start,end,config.frame_length,config.frame_shift,config.lpc_order,config.preemphasis,config.window_name,stop)
    check();f0=fill_unvoiced_inside(estimator(residual,11025,config),11025,start,end)
    check()
    try:pulses=detect_pulses(residual,f0,11025,start,end,config.negative_peak_threshold,config.pulse_inner_periods,config.pulse_outer_periods,stop)
    except ValueError:raise ValueError('m07_insufficient_pulses') from None
    result=PhonationAnalysisResult(11025,values,start,end,f0,coeff,residual,pulses,config)
    validate_result(result)
    return result

def validate_result(item):
    validate_config(item.config)
    n=len(item.signal)
    if item.sample_rate!=11025 or not 1<=n<=MAX_SAMPLES or not 0<=item.start_sample<item.end_sample<n:raise ValueError('m07_invalid_snapshot')
    if item.signal.ndim!=1 or item.residual.shape!=(n,) or item.f0_hz.shape!=(math.ceil(n/11025*1000)+1,):raise ValueError('m07_invalid_snapshot')
    length=item.end_sample-item.start_sample+1
    frames=1 if length<=item.config.frame_length else 1+(length-item.config.frame_length+item.config.frame_shift-1)//item.config.frame_shift
    if item.lpc_coefficients.shape!=(frames,item.config.lpc_order+1):raise ValueError('m07_invalid_snapshot')
    if item.pulses.ndim!=1 or not 2<=len(item.pulses)<=n or np.any(np.diff(item.pulses)<=0) or np.any(item.pulses<item.start_sample) or np.any(item.pulses>item.end_sample):raise ValueError('m07_invalid_snapshot')
    for value in (item.signal,item.residual,item.f0_hz,item.lpc_coefficients,item.pulses):
        if not np.isfinite(value).all():raise ValueError('m07_nonfinite_result')
    if np.any(item.f0_hz<0) or np.any(item.f0_hz>1000):raise ValueError('m07_invalid_f0')

def controls(source,target,count=21,mode='normalize'):
    if type(count)!=int or not 20<=count<=200:raise ValueError('m07_control_count')
    mode=F0AlignmentMode(mode);axis=build_f0_control_axis(source.f0_hz,target.f0_hz,count,mode)
    return dict(axis=axis.tolist(),source=np.round(sample_f0_on_axis(source.f0_hz,axis,mode),3).tolist(),target=np.round(sample_f0_on_axis(target.f0_hz,axis,mode),3).tolist())

def apply_controls(source,target,points,mode):
    axis=np.asarray(points['axis'],float);left=np.asarray(points['source'],float);right=np.asarray(points['target'],float)
    if not 2<=len(axis)<=200 or axis.ndim!=1 or left.shape!=axis.shape or right.shape!=axis.shape:raise ValueError('m07_invalid_controls')
    if not all(np.isfinite(v).all() for v in (axis,left,right)):raise ValueError('m07_nonfinite_parameter')
    if np.any(np.diff(axis)<=0) or axis[0]<0 or axis[-1]>(100 if mode=='normalize' else 10000):raise ValueError('m07_invalid_time_order')
    if any(np.any(v<0) or np.any(v>1000) or not np.any(v>0) for v in (left,right)):raise ValueError('m07_invalid_f0')
    return tuple(item.copy_with_f0(interpolate_f0_control_points(axis,value,item.f0_hz,F0AlignmentMode(mode))) for item,value in ((source,left),(target,right)))

def generate(source,target,kind,generation,reverse=False,stop=lambda:False):
    validate_result(source);validate_result(target);generation.validate()
    if type(generation.step_count)!=int or not 2<=generation.step_count<=50 or not math.isfinite(generation.output_peak_limit):raise ValueError('m07_generation_budget')
    if reverse:source,target=target,source
    kind=ContinuumType(kind)
    residual=make_residual_continuum(source,target,kind,generation.step_count,generation.energy_match,stop)
    audio=synthesize_from_residual(source,residual,generation.normalize_to_source,generation.output_peak_limit,stop)
    if not np.isfinite(audio).all():raise ValueError('m07_nonfinite_result')
    return PhonationContinuumResult(source.sample_rate,audio,kind)

def pcm16(audio):
    values=np.asarray(audio,dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('m07_nonfinite_result')
    peak=float(np.max(np.abs(values))) if values.size else 0.
    if peak>1:values=values/peak
    return np.round(np.clip(values,-1,1)*32767).astype(np.int16)
