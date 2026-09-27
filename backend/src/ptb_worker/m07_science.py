"""M07 child-side IO, F0 ports and versioned safe analysis serialization."""
import csv
import hashlib
import io
import json
import math
from dataclasses import asdict
import numpy as np
from scipy.io import wavfile
from phonetic_core.manipulation import m07_api as core
from phonetic_core.manipulation.m07_models import PhonationAnalysisConfig, PhonationAnalysisResult, PhonationGenerationConfig

MAX_SNAPSHOT=32_000_000
def encode(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8')
def sha(raw):return hashlib.sha256(raw).hexdigest()

def estimator(native=None):
    def estimate(audio,rate,config):
        # V2 F0 adapter first serializes FLOAT32 WAV. Preserve that rounding.
        rounded=np.asarray(audio,dtype=np.float32)
        if config.f0_backend=='parselmouth':
            import parselmouth
            try:
                pitch=parselmouth.Sound(rounded.astype(float),rate).to_pitch(time_step=config.f0_frame_interval_ms/1000,pitch_floor=config.min_f0_hz,pitch_ceiling=config.max_f0_hz)
            except parselmouth.PraatError:raise ValueError('m07_f0_analysis_failed') from None
            times=pitch.xs();values=pitch.selected_array['frequency']
            values=np.where(values>0,values,np.nan)
        elif config.f0_backend=='reaper':
            if native is None:raise ValueError('m07_reaper_unavailable')
            from phonetic_core.models.audio import AudioInput
            result=native(AudioInput(rounded,rate),config.f0_frame_interval_ms/1000,config.min_f0_hz,config.max_f0_hz,hilbert=False,no_highpass=False)
            times=result.times;values=result.values
        else:raise ValueError('m07_backend_unavailable')
        return core.track_to_millisecond_grid(times,values,len(audio)/rate)
    return estimate

def decode_audio(raw):
    if not 0<len(raw)<=8_000_000:raise ValueError('m07_input_budget')
    try:rate,data=wavfile.read(io.BytesIO(raw))
    except (ValueError,EOFError):raise ValueError('m07_invalid_audio') from None
    if data.ndim not in (1,2) or len(data)>480000 or (data.ndim==2 and data.shape[1]>8) or not 1000<=rate<=192000 or len(data)/rate>10:raise ValueError('m07_input_budget')
    if data.dtype==np.int16:audio=data.astype(float)/32768
    elif data.dtype==np.int32:audio=data.astype(float)/2147483648
    elif data.dtype==np.uint8:audio=(data.astype(float)-128)/128
    else:audio=data.astype(float)
    if audio.ndim==2:audio=audio.mean(axis=1)
    return int(rate),audio

def pack_analysis(item):
    return dict(sample_rate=item.sample_rate,start_sample=item.start_sample,end_sample=item.end_sample,config=asdict(item.config),**{k:getattr(item,k).tolist() for k in ('signal','residual','f0_hz','lpc_coefficients','pulses')})

def unpack_analysis(item):
    try:
        config=PhonationAnalysisConfig(**item['config'])
        result=PhonationAnalysisResult(item['sample_rate'],np.asarray(item['signal'],float),item['start_sample'],item['end_sample'],np.asarray(item['f0_hz'],float),np.asarray(item['lpc_coefficients'],float),np.asarray(item['residual'],float),np.asarray(item['pulses']),config)
        if result.pulses.dtype.kind not in 'iu':raise ValueError('m07_invalid_snapshot')
        core.validate_result(result);return result
    except (KeyError,TypeError,OverflowError):raise ValueError('m07_invalid_snapshot') from None

def csv_bytes(source,target):
    text=io.StringIO(newline='');writer=csv.writer(text,lineterminator='\n');writer.writerow(('time_ms','source_f0','target_f0'))
    for i in range(max(len(source.f0_hz),len(target.f0_hz))):writer.writerow((i,float(source.f0_hz[i]) if i<len(source.f0_hz) else 0.,float(target.f0_hz[i]) if i<len(target.f0_hz) else 0.))
    return text.getvalue().encode('utf8')

def compute(header,blobs,native=None):
    from ptb_api.m07_models import M07Request
    from importlib.metadata import version
    request=M07Request.model_validate(header['request']);c=request.analysis.model_dump()
    refs={k:request.model_dump()[k] for k in ('source','target')}
    if any(sha(blobs[k])!=refs[k]['sha256'] for k in refs):raise ValueError('m07_input_changed')
    if request.action=='analyze':
        pair=[]
        for role in ('source','target'):
            rate,audio=decode_audio(blobs[role]);pair.append(core.analyze(audio,rate,PhonationAnalysisConfig(**c),estimator(native)))
        applied=None
    else:
        raw=blobs['analysis']
        if len(raw)>MAX_SNAPSHOT:raise ValueError('m07_snapshot_budget')
        saved=json.loads(raw)
        if saved.get('schema_version')!='m07-analysis/1' or saved.get('inputs')!=refs or saved.get('analysis')!=c or saved.get('algorithm')!=core.ALGORITHM_VERSION:raise ValueError('m07_analysis_stale')
        pair=[unpack_analysis(saved[role]) for role in ('source','target')];applied=saved.get('applied_controls')
        if request.action=='apply':
            pair=core.apply_controls(*pair,request.controls.model_dump(),request.alignment)
            applied=dict(alignment=request.alignment,**request.controls.model_dump())
    source,target=pair
    metadata=dict(schema_version='m07/1',status='complete',action=request.action,inputs=refs,analysis=c,generation=request.generation.model_dump(),alignment=request.alignment,point_count=request.point_count,continuum_type=request.continuum_type,reverse_direction=request.reverse_direction,batch_id=request.batch_id,batch_group_count=request.batch_group_count,batch_group_index=request.batch_group_index,analysis_job_id=request.analysis_job_id,algorithm=core.ALGORITHM_VERSION,grid_ms=1,interval='inclusive_sample_bounds',source_ids=['SRC-ZAIWA','REF-ZAIWA','SRC-PRAAT','SRC-REAPER'],runtime={k:version(k) for k in ('numpy','scipy','praat-parselmouth')},applied_controls=applied,controls=core.controls(source,target,request.point_count,request.alignment),source_f0=source.f0_hz.tolist(),target_f0=target.f0_hz.tolist(),source_bounds=[source.start_sample,source.end_sample],target_bounds=[target.start_sample,target.end_sample],sample_rate_hz=11025,source_samples=len(source.signal),target_samples=len(target.signal))
    files=[]
    if request.action in ('analyze','apply'):
        snapshot=dict(schema_version='m07-analysis/1',inputs=refs,analysis=c,algorithm=core.ALGORITHM_VERSION,applied_controls=applied,source=pack_analysis(source),target=pack_analysis(target))
        raw=encode(snapshot)
        if len(raw)>MAX_SNAPSHOT:raise ValueError('m07_snapshot_budget')
        files.append(('analysis.m07.json',raw))
    else:
        result=core.generate(source,target,request.continuum_type,PhonationGenerationConfig(**request.generation.model_dump()),request.reverse_direction)
        metadata['sample_count']=len(result.audio_steps)
        for i in range(result.audio_steps.shape[1]):
            output=io.BytesIO();wavfile.write(output,11025,core.pcm16(result.audio_steps[:,i]));files.append((f'step{i+1:02d}.wav',output.getvalue()))
        output=io.BytesIO();wavfile.write(output,11025,core.pcm16(result.audio_steps.T.reshape(-1)));files.append(('combined_steps.wav',output.getvalue()))
    files.append(('edited_f0.csv',csv_bytes(source,target)))
    metadata['outputs']=[dict(name=n,size_bytes=len(v),sha256=sha(v),status='complete') for n,v in files]
    files.append(('m07.ptb.json',encode(metadata)))
    if sum(len(v) for _,v in files)>64_000_000:raise ValueError('m07_output_budget')
    return files
