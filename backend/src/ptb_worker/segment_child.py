"""Trusted worker CLI. Scientific imports happen after entry into the Windows Job."""
import os
import sys
import json
import struct
from pathlib import Path


def main():
    # Per-child only; no system settings or parent environment mutation.
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    from .segmentation import digest,MAX_MANIFEST_BYTES
    from .io.limits import Limits,LimitError,FormatError
    from .io.audio import decode_wav,slice_wav
    from .io.annotations import decode_textgrid
    from .io.parameter_exports import _build_pair,validate_table
    from phonetic_core.segmentation import plan_segments,slice_parameter_table
    from ptb_api.acoustic_models import AcousticResult
    from dataclasses import replace
    request=Path(sys.argv[1])
    with request.open('rb') as handle:
        header=json.loads(handle.readline(8192));limits=Limits(**header['limits'])
        sizes=[header['audio_size'],header['grid_size'],header['parent_size']]
        if any(type(n)!=int or n<0 for n in sizes) or sizes[0]>limits.input_bytes or sizes[1]>limits.text_bytes or sizes[2]>16_000_000:
            raise LimitError('segment_input_budget')
        audio_raw,grid_raw,parent_raw=(handle.read(n) for n in sizes)
        if any(len(b)!=n for b,n in zip((audio_raw,grid_raw,parent_raw),sizes)) or handle.read(1):raise FormatError('incomplete_segment_input')
    if digest(audio_raw)!=header['audio_sha256'] or digest(grid_raw)!=header['textgrid_sha256']:raise FormatError('segment_source_changed')
    audio=decode_wav(audio_raw,limits);tiers=decode_textgrid(grid_raw,limits)
    try:plan=plan_segments(tiers,header['layer'],audio.sample_rate_hz,len(audio.samples))
    except ValueError:raise FormatError('missing_or_invalid_tier') from None
    if not plan:raise FormatError('no_labelled_segments')
    parent=None;table=None
    if parent_raw:
        parent=AcousticResult.model_validate_json(parent_raw)
        source=next(x for x in parent.metadata.inputs if x.role=='audio')
        decoded=parent.metadata.decoded
        if (source.sha256!=digest(audio_raw) or decoded.sample_rate_hz!=audio.sample_rate_hz or
            decoded.sample_count!=len(audio.samples) or decoded.channels!=(audio.samples.shape[1] if audio.samples.ndim==2 else 1) or
            decoded.sample_dtype!=str(audio.samples.dtype)):raise FormatError('parent_result_source_mismatch')
        numeric={c.key:c for c in parent.numeric};text={c.key:c for c in parent.text}
        keys=parent.column_order;rows=[]
        for i,t in enumerate(parent.times_s):
            row=[]
            for key in keys:
                if key=='Time_s':row.append(t)
                elif key in text:row.append(text[key].values[i])
                else:
                    c=numeric[key];row.append(c.values[i] if c.nonfinite[i]==0 else [None,'+Infinity','-Infinity'][c.nonfinite[i]-1])
            rows.append(row)
        table=validate_table({'columns':[numeric[k].label if k in numeric else k for k in keys],
                              'kinds':['text' if k in text else 'number' for k in keys],'rows':rows},limits)
    def clean(value,fallback):
        value=''.join(c for c in value if c.isalnum() or c in '._-').strip('.')
        return value[:36] or fallback
    stem=clean(Path(header['audio_name']).stem,'audio');layer=clean(header['layer'],'tier')
    files=[];payloads=[];segments=[];used=0
    def add(name,kind,raw,segment_index):
        nonlocal used
        used+=len(raw)
        if used+8>limits.output_bytes:raise LimitError('segment_output_budget')
        files.append({'name':name,'format':kind,'segment_index':segment_index,'size_bytes':len(raw),'sha256':digest(raw)})
        payloads.append(raw)
    for segment in plan:
        base=f'{stem}_{layer}_{clean(segment.label,"label")}_{segment.start_s:.3f}_{segment.end_s:.3f}_{segment.index+1:04d}'
        raw=slice_wav(audio,segment.start_s,segment.end_s,replace(limits,output_bytes=limits.output_bytes-used))
        restored=decode_wav(raw,limits)
        import numpy as np
        if restored.sample_rate_hz!=audio.sample_rate_hz or restored.samples.dtype!=audio.samples.dtype or not np.array_equal(restored.samples,audio.samples[segment.first:segment.last],equal_nan=True):
            raise FormatError('segment_wav_readback_mismatch')
        add(base+'.wav','wav',raw,segment.index)
        part=slice_parameter_table(table,segment,audio.sample_rate_hz) if table else None
        if part:
            pair=_build_pair(part,replace(limits,output_bytes=limits.output_bytes-used))
            add(base+'.xlsx','xlsx',pair.xlsx,segment.index);add(base+'.ptb.sqlite','sqlite',pair.sqlite,segment.index)
        segments.append({'interval_index':segment.index,'label':segment.label,'start_s':segment.start_s,'end_s':segment.end_s,
                         'first_sample':segment.first,'last_sample':segment.last,
                         'parameter_status':'included' if part else 'no_frames' if table else 'no_parent_result',
                         'parameter_rows':len(part['rows']) if part else 0})
    manifest={'kind':'prepared_segments','schema_version':'m01-segments/1','audio_sha256':digest(audio_raw),
              'textgrid_sha256':digest(grid_raw),'layer':header['layer'],'sample_rate_hz':audio.sample_rate_hz,
              'sample_dtype':str(audio.samples.dtype),'channels':audio.samples.shape[1] if audio.samples.ndim==2 else 1,
              'parent_result_sha256':digest(parent_raw) if parent_raw else None,
              'parameter_time_policy':'original_frames_rebased_to_first_sample_with_Source_Time_s',
              'reestimated':False,'segments':segments,'files':files}
    encoded=json.dumps(manifest,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    if len(encoded)>MAX_MANIFEST_BYTES or len(encoded)+8+used>limits.output_bytes:raise LimitError('segment_output_budget')
    with open(sys.argv[2],'wb',buffering=0) as output:
        output.write(struct.pack('<Q',len(encoded)));output.write(encoded)
        for raw in payloads:
            for i in range(0,len(raw),65536):output.write(raw[i:i+65536])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        from .io.limits import LimitError
        code=str(exc) if str(exc) in ('parent_result_source_mismatch','no_labelled_segments','missing_or_invalid_tier') else 'segment_budget_exceeded' if isinstance(exc,(LimitError,MemoryError)) else 'invalid_segment_input'
        # Fixed error codes only; private paths and annotation text never become logs.
        encoded=json.dumps({'error':code}).encode()
        try:
            with open(sys.argv[2],'wb',buffering=0) as output:output.write(struct.pack('<Q',len(encoded))+encoded)
        except Exception:raise SystemExit(2) from None
