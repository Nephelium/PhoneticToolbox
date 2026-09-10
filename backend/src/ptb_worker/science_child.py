"""M01 scientific work inside a memory-limited Windows Job; no DB/HTTP/Qt."""
import os
import sys
import json
import struct


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    from .io.audio import decode_wav
    from .io.annotations import decode_textgrid
    from .io.lip import decode_lip
    from .io.limits import Limits,LimitError
    from .io.parameter_exports import table_from_frame,_build_pair
    from .managed_scratch import ReservedNativeScratch
    from .native.reaper import Reaper,REAPER_SHA256
    from .acoustic_result import to_core_config,build_acoustic_result
    from .segmentation import digest
    from phonetic_core.catalog import PARAMETER_MAPPING
    from phonetic_core.models.associations import AcousticAssociations
    from phonetic_core.ports.acoustic import AcousticBackends
    from phonetic_core.services.acoustic import analyze_audio
    from ptb_api.acoustic_models import AcousticRequest,AcousticInputSnapshot
    limits=Limits(input_bytes=64_000_000,output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=240)
    with open(sys.argv[1],'rb') as handle:
        header=json.loads(handle.readline(16384));request=AcousticRequest.model_validate(header['request'])
        blobs={}
        for item in header['inputs']:
            limit=limits.input_bytes if item['role']=='audio' else limits.text_bytes
            n=item['size_bytes']
            if type(n)!=int or not 0<n<=limit:raise LimitError('scientific_input_budget')
            raw=handle.read(n)
            if len(raw)!=n or digest(raw)!=item['sha256']:raise ValueError('Input changed')
            blobs[item['role']]=raw
        if handle.read(1):raise ValueError('Trailing input')
    audio=decode_wav(blobs['audio'],limits)
    associations=AcousticAssociations(tiers=decode_textgrid(blobs['textgrid'],limits) if 'textgrid' in blobs else (),
                                     lip=decode_lip(blobs['lip']) if 'lip' in blobs else None)
    inputs=[AcousticInputSnapshot(asset_id=i['id'],role=i['role'],sha256=i['sha256'],expires_at=i['expires_at']) for i in header['inputs']]
    with ReservedNativeScratch(header['native_scratch'],16_000_000) as scratch:
        native=Reaper(header['reaper_binary'],scratch,limits) if request.config.backend_policy.reaper!='disabled' else None
        result=analyze_audio(audio,to_core_config(request.config),associations,AcousticBackends(reaper=native))
        wire=build_acoustic_result(result,audio,request,inputs,native_sha256=REAPER_SHA256 if native else None)
        raw=wire.model_dump_json().encode()
        if len(raw)>16_000_000:raise LimitError('scientific_result_budget')
        table=table_from_frame(result.to_dataframe().rename(columns=PARAMETER_MAPPING),limits)
        pair=_build_pair(table,limits)
    payloads=[pair.xlsx,pair.sqlite,raw]
    names=[('result.xlsx','xlsx'),('result.ptb.sqlite','sqlite'),('result.ptb.json','json')]
    files=[dict(name=name,format=kind,size_bytes=len(blob),sha256=digest(blob)) for (name,kind),blob in zip(names,payloads)]
    manifest=dict(kind='prepared_analysis',files=files,audio_sha256=digest(blobs['audio']))
    encoded=json.dumps(manifest,separators=(',',':')).encode()
    if 8+len(encoded)+sum(map(len,payloads))>limits.output_bytes:raise LimitError('scientific_output_budget')
    with open(sys.argv[2],'wb',buffering=0) as output:
        output.write(struct.pack('<Q',len(encoded))+encoded)
        for raw in payloads:
            for offset in range(0,len(raw),65536):output.write(raw[offset:offset+65536])


if __name__=='__main__':
    try:main()
    except Exception:
        # Fixed code only; annotation text and private paths never go to logs.
        encoded=b'{"error":"invalid_segment_input"}'
        try:
            with open(sys.argv[2],'wb',buffering=0) as out:out.write(struct.pack('<Q',len(encoded))+encoded)
        except Exception:raise SystemExit(2) from None
