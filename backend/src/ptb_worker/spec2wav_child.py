"""Decode, calibrate and reconstruct in an owned, memory/time-limited child."""
import os
import sys
import json
import struct


def source(raw, settings):
    import io
    import cv2
    import numpy as np
    import soundfile as sf
    from phonetic_core.spec2wav.editing import rectify, analyze_audio
    if settings.mode == 'audio_draw':
        if not 0 < len(raw) <= 64_000_000:raise ValueError('spectrogram_budget')
        try:
            with sf.SoundFile(io.BytesIO(raw)) as handle:
                if not 8000 <= handle.samplerate <= 96000 or not 1 <= handle.channels <= 2 or not 2 <= handle.frames <= handle.samplerate*30:
                    raise ValueError('invalid_spectrogram_audio')
                audio=handle.read(dtype='float64',always_2d=True);sr=handle.samplerate
        except (RuntimeError,ValueError):raise ValueError('invalid_spectrogram_audio') from None
        _,_,gray,meta=analyze_audio(audio,sr,channel=settings.channel,n_fft=settings.audio_fft,dynamic_range=settings.dynamic_range)
        return gray,meta,(audio,sr)
    if not 0<len(raw)<=16_000_000:raise ValueError('spectrogram_budget')
    try:gray=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    except cv2.error:raise ValueError('invalid_spectrogram_image') from None
    if gray is None:raise ValueError('invalid_spectrogram_image')
    gray=rectify(gray,[p.model_dump() for p in settings.corners] if settings.corners else None)
    if settings.invert_image:gray=255-gray
    return gray,dict(width=gray.shape[1],height=gray.shape[0]),None


def preview(raw,config):
    import base64
    import cv2
    import numpy as np
    from ptb_api.spec2wav_models import Spec2WavConfig
    from .segmentation import digest
    settings=Spec2WavConfig.model_validate(config)
    gray,meta,_=source(raw,settings)
    if settings.mode!='audio_draw' and settings.auto_parameters:
        from .m09_parameters import predict
        meta['prediction']=predict(gray,settings.model_dump())
    png=cv2.imencode('.png',np.rint(gray).astype(np.uint8))[1].tobytes()
    return dict(image_base64=base64.b64encode(png).decode('ascii'),source_sha256=digest(raw),**meta)


def prepare(raw,config):
    import io
    import cv2
    import numpy as np
    import soundfile as sf
    from phonetic_core.spec2wav import reconstruct
    from phonetic_core.spec2wav.editing import paint, edit_audio
    from ptb_api.spec2wav_models import Spec2WavConfig
    from .segmentation import digest
    from .acoustic_errors import AcousticFailure
    settings=Spec2WavConfig.model_validate(config)
    try:
        gray,_,audio=source(raw,settings)
        strokes=[s.model_dump() for s in settings.strokes]
        if audio is not None:
            result=edit_audio(*audio,strokes,channel=settings.channel,n_fft=settings.audio_fft,dynamic_range=settings.dynamic_range)
            gray=result['target']
        else:
            if strokes:gray=np.rint(paint(gray,strokes)[0]).astype(np.uint8)
            if settings.algorithm=='neural_pipeline':
                from .vocoder_components import pipeline
                result=pipeline(gray,settings.model_dump())
            else:
                applied=settings.model_dump()
                prediction=None
                if settings.auto_parameters:
                    from .m09_parameters import predict
                    prediction=predict(gray,applied)
                    for key in ('win_length_ms','min_db','max_db'):applied[key]=prediction[key]
                args={k:v for k,v in applied.items() if k not in {'corners','mode','invert_image','vocoder','strokes','channel','audio_fft','dynamic_range','algorithm','auto_parameters','frequency_axis','reconstruction_profile','inference_device','phase_initialization'}}
                if settings.reconstruction_profile=='calibrated' or prediction or settings.frequency_axis=='log' or settings.phase_initialization!='auto':
                    from phonetic_core.spec2wav.adaptive import reconstruct_calibrated
                    result=reconstruct_calibrated(gray,frequency_axis=settings.frequency_axis,phase_initialization=settings.phase_initialization,**args)
                else:result=reconstruct(gray,**args)
                result['metadata']['phase_method']='griffin-lim'
                if prediction:
                    result['metadata'].update(prediction=prediction,applied_config=applied,
                                              parameter_backend='builtin-numpy-float32')
            if settings.vocoder=='bigvgan' and settings.algorithm!='neural_pipeline':
                from .vocoder_components import apply
                result=apply(result,device=settings.inference_device)
            from phonetic_core.spec2wav.adaptive import auto_gain
            result['audio'],gain=auto_gain(result['audio'])
            gain['output_gain']*=result['metadata'].get('output_gain',1.)
            result['metadata'].update(gain)
            result['metadata']['pcm16_clipped_samples']=0
            from phonetic_core.spec2wav.adaptive import comparison_image
            result['image']=comparison_image(result['audio'],result['sr'],result['metadata'],settings.model_dump(),gray.shape)
            result['metadata']['comparison_grid']=dict(shape=list(gray.shape),freq_start=settings.freq_start,
                freq_end=settings.freq_end,frequency_axis=settings.frequency_axis,time_start=settings.time_start,time_end=settings.time_end)
            if settings.mode=='image_draw':result['metadata']['schema_version']='m09-image-draw/1'
    except ValueError as exc:raise AcousticFailure(str(exc)) from None
    wav=io.BytesIO();sf.write(wav,result['audio'],result['sr'],format='WAV',subtype='FLOAT' if audio is not None else 'PCM_16')
    from ptb_api.protocol_version import CORE_VERSION
    metadata=result['metadata']|dict(config=settings.model_dump(),image_sha256=digest(raw),source_ids=['REF-GRIFFINLIM','M01-PY-NUMPY','M01-PY-SCIPY','PKG-OPENCV-CONTRIB-PYTHON','PKG-SOUNDFILE'],core_version=CORE_VERSION,image_shape=list(gray.shape))
    if settings.vocoder=='bigvgan' or settings.algorithm=='neural_pipeline':metadata['source_ids'].append('PKG-BIGVGAN-BASE')
    if result['metadata'].get('phase_initialization')=='pulse':metadata['source_ids'].append('WEBSITE-AOKI-SHORT-WINDOWS')
    metadata.update(source_sha256=digest(raw),source_kind='audio' if audio is not None else 'image',geometry='perspective-four-corners' if settings.corners else 'full-source',stroke_count=len(strokes))
    blobs=[cv2.imencode('.png',gray)[1].tobytes(),cv2.imencode('.png',result['image'])[1].tobytes(),wav.getvalue(),json.dumps(metadata,allow_nan=False).encode()]
    names=[('calibrated.png','png'),('reconstructed.png','png'),('reconstructed.wav','wav'),('reconstruction.ptb.json','json')]
    manifest=dict(kind='prepared_spec2wav',image_sha256=digest(raw),files=[dict(name=n,format=f,size_bytes=len(b),sha256=digest(b)) for (n,f),b in zip(names,blobs)])
    header=json.dumps(manifest).encode()
    return struct.pack('<Q',len(header))+header+b''.join(blobs)


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']='25000000'
    os.environ['OPENCV_IO_MAX_IMAGE_WIDTH']='1000000'
    os.environ['OPENCV_IO_MAX_IMAGE_HEIGHT']='1000000'
    from .segmentation import digest
    with open(sys.argv[1],'rb') as handle:
        header=json.loads(handle.readline(1_000_000));raw=handle.read(64_000_001)
    if len(raw)>64_000_000 or digest(raw)!=header['sha256']:raise ValueError('Input changed')
    payload=prepare(raw,header['config'])
    with open(sys.argv[2],'wb',buffering=0) as handle:
        for offset in range(0,len(payload),65536):handle.write(payload[offset:offset+65536])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        from .acoustic_errors import public_error
        header=json.dumps({'error':public_error(exc)}).encode()
        try:
            with open(sys.argv[2],'wb',buffering=0) as out:out.write(struct.pack('<Q',len(header))+header)
        except Exception:raise SystemExit(2) from None
