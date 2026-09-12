"""Decode, calibrate and reconstruct in an owned, memory/time-limited child."""
import os
import sys
import json
import struct


def prepare(raw,config):
    import io
    import cv2
    import numpy as np
    import soundfile as sf
    from phonetic_core.spec2wav import reconstruct
    from ptb_api.spec2wav_models import Spec2WavConfig
    from .segmentation import digest
    from .acoustic_errors import AcousticFailure
    settings=Spec2WavConfig.model_validate(config)
    if not 0<len(raw)<=16_000_000:raise AcousticFailure('spectrogram_budget')
    try:gray=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    except cv2.error:raise AcousticFailure('invalid_spectrogram_image') from None
    if gray is None or min(gray.shape)<2:raise AcousticFailure('invalid_spectrogram_image')
    if gray.size>25_000_000:raise AcousticFailure('spectrogram_budget')
    if settings.corners:
        # Explicit click order TL, TR, BR, BL. Reject crossed/degenerate quads.
        points=np.float32([[p.x*(gray.shape[1]-1),p.y*(gray.shape[0]-1)] for p in settings.corners])
        edges=np.roll(points,-1,axis=0)-points
        cross=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
        if not np.all(cross>1):raise AcousticFailure('invalid_image_corners')
        width=max(int(np.linalg.norm(points[1]-points[0])),int(np.linalg.norm(points[2]-points[3])))
        height=max(int(np.linalg.norm(points[3]-points[0])),int(np.linalg.norm(points[2]-points[1])))
        if min(width,height)<2 or width*height>1_000_000:raise AcousticFailure('invalid_image_corners')
        target=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
        gray=cv2.warpPerspective(gray,cv2.getPerspectiveTransform(points,target),(width,height))
    try:result=reconstruct(gray,**settings.model_dump(exclude={'corners'}))
    except ValueError as exc:raise AcousticFailure(str(exc)) from None
    wav=io.BytesIO();sf.write(wav,result['audio'],result['sr'],format='WAV',subtype='PCM_16')
    metadata=result['metadata']|dict(config=settings.model_dump(),image_sha256=digest(raw),source_ids=['REF-GRIFFINLIM','M01-PY-NUMPY','M01-PY-SCIPY','PKG-OPENCV-CONTRIB-PYTHON','PKG-SOUNDFILE'],core_version='3.0.0a1',image_shape=list(gray.shape))
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
        header=json.loads(handle.readline(16384));raw=handle.read(16_000_001)
    if len(raw)>16_000_000 or digest(raw)!=header['sha256']:raise ValueError('Input changed')
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
