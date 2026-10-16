"""Public astronaut synthetic videos and independent export readback."""
import json
from pathlib import Path
import sys
from fractions import Fraction
import av
import cv2
import numpy as np
import wave

def generate(directory):
    root=Path(__file__).resolve().parents[2]
    image=cv2.imread(str(root/'output/validation/m05/inputs/astronaut.png'))
    assert image is not None
    for name in ('first.mkv','second.mkv'):
        with av.open(str(directory/name),'w') as media:
            video=media.add_stream('ffv1',rate=10);video.width=image.shape[1];video.height=image.shape[0];video.pix_fmt='yuv420p'
            audio=media.add_stream('pcm_s16le',rate=8000);audio.layout='mono'
            samples=(np.sin(np.arange(8000)*2*np.pi*440/8000)*2000).astype(np.int16).reshape(1,-1)
            frame=av.AudioFrame.from_ndarray(samples,format='s16',layout='mono');frame.sample_rate=8000;frame.time_base=Fraction(1,8000);frame.pts=0
            for packet in audio.encode(frame):media.mux(packet)
            for i in range(6):
                frame=av.VideoFrame.from_ndarray(image,format='bgr24');frame.time_base=Fraction(1,10);frame.pts=i
                for packet in video.encode(frame):media.mux(packet)
            for stream in (video,audio):
                for packet in stream.encode():media.mux(packet)

def inspect(saved,out):
    bundles=[];standalone=[]
    for directory in sorted(saved.iterdir()):
        if not directory.is_dir():continue
        receipt=json.loads((directory/'saved-manifest.json').read_text('utf8'))
        files={p.name for p in directory.iterdir()}
        assert 'raw_recording.mp4' in files and 'lip-animation.mp4' in files
        assert receipt['save_options']==dict(video=True,animation=True,quality='high')
        metadata=json.loads((directory/'manifest.json').read_text('utf8'))
        frames=list(map(json.loads,(directory/'frames.jsonl').read_text('utf8').splitlines()))
        assert len(frames)==6 and all(r['detected'] for r in frames)
        original=json.loads((directory/'unaligned.lip.json').read_text('utf8'))
        aligned=json.loads((directory/'audio_recording.lip.json').read_text('utf8'))
        assert original['data']['relative_times']==aligned['data']['relative_times']
        assert aligned['data']['metadata']['lip_manual_offset']==receipt['lip_manual_offset']
        assert receipt['media']['animation']['lip_manual_offset']==receipt['lip_manual_offset']
        with av.open(str(directory/'raw_recording.mp4')) as media:
            assert len(list(media.decode(video=0)))==6
        bundles.append(dict(input=metadata['input_name'],offset=receipt['lip_manual_offset'],files=sorted(files)))
    decoded={}
    for path in saved.iterdir():
        if not path.is_file() or path.suffix not in ('.mp4','.gif'):continue
        with av.open(str(path)) as media:
            sound=len(media.streams.audio);frames=[(float(f.pts*f.time_base),f.to_ndarray(format='rgb24')) for f in media.decode(video=0)]
        assert max(frames[0][1].shape[:2])==1080
        assert sound==(1 if path.suffix=='.mp4' else 0)
        decoded[path.suffix]=frames
        image=frames[8][1];cv2.imwrite(str(out/('decoded'+path.suffix+'.png')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        green=(image[:,:,1].astype(int)>image[:,:,0].astype(int)+30)&(image[:,:,1].astype(int)>image[:,:,2].astype(int)+30)
        assert np.count_nonzero(green[270:810,270:810])>1000
        standalone.append(dict(name=path.name,frames=len(frames),audio_streams=sound,last_pts=frames[-1][0],mesh_pixels=int(np.count_nonzero(green))))
    assert len(decoded['.gif'])==len(decoded['.mp4'])
    assert abs(decoded['.gif'][-1][0]-decoded['.mp4'][-1][0])<=.01
    assert len(bundles)==4
    offsets={b['input']:b['offset'] for b in bundles if b['offset']}
    assert offsets=={'first.mkv':.125,'second.mkv':-.075}
    (out/'codec-readback.json').write_text(json.dumps(dict(bundles=bundles,standalone=standalone),ensure_ascii=False,indent=2),'utf8')

if __name__=='__main__':
    if sys.argv[1]=='generate':generate(Path(sys.argv[2]))
    else:inspect(Path(sys.argv[2]),Path(sys.argv[3]))
