import pytest
import soundfile as sf
from ptb_desktop.file_provider import FileProvider,FileAccessError
from ptb_desktop.task_bridge import TaskBridge

def test_duration_preflight_rejects_extra_sample_before_upload(tmp_path):
    class Service:
        calls=[]
        def import_stream(self,stream,name,role,size):
            self.calls.append((name,role,size));assert stream.tell()==0
            return {'asset_id':'reference','sha256':'a'*64}
    for name,count in [('exact.wav',8000*1800),('over.wav',8000*1800+1)]:
        with sf.SoundFile(tmp_path/name,'w',samplerate=8000,channels=1,subtype='PCM_16') as target:
            target.seek(count-1);target.write([0.])
    provider=FileProvider();grant=provider.choose('input',lambda:tmp_path);files=provider.list(grant['id'])
    bridge=TaskBridge(provider,Service());exact=next(f for f in files if f['name']=='exact.wav');over=next(f for f in files if f['name']=='over.wav')
    bridge.stream_input(exact['id'],'audio');assert len(bridge.service.calls)==1
    with pytest.raises(FileAccessError,match='30 分钟'):bridge.stream_input(over['id'],'audio')
    assert len(bridge.service.calls)==1
    bridge.stream_input(exact['id'],'audio');assert len(bridge.service.calls)==1
    with (tmp_path/'exact.wav').open('ab') as handle:handle.write(b'changed')
    with pytest.raises(FileAccessError,match='变化'):bridge.stream_input(exact['id'],'audio')
