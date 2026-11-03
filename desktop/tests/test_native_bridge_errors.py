"""The real native dispatch retains raw diagnostics and known display codes."""
import io,json
from types import SimpleNamespace
from urllib.error import HTTPError
import pytest
from ptb_desktop.file_provider import FileAccessError
from ptb_desktop.native_i18n import NativeLocale,native_error_payload
from ptb_desktop.task_bridge import TaskBridge


@pytest.mark.parametrize('operation',['reconstruct','reconstruction_preview'])
@pytest.mark.parametrize('code,key',[
    ('invalid_spectrogram_audio','invalidAudio'),('invalid_image_corners','invalidCorners'),
    ('spectrogram_budget','inputBudget'),('preview_busy','previewBusy'),
    ('preview_timeout','previewTimeout'),('input_unavailable','inputUnavailable'),
    ('unknown-server-code','invalidInput'),({'unexpected':'detail'},'invalidInput'),
])
def test_reconstruction_failure_descriptor_and_raw_error_survive_native_dispatch(operation,code,key):
    calls=[]
    def request(endpoint,method,body):
        calls.append((endpoint,method,body))
        raise HTTPError(endpoint,422,'controlled failure',{},io.BytesIO(json.dumps({'detail':code}).encode()))
    bridge=TaskBridge.__new__(TaskBridge)
    bridge.provider=SimpleNamespace(read=lambda _: (b'owned unit input','sha'),entries={'input':SimpleNamespace(name='原始图-{V}.png')})
    bridge.service=SimpleNamespace(import_input=lambda data,name,role:{'name':name,'role':role},request=request)
    config={'mode':'image','duration':2,'sample_rate':16000}
    with pytest.raises(FileAccessError) as caught:bridge.invoke({'op':operation,'id':'input','key':'literal-idempotency-key','config':config})
    language=NativeLocale();message_key='errors.reconstruction.'+key
    payload=json.loads(json.dumps(native_error_payload(caught.value),ensure_ascii=False))
    assert payload=={'ok':False,'error':language.text(message_key),'display':{'key':'native.'+message_key,'values':{}}}
    assert len(calls)==1
    assert calls[0][2]['config'] is config
    assert calls[0][2]['idempotency_key']=='literal-idempotency-key'
    assert calls[0][0].endswith('/preview' if operation=='reconstruction_preview' else '/create')
    for locale in ('en','ja'):
        language.select(locale)
        assert language.text(message_key)!=payload['error']
        assert language.error(payload['error'])==language.text(message_key)
