"""M16-R4: active Windows endpoints, stable IDs and safe device refresh."""
import hashlib
import json
import pytest
from ptb_desktop.recording.devices import devices, validate_config
from ptb_desktop.recording.service import RecordingService


class Backend:
    disconnected=set()
    def query_hostapis(self):return [{'name':'MME'},{'name':'Windows WASAPI'},{'name':'Windows WDM-KS'}]
    def query_devices(self):
        return [dict(name=name,hostapi=host,max_input_channels=ins,max_output_channels=outs,default_samplerate=48000)
                for name,host,ins,outs in [('default mapper',0,2,0),('same name',0,2,0),
                    ('same name',1,2,0),('same name',1,2,0),('speakers',1,0,2),
                    ('disconnected bluetooth',2,1,0),('empty endpoint',1,0,0),('stale endpoint',1,2,0)]]
    def check_input_settings(self,**kwargs):
        if kwargs['device'] in self.disconnected|{7}:raise ValueError('unavailable')
    def check_output_settings(self,**kwargs):pass


def test_active_endpoints_no_driver_aliases_or_name_based_merging():
    result=devices(Backend())
    assert [d['index'] for d in result]==[2,3,4]
    assert len({d['id'] for d in result})==3
    # Preserve original hashes so filtering never renumbers native devices.
    item={k:v for k,v in result[0].items() if k!='id'}
    assert result[0]['id']==hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()[:24]
    assert validate_config({'device':result[1]['id']},Backend())['device_index']==3


def test_disconnected_selected_device_cannot_silently_switch():
    backend=Backend();selected=devices(backend)[0]
    backend.disconnected={2}
    assert [d['index'] for d in devices(backend)]==[3,4]
    with pytest.raises(ValueError,match='刷新'):
        validate_config({'device':selected['id']},backend)


def test_refresh_refused_during_capture_without_touching_stream():
    service=RecordingService(backend=Backend());capture=object();service.capture=capture
    with pytest.raises(ValueError,match='先停止'):
        service.dispatch({'op':'devices'})
    assert service.capture is capture


def test_empty_wasapi_does_not_fall_back_to_disconnected_legacy_pins():
    backend=Backend();backend.query_devices=lambda:[dict(name='old bluetooth',hostapi=2,max_input_channels=1,max_output_channels=0,default_samplerate=48000)]
    assert devices(backend)==[]
