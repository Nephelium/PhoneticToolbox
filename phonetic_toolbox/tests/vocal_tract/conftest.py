"""A hard test boundary: never open a real audio output device."""
import pytest
import sounddevice as sd

@pytest.fixture(autouse=True)
def prohibit_physical_audio(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Physical audio playback is forbidden during classroom tests')
    for name in ['OutputStream','RawOutputStream','play']:
        monkeypatch.setattr(sd,name,forbidden)
