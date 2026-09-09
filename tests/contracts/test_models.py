import json
import math

import pytest
from pydantic import ValidationError
from ptb_api.models import Audio, Selection, Track, Viewport


def audio(**overrides):
    return dict(sample_rate_hz=44100, channels=2, channel_roles=['audio', 'egg'],
                sample_count=88200, origin='generated', **overrides)


def track():
    return dict(parameter_key='pF0', backend='praat', unit='Hz',
                times_s=[0.0, 0.01, 0.02], values=[120.0, None, None],
                validity=['valid', 'unvoiced', 'failed'], reason=[None, 'unvoiced', 'backend_failure'],
                analysis_config_hash='a' * 64, source_ids=['SRC-PRAAT'])


def test_half_open_eof_and_empty_roundtrip():
    for start, end in [(123, 22173), (88199, 88200), (88200, 88200)]:
        value = Viewport(audio=Audio(**audio()), selection=Selection(
            start_sample=start, end_sample=end, sample_rate_hz=44100), tracks=[])
        assert Viewport.model_validate_json(value.model_dump_json()) == value
        assert value.selection.end_sample - value.selection.start_sample == end - start


@pytest.mark.parametrize('start,end,rate', [(-1, 10, 44100), (11, 10, 44100),
    (0, 88201, 44100), (0, 1, 48000), (0.5, 1, 44100), (True, 1, 44100), ('0', 1, 44100)])
def test_invalid_selection(start, end, rate):
    with pytest.raises(ValidationError):
        Viewport.model_validate(dict(audio=audio(), tracks=[], selection=dict(
            start_sample=start, end_sample=end, sample_rate_hz=rate)))


def test_track_missingness_preserved():
    result = Track.model_validate_json(json.dumps(track()))
    assert result.values == [120.0, None, None]
    assert result.validity[1:] == ['unvoiced', 'failed']
    assert 'NaN' not in result.model_dump_json()


@pytest.mark.parametrize('field,value', [('times_s', [0.0, 0.0, 0.02]),
    ('times_s', [0.0, 0.02, 0.01]), ('times_s', [0.0, math.inf, 0.02]),
    ('values', [math.nan, None, None]), ('values', [math.inf, None, None]),
    ('values', [120.0]), ('values', [120.0, 0.0, None]),
    ('validity', ['valid', 'valid', 'failed']), ('reason', [None, None, 'backend_failure'])])
def test_invalid_trajectories(field, value):
    payload = track()
    payload[field] = value
    with pytest.raises(ValidationError):
        Track.model_validate(payload)


def test_channel_count_extra_fields_safe_integers():
    for update in [{'channels': 1}, {'sample_count': 2**53}, {'server_path': 'private'},
                   {'sample_count': True}, {'sample_rate_hz': 0}]:
        payload = audio()
        payload.update(update)
        with pytest.raises(ValidationError):
            Audio.model_validate(payload)
