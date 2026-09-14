"""M12 Windows real-file acceptance: grants, stale writes, independent lip save."""
from pathlib import Path
import hashlib
import pickle
import numpy as np
import pytest
from ptb_desktop.file_provider import FileProvider, FileAccessError
from ptb_desktop.annotation import AnnotationFiles
from ptb_worker.io.annotation import update_lip_offset, lip_preview
from phonetic_core.annotation import parse_document

GRID = 'File type = "ooTextFile short"\n"TextGrid"\n0\n2\n<exists>\n2\n"IntervalTier"\n"words"\n0\n2\n1\n0\n2\n"井井 æ"\n"TextTier"\n"events"\n0\n2\n1\n1\n"ʔ"\n'


def setup(tmp_path):
    root = tmp_path/'corpus'; root.mkdir()
    sub = root/'中文 session'; sub.mkdir()
    (sub/'audio_recording.wav').write_bytes(b'WAV test capability')
    (sub/'audio_recording.TextGrid').write_text(GRID, encoding='utf-8')
    (sub/'audio_recording.lab').write_text('ba1 ba2', encoding='utf-8')
    provider = FileProvider()
    grant = provider.choose('input', lambda: root)
    annotation = AnnotationFiles(provider)
    files = annotation.scan(grant['id'])
    return sub, provider, annotation, files, grant


def test_m12_recursive_scan_and_versioned_textgrid_save(tmp_path):
    sub, provider, annotation, files, _ = setup(tmp_path)
    assert len(files) == 3 and all(f['name'].startswith('中文 session/') for f in files)
    audio = next(f for f in files if f['kind'] == 'audio')
    grid = next(f for f in files if f['kind'] == 'textgrid')
    sha = provider.read(grid['id'])[1]
    target = annotation.target(dict(role='textgrid', id=audio['id'], suffix='_webedit'))
    result = annotation.save(dict(target=target['id'], source=dict(id=grid['id'], sha256=sha), text=GRID.replace('井井', '秋叶')))
    assert (sub/'audio_recording.TextGrid').read_text('utf-8') == GRID
    output = (sub/result['name']).read_text('utf-8')
    assert parse_document(output)['tiers'][1]['points'][0]['mark'] == 'ʔ'
    assert provider.read(result['file']['id'])[1] == result['sha256']


def test_m12_conflict_preserves_other_window_output(tmp_path):
    sub, provider, annotation, files, _ = setup(tmp_path)
    audio = next(f for f in files if f['kind'] == 'audio'); grid = next(f for f in files if f['kind'] == 'textgrid')
    first = annotation.target(dict(role='textgrid', id=audio['id'], suffix='_webedit'))
    stale = annotation.target(dict(role='textgrid', id=audio['id'], suffix='_webedit'))
    body = dict(source=dict(id=grid['id'], sha256=provider.read(grid['id'])[1]), text=GRID)
    annotation.save(dict(target=first['id'], **body))
    with pytest.raises(FileAccessError, match='其他窗口'):
        annotation.save(dict(target=stale['id'], **body))
    assert (sub/'audio_recording_webedit.TextGrid').read_text('utf-8') == GRID


def test_m12_changed_source_and_invalid_suffix_cannot_write(tmp_path):
    sub, provider, annotation, files, _ = setup(tmp_path)
    audio = next(f for f in files if f['kind'] == 'audio'); grid = next(f for f in files if f['kind'] == 'textgrid')
    for suffix in ['../escape', '\\escape', ':stream', 'bad.', 'bad ']:
        with pytest.raises(FileAccessError): annotation.target(dict(role='textgrid', id=audio['id'], suffix=suffix))
    target = annotation.target(dict(role='textgrid', id=audio['id'], suffix=''))
    sha = provider.read(grid['id'])[1]
    (sub/'audio_recording.TextGrid').write_text(GRID.replace('井井', '外部'), encoding='utf-8')
    with pytest.raises(FileAccessError): annotation.save(dict(target=target['id'], source=dict(id=grid['id'], sha256=sha), text=GRID))
    assert '外部' in (sub/'audio_recording.TextGrid').read_text('utf-8')


def test_m12_write_failure_does_not_replace_original(tmp_path, monkeypatch):
    sub, provider, annotation, files, _ = setup(tmp_path)
    audio = next(f for f in files if f['kind'] == 'audio'); grid = next(f for f in files if f['kind'] == 'textgrid')
    target = annotation.target(dict(role='textgrid', id=audio['id'], suffix=''))
    def fail(*args): raise OSError('injected disk failure')
    monkeypatch.setattr('ptb_desktop.annotation.os.replace', fail)
    with pytest.raises(OSError): annotation.save(dict(target=target['id'], source=dict(id=grid['id'], sha256=provider.read(grid['id'])[1]), text=GRID.replace('井井', '编辑')))
    assert (sub/'audio_recording.TextGrid').read_text('utf-8') == GRID
    assert not list(sub.glob('.ptb-annotation-*'))


@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('protocol', [2, 4, 5])
def test_m12_pickle_offset_preserves_all_recording_fields(order, protocol):
    original = {'open': np.array([.2, np.nan, .8]), 'outer_width': [.5, .7, .9], 'relative_times': np.array([0, .1, .2]),
                'landmarks': np.array(np.arange(24).reshape(3, 4, 2), dtype=np.float64, order=order),
                'metadata': {'lip_manual_offset': .01, 'note': '中文元数据', 'calibration': [1., 2.]}, 'unrelated': {'tuple': ('a', 1)}}
    payload = pickle.dumps(original, protocol=protocol)
    updated = update_lip_offset(payload, 'audio_recording.pkl', -.025)
    result = pickle.loads(updated)  # Test fixture generated by this test, never untrusted input.
    assert result['metadata']['lip_manual_offset'] == -.025
    assert result['metadata']['calibration'] == [1., 2.]
    assert result['metadata']['note'] == '中文元数据'
    assert result['unrelated'] == original['unrelated']
    np.testing.assert_array_equal(result['landmarks'], original['landmarks'])
    np.testing.assert_array_equal(result['open'], original['open'])
    assert lip_preview(updated, 'audio_recording.pkl')['data']['metadata']['lip_manual_offset'] == -.025


def test_m12_reject_executable_pickle_and_invalid_offset(tmp_path):
    marker = tmp_path/'must-not-exist'
    class Payload:
        def __reduce__(self): return (Path.write_text, (marker, 'executed'))
    raw = pickle.dumps(Payload())
    with pytest.raises(ValueError): update_lip_offset(raw, 'audio_recording.pkl', 0)
    assert not marker.exists()
    for offset in [float('nan'), float('inf'), True, 3601]:
        with pytest.raises(ValueError): update_lip_offset(b'', 'audio_recording.pkl', offset)


def test_m12_save_lip_is_independent_from_textgrid(tmp_path):
    sub, provider, annotation, _, grant = setup(tmp_path)
    record = {'relative_times':[.1,.2,.3], 'open':[.2,.4,.7], 'metadata':{'lip_manual_offset':0,'other':'keep'}}
    path = sub/'audio_recording.pkl';path.write_bytes(pickle.dumps(record))
    files = annotation.scan(grant['id']);lip = next(f for f in files if f['kind']=='lip_pickle')
    before = hashlib.sha256((sub/'audio_recording.TextGrid').read_bytes()).hexdigest()
    target = annotation.target(dict(role='lip', id=lip['id']))
    result = annotation.save(dict(target=target['id'], source=dict(id=lip['id'], sha256=provider.read(lip['id'])[1]), offset=.014))
    assert annotation.lip(result['file']['id'])['wire']['data']['metadata']['lip_manual_offset'] == .014
    assert pickle.loads(path.read_bytes())['metadata']['other'] == 'keep'
    assert hashlib.sha256((sub/'audio_recording.TextGrid').read_bytes()).hexdigest() == before
