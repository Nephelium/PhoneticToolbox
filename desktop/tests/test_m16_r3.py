"""M16-R3 display coverage and retained recording history, synthetic data only."""
import copy
import numpy as np
from ptb_desktop.recording.service import RecordingService
from ptb_desktop.recording.storage import Project, save_pcm


def recording(tmp_path):
    service = RecordingService()
    service.project = Project(tmp_path / 'project', True)
    sr = 48000
    t = np.arange(sr * 11) / sr
    x = np.column_stack((.3 * np.sin(2*np.pi*1000*t), .2 * np.sin(2*np.pi*3000*t))).astype('float32')
    spans = [save_pcm(service.project.root, f'takes/test/raw/{i}.f32', x[i:i+131072]) for i in range(0, len(x), 131072)]
    task = dict(id='T001', prompt='示例', title='', filename_stem='test', group='', note='', enabled=True, skipped=False)
    data = copy.deepcopy(service.project.data)
    data.update(tasks=[task], takes=[dict(id='test', config=dict(sample_rate=sr, channels=2, roles=['microphone']*2), task_snapshot=copy.deepcopy(task), head=0, versions=[dict(id='raw', kind='raw', spans=spans)])], selected={'T001':'test'})
    service.project.commit(data)
    return service, x


def test_preview_covers_full_time_and_zoom_channel(tmp_path):
    service, x = recording(tmp_path)
    try:
        for start, end, channel, hz in [(0, len(x), 0, 1000), (48000*7,48000*10,1,3000), (31,97,0,None)]:
            out = service.dispatch(dict(op='preview', id='test', start=start, end=end, spectrum=True, channel=channel))
            spec = out['spectrum']
            assert out['spectrum_window_frames'] == end-start
            assert spec['time_edges'][0] == 0
            assert spec['time_edges'][-1] == (end-start)/48000
            assert 0 < len(spec['rows']) <= 640
            assert max(spec['frequencies']) <= 5000
            if hz:
                peaks = np.asarray(spec['frequencies'])[np.argmax(spec['rows'], axis=1)]
                assert np.max(np.abs(peaks-hz)) <= 48000/1024
        assert service.dispatch(dict(op='preview',id='test',spectrum=False))['spectrum'] is None
    finally:
        service.close()


def test_remove_task_persists_but_keeps_take_snapshot_and_pcm(tmp_path):
    service, x = recording(tmp_path)
    original = copy.deepcopy(service.project.data['takes'])
    service.dispatch(dict(op='tasks',tasks=[]))
    root = service.project.root
    service.close()
    project = Project(root)
    try:
        assert project.data['tasks'] == []
        assert project.data['takes'] == original
        assert all((root/span['file']).exists() for span in original[0]['versions'][0]['spans'])
    finally:
        project.close()
