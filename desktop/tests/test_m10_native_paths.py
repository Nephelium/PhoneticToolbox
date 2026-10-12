"""Real M10 resource loading through a private short entry, with owned cleanup."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from scripts.verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'desktop/src'), str(ROOT / 'backend/src')]
pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Win32 native resource entry')


def deep_directory(scratch):
    root = scratch / '中文声道资源'
    while len(str(root)) < 285:
        root /= 'long-native-resource-directory'
    root.mkdir(parents=True)
    return root


def test_real_runtime_deep_chinese_resources_metadata_snapshot_and_close():
    recipe = 'Real current three DLLs and JD2 speaker in a deep Chinese directory; metadata/snapshot with playback disabled'
    with verification_run('native-paths', 'm10-runtime', recipe) as (out, scratch):
        deep = deep_directory(scratch)
        original = deep / 'native'
        shutil.copytree(ROOT / 'resources/vocal_tract/native', original)
        # Windows keeps loaded DLL files open for the process lifetime. A short
        # lived child lets the parent clean every copied resource after close.
        code = '''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'desktop/src'),str(root/'backend/src'),str(root/'packages/phonetic_core/src')]
from ptb_desktop.vocal_tract.runtime import Runtime
original=Path(sys.argv[2]);runtime=Runtime(original,Path(sys.argv[3]),playback_allowed=False)
entry=runtime.resource_entry
try:
    assert entry.owned and str(entry.path).isascii() and len(str(entry.path))<140
    assert entry.path.samefile(original)
    assert runtime.engine.speaker_path.samefile(original/'JD2.speaker')
    for library,name in [(runtime.engine.lib,'VocalTractLabApi.dll'),(runtime.engine.analysis,'VocalTractLabAnalysis.dll'),(runtime.engine.geom,'geometry_p2.dll')]:
        assert Path(library._name).samefile(original/name)
        assert str(library._name).isascii() and len(str(library._name))<160
    metadata=runtime.invoke('meta')
    assert metadata['geometry_method']=='m10/r16'
    assert metadata['audio_locked'] and metadata['output_devices']==[]
    assert metadata['speaker_sha256']==hashlib.sha256((original/'JD2.speaker').read_bytes()).hexdigest()
    snapshot=runtime.engine.snapshot(runtime.engine.params)
    assert snapshot['frequency'] and snapshot['resonances'] and snapshot['oral_min_area']>0
    result=dict(original_length=len(str(original)),entry_length=len(str(entry.path)),same_three_libraries=True,same_speaker=True,speaker_sha256=metadata['speaker_sha256'],geometry_method=metadata['geometry_method'],snapshot_resonances=snapshot['resonances'],physical_audio_access=False)
finally:
    runtime.close()
runtime.close()
assert not entry.path.exists() and not entry.path.parent.exists()
result['short_entry_cleaned']=True
Path(sys.argv[4]).write_text(json.dumps(result),encoding='utf8')
'''
        result_file = scratch / 'runtime-result.json'
        child = subprocess.run([sys.executable, '-I', '-B', '-X', 'utf8', '-c', code,
                                str(ROOT), str(original), str(deep / '用户设置'), str(result_file)],
                               capture_output=True, text=True, encoding='utf8', timeout=30,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        assert child.returncode == 0, child.stderr
        result = json.loads(result_file.read_text(encoding='utf8'))
        assert result['same_three_libraries'] and result['same_speaker'] and result['short_entry_cleaned']
        (out / 'report.json').write_text(json.dumps(result), encoding='utf8')
        assert (original / 'JD2.speaker').is_file()
        assert (original / 'geometry_p2.dll').is_file()
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'


@pytest.mark.parametrize('failure_stage', ['engine', 'profile'])
def test_runtime_initialization_failure_closes_engine_before_resource_entry(monkeypatch, failure_stage):
    from ptb_desktop.vocal_tract import runtime as module
    from ptb_worker import native_runtime

    recipe = 'Constructor failures release the owned short resource entry and close an initialized engine first'
    with verification_run('native-paths', 'm10-failure', recipe) as (out, scratch):
        original = deep_directory(scratch)
        (original / 'original.txt').write_text('preserved resource', encoding='utf8')
        events = []
        observed = []
        class Engine:
            def __init__(self, *, resource_dir):
                observed.append(Path(resource_dir))
                if failure_stage == 'engine': raise RuntimeError('engine fixture failure')
            def close(self):
                assert observed[0].exists()
                events.append('engine closed with resource entry alive')
        def broken_profile(*args, **kwargs):
            raise RuntimeError('profile fixture failure')
        monkeypatch.setattr(module, 'Engine', Engine)
        monkeypatch.setattr(module, 'ProfileStore', broken_profile)
        before = set(native_runtime._working_entries)
        with pytest.raises(RuntimeError, match=failure_stage + ' fixture failure'):
            module.Runtime(original, scratch / 'profile', playback_allowed=False)
        assert observed and not observed[0].exists() and not observed[0].parent.exists()
        assert set(native_runtime._working_entries) == before
        assert events == ([] if failure_stage == 'engine' else ['engine closed with resource entry alive'])
        assert (original / 'original.txt').read_text(encoding='utf8') == 'preserved resource'
        (out / 'report.json').write_text(json.dumps(dict(stage=failure_stage, cleaned_entry=True, events=events)), encoding='utf8')
    assert json.loads((out / 'cleanup.json').read_text(encoding='utf-8-sig'))['status'] == 'deleted'
