"""P02: generated resources remain checked against author inputs, never exempted."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from check_architecture import check_assets, check_python
from manual.build import build
from manual.core import ManualError, dump


@pytest.fixture
def reader():
    # Owned tiny fixtures are removed at the end, including failing tests.
    with TemporaryDirectory(prefix='ptb-resource-check-') as folder:
        root = Path(folder)
        source = root / 'manual'
        output = root / 'frontend/public/manual'
        (source / 'assets').mkdir(parents=True)
        (source / 'chapters').mkdir()
        (source / 'assets/example.txt').write_bytes(b'example')
        chapter = dict(schemaVersion='ptb-manual-chapter/1', id='intro', title='Intro',
                       body=dict(type='doc', content=[dict(type='paragraph', content=[dict(type='text', text='Text')])]))
        project = dict(schemaVersion='ptb-manual/1', id='test', title='Test',
                       chapters=[dict(id='intro', title='Intro', path='chapters/intro.json', status='reviewed')],
                       assets=[dict(id='example', kind='example', path='assets/example.txt',
                                    distribution='software-only', git=False,
                                    sha256=hashlib.sha256(b'example').hexdigest())], references=[])
        (source / 'project.json').write_bytes(dump(project))
        (source / 'chapters/intro.json').write_bytes(dump(chapter))
        manifest = dict(resources=[], generated_resources=[dict(path='frontend/public/manual',
                        kind='manual-reader', source_project='manual', distribution='software')])
        build(source, output)
        yield root, source, output, manifest


def test_generated_tree_is_verified_and_can_be_absent(reader):
    root, source, output, manifest = reader
    assert not check_assets(root, manifest, set())
    output.rename(root / 'not-installed')
    assert not check_assets(root, manifest, set())


@pytest.mark.parametrize('name', ['assets/example.txt', 'chapters/intro.json', 'project.json', 'build-report.json'])
def test_corrupt_or_missing_generated_member_fails(reader, name):
    root, _, output, manifest = reader
    file = output / name
    file.write_bytes(b'corrupt')
    assert check_assets(root, manifest, set())
    file.unlink()
    assert check_assets(root, manifest, set())


def test_forged_build_report_cannot_bless_changed_media(reader):
    root, _, output, manifest = reader
    (output / 'assets/example.txt').write_bytes(b'changed')
    report = json.loads((output / 'build-report.json').read_text('utf-8'))
    report['hashes']['assets/example.txt'] = hashlib.sha256(b'changed').hexdigest()
    (output / 'build-report.json').write_bytes(dump(report))
    assert any('hash mismatch' in e for e in check_assets(root, manifest, set()))


def test_unregistered_or_stale_file_inside_generated_tree_fails(reader):
    root, source, output, manifest = reader
    (output / 'secret.txt').write_bytes(b'not a generated asset')
    assert any('undeclared' in e for e in check_assets(root, manifest, set()))
    with pytest.raises(ManualError, match='未登记'):
        build(source, output)
    assert (output / 'secret.txt').is_file()


def test_source_change_requires_regeneration(reader):
    root, source, output, manifest = reader
    chapter = json.loads((source / 'chapters/intro.json').read_text('utf-8'))
    chapter['body']['content'][0]['content'][0]['text'] = 'New text'
    (source / 'chapters/intro.json').write_bytes(dump(chapter))
    assert check_assets(root, manifest, set())
    build(source, output)
    assert not check_assets(root, manifest, set())


def test_successful_build_retires_only_known_old_outputs(reader):
    root, source, output, manifest = reader
    project = json.loads((source / 'project.json').read_text('utf-8'))
    project['assets'] = []
    (source / 'project.json').write_bytes(dump(project))
    build(source, output)
    assert not (output / 'assets/example.txt').exists()
    assert (source / 'assets/example.txt').read_bytes() == b'example'
    assert not check_assets(root, manifest, set())


def test_changed_old_output_is_preserved(reader):
    _, source, output, _ = reader
    project = json.loads((source / 'project.json').read_text('utf-8'))
    project['assets'] = []
    (source / 'project.json').write_bytes(dump(project))
    (output / 'assets/example.txt').write_bytes(b'user edit')
    with pytest.raises(ManualError, match='已变化'):
        build(source, output)
    assert (output / 'assets/example.txt').read_bytes() == b'user edit'


def test_failed_build_does_not_retire_previous_reader(reader, monkeypatch):
    _, source, output, _ = reader
    import manual.build as builder
    project = json.loads((source / 'project.json').read_text('utf-8'))
    project['assets'] = []
    (source / 'project.json').write_bytes(dump(project))
    def fail(*args):
        raise OSError('simulated write failure')
    monkeypatch.setattr(builder, 'write_atomic', fail)
    with pytest.raises(OSError, match='simulated'):
        builder.build(source, output)
    assert (output / 'assets/example.txt').is_file()


def test_atomic_replace_failure_preserves_old_file_and_cleans_temp(reader, monkeypatch):
    _, _, output, _ = reader
    import manual.build as builder
    file = output / 'chapters/intro.json'
    old = file.read_bytes()
    def fail(*args):
        raise OSError('simulated replace failure')
    monkeypatch.setattr(builder.os, 'replace', fail)
    with pytest.raises(OSError, match='replace failure'):
        builder.write_atomic(file, b'new content')
    assert file.read_bytes() == old
    assert not list(file.parent.glob('*.tmp-*'))


def test_public_build_still_refuses_old_private_media(reader):
    _, source, output, _ = reader
    with pytest.raises(ManualError, match='公开输出目录'):
        build(source, output, 'public')
    assert (output / 'assets/example.txt').is_file()


@pytest.mark.parametrize('value', ['../outside', '/outside', 'assets/../../outside', 'assets\\outside'])
def test_generated_paths_cannot_escape(reader, value):
    root, _, _, manifest = reader
    manifest['generated_resources'][0]['source_project'] = value
    assert check_assets(root, manifest, set())


def test_links_cannot_hide_generated_or_static_resources(reader):
    root, source, output, manifest = reader
    outside = root / 'outside'
    outside.mkdir()
    (outside / 'keep').write_bytes(b'keep')
    link = output / 'assets/linked'
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != 'nt':
            pytest.skip('symlink creation unavailable')
        script = root / 'junction.ps1'
        script.write_text('param([string]$Link,[string]$Target)\nNew-Item -ItemType Junction -Path $Link -Target $Target -ErrorAction Stop | Out-Null\n', encoding='utf-8')
        subprocess.run(['powershell.exe', '-NoProfile', '-File', str(script), str(link), str(outside)],
                       check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        assert any('linked' in e for e in check_assets(root, manifest, set()))
        with pytest.raises(ManualError, match='联接|符号链接'):
            build(source, output)
        with pytest.raises(ManualError, match='联接|符号链接'):
            build(source, link / 'reading')
    finally:
        if link.is_symlink():
            link.unlink()
        else:
            # Windows RemoveDirectory removes this junction itself, not its target.
            os.rmdir(link)
    assert (outside / 'keep').read_bytes() == b'keep'


def test_reader_cannot_replace_author_project_or_its_parent(reader):
    root, source, _, _ = reader
    for destination in [source, source / 'ordinary-output', root]:
        with pytest.raises(ManualError, match='覆盖源工程'):
            build(source, destination)
    assert (source / 'assets/example.txt').read_bytes() == b'example'


@pytest.mark.parametrize('code', [
    'import cv2', 'import cv2 as vision', 'from cv2 import *',
    'from cv2 import VideoCapture', 'from cv2 import imread, imwrite',
    'from cv2 import imshow', 'from cv2 import line, VideoWriter',
    'from cv2.cuda import GpuMat', '__import__("cv2")',
    'import importlib; importlib.import_module("cv2")', 'from .cv2 import line',
])
def test_opencv_namespace_io_and_devices_stay_forbidden(code):
    assert check_python(code, 'core')


def test_only_explicit_array_symbols_are_allowed():
    assert not check_python('from cv2 import LINE_8, circle, getPerspectiveTransform, line as draw_line, warpPerspective', 'core')


def test_static_resources_need_digest_and_explicit_origin(reader):
    root, _, _, manifest = reader
    file = root / 'frontend/public/selection.js'
    file.write_bytes(b'project code')
    item = dict(path='frontend/public/selection.js', sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    manifest['resources'] = [item]
    assert check_assets(root, manifest, set())
    item.update(origin='project', description='Owned interaction code')
    assert not check_assets(root, manifest, set())
    item['source_id'] = 'invented'
    assert check_assets(root, manifest, set())
    item.pop('origin'); item['source_id'] = 'known'
    assert not check_assets(root, manifest, {'known'})
    file.write_bytes(b'changed')
    assert any('hash mismatch' in e for e in check_assets(root, manifest, {'known'}))
