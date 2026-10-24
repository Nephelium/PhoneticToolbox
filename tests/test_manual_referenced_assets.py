"""Post-freeze publishing helper: select reader media without changing authors."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/manual'))
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('ptb_manual_referenced_build', ROOT / 'scripts/manual/build.py')
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)
from verification_artifacts import verification_run


@pytest.fixture
def owned_run():
    with verification_run('manual-referenced-assets', 'test', 'Optional reader asset selection; author/source media immutable') as (out, scratch):
        yield out, scratch


def snapshot(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def fixture_project(folder):
    folder.mkdir()
    assets = []
    for key, kind, distribution in [('picture', 'image', 'public'), ('voice', 'audio', 'public'),
                                    ('movie', 'video', 'public'), ('unused', 'image', 'public'),
                                    ('restricted', 'image', 'software-only')]:
        data = ('Neutral fixture ' + key).encode()
        target = folder / 'assets' / (key + '.bin')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        assets.append(dict(id=key, kind=kind, path='assets/' + target.name, distribution=distribution,
                           git=distribution == 'public', sha256=hashlib.sha256(data).hexdigest()))
    chapter = dict(schemaVersion='ptb-manual-chapter/1', id='one', title='Nested media', body=dict(type='doc', content=[
        dict(type='columns', content=[dict(type='column', content=[
            dict(type='admonition', content=[dict(type='image', attrs=dict(assetId='picture'))])])]),
        dict(type='table', content=[dict(type='tableRow', content=[dict(type='tableCell', content=[
            dict(type='audio', attrs=dict(assetId='voice'))])])]),
        dict(type='blockquote', content=[dict(type='bulletList', content=[dict(type='listItem', content=[
            dict(type='video', attrs=dict(assetId='movie'))])])]),
        dict(type='footnote', content=[dict(type='image', attrs=dict(assetId='picture'))]),
        dict(type='image', attrs=dict(assetId='restricted')),
    ]))
    descriptor = dict(id='one', title=chapter['title'], path='chapters/one.json', status='verified')
    project = dict(schemaVersion='ptb-manual/1', id='fixture', title='Neutral reader fixture', chapters=[descriptor], assets=assets)
    (folder / 'chapters').mkdir()
    (folder / descriptor['path']).write_bytes(BUILD.dump(chapter))
    (folder / 'project.json').write_bytes(BUILD.dump(project))
    return project


def test_nested_image_audio_video_and_public_placeholders_keep_author(owned_run):
    out, scratch = owned_run
    author = scratch / 'author'
    fixture_project(author)
    before = snapshot(p for p in author.rglob('*') if p.is_file())
    default, selected = scratch / 'default', scratch / 'selected'
    normal = BUILD.build(author, default, 'public')
    filtered = BUILD.build(author, selected, 'public', referenced_assets_only=True)
    assert normal['assets'] == 4
    assert 'referencedAssetsOnly' not in normal and 'unreferencedAssets' not in normal
    project = json.loads((selected / 'project.json').read_text('utf-8'))
    assert [a['id'] for a in project['assets']] == ['picture', 'voice', 'movie']
    assert filtered['assets'] == 3 and filtered['unreferencedAssets'] == ['unused']
    assert filtered['skippedAssets'] == normal['skippedAssets'] == ['restricted']
    assert any('占位' in warning for warning in filtered['warnings'])
    assert filtered['mediaBytes'] == sum((selected / a['path']).stat().st_size for a in project['assets'])
    assert (selected / 'chapters/one.json').read_bytes() == (default / 'chapters/one.json').read_bytes()
    assert b'restricted' in (selected / 'chapters/one.json').read_bytes()
    assert not (selected / 'assets/unused.bin').exists()
    assert not (selected / 'assets/restricted.bin').exists()
    assert snapshot(Path(p) for p in before) == before
    (out / 'result.json').write_text(json.dumps(dict(defaultAssets=4, selectedAssets=3, sourceUnchanged=True)), encoding='utf-8')


def test_public_selection_rejects_reusing_old_directory_without_deleting_media(owned_run):
    _, scratch = owned_run
    author = scratch / 'author'
    fixture_project(author)
    target = scratch / 'existing-reader'
    BUILD.build(author, target, 'public')
    before = snapshot(p for p in target.rglob('*') if p.is_file())
    with pytest.raises(BUILD.ManualError, match='新的独立目录'):
        BUILD.build(author, target, 'public', referenced_assets_only=True)
    assert snapshot(Path(p) for p in before) == before


def test_actual_manual_default15_filtered8_datav_and_exact_chapters(owned_run):
    out, scratch = owned_run
    author = ROOT / 'manual'
    project = json.loads((author / 'project.json').read_text('utf-8'))
    public = [a for a in project['assets'] if a.get('distribution') == 'public' and a.get('git') is not False]
    assert len(public) == 15
    author_files = [author / 'project.json', *[author / c['path'] for c in project['chapters']], *[author / a['path'] for a in public]]
    before = snapshot(author_files)
    default, selected = scratch / 'default-reader', scratch / 'referenced-reader'
    normal = BUILD.build(author, default, 'public')
    command = [sys.executable, '-B', '-X', 'utf8', str(ROOT / 'scripts/manual/build.py'), '--project', str(author),
               '--output', str(selected), '--distribution', 'public', '--referenced-assets-only']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert run.returncode == 0, run.stdout + run.stderr
    filtered = json.loads(run.stdout)
    actual = json.loads((selected / 'project.json').read_text('utf-8'))
    assert normal['assets'] == 15 and filtered['assets'] == 8
    assert len(filtered['unreferencedAssets']) == 7
    assert all('M19-DATAV' in a['sourceIds'] for a in actual['assets'])
    assert all(not {'M19-OSM', 'M19-HK-HAD'}.intersection(a['sourceIds']) for a in actual['assets'])
    expected = {
        'm19-current-data-download-2fbc8590', 'm19-current-dialect-distribution-b7bad4de',
        'm19-current-phonology-broad-filter-485894f8', 'm19-current-two-maps-397a4241',
        'm19-current-four-maps-5d520a8c', 'm19-current-two-place-comparison-bfd6f0e2',
        'm19-current-custom-point-editor-da6ae173', 'm19-current-custom-map-export-827444eb',
    }
    assert {a['id'] for a in actual['assets']} == expected
    for descriptor in project['chapters']:
        assert (selected / descriptor['path']).read_bytes() == (default / descriptor['path']).read_bytes()
    for asset in public:
        assert (default / asset['path']).read_bytes() == (author / asset['path']).read_bytes()
        if asset['id'] not in expected:
            assert not (selected / asset['path']).exists()
            assert (author / asset['path']).is_file()
    assert snapshot(author_files) == before
    (out / 'result.json').write_text(json.dumps(dict(defaultAssets=normal['assets'], selectedAssets=filtered['assets'],
        selectedIds=sorted(expected), excludedIds=filtered['unreferencedAssets'], authorFilesUnchanged=len(before),
        chaptersByteIdentical=len(project['chapters']), command=command), ensure_ascii=False, indent=2), encoding='utf-8')


def test_fresh_git_public8_needs_no_old_unreferenced_media_and_matches_full_author(owned_run):
    out, scratch = owned_run
    author = ROOT / 'manual'
    project = json.loads((author / 'project.json').read_text('utf-8'))
    chapter_sources = [author / descriptor['path'] for descriptor in project['chapters']]
    chapter_data = [json.loads(path.read_text('utf-8')) for path in chapter_sources]
    referenced = BUILD.referenced_asset_ids(chapter_data)
    selected = [asset for asset in project['assets'] if asset['id'] in referenced
                and asset.get('distribution') == 'public' and asset.get('git') is not False]
    assert len(selected) == 8
    public = [asset for asset in project['assets'] if asset.get('distribution') == 'public' and asset.get('git') is not False]
    original_files = [author / 'project.json', *chapter_sources, *[author / a['path'] for a in public]]
    before = snapshot(original_files)
    checkout = scratch / 'fresh-git-manual'
    for source in [author / 'project.json', *chapter_sources, *[author / a['path'] for a in selected]]:
        target = checkout / source.relative_to(author)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    checkout_before = snapshot(p for p in checkout.rglob('*') if p.is_file())
    assert len(list((checkout / 'assets').rglob('*.png'))) == 8
    with pytest.raises(BUILD.ManualError, match='缺少媒体'):
        BUILD.build(checkout, scratch / 'default-must-still-fail', 'public')
    expected, actual = scratch / 'full-author-public8', scratch / 'fresh-git-public8'
    BUILD.build(author, expected, 'public', referenced_assets_only=True)
    report = BUILD.build(checkout, actual, 'public', referenced_assets_only=True)
    expected_hashes = {p.relative_to(expected).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in expected.rglob('*') if p.is_file()}
    actual_hashes = {p.relative_to(actual).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in actual.rglob('*') if p.is_file()}
    assert report['assets'] == 8 and len(report['unreferencedAssets']) == 7
    assert actual_hashes == expected_hashes
    assert snapshot(original_files) == before
    assert snapshot(Path(p) for p in checkout_before) == checkout_before
    (out / 'result.json').write_text(json.dumps(dict(success=True, freshGitMedia=8,
        comparedOutputFiles=len(actual_hashes), hashes=actual_hashes, originalAuthorUnchanged=True,
        freshCheckoutUnchanged=True, defaultValidationStillRejectsMissingMedia=True), indent=2), encoding='utf-8')


@pytest.mark.parametrize('failure', ['missing', 'wrong-sha'])
def test_referenced_projection_keeps_referenced_media_failure(failure, owned_run):
    _, scratch = owned_run
    author = scratch / 'author'
    fixture_project(author)
    source = author / 'assets/picture.bin'
    if failure == 'missing':
        source.unlink()
        expected = 'picture: 缺少媒体'
    else:
        source.write_bytes(b'Corrupt owned fixture')
        expected = 'picture: 媒体 SHA-256 不一致'
    with pytest.raises(BUILD.ManualError, match=expected):
        BUILD.build(author, scratch / 'must-fail', 'public', referenced_assets_only=True)
    assert not (scratch / 'must-fail').exists()
