import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ptb_desktop import cache_location as location
from ptb_desktop.startup_cache import cache_root
from ptb_desktop.updates import UpdateService


@pytest.fixture
def profile(tmp_path, monkeypatch):
    data = tmp_path / 'user-data'
    monkeypatch.setenv('LOCALAPPDATA', str(data))
    monkeypatch.delenv(location.ENVIRONMENT, raising=False)
    monkeypatch.delenv(location.MODE_ENVIRONMENT, raising=False)
    return data


def test_cancel_does_not_prepare_hidden_cache_or_save_location(profile, tmp_path):
    assert not location.configure_launcher(tmp_path / 'app.exe', [], lambda reason: None)
    assert not location.configuration_path().exists()
    assert not cache_root().exists()
    assert location.ENVIRONMENT not in os.environ


def test_simultaneous_launchers_share_one_explicit_choice(profile, tmp_path):
    source = '''import json,sys,time
from pathlib import Path
from ptb_desktop.cache_location import configure_launcher,portable_directory
asked=[]
def choose(reason):
    asked.append(reason)
    time.sleep(.15)
    return Path(sys.argv[1])
assert configure_launcher(Path(sys.argv[1])/'app.exe',[],choose)
print(json.dumps({'asked':len(asked),'directory':str(portable_directory())}))
'''
    processes = [subprocess.Popen([sys.executable, '-B', '-c', source, str(tmp_path / 'chosen')],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    replies = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        replies.append(json.loads(stdout))
    assert sorted(reply['asked'] for reply in replies) == [0, 1]
    assert replies[0]['directory'] == replies[1]['directory']


@pytest.mark.parametrize('language',['python','node'])
def test_qa_profiles_use_real_location_preferences_without_runtime_bypass(profile,tmp_path,language):
    if language=='python':
        from portable_cache_qa import seed_portable_profile
        directory=seed_portable_profile(profile)
        assert seed_portable_profile(profile)==directory
    else:
        script=Path(__file__).resolve().parents[2]/'scripts/portable-cache-qa.mjs'
        source="const {seedPortableProfile}=await import(process.argv[1]); await seedPortableProfile(process.argv[2]);"
        subprocess.run(['node','--input-type=module','-e',source,script.as_uri(),str(profile)],check=True)
    def must_not_ask(reason):raise AssertionError('Explicit test preference was not recognized')
    assert location.configure_launcher(tmp_path/'app.exe',[],must_not_ask)
    assert location.portable_directory()==profile/location.DIRECTORY_NAME


def test_replacement_exe_reuses_selected_directory_and_deleted_cache_rebuilds(profile, tmp_path):
    parent = tmp_path / '井井的工具 缓存'
    assert location.configure_launcher(tmp_path / 'v3.0.3.exe', [], lambda reason: parent)
    directory = parent / location.DIRECTORY_NAME
    assert cache_root() == directory / 'startup-cache'
    assert location.update_directory() == directory / 'updates'
    assert location.browser_directory() == directory / 'browser-cache'
    original = location.configuration_path().read_bytes()
    protected = profile / 'research.wav'
    protected.write_bytes(b'research')

    def must_not_ask(reason):
        raise AssertionError('A replacement EXE must retain the chosen location')

    assert location.configure_launcher(tmp_path / 'new-name.exe', [], must_not_ask)
    assert location.configuration_path().read_bytes() == original
    assert directory.is_relative_to(tmp_path)
    shutil.rmtree(directory)
    assert location.configure_launcher(tmp_path / 'app.exe', [], must_not_ask)
    assert directory.is_dir() and protected.read_bytes() == b'research'


def test_nonempty_foreign_directory_is_not_adopted(profile, tmp_path):
    parent = tmp_path / 'first'
    foreign = parent / location.DIRECTORY_NAME
    foreign.mkdir(parents=True)
    (foreign / 'recording.wav').write_bytes(b'original')
    answers = iter((parent, tmp_path / 'second'))
    reasons = []
    def choose(reason):
        reasons.append(reason)
        return next(answers)
    assert location.configure_launcher(tmp_path / 'app.exe', [], choose)
    assert reasons[0] == '' and '已有其他内容' in reasons[1]
    assert (foreign / 'recording.wav').read_bytes() == b'original'
    assert not (foreign / 'cache-directory.json').exists()


def test_corrupt_preference_prompts_and_retains_old_cache(profile, tmp_path):
    previous = cache_root()
    previous.mkdir(parents=True)
    (previous / 'old-cache.bin').write_bytes(b'old')
    location.configuration_path().write_text('{broken', encoding='utf8')
    reasons = []
    def choose(reason):
        reasons.append(reason)
        return tmp_path / 'new'
    assert location.configure_launcher(tmp_path / 'app.exe', [], choose)
    assert reasons[0]
    assert location.location_status()['legacy_path'] == str(previous)
    assert (previous / 'old-cache.bin').read_bytes() == b'old'


def test_settings_directory_cannot_be_used_for_disposable_cache(profile, tmp_path):
    config = location.configuration_path()
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'schema': location.SCHEMA,
                                 'directory': str(config.parent / location.DIRECTORY_NAME)}), encoding='utf8')
    reasons = []
    def cancel(reason):
        reasons.append(reason)
        return None
    assert not location.configure_launcher(tmp_path / 'app.exe', [], cancel)
    assert '分开' in reasons[0]
    assert not (config.parent / location.DIRECTORY_NAME).exists()


def test_unwritable_selection_prompts_again_without_fallback(profile, tmp_path, monkeypatch):
    denied = tmp_path / 'denied'
    original = Path.open
    def open_file(path, *args, **kwargs):
        if path.is_relative_to(denied) and path.name.startswith('.write-check-'):
            raise PermissionError('所选目录不可写')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_file)
    reasons = []
    answers = iter((denied, tmp_path / 'writable'))
    def choose(reason):
        reasons.append(reason)
        return next(answers)
    assert location.configure_launcher(tmp_path / 'app.exe', [], choose)
    assert '不可写' in reasons[1]
    assert not (location.configuration_path().parent / 'startup-cache').exists()


def test_installer_ignores_portable_preference_and_inherited_environment(profile, tmp_path, monkeypatch):
    assert location.configure_launcher(tmp_path / 'app.exe', [], lambda reason: tmp_path / 'portable')
    saved = location.configuration_path().read_bytes()
    installed = tmp_path / 'installed'
    installed.mkdir()
    (installed / '.ptb-installed.json').write_text('{"schema":"ptb-install/1","kind":"installer"}', encoding='utf8')
    def must_not_ask(reason):
        raise AssertionError('Installer must not select a portable directory')
    assert location.configure_launcher(installed / 'app.exe', ['--ptb-prepare-cache'], must_not_ask)
    assert location.portable_directory() is None
    assert cache_root() == location.configuration_path().parent / 'startup-cache'
    assert location.configuration_path().read_bytes() == saved


def test_update_helper_keeps_its_parent_cache_identity(profile, tmp_path):
    assert location.configure_launcher(tmp_path / 'app.exe', [], lambda reason: tmp_path / 'portable')
    chosen = cache_root()
    def must_not_ask(reason):
        raise AssertionError('Update helper must not ask for another cache')
    assert location.configure_launcher(tmp_path / 'helper.exe', ['--ptb-apply-update'], must_not_ask)
    assert cache_root() == chosen


def test_junction_selection_cannot_write_into_research_directory(profile, tmp_path):
    outside = tmp_path / 'research'
    outside.mkdir()
    (outside / 'recording.wav').write_bytes(b'original')
    link = tmp_path / location.DIRECTORY_NAME
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)], capture_output=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match='reparse'):
        location._validate_directory(link, create=True)
    assert (outside / 'recording.wav').read_bytes() == b'original'
    assert not (outside / 'cache-directory.json').exists()


def test_update_preferences_survive_cache_removal(profile, tmp_path):
    cache = tmp_path / 'cache'
    state = tmp_path / 'settings'
    service = UpdateService(current_version='3.0.3', data_root=cache, state_root=state)
    service.set_preferences({'autoCheck': False, 'source': 'github'})
    download = cache / 'downloads' / ('a' * 32)
    download.mkdir(parents=True)
    (download / 'package.zip').write_bytes(b'cache')
    assert cache.is_relative_to(tmp_path)
    shutil.rmtree(cache)
    replacement = UpdateService(current_version='3.0.4', data_root=cache, state_root=state)
    assert replacement.preferences()['autoCheck'] is False
    assert replacement.preferences()['source'] == 'github'
    assert not cache.exists()


def test_native_open_is_limited_to_current_or_legacy_cache(profile, tmp_path, monkeypatch):
    from PyQt6.QtGui import QDesktopServices
    assert location.configure_launcher(tmp_path / 'app.exe', [], lambda reason: tmp_path / 'portable')
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toLocalFile()) or True)
    assert location.open_directory() == {'opened': True}
    assert Path(opened[0]) == location.portable_directory()
    with pytest.raises(ValueError, match='不存在'):
        location.open_directory(legacy=True)
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: False)
    with pytest.raises(OSError, match='复制路径'):
        location.open_directory()
