"""Publication must retain approved Chinese text and ship all supported translations."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('localize_update_manifest', Path(__file__).resolve().parents[2] / 'release/localize_update_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_release_notes_preserve_packages_and_chinese(tmp_path):
    texts = {'': '# 中文\n', '.en': '# English\n', '.ja': '# 日本語\n'}
    for suffix, text in texts.items():
        (tmp_path / f'3.1.4{suffix}.md').write_text(text, 'utf-8')
    original = {'schemaVersion': 'ptb-release/1', 'version': '3.1.4', 'notes': texts[''], 'packages': [{'sha256': 'unchanged'}]}
    result = module.attach_notes(original, tmp_path)
    assert result['notesByLocale'] == {'zh-CN': texts[''], 'en': texts['.en'], 'ja': texts['.ja']}
    assert result['notes'] == original['notes'] and result['packages'] == original['packages']
    assert 'notesByLocale' not in original
    with pytest.raises(ValueError, match='before signing'):
        module.attach_notes(original | {'signature': {}}, tmp_path)
    with pytest.raises(ValueError, match='differ'):
        module.attach_notes(original | {'notes': 'other'}, tmp_path)


def test_missing_translation_stops_publication(tmp_path):
    (tmp_path / '3.1.4.md').write_text('# 中文\n', 'utf-8')
    with pytest.raises(FileNotFoundError):
        module.attach_notes({'schemaVersion': 'ptb-release/1', 'version': '3.1.4'}, tmp_path)


@pytest.mark.parametrize('version', ['', '../3.1.4', '3.1.4/ja', None])
def test_invalid_version_cannot_escape_notes_directory(tmp_path, version):
    with pytest.raises(ValueError, match='version'):
        module.attach_notes({'schemaVersion': 'ptb-release/1', 'version': version}, tmp_path)
