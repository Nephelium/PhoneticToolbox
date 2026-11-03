import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('manual_reader_pins', ROOT / 'scripts/manual/register_reader_resources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pins_require_equal_real_author_and_generated_bytes_and_preserve_provenance(tmp_path):
    (tmp_path / 'contracts').mkdir()
    source = tmp_path / 'manual/assets/example.png'
    generated = tmp_path / 'frontend/public/manual/assets/example.png'
    source.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    source.write_bytes(b'current author bytes')
    generated.write_bytes(b'old reader bytes')
    book = {'assets': [{'path': 'assets/example.png'}]}
    (tmp_path / 'manual/project.json').write_text(json.dumps(book), encoding='utf-8')
    path = tmp_path / 'contracts/resource-manifest.json'
    original = {'resources': [{'path': 'frontend/public/manual/assets/example.png', 'sha256': 'old', 'source_id': 'retained-source'},
                              {'path': 'other/resource', 'sha256': 'unrelated', 'source_id': 'other'}]}
    path.write_text(json.dumps(original), encoding='utf-8')
    with pytest.raises(ValueError, match='Rebuild'):
        module.refresh(tmp_path)
    assert json.loads(path.read_text('utf-8')) == original
    generated.write_bytes(source.read_bytes())
    assert module.refresh(tmp_path, check=True)['changed']
    assert json.loads(path.read_text('utf-8')) == original
    module.refresh(tmp_path)
    updated = json.loads(path.read_text('utf-8'))
    assert updated['resources'][0]['source_id'] == 'retained-source'
    assert updated['resources'][1] == original['resources'][1]
    assert not module.refresh(tmp_path, check=True)['changed']
