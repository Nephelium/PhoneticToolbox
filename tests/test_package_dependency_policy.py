"""Reviewed trim boundaries and stale-cache protection, without scientific substitution."""
import sys
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'release'))
from release_content_policy import dependency_exclusion, assert_dependency_policy
from prepare_lean_runtimes import select
from compact_payload import prepare


@pytest.mark.parametrize('path', [
    'Lib/idlelib/editor.py', 'Lib/pydoc_data/topics.py',
    'Lib/site-packages/cv2/data/haarcascade_eye.xml',
    'Lib/site-packages/numpy/_core/tests/test_multiarray.py',
    'Lib/site-packages/numpy/random/tests/data/mt19937-testset-1.csv'])
def test_reviewed_files_are_excluded_from_both_scientific_prefixes(path):
    for runtime in ('egg', 'm05'):
        assert not select(runtime, Path(path), set())[0]
        with pytest.raises(ValueError, match='prepare a new tested stage'):
            assert_dependency_policy({'runtimes/' + runtime + '/' + path})


@pytest.mark.parametrize('path', [
    'Lib/site-packages/numpy/_core/tests/_natype.py',
    'Lib/site-packages/numpy/_core/tests/__init__.py',
    'Lib/site-packages/numpy/testing/__init__.py',
    'Lib/site-packages/numpy/testing/tests/test_utils.py',
    'Lib/site-packages/numpy/_core/_multiarray_tests.cp311-win_amd64.pyd',
    'Lib/site-packages/numpy/_core/tests/native-helper.pyd',
    'Lib/site-packages/numpy/random/tests/data/LICENSE.txt',
    'Lib/site-packages/cv2/data/__init__.py',
    'Lib/site-packages/cv2/cv2.pyd', 'Library/bin/mkl_avx512.2.dll',
    'Library/bin/mkl_intel_thread.2.dll', 'Library/bin/mkl_sequential.2.dll',
    'Lib/idlelib/LICENSE.txt'])
def test_public_support_native_variants_and_licenses_remain(path):
    assert dependency_exclusion(path) is None
    for runtime in ('egg', 'm05'):
        assert select(runtime, Path(path), set())[0]


@pytest.mark.parametrize('path', [
    'PyQt6/Qt6/translations/qt_ar.qm', 'PyQt6/Qt6/translations/qtbase_zh_CN.qm',
    'PyQt6/Qt6/translations/qtwebengine_locales/fr.pak',
    'PyQt6/Qt6/resources/icudtl.dat', 'frontend/dist/manual/example.wav'])
def test_language_resources_and_unrelated_application_data_untouched(path):
    assert dependency_exclusion(path) is None


def test_public_support_is_required_even_when_no_excluded_files_remain():
    base = 'runtimes/egg/Lib/site-packages/'
    with pytest.raises(ValueError, match='Missing public NumPy'):
        assert_dependency_policy({base + 'numpy/__init__.py'})
    assert_dependency_policy({base + name for name in (
        'numpy/__init__.py', 'numpy/testing/__init__.py', 'numpy/_core/tests/_natype.py')})


def test_mutation_after_functional_comparison_is_rejected(tmp_path):
    stage = tmp_path / 'stage'
    runtime = stage / 'egg'
    runtime.mkdir(parents=True)
    source = runtime / 'python.exe'
    source.write_bytes(b'original compared bytes')
    record = dict(path='python.exe', size=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (stage / 'selection-manifest.json').write_text(json.dumps(dict(runtimes=[dict(runtime='egg', selected=[record])])), 'utf8')
    source.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='changed after its scientific/media comparison'):
        prepare(stage, tmp_path / 'archive')
