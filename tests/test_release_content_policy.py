"""P19: removing author documents must leave the rendered help and runtime data."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_policy_test',ROOT/'scripts/release_content_policy.py')
policy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_real_in_app_help_references_remain_packaged():
    base=ROOT/'frontend/dist/manual'
    book=json.loads((base/'project.json').read_text('utf8'))
    assert len(book['chapters'])>=17
    for row in [*book['chapters'],*book['assets']]:
        path=base/row['path']
        assert path.is_file()
        assert policy.exclusion(path.relative_to(ROOT).as_posix()) is None
    assert policy.exclusion('docs/manual/ipa-content-authoring.md')
    assert policy.exclusion('contracts/generated/api.ts')


def test_format_schemas_notices_and_import_metadata_are_retained():
    for path in ['contracts/recording/project.schema.json','contracts/resource-manifest.json',
                 'backend/src/ptb_api.egg-info/PKG-INFO','resources/vocal_tract/THIRD_PARTY_NOTICES.md',
                 'resources/vocal_tract/sources/VTL2.4-API-source.zip',
                 'preview-fixtures/EGG-SYN-PCM16.npz']:
        assert policy.exclusion(path) is None


def test_m18_papers_are_external_content_not_executable_resources():
    assert policy.exclusion('output/paper-reading/2610.00735v1/original.pdf')
    assert policy.exclusion('frontend/dist/paper-reading-content/translation.pdf')
    assert not list((ROOT/'frontend/dist').rglob('*.pdf'))
    assert policy.exclusion('frontend/src/modules/paper-reading/PaperReadingPage.vue') is None
