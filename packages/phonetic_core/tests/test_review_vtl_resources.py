import pytest
from phonetic_core.vocal_tract.native_resources import resolve_libraries


@pytest.mark.parametrize('platform,arch,names',[
    ('win32','AMD64',('VocalTractLabApi.dll','VocalTractLabAnalysis.dll','geometry_p2.dll')),
    ('darwin','arm64',('libVocalTractLabApi.dylib','libVocalTractLabAnalysis.dylib','libgeometry_p2.dylib')),
    ('linux','x86_64',('libVocalTractLabApi.so','libVocalTractLabAnalysis.so','libgeometry_p2.so')),
])
def test_native_selection_is_explicitly_platform_and_arch_scoped(tmp_path,platform,arch,names):
    normalized='x86_64' if arch=='AMD64' else arch
    target=tmp_path/(platform+'-'+normalized);target.mkdir()
    for name in names:(target/name).write_bytes(b'fixture, not a real library')
    paths=resolve_libraries(tmp_path,platform=platform,arch=arch)
    assert tuple(path.name for path in paths)==names
    assert all(path.parent==target for path in paths)


def test_macos_cannot_fall_back_to_windows_dlls(tmp_path):
    for name in ('VocalTractLabApi.dll','VocalTractLabAnalysis.dll','geometry_p2.dll'):(tmp_path/name).write_bytes(b'windows')
    with pytest.raises(RuntimeError,match='native_resource_unavailable'):
        resolve_libraries(tmp_path,platform='darwin',arch='arm64')


def test_windows_missing_machine_uses_native_api_only_when_not_explicit(monkeypatch,tmp_path):
    import phonetic_core.vocal_tract.native_resources as resources
    monkeypatch.setattr(resources.sys,'platform','win32')
    monkeypatch.setattr(resources.host,'machine',lambda:'')
    monkeypatch.setattr(resources,'_windows_process_architecture',lambda:'x86_64')
    for name in ('VocalTractLabApi.dll','VocalTractLabAnalysis.dll','geometry_p2.dll'):
        (tmp_path/name).write_bytes(b'fixture')
    assert len(resolve_libraries(tmp_path))==3
    with pytest.raises(RuntimeError,match='native_platform_unavailable'):
        resolve_libraries(tmp_path,arch='')
