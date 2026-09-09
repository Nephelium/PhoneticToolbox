import ast
import json
from pathlib import Path
import pytest
from phonetic_toolbox.services.vocal_tract.profile import ProfileStore
from .test_http import server, call


def test_saved_keyframes_persist_independently_of_server_origin(server,tmp_path):
    frame={'params':server.app.engine.presets['u'],'preset':'u','lip_width':1.2,'f0':130,'duration':.5}
    assert call(server,'/api/keyframes',{'frames':[frame]})[0]==200
    saved=json.loads(call(server,'/api/keyframes')[2])['frames']
    assert saved==[frame]
    store=ProfileStore(server.app.profile.directory)
    assert store.load_frames(server.app.engine)==saved
    assert call(server,'/api/keyframes',{'frames':[{**frame,'lip_width':20}]})[0]==400
    assert store.load_frames(server.app.engine)==saved
    assert call(server,'/api/keyframes',{'frames':[]})[0]==200


def test_corrupt_profile_is_reported_without_overwriting(server,tmp_path):
    store=ProfileStore(tmp_path);path=tmp_path/'keyframes.json';path.write_text('broken',encoding='utf-8')
    with pytest.raises(ValueError):store.load_frames(server.app.engine)
    assert path.read_text(encoding='utf-8')=='broken'


def test_production_layers_do_not_import_prototype_or_gui_from_core():
    package=Path(__file__).resolve().parents[2]
    for directory in ['core/vocal_tract','services/vocal_tract']:
        for path in (package/directory).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.ImportFrom):
                    assert 'prototypes' not in (node.module or '')
                    if directory.startswith('core'):assert not any(x in (node.module or '') for x in ['.services','.gui'])


def test_resources_are_pinned_and_readonly_engine_does_not_copy_dlls():
    package=Path(__file__).resolve().parents[2]
    resources=package/'resources/vocal_tract'
    import hashlib
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(resources/'native/VocalTractLabAnalysis.dll')==digest(resources/'native/VocalTractLabApi.dll')
    assert (resources/'sources/VTL2.4-API-source.zip').is_file()
    assert 'copy2' not in (package/'core/vocal_tract/engine.py').read_text(encoding='utf-8')
