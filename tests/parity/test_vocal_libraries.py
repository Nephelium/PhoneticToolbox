"""M10 library persistence, legacy adoption and explicit old-geometry import."""
import copy
import json
from pathlib import Path
import pytest
from phonetic_core.vocal_tract.engine import Engine
from phonetic_core.vocal_tract.session import Runtime
from phonetic_core.vocal_tract.preset_pack import make_pack, validate_pack
from ptb_worker.vocal_web import MemoryProfile, BrowserAudio
from ptb_desktop.vocal_tract.profile import ProfileStore

ROOT=Path(__file__).resolve().parents[2]

def test_checkpoint_restores_all_sets_including_legacy_duplicate_names():
    from ptb_worker.browser_vocal import BrowserVocalRuntime
    host=BrowserVocalRuntime(ROOT/'resources/vocal_tract/native')
    try:
        r=host.runtime
        library=r.invoke('library/load')
        library['sets'][0]['presets']=[pose(r.engine),{**pose(r.engine),'id':'old-duplicate'}]
        r.invoke('library/save',library)
        r.invoke('library/edit',{'action':'copy','name':'第二套'})
        expected=r.invoke('library/load')
        r.invoke('speaker/select',{'speaker':'W02'})
        r.invoke('library/edit',{'action':'create','name':'女声套装'})
        checkpoint=json.loads(json.dumps(host.checkpoint()))
    finally:host.runtime.close()
    restored=BrowserVocalRuntime(ROOT/'resources/vocal_tract/native',checkpoint)
    try:
        assert len(restored.runtime.invoke('library/load')['sets'])==2
        restored.runtime.invoke('speaker/select',{'speaker':'JD2'})
        assert restored.runtime.invoke('library/load')==expected
    finally:restored.runtime.close()

@pytest.fixture
def engine():
    e=Engine(resource_dir=ROOT/'resources/vocal_tract/native')
    yield e
    e.close()

def pose(e, label='a', f0=120):
    return dict(id='own-a',name=label,params=list(e.presets['a']),f0=f0,duration=.2,lip_width=1.)

@pytest.mark.parametrize('disk',[False,True])
def test_named_sets_keep_same_symbols_and_frames_independent(engine,tmp_path,disk):
    profile=ProfileStore(tmp_path) if disk else MemoryProfile()
    r=Runtime(engine,profile,BrowserAudio(engine));a=pose(engine)
    r.invoke('presets/save',{'presets':[a]});r.invoke('favorites/save',{'ids':['own-a']})
    r.invoke('keyframes',{'frames':[a]});sequence=r.invoke('keyframes/load')
    first=r.invoke('library/load')['active']
    second=r.invoke('library/edit',{'action':'create','name':'同事的构型'})['active']
    assert r.invoke('presets/load')==[]
    r.invoke('presets/save',{'presets':[pose(engine,f0=210)]})
    duplicate=r.invoke('library/edit',{'action':'copy','name':'实验组合'})['active']
    assert duplicate!=second and r.invoke('presets/load')[0]['f0']==210
    r.invoke('library/edit',{'action':'rename','name':'实验组合二'})
    r.invoke('library/edit',{'action':'select','id':first})
    assert r.invoke('presets/load')[0]['f0']==120
    assert r.invoke('favorites/load')==['own-a']
    assert r.invoke('keyframes/load')==sequence
    r.invoke('speaker/select',{'speaker':'W02'})
    assert len(r.invoke('library/load')['sets'])==1 and r.invoke('presets/load')==[]
    r.invoke('speaker/select',{'speaker':'JD2'})
    assert len(r.invoke('library/load')['sets'])==3 and r.invoke('presets/load')[0]['f0']==120
    if disk:assert ProfileStore(tmp_path).load_library(engine)==r.invoke('library/load')

def test_legacy_read_is_non_destructive_and_current_files_win(engine,tmp_path):
    old=tmp_path/'old';old.mkdir();current=tmp_path/'current'
    source=old/'presets.json';source.write_text(json.dumps({'version':2,'presets':[pose(engine)]}),encoding='utf-8')
    (old/'favorites.json').write_text(json.dumps({'version':1,'ids':['own-a']}),encoding='utf-8')
    original=source.read_bytes();profile=ProfileStore(current,legacy_directory=old)
    assert profile.load_presets(engine)[0]['name']=='a' and profile.load_favorites()==['own-a']
    assert not (current/'preset-libraries.json').exists()
    profile.save_presets(engine,[pose(engine,f0=190)])
    assert source.read_bytes()==original and profile.load_presets(engine)[0]['f0']==190
    profile.set_speaker('W02');assert profile.load_presets(engine)==[]
    clean=tmp_path/'clean';clean.mkdir();(clean/'presets.json').write_text(json.dumps({'version':2,'presets':[pose(engine,f0=230)]}),encoding='utf-8')
    assert ProfileStore(clean,legacy_directory=old).load_presets(engine)[0]['f0']==230
    source.write_text('{damaged',encoding='utf-8')
    with pytest.raises(ValueError):ProfileStore(tmp_path/'fresh',legacy_directory=old).load_presets(engine)
    assert source.read_text('utf-8')=='{damaged'

def test_custom_names_and_duplicate_recovery_share_one_contract(engine,tmp_path):
    profile=ProfileStore(tmp_path);item=pose(engine,'低 a 构型')
    saved=profile.save_presets(engine,[item]);pack=make_pack(engine,{'presets':saved})
    assert pack['version']==3 and validate_pack(engine,pack)['presets'][0]['name']=='低 a 构型'
    duplicate={**item,'id':'second','name':'/低 a 构型/'}
    with pytest.raises(ValueError,match='重复'):profile.save_presets(engine,[item,duplicate])
    assert profile.load_presets(engine)==saved
    duplicate['name']='低 a 构型二';assert len(profile.save_presets(engine,[item,duplicate]))==2
    for bad in ['', 'a\nb', 'a\x7fb']:
        with pytest.raises(ValueError):profile.save_presets(engine,[{**item,'name':bad}])

def test_multiple_legacy_duplicates_can_be_repaired_one_at_a_time(engine,tmp_path):
    items=[{**pose(engine),'id':f'old-{i}','name':'a' if i<3 else 'i'} for i in range(5)]
    source=tmp_path/'presets.json';source.write_text(json.dumps({'version':2,'presets':items}),encoding='utf-8')
    favorite=tmp_path/'favorites.json';favorite.write_text(json.dumps({'version':1,'ids':['old-0','removed']}),encoding='utf-8')
    original=source.read_bytes(),favorite.read_bytes();profile=ProfileStore(tmp_path)
    library=profile.load_library(engine);assert library['sets'][0]['favorites']==['old-0']
    for i in [1,2,4]:
        items[i]['name']=f'音标{i}'
        assert len(profile.save_presets(engine,items))==5
    assert (source.read_bytes(),favorite.read_bytes())==original

def test_old_pack_import_requires_confirmation_and_preserves_original(engine):
    r=Runtime(engine,MemoryProfile(),BrowserAudio(engine));r.invoke('presets/save',{'presets':[pose(engine)]})
    original=r.invoke('library/load');pack=make_pack(engine,{'title':'旧同事套装','presets':[pose(engine,f0=200)]})
    pack.update(version=2,geometry_method='m10/r20');before=copy.deepcopy(pack)
    preview=validate_pack(engine,pack)
    assert preview['geometry_method']=='m10/r20' and preview['target_geometry_method']=='m10/r21'
    with pytest.raises(ValueError,match='确认'):r.invoke('library/import',{'document':preview})
    assert r.invoke('library/load')==original
    imported=r.invoke('library/import',{'document':preview,'name':'旧同事套装','recompute':True})
    assert imported['added']==1 and len(imported['library']['sets'])==2 and pack==before
    assert r.invoke('presets/load')[0]['f0']==200
    r.invoke('library/edit',{'action':'select','id':original['active']})
    assert r.invoke('presets/load')[0]['f0']==120
    for update in [{'geometry_method':'future'},{'speaker_sha256':'wrong'},{'speaker':'W02'},{'parameters':list(reversed(engine.names))}]:
        with pytest.raises(ValueError):validate_pack(engine,{**pack,**update})

def test_failed_atomic_write_keeps_last_library_and_retry_succeeds(engine,tmp_path,monkeypatch):
    profile=ProfileStore(tmp_path);profile.save_presets(engine,[pose(engine)])
    before=(tmp_path/'preset-libraries.json').read_bytes();write=profile._write_library
    monkeypatch.setattr(profile,'_write_library',lambda value:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):profile.save_presets(engine,[pose(engine,f0=220)])
    assert (tmp_path/'preset-libraries.json').read_bytes()==before
    monkeypatch.setattr(profile,'_write_library',write);profile.save_presets(engine,[pose(engine,f0=220)])
    assert profile.load_presets(engine)[0]['f0']==220


@pytest.mark.parametrize('disk',[False,True])
@pytest.mark.parametrize('speaker',['JD2','W02'])
def test_delete_saved_pose_keeps_other_sets_models_frames_and_builtins(engine,tmp_path,disk,speaker):
    profile=ProfileStore(tmp_path) if disk else MemoryProfile()
    r=Runtime(engine,profile,BrowserAudio(engine));r.invoke('speaker/select',{'speaker':speaker})
    a=pose(engine);other={**pose(engine,'i'),'id':'own-i'}
    r.invoke('presets/save',{'presets':[a,other]});r.invoke('favorites/save',{'ids':['a','own-a','own-i']})
    r.invoke('keyframes',{'frames':[a]});sequence=r.invoke('keyframes/load')
    first=r.invoke('library/load')['active'];r.invoke('library/edit',{'action':'copy','name':'独立副本'})
    before=r.invoke('library/load');next_value=copy.deepcopy(before);current=next_value['sets'][-1]
    current['presets']=[p for p in current['presets'] if p['id']!='own-a']
    current['favorites']=[key for key in current['favorites'] if key!='own-a']
    assert r.invoke('library/save',next_value)==next_value
    assert r.invoke('favorites/load')==['a','own-i'] and r.invoke('presets/load')[0]['id']=='own-i'
    assert r.invoke('keyframes/load')==sequence and 'a' in engine.presets
    assert r.invoke('library/load')['sets'][0]==before['sets'][0]
    # Removing the last saved pose leaves builtins and an empty usable set.
    current['presets']=[];current['favorites']=['a'];r.invoke('library/save',next_value)
    assert r.invoke('presets/load')==[] and r.invoke('favorites/load')==['a']
    if disk:
        reopened=ProfileStore(tmp_path);reopened.set_speaker(speaker)
        assert reopened.load_library(engine)==next_value
    other_speaker='W02' if speaker=='JD2' else 'JD2'
    r.invoke('speaker/select',{'speaker':other_speaker});assert r.invoke('presets/load')==[]
    r.invoke('presets/save',{'presets':[pose(engine,f0=210)]});other_model=r.invoke('library/load')
    r.invoke('speaker/select',{'speaker':speaker});assert r.invoke('library/load')==next_value
    r.invoke('library/edit',{'action':'select','id':first});assert len(r.invoke('presets/load'))==2
    r.invoke('speaker/select',{'speaker':other_speaker});assert r.invoke('library/load')==other_model


def test_delete_snapshot_failure_keeps_disk_and_legacy_duplicate_names_can_be_retained(engine,tmp_path,monkeypatch):
    profile=ProfileStore(tmp_path);library=profile.load_library(engine)
    current=library['sets'][0];current['presets']=[pose(engine),{**pose(engine),'id':'old-a'},{**pose(engine,'u'),'id':'own-u'}]
    current['favorites']=['own-u','a'];profile.save_library(engine,library)
    before=(tmp_path/'preset-libraries.json').read_bytes();write=profile._write_library
    next_value=copy.deepcopy(library);next_value['sets'][0]['presets'].pop();next_value['sets'][0]['favorites']=['a']
    monkeypatch.setattr(profile,'_write_library',lambda value:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):profile.save_library(engine,next_value)
    assert (tmp_path/'preset-libraries.json').read_bytes()==before
    monkeypatch.setattr(profile,'_write_library',write);profile.save_library(engine,next_value)
    assert [p['id'] for p in profile.load_presets(engine)]==['own-a','old-a']
    assert profile.load_favorites()==['a']
