"""M01-E directory capabilities; no access to personal or legacy input data."""
from pathlib import Path
import os
import subprocess
import pytest
from ptb_desktop.file_provider import FileProvider, FileAccessError


def fixtures(tmp_path):
    folder=tmp_path/'input';folder.mkdir()
    (folder/'a.wav').write_bytes(b'original input')
    (folder/'a.TextGrid').write_text('labels',encoding='utf-8')
    (folder/'secret.txt').write_text('not a supported resource')
    return folder


def test_directory_handles_are_scoped_and_cancel_preserves_existing(tmp_path):
    folder=fixtures(tmp_path);a,b=FileProvider(),FileProvider()
    grant=a.choose('input',lambda:folder)
    assert a.choose('input',lambda:None) is None
    files=a.list(grant['id']);assert {f['name'] for f in files}=={'a.wav','a.TextGrid'}
    assert all('path' not in f for f in files)
    with pytest.raises(FileAccessError):b.list(grant['id'])
    wav=next(f for f in files if f['kind']=='audio')
    with pytest.raises(FileAccessError):b.read(wav['id'])
    assert a.read(wav['id'])[0]==b'original input'
    a.close()
    with pytest.raises(FileAccessError):a.read(wav['id'])


def test_replaced_file_and_wrong_directory_are_rejected(tmp_path):
    folder=fixtures(tmp_path);p=FileProvider();d=p.choose('input',lambda:folder)
    wav=next(f for f in p.list(d['id']) if f['kind']=='audio')
    (folder/'a.wav').write_bytes(b'changed')
    with pytest.raises(FileAccessError):p.read(wav['id'])
    with pytest.raises(FileAccessError):p.list(str(folder))
    with pytest.raises(FileAccessError):p.read('../a.wav')


def test_same_directory_result_plan_cannot_overwrite_wav_or_existing_output(tmp_path):
    folder=fixtures(tmp_path);p=FileProvider();d=p.choose('input',lambda:folder)
    out=p.choose('output',lambda:folder);wav=next(f for f in p.list(d['id']) if f['kind']=='audio')
    assert p.output_target(out['id'],wav['id'],'xlsx')==folder/'a.xlsx'
    with pytest.raises(FileAccessError):p.output_target(out['id'],wav['id'],'wav')
    with pytest.raises(FileAccessError):p.output_target(out['id'],wav['id'],'../escape')
    (folder/'a.xlsx').write_bytes(b'previous result')
    with pytest.raises(FileAccessError):p.output_target(out['id'],wav['id'],'xlsx')
    assert (folder/'a.wav').read_bytes()==b'original input'


def test_file_budget_and_output_only_grant(tmp_path):
    folder=fixtures(tmp_path);p=FileProvider(max_bytes=4)
    out=p.choose('output',lambda:folder)
    with pytest.raises(FileAccessError):p.list(out['id'])
    inp=p.choose('input',lambda:folder);wav=next(f for f in p.list(inp['id']) if f['kind']=='audio')
    with pytest.raises(FileAccessError):p.read(wav['id'])


def test_same_physical_file_has_one_id_across_read_grants_and_hardlinks_are_rejected(tmp_path):
    folder=fixtures(tmp_path);p=FileProvider()
    a=p.choose('input',lambda:folder);b=p.choose('association',lambda:folder)
    assert p.list(a['id'])==p.list(b['id'])
    wav=next(f for f in p.list(a['id']) if f['kind']=='audio')
    os.link(folder/'a.wav',tmp_path/'linked.wav')
    with pytest.raises(FileAccessError):p.read(wav['id'])
    assert all(f['kind']!='audio' for f in p.list(a['id']))


def test_directory_entry_budget_fails_explicitly(tmp_path):
    folder=fixtures(tmp_path);p=FileProvider(max_entries=1)
    d=p.choose('input',lambda:folder)
    with pytest.raises(FileAccessError):p.list(d['id'])


def test_host_network_allowlist_is_bound_to_its_owned_service():
    from ptb_desktop.host import permitted_url
    service='http://127.0.0.1:12345'
    for url in ('ptbapp://app/index.html','qrc:///qtwebchannel/qwebchannel.js',service+'/api/v1/health'):
        assert permitted_url(url,service)
    for url in ('file:///C:/Users','ptbapp://foreign/index.html','qrc:///unrelated.js',
                'http://127.0.0.1:54321/api/v1/health',service+'.evil.test/api/v1/health',
                service+'/other','https://example.com'):
        assert not permitted_url(url,service)


@pytest.mark.skipif(os.name!='nt',reason='Windows junction acceptance')
def test_junction_directory_and_replaced_root_escape_are_rejected(tmp_path):
    folder=fixtures(tmp_path);outside=tmp_path/'outside';outside.mkdir()
    junction=tmp_path/'link'
    # New owned test paths only; mklink does not modify either target.
    subprocess.run(['cmd','/c','mklink','/J',str(junction),str(outside)],check=True,capture_output=True)
    p=FileProvider()
    with pytest.raises(FileAccessError):p.choose('input',lambda:junction)
    d=p.choose('input',lambda:folder);wav=next(f for f in p.list(d['id']) if f['kind']=='audio')
    moved=tmp_path/'moved';folder.rename(moved)
    subprocess.run(['cmd','/c','mklink','/J',str(folder),str(outside)],check=True,capture_output=True)
    with pytest.raises(FileAccessError):p.read(wav['id'])
