import json
from pathlib import Path
import subprocess
import sys
import pytest
from ptb_desktop.update_cache import clean_update_cache, RETENTION_SECONDS
from ptb_desktop.updates import _atomic_json


def folder(root,family,key):
    value=root/family/(key*32);value.mkdir(parents=True);(value/'keep').write_bytes(b'fixture');return value


def test_first_seen_grace_expiry_and_session_download_protection(tmp_path):
    root=tmp_path/'updates'
    stale=folder(root,'downloads','a');session=folder(root,'downloads','b')
    unrelated=tmp_path/'user';unrelated.mkdir();(unrelated/'record.wav').write_bytes(b'user')
    assert clean_update_cache(root,now=1)['removedDirectories']==0
    result=clean_update_cache(root,now=1+RETENTION_SECONDS,protected_downloads=['b'*32])
    assert result['removedDirectories']==1 and not stale.exists() and session.exists()
    assert (unrelated/'record.wav').read_bytes()==b'user'


def test_pending_plan_protects_helper_and_package(tmp_path):
    root=tmp_path/'updates';download=folder(root,'downloads','a');helper=folder(root,'helpers','b');pending=folder(root,'apply','c')
    _atomic_json(pending/'request.json',{'package':str(download/'payload.zip'),'helperExe':str(helper/'PhoneticToolbox.exe')})
    _atomic_json(pending/'status.json',{'state':'waiting'})
    clean_update_cache(root,now=1)
    assert clean_update_cache(root,now=1+RETENTION_SECONDS*2)['removedDirectories']==0
    _atomic_json(pending/'status.json',{'state':'failed'})
    assert clean_update_cache(root,now=1+RETENTION_SECONDS*2)['removedDirectories']==3


def test_unknown_pending_plan_retains_all_and_reports_partial(tmp_path):
    root=tmp_path/'updates';download=folder(root,'downloads','a');helper=folder(root,'helpers','b');pending=folder(root,'apply','c')
    clean_update_cache(root,now=1)
    result=clean_update_cache(root,now=1+RETENTION_SECONDS*2)
    assert result['state']=='partial' and result['removedDirectories']==0
    assert download.exists() and helper.exists() and pending.exists()


def test_delete_failure_keeps_registration_and_reports_partial(tmp_path):
    root=tmp_path/'updates';download=folder(root,'downloads','a');clean_update_cache(root,now=1)
    def denied(path):raise PermissionError('fixture failure')
    result=clean_update_cache(root,now=1+RETENTION_SECONDS,remove=denied)
    assert result['state']=='partial' and result['removedDirectories']==0 and download.exists()
    assert 'downloads/'+'a'*32 in json.loads((root/'cache-index.json').read_text('utf-8'))['entries']


def test_link_within_old_owned_directory_blocks_recursive_delete(tmp_path):
    root=tmp_path/'updates';download=folder(root,'downloads','a');outside=tmp_path/'user';outside.mkdir();(outside/'audio.wav').write_bytes(b'user')
    clean_update_cache(root,now=1)
    try:(download/'link').symlink_to(outside,target_is_directory=True)
    except OSError:
        if sys.platform!='win32':pytest.skip('symlink privilege unavailable')
        created=subprocess.run(['cmd','/c','mklink','/J',str(download/'link'),str(outside)],capture_output=True)
        if created.returncode:pytest.skip('Windows junction unavailable')
    result=clean_update_cache(root,now=1+RETENTION_SECONDS)
    assert result['state']=='partial' and download.exists() and (outside/'audio.wav').read_bytes()==b'user'
