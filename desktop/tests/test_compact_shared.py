"""P19: content sharing preserves both paths; corrupt maps never complete."""
import hashlib
from io import BytesIO
import json
import os
import tarfile

import pytest
from ptb_desktop import compact_host


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(root):
    source='runtimes/m05/Lib/site-packages/shared.pyd'
    body=b'original shared native bytes'
    with tarfile.open(root/'runtime-payload.tar.xz','w:xz') as archive:
        info=tarfile.TarInfo(source);info.size=len(body);archive.addfile(info,BytesIO(body))
    row=dict(size=len(body),sha256=sha(body))
    (root/'runtime-files.json').write_text(json.dumps(dict(schema='ptb-runtime-files/1',files={source:row})),'utf8')
    science=dict(schema='ptb-runtime-archive/1',archive='runtime-payload.tar.xz',manifest='runtime-files.json',
                 size=(root/'runtime-payload.tar.xz').stat().st_size,sha256=sha((root/'runtime-payload.tar.xz').read_bytes()),
                 manifestSha256=sha((root/'runtime-files.json').read_bytes()),files=1,expandedBytes=len(body))
    (root/'desktop-bundle.json').write_text(json.dumps(dict(runtimeArchive=science)),'utf8')
    with tarfile.open(root/'host-payload.tar.xz','w:xz') as archive:
        info=tarfile.TarInfo('QtCore.dll');info.size=2;archive.addfile(info,BytesIO(b'Qt'))
    files={'QtCore.dll':dict(size=2,sha256=sha(b'Qt')),'one/shared.pyd':row,'two/shared.pyd':row}
    value=dict(schema='ptb-host-files/2',files=files,sharedFiles={'one/shared.pyd':source,'two/shared.pyd':source},
               runtimeArchiveSha256=science['sha256'],runtimeManifestSha256=science['manifestSha256'])
    descriptor=dict(schema='ptb-host-archive/2',size=(root/'host-payload.tar.xz').stat().st_size,
                    sha256=sha((root/'host-payload.tar.xz').read_bytes()),expandedBytes=2+2*len(body),files=3,
                    sharedFiles=2,sharedBytes=2*len(body))
    save(root,value,descriptor)
    return value,descriptor,source


def save(root,value,descriptor):
    raw=json.dumps(value).encode()
    (root/'host-files.json').write_bytes(raw)
    descriptor['manifestSha256']=sha(raw)
    (root/'host-archive.json').write_text(json.dumps(descriptor),'utf8')


def test_link_identity_and_completed_children(tmp_path):
    value,_,source=fixture(tmp_path)
    compact_host.expand(tmp_path);compact_host.expand(tmp_path)
    for target in value['sharedFiles']:
        assert os.path.samefile(tmp_path/source,tmp_path/target)
        assert (tmp_path/target).read_bytes()==b'original shared native bytes'
    receipt=json.loads((tmp_path/'.ptb-host-ready.json').read_text('utf8'))
    assert receipt['restoration']['linkedFiles']==2
    assert receipt['restoration']['copiedFiles']==0


def test_copy_fallback_when_volume_has_no_hardlinks(tmp_path,monkeypatch):
    value,_,source=fixture(tmp_path)
    def unsupported(*args):raise OSError('links unsupported')
    monkeypatch.setattr(compact_host.os,'link',unsupported)
    compact_host.expand(tmp_path)
    for target in value['sharedFiles']:
        assert not os.path.samefile(tmp_path/source,tmp_path/target)
        assert (tmp_path/target).read_bytes()==(tmp_path/source).read_bytes()
    receipt=json.loads((tmp_path/'.ptb-host-ready.json').read_text('utf8'))
    assert receipt['restoration']['copiedFiles']==2


@pytest.mark.parametrize('damage',['outside','different-content','wrong-science','unknown-target','collision','corrupt-source','case-alias','manifest-rebound'])
def test_bad_sharing_fails_before_ready(tmp_path,damage):
    value,descriptor,source=fixture(tmp_path)
    if damage=='outside':value['sharedFiles']['one/shared.pyd']='runtimes/egg/../../outside'
    elif damage=='different-content':value['files']['one/shared.pyd']['sha256']='0'*64
    elif damage=='wrong-science':value['runtimeArchiveSha256']='0'*64
    elif damage=='unknown-target':value['sharedFiles']['unknown.pyd']=source
    elif damage=='collision':
        target=tmp_path/'one/shared.pyd';target.parent.mkdir();target.write_bytes(b'keep')
    elif damage=='corrupt-source':
        from ptb_desktop.compact_runtime import expand
        expand(tmp_path);(tmp_path/source).write_bytes(b'damaged')
    elif damage=='case-alias':
        value['files']['ONE/SHARED.PYD']=value['files']['one/shared.pyd']
        descriptor['files']+=1;descriptor['expandedBytes']+=value['files']['one/shared.pyd']['size']
    elif damage=='manifest-rebound':
        compact_host.expand(tmp_path)
        value['sharedFiles']['one/shared.pyd']='runtimes/egg/missing'
    save(tmp_path,value,descriptor)
    with pytest.raises((ValueError,FileExistsError)):compact_host.expand(tmp_path)
    if damage!='manifest-rebound':assert not (tmp_path/'.ptb-host-ready.json').exists()
    if damage=='collision':assert target.read_bytes()==b'keep'
