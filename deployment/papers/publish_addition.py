"""Publish one reviewed paper while preserving every existing entry and asset.

Credentials remain in the external SSH helper. The Linux publisher holds an
exclusive lock and checks the expected full-catalog SHA before any write.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from uuid import uuid4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--paper-id',required=True)
    p.add_argument('--ssh-helper',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo/'desktop/src'))
    from ptb_documents.papers import validate_catalog, catalog_papers, asset_kinds, paper_asset
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    before_raw=a.baseline.read_bytes();raw=(a.stage/'catalog.json').read_bytes()
    before=validate_catalog(json.loads(before_raw));after=validate_catalog(json.loads(raw))
    old={x['id']:x for x in catalog_papers(before)};new={x['id']:x for x in catalog_papers(after)}
    assert set(new)==set(old)|{a.paper_id} and a.paper_id not in old,'Expected exactly one new paper'
    assert all(new[k]==v for k,v in old.items()),'Existing paper changed'
    assert {k:v for k,v in before.items() if k not in ('papers','journalPapers')}=={k:v for k,v in after.items() if k not in ('papers','journalPapers')},'Notice or unrelated catalog field changed'
    for group in ('papers','journalPapers'):
        assert [x for x in after.get(group,[]) if x['id']!=a.paper_id]==before.get(group,[]),'Existing source group/order changed'
    paper=new[a.paper_id]
    assets=[paper_asset(paper,kind) for kind in asset_kinds(paper)]
    for asset in assets:
        assert asset['path'].startswith(a.paper_id+'/')
        data=(a.stage/asset['path']).read_bytes()
        assert len(data)==asset['size'] and sha(data)==asset['sha256'] and data.startswith(b'%PDF-')
    spec=importlib.util.spec_from_file_location('paper_private_ssh',a.ssh_helper)
    sys.path.insert(0,str(a.ssh_helper.parent));helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    def execute(client,command):
        i,o,e=client.exec_command(command,timeout=90);i.close();data=o.read();err=e.read()
        assert o.channel.recv_exit_status()==0,err.decode('utf8',errors='replace')
        return data
    root='/var/www/phonetictoolbox-coming-soon'
    existing_assets=[paper_asset(v,k) for v in old.values() for k in asset_kinds(v)]
    if before.get('notice'):existing_assets.append(before['notice']['pdf'])
    protected=[root+'/index.html',root+'/papers/catalog.json']+[root+'/papers/'+v['path'] for v in existing_assets]
    command='sha256sum '+' '.join(shlex.quote(x) for x in protected)
    with helper.connect() as client:
        hashes=execute(client,command).decode().splitlines()
        assert hashes[1].split()[0]==sha(before_raw),'Server catalog changed; refresh baseline'
        for line,asset in zip(hashes[2:],existing_assets):assert line.split()[0]==asset['sha256']
        report=dict(published=False,paper=a.paper_id,existingPapersPreserved=len(old),existingAssetsPreserved=len(existing_assets),previousCatalogSha256=sha(before_raw),newCatalogSha256=sha(raw))
        if a.apply:
            remote='/home/admin/papers-upload/addition-'+uuid4().hex
            with client.open_sftp() as sftp:
                sftp.mkdir(remote,0o700);sftp.mkdir(remote+'/'+a.paper_id,0o700)
                for name,body in [('previous-catalog.json',before_raw),('catalog.json',raw)]:
                    with sftp.open(remote+'/'+name,'wb') as stream:stream.write(body)
                for asset in assets:sftp.put(str(a.stage/asset['path']),remote+'/'+asset['path'])
                sftp.put(str(Path(__file__).with_name('publish.py')),remote+'/publish.py')
            receipt=json.loads(execute(client,'python3 '+shlex.quote(remote+'/publish.py')+' '+shlex.quote(remote)+' --expected-catalog-sha256 '+sha(before_raw)))
            actual=execute(client,command).decode().splitlines()
            assert actual[0]==hashes[0] and actual[2:]==hashes[2:],'Existing home page or paper asset changed'
            assert actual[1].split()[0]==sha(raw)
            report.update(published=True,receipt=receipt,serverStage=remote,homeAndExistingAssetHashesUnchanged=True)
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
