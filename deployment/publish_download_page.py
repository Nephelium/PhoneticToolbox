"""Publish the reviewed static page after its two release downloads exist.

Uses existing nginx configuration. Preserves paper/feedback routes and saves
the previous index privately for rollback. No credentials in arguments/logs.
"""
import argparse,hashlib,importlib.util,json,re,shlex,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

REMOTE=r'''
import hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
p=json.loads(__import__('sys').stdin.readline())
root=Path('/var/www/phonetictoolbox-coming-soon');assert root.resolve()==root
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for relative,row in p['downloads'].items():
    file=root/relative;assert file.resolve().is_relative_to(root/'releases')
    assert file.stat().st_size==row['size'] and sha(file)==row['sha256']
catalog=root/'papers/catalog.json';catalog_before=sha(catalog)
subprocess.run(['systemctl','is-active','--quiet','ptb-feedback.service'],check=True)
archive=Path('/home/admin')/('ptb-page-'+p['operation']+'.tar.gz')
assert archive.resolve()==archive and sha(archive)==p['archive_sha256']
target=root/'site-releases'/p['release'];assert target.resolve().is_relative_to(root/'site-releases')
target.mkdir(mode=0o755,parents=True,exist_ok=True)
with tarfile.open(archive,'r:gz') as tar:
    assert set(tar.getnames())==set(p['files'])
    for member in tar.getmembers():
        assert member.isfile() and '..' not in Path(member.name).parts and not member.name.startswith('/')
        content=tar.extractfile(member).read();row=p['files'][member.name]
        assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
        file=target/member.name;assert file.resolve().is_relative_to(target)
        file.parent.mkdir(parents=True,exist_ok=True)
        if file.exists():assert sha(file)==row['sha256']
        else:file.write_bytes(content);os.chmod(file,0o644)
backup=Path('/var/backups/phonetictoolbox-website')/p['operation'];backup.mkdir(parents=True,mode=0o700)
(backup/'index.html').write_bytes((root/'index.html').read_bytes())
pending=root/('index.'+p['operation']+'.tmp');pending.write_bytes((target/'index.html').read_bytes());os.chmod(pending,0o644)
os.replace(pending,root/'index.html')
assert sha(catalog)==catalog_before
subprocess.run(['systemctl','is-active','--quiet','ptb-feedback.service'],check=True)
archive.unlink()
print(json.dumps({'success':True,'release':p['release'],'indexSha256':sha(root/'index.html'),'backup':str(backup),'papersUnchanged':True,'feedbackActive':True,'nginxConfigurationChanged':False}))
'''

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ssh-helper',type=Path,required=True);ap.add_argument('--package-report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    release,files,archive=load('page_builder',ROOT/'deployment/temporary-site/publish.py').build()
    packages=json.loads(args.package_report.read_text('utf8'));version=packages['version'];assert re.fullmatch(r'\d+\.\d+\.\d+',version)
    names=['PhoneticToolbox-windows-x64.exe',f'PhoneticToolbox-{version}-windows-x64-setup.exe']
    html=(ROOT/'temporary-site/index.html').read_text('utf8')
    downloads={f'releases/windows-x64/{version}/'+n:packages['files'][n] for n in names}
    assert all('href="/'+n+'"' in html for n in downloads)
    payload={'release':release,'operation':'downloads-'+uuid.uuid4().hex,'files':files,'downloads':downloads,'archive_sha256':hashlib.sha256(archive).hexdigest()}
    if not args.apply:print(json.dumps({'release':release,'downloads':downloads}));return
    sys.path.insert(0,str(args.ssh_helper.parent));helper=load('page_ssh',args.ssh_helper)
    with helper.connect() as client:
        with client.open_sftp() as sftp:
            with sftp.open('/home/admin/ptb-page-'+payload['operation']+'.tar.gz','wb') as f:f.set_pipelined(True);f.write(archive)
        stdin,stdout,stderr=client.exec_command('sudo -n python3 -c '+shlex.quote(REMOTE),timeout=300)
        stdin.write(json.dumps(payload)+'\n');stdin.flush();stdin.channel.shutdown_write()
        out=stdout.read().decode();error=stderr.read().decode();assert stdout.channel.recv_exit_status()==0,error
        receipt=json.loads(out);assert receipt['success'];args.output.write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(out)
if __name__=='__main__':main()
