"""Install the explicitly requested anonymous website counters, with rollback."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import uuid

HERE = Path(__file__).resolve().parent
SERVICE = '''[Unit]
Description=PhoneticToolbox anonymous website counters
After=nginx.service

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /opt/phonetictoolbox-website-stats/collect_stats.py
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/phonetictoolbox-website-stats /var/www/phonetictoolbox-coming-soon
RestrictAddressFamilies=AF_UNIX
'''
TIMER = '''[Unit]
Description=Refresh PhoneticToolbox website counters every minute

[Timer]
OnBootSec=20s
OnUnitActiveSec=60s
AccuracySec=5s
Unit=ptb-website-stats.service

[Install]
WantedBy=timers.target
'''
REMOTE = r'''
from pathlib import Path
import hashlib,json,os,subprocess
conf=Path('/etc/nginx/conf.d/phonetictoolbox-coming-soon.conf')
root=Path('/var/www/phonetictoolbox-coming-soon')
backup=Path('/var/backups/phonetictoolbox-website')/PAYLOAD['operation']
backup.mkdir(parents=True,mode=0o700)
oldconf=conf.read_bytes()
catalog=root/'papers/catalog.json'
before=hashlib.sha256(catalog.read_bytes()).hexdigest()
files={Path(k):v.encode() for k,v in PAYLOAD['files'].items()}
prior={p:p.read_bytes() if p.exists() else None for p in [conf,*files]}
for i,(p,data) in enumerate(prior.items()):
    if data is not None:(backup/str(i)).write_bytes(data)
(backup/'paths.json').write_text(json.dumps({str(p):i for i,p in enumerate(prior)}))
def write(p,data,mode=0o644):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.stats.tmp');t.write_bytes(data);os.chmod(t,mode);os.replace(t,p)
def run(*args):return subprocess.run(args,check=True,capture_output=True,text=True)
timer_was_active=subprocess.run(['systemctl','is-active','--quiet','ptb-website-stats.timer']).returncode==0
try:
    for p,data in files.items():write(p,data)
    state=Path('/var/lib/phonetictoolbox-website-stats');state.mkdir(exist_ok=True,mode=0o700);os.chmod(state,0o700)
    anchor='    root /var/www/phonetictoolbox-coming-soon;'
    text=oldconf.decode();assert text.count(anchor)==1
    if 'access_log /var/log/nginx/ptb-website-stats.log' not in text:
        # Keep the inherited ordinary log as well. Feedback retains access_log off.
        text=text.replace(anchor,'    access_log /var/log/nginx/access.log combined;\n    access_log /var/log/nginx/ptb-website-stats.log ptb_website_stats if=$ptb_stats_loggable;\n\n'+anchor)
    write(conf,text.encode())
    run('nginx','-t')
    run('systemctl','daemon-reload')
    run('systemctl','start','ptb-website-stats.service')
    run('systemctl','enable','--now','ptb-website-stats.timer')
    run('systemctl','reload','nginx')
    run('systemctl','is-active','--quiet','ptb-feedback.service')
    assert before==hashlib.sha256(catalog.read_bytes()).hexdigest()
except Exception:
    if not timer_was_active:subprocess.run(['systemctl','disable','--now','ptb-website-stats.timer'],capture_output=True)
    for p,data in prior.items():
        if data is None:p.unlink(missing_ok=True)
        else:write(p,data)
    subprocess.run(['systemctl','daemon-reload'],capture_output=True)
    run('nginx','-t');run('systemctl','reload','nginx')
    raise
receipt={'success':True,'backup':str(backup),'timer':run('systemctl','is-active','ptb-website-stats.timer').stdout.strip(),'statistics':json.loads((root/'site-stats.json').read_text()),'papers_catalog_unchanged':True,'feedback_active':True}
(backup/'receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ssh-helper',type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    payload={'operation':'stats-'+uuid.uuid4().hex,'files':{
        '/opt/phonetictoolbox-website-stats/collect_stats.py':(HERE/'collect_stats.py').read_text('utf-8'),
        '/etc/nginx/conf.d/00-phonetictoolbox-stats.conf':(HERE/'website-stats.nginx.conf').read_text('utf-8'),
        '/etc/systemd/system/ptb-website-stats.service':SERVICE,
        '/etc/systemd/system/ptb-website-stats.timer':TIMER}}
    if not args.apply:
        print(json.dumps({'apply':False,'files':list(payload['files']),'site_config':'/etc/nginx/conf.d/phonetictoolbox-coming-soon.conf'},indent=2));return
    assert args.ssh_helper and args.ssh_helper.is_file()
    sys.path.insert(0,str(args.ssh_helper.parent))
    spec=importlib.util.spec_from_file_location('ptb_private_ssh',args.ssh_helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    program='PAYLOAD='+repr(payload)+'\n'+REMOTE
    with module.connect() as client:
        stdin,stdout,stderr=client.exec_command('sudo -n python3 -',timeout=90)
        stdin.write(program);stdin.channel.shutdown_write()
        output=stdout.read().decode();error=stderr.read().decode();code=stdout.channel.recv_exit_status()
        if code:raise RuntimeError(error)
        receipt=json.loads(output)
    report=HERE.parents[1]/'output/validation/website'/payload['operation'];report.mkdir(parents=True)
    (report/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
