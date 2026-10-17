"""Prepare or apply the authorized Mac counter update, preserving live totals.

Uses the existing external SSH helper; no credentials are copied or logged.
Only the collector and its nginx statistics map are replaced. The existing
service/timer definitions, homepage, applications and databases are untouched.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shlex
import sys
import uuid

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,tempfile
p=json.loads(sys.stdin.readline())
collector=Path('/opt/phonetictoolbox-website-stats/collect_stats.py')
config=Path('/etc/nginx/conf.d/00-phonetictoolbox-stats.conf')
state=Path('/var/lib/phonetictoolbox-website-stats/state.json')
public=Path('/var/www/phonetictoolbox-coming-soon/site-stats.json')
history=public.with_name('site-stats-history.json')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(*args):return subprocess.run(args,check=True,capture_output=True,text=True)
def atomic(path,raw,mode):
    tmp=path.with_name(path.name+'.'+p['operation']+'.tmp')
    tmp.write_bytes(raw);os.chmod(tmp,mode);os.replace(tmp,path)
before={str(path):sha(path) for path in [collector,config]}
if p.get('expected'):assert before==p['expected'],'Statistics sources changed after review'
old=json.loads(state.read_text())
with tempfile.TemporaryDirectory(prefix='ptb-macos-stats-') as tmp:
    root=Path(tmp);root.chmod(0o755)
    for name,text in p['sources'].items():(root/name).write_text(text)
    integration=run(sys.executable,'-B',str(root/'test_nginx_stats.py'))
    checks=json.loads(integration.stdout.strip())
    (root/'state.json').write_bytes(state.read_bytes())
    spec=importlib.util.spec_from_file_location('candidate_stats',root/'collect_stats.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    preview=module.collect(Path('/var/log/nginx'),root/'state.json',root/'site-stats.json',backfill_access_logs=True)
    candidate=json.loads((root/'state.json').read_text())
    assert all(candidate['counts'][k]>=old['counts'][k] for k in ['page','portable','installer','web'])
    prior={point[0]:point[:5] for point in old['history']}
    assert all(point[:5]==prior[point[0]] for point in candidate['history'] if point[0] in prior)
    result={'success':True,'applied':False,'before':before,'integration':checks,'preview':preview,
            'backfill':candidate['macosBackfill'],'existingHistoryValuesPreserved':True}
    if p['apply']:
        timer_active=subprocess.run(['systemctl','is-active','--quiet','ptb-website-stats.timer']).returncode==0
        assert timer_active,'Existing statistics timer is not active'
        run('systemctl','stop','ptb-website-stats.timer')
        # Wait for any current atomic collection instead of interrupting it.
        with state.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
        backup=Path('/var/backups/phonetictoolbox-website')/p['operation']
        backup.mkdir(parents=True,mode=0o700)
        paths=[collector,config,state,public,history]
        originals={path:path.read_bytes() for path in paths}
        for i,(path,raw) in enumerate(originals.items()):(backup/str(i)).write_bytes(raw)
        (backup/'paths.json').write_text(json.dumps([str(path) for path in paths]))
        try:
            assert {str(path):sha(path) for path in [collector,config]}==before
            atomic(collector,p['sources']['collect_stats.py'].encode(),0o644)
            atomic(config,p['sources']['website-stats.nginx.conf'].encode(),0o644)
            run('nginx','-t');run('systemctl','reload','nginx')
            run(sys.executable,'-B',str(collector),'--backfill-macos-access-logs')
            run('systemctl','start','ptb-website-stats.service')
            actual=json.loads(public.read_text())
            assert actual['downloads']['total']==actual['downloads']['windows']+actual['downloads']['macos']
            assert all(json.loads(state.read_text())['counts'][k]>=old['counts'][k] for k in ['page','portable','installer','web'])
            run('systemctl','start','ptb-website-stats.timer')
            run('systemctl','is-active','--quiet','ptb-website-stats.timer')
            result.update(applied=True,backup=str(backup),statistics=actual,
                          backfill=json.loads(state.read_text())['macosBackfill'],after={str(path):sha(path) for path in [collector,config]})
            (backup/'receipt.json').write_text(json.dumps(result,indent=2))
        except BaseException:
            for path,raw in originals.items():atomic(path,raw,0o600 if path==state else 0o644)
            run('nginx','-t');run('systemctl','reload','nginx')
            run('systemctl','start','ptb-website-stats.timer')
            raise
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh-helper', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-report', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    payload = {'operation': 'macos-statistics-' + uuid.uuid4().hex, 'apply': args.apply,
               'sources': {name: (HERE/name).read_text('utf-8') for name in
                           ['collect_stats.py', 'website-stats.nginx.conf', 'test_nginx_stats.py']}}
    if args.expected_report:
        payload['expected'] = json.loads(args.expected_report.read_text('utf-8'))['before']
    sys.path.insert(0, str(args.ssh_helper.parent))
    spec = importlib.util.spec_from_file_location('stats_private_ssh', args.ssh_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with helper.connect() as client:
        stdin, stdout, stderr = client.exec_command('sudo -n python3 -c ' + shlex.quote(REMOTE), timeout=60)
        stdin.write(json.dumps(payload) + '\n')
        stdin.flush()
        stdin.channel.shutdown_write()
        output, error = stdout.read().decode('utf-8'), stderr.read().decode('utf-8')
        if stdout.channel.recv_exit_status():
            raise RuntimeError(error)
        result = json.loads(output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
