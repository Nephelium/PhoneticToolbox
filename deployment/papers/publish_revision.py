"""Publish a reviewed translation or editorial notice with a catalogue guard.

Credentials stay in the external SSH helper. --apply is required to write;
the original/analysis assets, other papers and home page must stay identical.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from uuid import uuid4


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument('--paper-id')
    kind.add_argument('--notice', action='store_true')
    parser.add_argument('--ssh-helper', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'desktop/src'))
    from ptb_desktop.papers import validate_catalog
    baseline = args.baseline.read_bytes()
    raw = (args.stage / 'catalog.json').read_bytes()
    old = validate_catalog(json.loads(baseline))
    new = validate_catalog(json.loads(raw))
    old_by_id = {p['id']: p for p in old['papers']}
    new_by_id = {p['id']: p for p in new['papers']}
    assert old_by_id.keys() == new_by_id.keys(), 'Unexpected paper addition/removal'
    if args.notice:
        assert old_by_id == new_by_id, 'Scientific paper changed'
        assert {k: v for k, v in old.items() if k != 'notice'} == {k: v for k, v in new.items() if k != 'notice'}, 'Unrelated catalogue field changed'
        after_notice = new['notice']; owner_id = after_notice['id']
        if old.get('notice'):
            assert old['notice']['id'] == owner_id, 'Notice identity changed'
            assert after_notice['updatedAt'] >= old['notice']['updatedAt']
        changed = {'notice'}; asset = after_notice['pdf']
    else:
        assert {k: v for k, v in old.items() if k != 'papers'} == {k: v for k, v in new.items() if k != 'papers'}, 'Unrelated catalogue field changed'
        before_paper = old_by_id[args.paper_id]
        after_paper = new_by_id[args.paper_id]; owner_id = args.paper_id
        for pid, paper in old_by_id.items():
            if pid != args.paper_id:
                assert new_by_id[pid] == paper, 'Unrelated paper changed'
        changed = {key for key in set(before_paper) | set(after_paper)
                   if before_paper.get(key) != after_paper.get(key)}
        assert changed == {'translation', 'translationNote', 'updatedAt'}, changed
        assert after_paper['updatedAt'] > before_paper.get('updatedAt', before_paper['publishedAt'])
        assert before_paper['license']['id'] == after_paper['license']['id'] == 'CC-BY-4.0'
        asset = after_paper['translation']
    assert asset['path'].startswith(owner_id + '/'), 'Unexpected asset directory'
    data = (args.stage / asset['path']).read_bytes()
    assert len(data) == asset['size'] and digest(data) == asset['sha256'] and data.startswith(b'%PDF-')
    spec = importlib.util.spec_from_file_location('paper_private_ssh', args.ssh_helper)
    sys.path.insert(0, str(args.ssh_helper.parent))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    def execute(client, command):
        stdin, stdout, stderr = client.exec_command(command, timeout=30)
        stdin.close()
        result, error = stdout.read(), stderr.read()
        assert stdout.channel.recv_exit_status() == 0, error.decode('utf8', errors='replace')
        return result

    server_root = '/var/www/phonetictoolbox-coming-soon'
    hashes_command = 'sha256sum ' + server_root + '/papers/catalog.json ' + server_root + '/index.html'
    with helper.connect() as client:
        hashes = execute(client, hashes_command).decode().splitlines()
        assert hashes[0].split()[0] == digest(baseline), 'Server catalogue changed; refresh baseline'
        report = {'paper': owner_id, 'published': False,
                  'previousCatalogSha256': digest(baseline), 'changedFields': sorted(changed),
                  'newAsset': asset, 'homePageSha256': hashes[1].split()[0]}
        if args.apply:
            remote = '/home/admin/papers-upload/translation-revision-' + uuid4().hex
            with client.open_sftp() as sftp:
                sftp.mkdir(remote, 0o700)
                sftp.mkdir(remote + '/' + owner_id, 0o700)
                with sftp.open(remote + '/previous-catalog.json', 'wb') as stream:
                    stream.write(baseline)
                sftp.put(str(args.stage / 'catalog.json'), remote + '/catalog.json')
                sftp.put(str(args.stage / asset['path']), remote + '/' + asset['path'])
                sftp.put(str(Path(__file__).with_name('publish.py')), remote + '/publish.py')
            # Existing publisher hashes every referenced asset, then replaces
            # catalogue last. Old content-addressed files are retained.
            receipt = json.loads(execute(client, 'python3 ' + remote + '/publish.py ' + remote))
            after = execute(client, hashes_command).decode().splitlines()
            assert after[0].split()[0] == digest(raw)
            assert after[1].split()[0] == hashes[1].split()[0], 'Home page changed'
            report.update(published=True, catalogSha256=digest(raw), receipt=receipt,
                          serverStage=remote, homePageUnchanged=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
