"""Preview/apply cleanup of a successful local acceptance's disposable copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
from owned_qa_paths import DEFAULT_QA_ROOT, qa_root


def no_links(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Linked cleanup path: ' + str(part))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--reviewed-incomplete', action='store_true', help='Cleanup an abandoned run after reviewing diagnostics; acceptance stays failed')
    args = parser.parse_args()
    file = args.report.absolute()
    out = file.parent
    no_links(file)
    if out.parent != ROOT / 'output/validation' or not out.name.startswith('local-package-') or file.name != 'report.json':
        raise ValueError('Expected the owned acceptance report')
    report = json.loads(file.read_text('utf8'))
    if report.get('schema') != 'ptb-local-acceptance/1' or (report.get('success') is not True and not args.reviewed_incomplete):
        raise ValueError('Acceptance has not succeeded; inspect failures before cleanup')
    token = out.name.removeprefix('local-package-')
    if len(token) != 8 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('Unexpected run identity')
    external = Path(report['external']).absolute()
    if external != qa_root(report.get('qaRoot', DEFAULT_QA_ROOT)) / ('local-' + token):
        raise ValueError('External QA ownership mismatch')
    targets = [external, out / 'owned-setup', out / 'owned-update.zip', out / 'portable-origin', out / 'installer-origin']
    for kind in ('portable', 'installer'):
        targets.extend(out / (kind + '-handoff') / part for part in ('profile', 'temp', 'vocal'))
        for retry in out.glob(kind + '-handoff-retry-*'):
            targets.extend(retry / part for part in ('profile', 'temp', 'vocal'))
    for evidence in [*out.glob('portable-handoff*'), *external.glob('portable-handoff*')]:
        handoff = evidence / 'handoff.json'
        if handoff.is_file():
            request = Path(json.loads(handoff.read_text('utf8'))['request'])
            no_links(request)
            metadata = json.loads(handoff.read_text('utf8'))
            new_app = Path(metadata['origin']).absolute().parent.parent / ('PhoneticToolbox-' + report['version'] + '-' + request.parent.name[:8])
            if new_app.exists():
                if new_app.parent not in (out, external) or not new_app.name.startswith('PhoneticToolbox-' + report['version'] + '-'):
                    raise ValueError('Portable handoff target is outside the owned run')
                no_links(new_app)
                descriptor = json.loads((new_app / 'PhoneticToolbox/application.json').read_text('utf8'))
                if descriptor['executable']['sha256'] != report['exeSha256']:
                    raise ValueError('Portable fixture identity differs')
                if not new_app.is_relative_to(external):targets.append(new_app)
    # Exclude this cleanup process and its command-only parent, which reference
    # the report; no executable or command may still reference a removal target.
    processes = json.loads(subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
        '[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress'], encoding='utf8'))
    by_pid = {p['ProcessId']: p for p in processes}
    ignored = {os.getpid()}
    parent = by_pid.get(os.getpid(), {}).get('ParentProcessId')
    while parent and parent not in ignored:
        ignored.add(parent)
        parent = by_pid.get(parent, {}).get('ParentProcessId')
    for process in processes:
        if process['ProcessId'] in ignored:
            continue
        command = (process['CommandLine'] or '').casefold()
        executable = (process['ExecutablePath'] or '').casefold()
        if any(str(p).casefold() in command or executable.startswith(str(p).casefold() + os.sep) for p in targets):
            raise ValueError('Owned test target still in use by PID ' + str(process['ProcessId']))
    rows = []
    runtime_entries = []
    retained = out / 'evidence'
    for target in targets:
        no_links(target)
        if not target.exists():
            continue
        paths = [target] if target.is_file() else []
        if target.is_dir():
            for directory, folders, files in os.walk(target, followlinks=False):
                for name in list(folders) + files:
                    path = Path(directory) / name
                    if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                        destination = path.resolve()
                        if (name not in folders or name not in ('r', '_internal') or
                                not re.fullmatch('ptb-rt-[0-9a-f]{32}', path.parent.name) or
                                not path.parent.resolve().is_relative_to(external) or
                                not any(destination.is_relative_to(p) for p in targets if p.is_dir())):
                            raise ValueError('Unowned link inside disposable QA tree: ' + str(path))
                        no_links(destination)
                        runtime_entries.append({'path': str(path), 'target': str(destination)})
                        folders.remove(name)
                paths.extend(Path(directory) / name for name in files)
        for path in paths:
            rows.append({'path': str(path), 'bytes': path.stat().st_size})
            cached_runtime = 'startup-cache' in path.parts and path.name not in ('last-launch.json', 'owner.json')
            extracted_bootstrap = any(part.startswith('_MEI') for part in path.parts)
            if args.apply and not cached_runtime and not extracted_bootstrap and path.suffix.lower() in {'.json', '.log', '.png', '.txt'}:
                prefix = 'external' if path.is_relative_to(external) else 'internal'
                relative = path.relative_to(external if prefix == 'external' else out)
                destination = retained / prefix / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                with path.open('rb') as source, destination.open('rb') as copied:
                    if hashlib.file_digest(source, 'sha256').digest() != hashlib.file_digest(copied, 'sha256').digest():
                        raise ValueError('Retained evidence differs')
    receipt = {'schema': 'ptb-local-qa-cleanup/1', 'status': 'planned', 'targets': list(map(str, targets)),
               'files': len(rows), 'logicalBytes': sum(row['bytes'] for row in rows),
               'retainedEvidence': str(retained), 'scope': 'Generated QA copies only; current EXE/build inputs, user files and source untouched'}
    receipt['acceptanceSuccess'] = report.get('success') is True
    receipt['reviewedIncomplete'] = args.reviewed_incomplete
    receipt['ownedRuntimeEntries'] = runtime_entries
    (out / 'cleanup-plan.json').write_text(json.dumps(receipt, indent=2) + '\n', 'utf8')
    if args.apply:
        for entry in runtime_entries:
            path = Path(entry['path'])
            if path.resolve() != Path(entry['target']):
                raise ValueError('Runtime entry target changed')
            # Remove the validated private junction entry, never its target.
            os.rmdir(path)
        for target in targets:
            # Recheck final absolute roots immediately before recursive removal.
            no_links(target)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        receipt['status'] = 'deleted'
        (out / 'cleanup.json').write_text(json.dumps(receipt, indent=2) + '\n', 'utf8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
