"""Validate v3 documents, metadata, local links and dependency registration."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ('docs/', 'contracts/', 'backend/', 'desktop/src/', 'desktop/tests/', 'frontend/src/',
          'frontend/tests/', 'frontend/scripts/', 'packages/phonetic_core/', 'scripts/', 'release/', 'third_party/')


def main():
    listed = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    paths = sorted(set(listed.decode('utf-8').split('\0')) - {''})
    errors, checked, archived_links = [], 0, []
    archived = {d['path']: d['content_sha256_lf'] for d in json.loads(
        (ROOT / 'docs/baseline/archived-documents.json').read_text('utf-8'))['documents']}
    for relative in paths:
        if relative not in {'README.md', 'AGENTS.md', 'ARCHITECTURE.md', 'frontend/package.json'} and not relative.startswith(SCOPES):
            continue
        path = ROOT / relative
        if path.suffix not in {'.md', '.json', '.py', '.toml', '.ts', '.vue', '.mjs'} or not path.is_file():
            continue
        checked += 1
        raw = path.read_bytes()
        # Git may convert text line endings during checkout. Preserve all text
        # bytes except CRLF/LF when auditing unchanged historical content.
        if relative in archived and hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() != archived[relative]:
            errors.append(relative + ': historical snapshot hash changed; review required')
        if raw.startswith(b'\xef\xbb\xbf'):
            errors.append(relative + ': UTF-8 BOM')
        try:
            content = raw.decode('utf-8')
            if path.suffix == '.json':
                json.loads(content)
            if path.suffix == '.py':
                ast.parse(content)
            if path.suffix == '.md':
                # Exclude code fences; examples are not claimed document links.
                prose = re.sub(r'```.*?```', '', content, flags=re.S)
                for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', prose):
                    link = link.strip().strip('<>')
                    parsed = urlsplit(link)
                    if not parsed.path or parsed.scheme or parsed.netloc:
                        continue
                    target = (path.parent / unquote(parsed.path)).resolve()
                    if not target.exists():
                        destination = archived_links if relative in archived else errors
                        destination.append(relative + ': missing link ' + link)
        except (ValueError, SyntaxError, UnicodeError) as exc:
            errors.append(relative + ': ' + str(exc))
    registry = json.loads((ROOT / 'third_party/source-registry.json').read_text('utf-8'))
    ids = [s['id'] for s in registry['sources']]
    if len(ids) != len(set(ids)) or len(ids) != registry['total_records']:
        errors.append('source ID/count mismatch')
    for source in registry['sources']:
        if source['id'].startswith('P02-'):
            for field in ['authors', 'actual_included_version', 'license', 'urls', 'verification_date', 'distribution_status']:
                if not source.get(field):
                    errors.append(f'{source["id"]}: missing {field}')
    ledger = json.loads((ROOT / 'docs/plans/task-ledger.json').read_text('utf-8'))['tasks']
    if len({t['id'] for t in ledger}) != len(ledger):
        errors.append('duplicate task ID')
    report = {'files_checked': checked, 'source_records': len(ids), 'tasks': len(ledger),
              'errors': errors, 'unresolved_links_in_unchanged_historical_snapshots': archived_links}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
