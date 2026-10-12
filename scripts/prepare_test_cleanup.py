"""Build a reviewed test-only manifest; never delete or adopt external inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from verification_artifacts import no_links

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ('output/validation', 'output/manual-work', 'tools/manual-studio/test-output')
PROTECTED = ('output/validation/p06', 'output/validation/m01',
             'output/validation/m03-runtime', 'output/validation/p03')
OPAQUE = {'startup-cache', 'site-packages', 'node_modules', 'avd', 'runtime', 'payload', '_internal'}


def overlaps(first, second):
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def project_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or p.drive or '..' in p.parts or not p.parts:
        raise ValueError('Use an explicit project-relative path')
    path = Path(os.path.abspath(root / p))
    if path == root or not path.is_relative_to(root):
        raise ValueError('Path must stay within this project')
    no_links(path)
    return path


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_files(path):
    no_links(path)
    if path.is_file():
        yield path
        return
    for directory, folders, files in os.walk(path, followlinks=False):
        parent = Path(directory)
        no_links(parent)
        for name in folders + files:
            child = parent / name
            if child.is_symlink() or getattr(child.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError(f'Linked artifact path: {child}')
        for name in sorted(files):
            yield parent / name


def ready_scratch(root):
    for relative in SCOPES:
        base = root / relative
        if not base.exists():
            continue
        no_links(base)
        for directory, folders, files in os.walk(base, followlinks=False):
            parent = Path(directory)
            safe = []
            for name in folders:
                if name in OPAQUE:
                    continue
                try:
                    no_links(parent / name)
                    safe.append(name)
                except ValueError:
                    pass
            folders[:] = safe
            if 'run.json' not in files or not (parent / 'scratch').exists():
                continue
            try:
                no_links(parent / 'run.json')
                no_links(parent / 'scratch-manifest.json')
                marker = load_json(parent / 'run.json')
                proof = load_json(parent / 'scratch-manifest.json')
            except (OSError, ValueError):
                continue
            if marker.get('schema') == 'ptb-test-run/1' and marker.get('state') == 'ready' and proof.get('schema') == 'ptb-test-scratch-files/1':
                yield dict(path=(parent / 'scratch').relative_to(root).as_posix(),
                           group='completed_scratch', reason='Completed owned scratch; recipe and evidence retained',
                           evidence=[dict(path=(parent / 'run.json').relative_to(root).as_posix(),
                                          require={'schema': 'ptb-test-run/1', 'state': 'ready'})],
                           scratch_manifest=(parent / 'scratch-manifest.json').relative_to(root).as_posix())


def prepare(root, rules):
    root = root.absolute()
    if rules.get('schema') != 'ptb-test-cleanup-rules/1':
        raise ValueError('Unexpected cleanup rules schema')
    tracked = {os.path.normcase(t).replace('\\', '/') for t in
               subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode('utf-8').split('\0')}
    protected = [project_path(root, p) for p in (*PROTECTED, *rules.get('protect', []))]
    candidates = list(rules.get('targets', []))
    if rules.get('discover_completed_scratch', True):
        candidates.extend(ready_scratch(root))
    evidence, targets, skipped, covered, hashes = {}, [], [], [], {}
    for candidate in candidates:
        relative = candidate['path']
        try:
            path = project_path(root, relative)
            if not any(path != root / scope and path.is_relative_to(root / scope) for scope in SCOPES):
                raise ValueError('Outside test-only scope')
            if any(overlaps(path, p) for p in protected):
                raise ValueError('Current dependency or active acceptance scope is protected')
            if any(overlaps(path, p) for p in covered):
                raise ValueError('Overlapping target')
            if not path.exists():
                continue
            rel = path.relative_to(root).as_posix()
            tracked_key = os.path.normcase(rel).replace('\\', '/')
            if any(t == tracked_key or t.startswith(tracked_key + '/') for t in tracked):
                raise ValueError('Git tracked content is protected')
            ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', '--', rel], check=False)
            if ignored.returncode != 0:
                raise ValueError('Only Git-ignored test artifacts qualify')
            proof_paths = []
            for proof in candidate.get('evidence', []):
                source = project_path(root, proof['path'])
                if overlaps(path, source):
                    raise ValueError('Evidence must survive outside the deletion target')
                if proof.get('require'):
                    value = load_json(source)
                    if any(value.get(k) != v for k, v in proof['require'].items()):
                        raise ValueError('Evidence requirements are not satisfied')
                evidence[proof['path']] = hashlib.sha256(source.read_bytes()).hexdigest()
                proof_paths.append(proof['path'])
            if not proof_paths:
                raise ValueError('An explicit retained recipe or completion record is required')
            rows = []
            for file in safe_files(path):
                st = file.stat()
                # Cache only within this inventory and by actual file identity.
                identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                if not st.st_ino or identity not in hashes:
                    with file.open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    hashes[identity] = digest
                rows.append(dict(path=file.relative_to(root).as_posix(), bytes=st.st_size, sha256=hashes[identity]))
            if candidate.get('scratch_manifest'):
                manifest_path = project_path(root, candidate['scratch_manifest'])
                proof = load_json(manifest_path)
                evidence[candidate['scratch_manifest']] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                proof_paths.append(candidate['scratch_manifest'])
                actual = {Path(row['path']).relative_to(path.relative_to(root)).as_posix():
                          (row['bytes'], row['sha256']) for row in rows}
                expected = {row['path']: (row['bytes'], row['sha256']) for row in proof['files']}
                if actual != expected:
                    raise ValueError('Scratch has changed since its finalizer inventory')
            covered.append(path)
            targets.append(dict(path=rel, group=candidate['group'], reason=candidate['reason'],
                                bytes=sum(row['bytes'] for row in rows), files=rows,
                                evidence=proof_paths))
            print(json.dumps(dict(prepared=rel, files=len(rows), bytes=targets[-1]['bytes']), ensure_ascii=False), flush=True)
        except (ValueError, OSError, KeyError) as exc:
            skipped.append(dict(path=relative, reason=str(exc)))
    return dict(schema='ptb-reviewed-cleanup/1', root=str(root),
                created_at=datetime.now(timezone.utc).isoformat(),
                scope='Owned test temporaries only; no deletion by this script',
                protected_paths=[p.relative_to(root).as_posix() for p in protected],
                retained_evidence_sha256=evidence, targets=targets, skipped=skipped,
                bytes=sum(t['bytes'] for t in targets),
                logical_bytes_note='File lengths; hardlinks and sparse files are not physical reclaimed space')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/maintenance/test-temporaries-plan.json')
    args = parser.parse_args()
    output = args.out.absolute()
    no_links(output)
    if not output.is_relative_to(ROOT / 'output/maintenance'):
        raise ValueError('Write the plan only inside output/maintenance')
    result = prepare(ROOT, load_json(args.rules))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(json.dumps(dict(targets=len(result['targets']), bytes=result['bytes'],
                         skipped=len(result['skipped']), report=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
