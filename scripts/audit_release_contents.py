"""Read a real compact EXE without executing or extracting its native payloads."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from release_content_policy import exclusion


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(exe):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'release'))
    from read_distribution import DistributionReader
    exe = Path(exe).resolve()
    archive = DistributionReader(exe)
    members = {name.replace('\\', '/'): dict(compressed_bytes=row[1], expanded_bytes=row[2], type=row[4])
               for name, row in archive.toc.items()}
    host_manifest = json.loads(archive.extract('host-files.json'))
    host = host_manifest['files']
    science = json.loads(archive.extract('runtime-files.json'))['files']
    groups = defaultdict(lambda: dict(files=0, compressed_bytes=0, expanded_bytes=0))
    omissions = []
    for name, row in members.items():
        group = ('scientific_archive' if name == 'runtime-payload.tar.xz' else
                 'host_archive' if name == 'host-payload.tar.xz' else
                 'manual' if name.startswith('frontend/dist/manual/') else
                 'browser_face_model_and_wasm' if name.startswith('frontend/dist/m05/') else
                 'application_source_files' if name.startswith(('backend/src/', 'desktop/src/', 'packages/phonetic_core/src/')) else
                 'other')
        value = groups[group]
        value['files'] += 1
        for key in ('compressed_bytes', 'expanded_bytes'):
            value[key] += row[key]
        reason = exclusion(name)
        if reason:
            omissions.append(dict(path=name, reason=reason, **row))
    def by_identity(files):
        result = defaultdict(list)
        for name, row in files.items():
            result[(row['size'], row['sha256'])].append(name)
        return result
    hg, sg = by_identity(host), by_identity(science)
    shared = [dict(size=size, sha256=digest, host=hg[(size,digest)], science=sg[(size,digest)])
              for size,digest in sorted(hg.keys() & sg.keys(), reverse=True)]
    all_names = list(members) + list(host) + list(science)
    mfa_payload = [name for name in all_names if name.startswith('runtimes/mfa/') or any(
        marker in name.lower() for marker in ('/montreal_forced_aligner/', '/kalpy/', '/kaldi/'))]
    obsolete_project_code = [name for name in science if '/site-packages/' in name and any(
        '/site-packages/'+package+'/' in name for package in ('phonetic_core','ptb_worker','ptb_api','ptb_desktop'))]
    return dict(schema='ptb-package-audit/1', exe=str(exe), bytes=exe.stat().st_size, sha256=sha(exe),
                bundle=json.loads(archive.extract('desktop-bundle.json')), groups=dict(groups),
                host=dict(files=len(host), expanded_bytes=sum(v['size'] for v in host.values())),
                science=dict(files=len(science), expanded_bytes=sum(v['size'] for v in science.values())),
                projected_expanded_files_bytes=sum(v['expanded_bytes'] for v in members.values()) +
                    sum(v['size'] for v in host.values()) + sum(v['size'] for v in science.values()),
                exclusions=dict(status=('excluded candidates remain in this EXE' if omissions else 'no excluded development resources in this EXE'),files=omissions,
                    compressed_bytes=sum(v['compressed_bytes'] for v in omissions),
                    expanded_bytes=sum(v['expanded_bytes'] for v in omissions)),
                shared_native_files=dict(status=('included in host-files/2; runtime verification separate' if host_manifest.get('schema')=='ptb-host-files/2' else 'candidate only; no deduplicated EXE built or verified'),
                    shared_paths=host_manifest.get('sharedFiles',{}),
                    unique_identities=len(shared), host_path_bytes=sum(v['size']*len(v['host']) for v in shared),
                    unique_bytes=sum(v['size'] for v in shared), entries=shared),
                mfa_environment_files=mfa_payload, installed_project_source_candidates=obsolete_project_code,
                members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exe', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    report = audit(options.exe)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    # Every audit gets a new path; never replace earlier measurements.
    with options.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key:report[key] for key in ('bytes','sha256','host','science')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
