"""Keep the portable EXE in dist and stage its current-user setup and update ZIP."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_artifacts import build_workspace

LIMIT = 500_000_000


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'output/release-staging' / ('installer-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    options = parser.parse_args()
    source, output = options.package.resolve(), options.output.resolve()
    if source.parent != ROOT / 'dist' or output.parent != ROOT / 'output/release-staging':
        raise ValueError('Expected new owned release directories')
    metadata = json.loads((source / 'application.json').read_text('utf8'))
    executable = source / 'PhoneticToolbox.exe'
    if (metadata.get('schema') != 'ptb-desktop-release/1' or metadata.get('layout') != 'onefile/1' or
            metadata.get('entry') != executable.name or executable.stat().st_size > LIMIT or
            metadata['executable'] != dict(size=executable.stat().st_size, sha256=digest(executable))):
        raise ValueError('Single-file build did not pass identity and size checks')
    output.mkdir(parents=True, exist_ok=False)
    with build_workspace(output, 'installer'):
        # Portable stays in dist; stage only the installer and update archive.
        package = source
        artifacts = output / 'artifacts'; artifacts.mkdir()
        version = metadata['version']
        portable = executable
        archive = artifacts / f'PhoneticToolbox-{version}-windows-x64-portable.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as stream:
            for path in (package / 'PhoneticToolbox.exe', package / 'application.json'):
                stream.write(path, 'PhoneticToolbox/' + path.name)
        compiler = ROOT / 'output/release-tools/inno-7.1.0/compiler/ISCC.exe'
        with (output / 'installer-build.log').open('wb') as log:
            persistent=json.loads((source/'build-info.json').read_text('utf8')).get('startup_cache')=='content-addressed/1'
            subprocess.run([str(compiler), '/DCompactOnefile', *(['/DPersistentCache'] if persistent else []), '/DPackageDir=' + str(package), '/DReleaseDir=' + str(artifacts),
                            '/DVersion=' + version, str(ROOT / 'release/installer.iss')],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        setup = artifacts / f'PhoneticToolbox-{version}-windows-x64-setup.exe'
        report = dict(schema='ptb-compact-package/1', version=version, layout='onefile/1',
                      limitBytes=LIMIT, mfa='environment and models excluded', files={})
        for path in (portable, archive, setup):
            report['files'][path.name] = dict(size=path.stat().st_size, sha256=digest(path))
            if path.stat().st_size > LIMIT:
                raise RuntimeError('Release artifact exceeds 500 MB: ' + path.name)
        if digest(portable) != metadata['executable']['sha256']:
            raise RuntimeError('Portable EXE is not the actual direct application')
        with zipfile.ZipFile(archive) as stream:
            if stream.testzip() is not None or len(stream.infolist()) != 2:
                raise RuntimeError('Update archive failed its CRC/layout check')
        (output / 'package-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
        print(json.dumps(report), flush=True)
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                        str(ROOT / 'release/cleanup_old_builds.ps1'), '-KeepBuild', str(source),
                        '-KeepStage', str(output)], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
