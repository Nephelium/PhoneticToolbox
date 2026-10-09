"""Build a local EXE and retire older owned build folders after success."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from build_artifacts import build_workspace

ROOT = Path(__file__).resolve().parents[1]


def copy_runtime_frontend(source, target, audit, optimize_media=False):
    """Freeze current chapters and only the media they actually use."""
    shutil.copytree(source, target, ignore=lambda folder, names: ['manual'] if Path(folder) == source and 'manual' in names else [])
    manual = source / 'manual'
    project = json.loads((manual / 'project.json').read_text('utf8'))
    used = set()
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('assetId'), str):
                used.add(value['assetId'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    selected = target / 'manual'
    selected.mkdir()
    for chapter in project['chapters']:
        relative = chapter['path']
        path = manual / relative
        if not path.resolve().is_relative_to(manual.resolve()):
            raise ValueError('Manual chapter path escape')
        visit(json.loads(path.read_text('utf8'))['body'])
        destination = selected / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    assets = [row for row in project['assets'] if row['id'] in used]
    if {row['id'] for row in assets} != used:
        raise ValueError('Referenced manual asset is missing')
    packaging = []
    if optimize_media:
        sys.path.insert(0, str(ROOT / 'scripts/manual'))
        from pack_media import package_asset
    for row in assets:
        path = manual / row['path']
        if not path.resolve().is_relative_to(manual.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Manual asset identity mismatch')
        if optimize_media:
            _, item = package_asset(row, path, selected)
            packaging.append(item)
        else:
            destination = selected / row['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    original_count = len(project['assets'])
    project['assets'] = assets
    (selected / 'project.json').write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', 'utf8')
    audit.write_text(json.dumps(dict(policy='Editable source media preserved; package includes referenced media only and optional compact derivatives',
                                   registered=original_count, included=len(assets),
                                   sourceBytes=sum(item['sourceBytes'] for item in packaging) if optimize_media else sum((manual / row['path']).stat().st_size for row in assets),
                                   includedBytes=sum(item['packageBytes'] for item in packaging) if optimize_media else sum((manual / row['path']).stat().st_size for row in assets),
                                   optimized=optimize_media, media=packaging), indent=2) + '\n', 'utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='PhoneticToolbox-v3-LocalPreview-20260927')
    parser.add_argument('--lean-qt', action='store_true',
                        help='Widgets/WebEngine build without unused QML plugins, debug resources or extra WebEngine locales')
    parser.add_argument('--runtime-stage',type=Path,
                        help='Build a distributable onedir package with inventoried independent runtimes')
    parser.add_argument('--compact-onefile', action='store_true',
                        help='True one-file release with opaque compressed EGG/M05 runtimes and no bundled MFA')
    parser.add_argument('--persistent-cache', action='store_true',
                        help='Retain verified runtime files across launches and reuse unchanged components across versions')
    parser.add_argument('--runtime-archive-cache', type=Path,
                        help='Reuse an owned previously checksummed archive of the identical runtime stage')
    parser.add_argument('--runtime-bundle-cache', type=Path,
                        help='Reuse a completed compact build snapshot without expanding its scientific runtimes')
    parser.add_argument('--host-archive-cache', type=Path,
                        help='Reuse an owned host archive when all analyzed native inputs still match')
    options = parser.parse_args()
    # Every compact distribution uses the shared persistent launcher, even if
    # a caller omits the historical opt-in flag. Future releases inherit the
    # same integrity checks, startup card and workbench-ready handshake.
    if options.compact_onefile:options.persistent_cache=True
    if options.persistent_cache and not options.compact_onefile:
        parser.error('Persistent startup cache requires --compact-onefile')
    if options.runtime_stage and options.runtime_bundle_cache:
        parser.error('Choose a runtime stage or a completed compact snapshot')
    if options.runtime_bundle_cache and not options.compact_onefile:
        parser.error('A compact snapshot requires --compact-onefile')
    if options.compact_onefile and (not (options.runtime_stage or options.runtime_bundle_cache) or not options.lean_qt):
        parser.error('The compact release requires an inventoried runtime stage and lean Qt')
    name = options.name
    if not name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Use a plain artifact name')
    destination = ROOT / 'dist' / name
    work = ROOT / 'output' / ('build-' + name)
    if destination.exists() or work.exists():
        raise ValueError('Choose a new name; existing artifacts are retained')
    if sys.platform != 'win32' or not (ROOT / 'frontend/dist/index.html').is_file():
        raise ValueError('Windows and a built frontend are required')
    # A Python schema check alone does not prove that the actual reader accepts
    # exported editor attributes. Fail before spending time on native archives.
    subprocess.run(['node',str(ROOT/'frontend/scripts/verify-manual-runtime.mjs'),
                    str(ROOT/'frontend/dist/manual')],cwd=ROOT,check=True)
    work.mkdir(parents=True)
    with build_workspace(work, 'build'):
        snapshot = work / 'snapshot'
        snapshot.mkdir()
        runtime_stage=options.runtime_stage.resolve() if options.runtime_stage else None
        bundle_cache=options.runtime_bundle_cache.resolve() if options.runtime_bundle_cache else None
        distributable=bool(runtime_stage or bundle_cache)
        if bundle_cache:
            if not bundle_cache.is_relative_to(ROOT / 'output'):
                raise ValueError('Compact snapshot must be an owned build output')
            cached_manifest=json.loads((bundle_cache / 'desktop-bundle.json').read_text('utf8'))
            if (not cached_manifest.get('portable') or cached_manifest.get('schema') != 'desktop-bundle/1' or
                    set(cached_manifest['runtimes']) != {'PTB_EGG_PYTHON', 'PTB_M05_PYTHON'} or cached_manifest.get('mfa')):
                raise ValueError('Expected a completed compact snapshot without MFA')
            cached_archive=bundle_cache / 'runtime-archive'
            descriptor=cached_manifest['runtimeArchive']
            for filename, expected in (('runtime-payload.tar.xz', descriptor['sha256']),
                                       ('runtime-files.json', descriptor['manifestSha256'])):
                with (cached_archive / filename).open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                        raise ValueError('Cached scientific archive identity differs')
            shutil.copytree(cached_archive, snapshot / 'runtime-archive')
        if options.host_archive_cache:
            host_cache=options.host_archive_cache.resolve()
            if not options.compact_onefile or not host_cache.is_relative_to(ROOT / 'output'):
                raise ValueError('Host cache must be an owned compact build output')
            shutil.copytree(host_cache, work / 'host-archive')
        if runtime_stage is not None:
            if not runtime_stage.is_relative_to(ROOT/'output/release-staging') or not (runtime_stage/'stage-report.json').is_file():
                raise ValueError('Expected an inventoried owned runtime stage')
            stage_report=json.loads((runtime_stage/'stage-report.json').read_text('utf8'))
            if any(stage_report['runtimes'][key].get('import_returncode')!=0 for key in ('egg','m05')):
                raise ValueError('Runtime stage did not pass isolated imports')
        # Analysis and real-file workers must consume exactly the same code.
        from source_snapshot import freeze, import_paths, validate
        freeze(ROOT, snapshot)
        from source_snapshot import identity
        single_inputs = ('release/version.json', 'third_party/source-registry.json',
                         'phonetic_toolbox/core/acoustic/reaper.exe',
                         'tests/fixtures/m14/public.xlsx', 'tests/fixtures/m03/EGG-SYN-PCM16.npz')
        for relative in single_inputs:
            source, target = ROOT / relative, snapshot / relative
            expected = identity(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if identity(source) != expected or identity(target) != expected:
                raise RuntimeError('Build input changed while snapshotting: ' + relative)
        runtimes = {
            'PTB_EGG_PYTHON': str(ROOT / '.venv/m03-compatible/python.exe'),
            'PTB_M05_PYTHON': str(ROOT / '.venv/m05/Scripts/python.exe'),
        }
        mfa_root = ROOT / 'output/m11c-028b881d'
        if (mfa_root / 'registry.json').is_file():
            runtimes['PTB_M11_COMPONENT_ROOT'] = str(mfa_root)
        config = dict(kind='distributable-preview' if distributable else 'local-only-preview', source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), runtimes=runtimes,
            portable=distributable, modules='M01-M18; per-module validation remains separate',
            bundle_profile='widgets-webengine-zh-en/1' if options.lean_qt else 'full/1')
        if distributable:
            bindings=(cached_manifest['runtimes'] if bundle_cache else
                      {key:dict(path='runtimes/'+folder+'/python.exe',sha256=stage_report['runtimes'][folder]['sha256'])
                       for key,folder in (('PTB_EGG_PYTHON','egg'),('PTB_M05_PYTHON','m05'))})
            manifest=dict(schema='desktop-bundle/1',portable=True,platform='win32',architecture='x86_64',
                          version=json.loads((snapshot/'release/version.json').read_text('utf8'))['frontend_version'],
                          runtimes=bindings)
            if options.compact_onefile:
                sys.path.insert(0, str(ROOT / 'release'))
                from compact_payload import prepare
                cache = options.runtime_archive_cache.resolve() if options.runtime_archive_cache else None
                if cache and not cache.is_relative_to(ROOT / 'output'):
                    raise ValueError('Runtime archive cache must be an owned build output')
                manifest['runtimeArchive'] = (descriptor if bundle_cache else
                                              prepare(runtime_stage, snapshot / 'runtime-archive', cache))
                config['bundle_profile'] = 'compact-onefile-no-mfa/1'
                config['mfa'] = dict(bundled=False, environment='user-configured external environment', models='not_bundled')
            else:
                manifest['mfa'] = dict(registry='runtimes/mfa/registry-bundled.json',
                                      sha256=hashlib.sha256((runtime_stage/'mfa/registry-bundled.json').read_bytes()).hexdigest())
            (snapshot/'desktop-bundle.json').write_text(json.dumps(manifest,indent=2),'utf8')
            config['runtimes']=bindings
        else:(snapshot / 'local-preview.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        frozen_name='PhoneticToolbox' if distributable else name
        onedir = bool(runtime_stage) and not options.compact_onefile
        frozen_dist = work / 'frozen' if options.persistent_cache else destination
        args = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onedir' if onedir or options.persistent_cache else '--onefile', '--windowed',
                '--name', frozen_name, '--distpath', str(frozen_dist), '--workpath', str(work / 'pyinstaller'),
                '--specpath', str(work)]
        if options.lean_qt:
            # Snapshot the policy too, so the resulting artifact remains auditable.
            hooks = snapshot / 'bundle-hooks'
            shutil.copytree(ROOT / 'scripts/bundle_hooks', hooks)
            args += ['--additional-hooks-dir', str(hooks)]
        for source_path in import_paths(snapshot):
            args += ['--paths', source_path]
        # Another module can rebuild frontend/dist while PyInstaller is analyzing.
        # Freeze data before analysis, just as we freeze the scientific source above.
        from release_content_policy import exclusion
        data = []
        excluded_data = []
        for relative in ('frontend/dist', 'resources/vocal_tract', 'resources/m05',
                         'contracts', 'backend/migrations', 'docs/manual', 'third_party/licenses'):
            source, target = ROOT / relative, snapshot / relative
            reason = exclusion(relative)
            if reason:
                excluded_data.extend(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, reason=reason)
                                     for p in sorted(source.rglob('*')) if p.is_file())
                continue
            if options.compact_onefile and relative == 'frontend/dist':
                copy_runtime_frontend(source, target, work / 'manual-package-selection.json', optimize_media=True)
                data.append((target, relative))
                continue
            shutil.copytree(source, target)
            expected = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob('*') if p.is_file()}
            copied = {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in target.rglob('*') if p.is_file()}
            if expected != copied:
                raise RuntimeError('Build input changed while snapshotting: ' + relative)
            if relative == 'contracts':
                for path in sorted(target.rglob('*')):
                    if not path.is_file():
                        continue
                    packaged_path = path.relative_to(snapshot).as_posix()
                    reason = exclusion(packaged_path)
                    if reason:
                        excluded_data.append(dict(path=packaged_path, bytes=path.stat().st_size, reason=reason))
                    else:
                        data.append((path, str(Path(packaged_path).parent)))
            else:
                data.append((target, relative))
        (work / 'runtime-data-exclusions.json').write_text(json.dumps(excluded_data, indent=2) + '\n', 'utf8')
        data += [(snapshot / r, r) for r in ('backend/src', 'desktop/src', 'packages/phonetic_core/src')]
        if options.compact_onefile:
            data += [(snapshot / 'runtime-archive' / filename, '.') for filename in ('runtime-payload.tar.xz', 'runtime-files.json')]
            probe = bundle_cache / 'runtime-probe.py' if bundle_cache else runtime_stage / 'validation/probe.py'
            if not probe.is_file():
                raise ValueError('Compact stage lacks the actual scientific entry probe')
            shutil.copyfile(probe, snapshot / 'runtime-probe.py')
            data.append((snapshot / 'runtime-probe.py', 'preview-fixtures'))
        # Match the runtime window's visible footprint; the original K2 image stays intact.
        import importlib.util
        icon_spec = importlib.util.spec_from_file_location('ptb_build_icon', snapshot / 'desktop/src/ptb_desktop/app_icon.py')
        icon_module = importlib.util.module_from_spec(icon_spec)
        icon_spec.loader.exec_module(icon_module)
        icon_source = next((snapshot / 'frontend/dist/assets').glob('k2-*.png'))
        icon_target = snapshot / 'PhoneticToolbox-v3.ico'
        icon_module.write_windows_icon(icon_source, icon_target)
        args += ['--icon', str(icon_target)]
        data += [(snapshot / ('desktop-bundle.json' if distributable else 'local-preview.json'), '.'),
                 (snapshot / 'source-snapshot.json', '.'),
                 (snapshot/'release/version.json','release'),
                 (snapshot / 'third_party/source-registry.json', 'third_party'),
                 (snapshot / 'phonetic_toolbox/core/acoustic/reaper.exe', 'resources/research'),
                 (snapshot / 'tests/fixtures/m14/public.xlsx', 'preview-fixtures'),
                 (snapshot / 'tests/fixtures/m03/EGG-SYN-PCM16.npz', 'preview-fixtures')]
        # Scientific runtimes are copied AFTER PyInstaller. Even --add-data makes
        # PyInstaller inspect PE dependencies and can substitute MFA's older Qt DLLs
        # into the host. These independent process payloads must stay opaque here.
        # The M17 help embeds these texts; retain standalone license files as well.
        for filename in ('OFL-PTBIPAPlus.txt', 'OFL-Noto.txt'):
            target = snapshot / 'ipa-plus-licenses' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'frontend/src/assets/ipa-plus' / filename, target)
            data.append((target, 'third_party/licenses/ipa-plus'))
        for source, target in data:
            args += ['--add-data', str(source) + ';' + target]
        for package in ('ptb_worker', 'ptb_api', 'ptb_desktop', 'phonetic_core', 'uvicorn'):
            args += ['--collect-submodules', package]
        # Installed wheels can lag behind the checkout. Fixed child names and
        # namespace packages must be analyzed from this exact source snapshot.
        for relative in ('backend/src', 'desktop/src', 'packages/phonetic_core/src'):
            source_root = snapshot / relative
            for source in sorted(source_root.rglob('*.py')):
                parts = list(source.relative_to(source_root).with_suffix('').parts)
                if any(part.endswith('.egg-info') for part in parts):
                    continue
                if parts[-1] == '__init__':
                    parts.pop()
                if parts:
                    args += ['--hidden-import', '.'.join(parts)]
        args += ['--collect-submodules', 'docx']
        for package in ('phonetic-core', 'ptb-api', 'ptb-desktop', 'numpy', 'scipy',
                        'praat-parselmouth', 'pandas', 'soundfile', 'python-docx', 'openpyxl', 'xlrd'):
            args += ['--copy-metadata', package]
        from m11_bundle import arguments
        args += arguments(snapshot)
        args += ['--collect-all', '_sounddevice_data', '--collect-data', 'docx',
                 '--hidden-import', '_cffi_backend', '--hidden-import', 'xlrd', '--exclude-module', 'matplotlib',
                 '--exclude-module', 'IPython', str(snapshot / 'scripts/v3_local_preview_entry.py')]
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(import_paths(snapshot))
        env['PYTHONNOUSERSITE'] = '1'
        env.pop('PYTHONHOME', None)
        if options.lean_qt:
            env['PTB_BUNDLE_AUDIT_DIR'] = str(work)
        windows = Path(env.get('SystemRoot', 'C:/Windows'))
        env['PATH'] = os.pathsep.join(map(str, (Path(sys.executable).parent, Path(sys.base_prefix),
                                              windows / 'System32', windows)))
        validate(snapshot)
        with (work / 'build.log').open('wb') as log:
            if options.compact_onefile:
                hook = 'persistent_cache_runtime.py' if options.persistent_cache else 'compact_host_runtime.py'
                args[-1:-1] = ['--runtime-hook', str(snapshot / 'bundle-hooks' / hook)]
                spec_args = args[4:]
                filtered = []
                i = 0
                while i < len(spec_args):
                    item = spec_args[i]
                    if item == '--noconfirm':
                        i += 1; continue
                    if item in ('--distpath', '--workpath'):
                        i += 2; continue
                    filtered.append(item); i += 1
                subprocess.run([sys.executable, '-m', 'PyInstaller.utils.cliutils.makespec', *filtered],
                               cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                spec = work / (frozen_name + '.spec')
                contents = spec.read_text('utf8')
                insertion = ('import sys\nsys.path.insert(0, ' + repr(str(ROOT / 'release')) +
                             ')\nfrom compact_host import pack\npack(a, ' + repr(str(work / 'host-archive')) +
                             ', runtime_archive=' + repr(str(snapshot / 'runtime-archive')) +
                             ', application_native=' + repr(options.persistent_cache) + ')\n')
                if contents.count('pyz = PYZ(a.pure)') != 1:
                    raise RuntimeError('Unexpected PyInstaller spec structure')
                spec.write_text(contents.replace('pyz = PYZ(a.pure)', insertion + 'pyz = PYZ(a.pure)'), 'utf8')
                args = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--distpath', str(frozen_dist),
                        '--workpath', str(work / 'pyinstaller'), str(spec)]
            subprocess.run(args, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        validate(snapshot)
        package=destination/frozen_name if onedir else destination
        exe = package / (frozen_name + '.exe')
        if options.persistent_cache:
            package.mkdir(parents=True,exist_ok=False)
            from cached_launcher import build as build_cached_launcher
            build_cached_launcher(frozen_dist/frozen_name,exe,work,snapshot,work/'host-archive',env=env)
            config['startup_cache']='content-addressed/1'
        if onedir:
            import importlib.util
            qt_package=Path(next(iter(importlib.util.find_spec('PyQt6').submodule_search_locations)))
            qt_bin=qt_package/'Qt6/bin'
            qt_hashes={}
            for binary in (package/'_internal').glob('Qt6*.dll'):
                original=qt_bin/binary.name
                actual=hashlib.sha256(binary.read_bytes()).hexdigest()
                if not original.is_file() or actual!=hashlib.sha256(original.read_bytes()).hexdigest():
                    raise RuntimeError('Host Qt library did not come from the reviewed host environment: '+binary.name)
                qt_hashes[binary.name]=actual
            (work/'host-qt-identity.json').write_text(json.dumps(qt_hashes,indent=2)+'\n','utf8')
            for folder in ('egg','m05','mfa'):
                shutil.copytree(runtime_stage/folder,package/'_internal/runtimes'/folder)
            (work/'runtime-payload-policy.json').write_text(json.dumps(dict(
                policy='Independent runtimes copied after host dependency analysis',
                runtimes=['egg','m05','mfa'],hostQt=qt_hashes),indent=2)+'\n','utf8')
            (package/'application.json').write_text(json.dumps(dict(schema='ptb-desktop-release/1',
                version=manifest['version'],entry='PhoneticToolbox.exe'),indent=2)+'\n','utf8')
        if options.compact_onefile:
            if exe.stat().st_size > 500_000_000:
                raise RuntimeError('Compact EXE exceeds the authorized 500 MB size limit: ' + str(exe.stat().st_size))
            (package / 'application.json').write_text(json.dumps(dict(schema='ptb-desktop-release/1',
                version=manifest['version'], entry='PhoneticToolbox.exe', layout='onefile/1',
                executable=dict(size=exe.stat().st_size, sha256=hashlib.sha256(exe.read_bytes()).hexdigest())), indent=2) + '\n', 'utf8')
        metadata = config | dict(source_snapshot_sha256=hashlib.sha256((snapshot/'source-snapshot.json').read_bytes()).hexdigest(),
                                  exe=exe.name, bytes=exe.stat().st_size,
                                  sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
        (destination / 'build-info.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata), flush=True)
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                        str(ROOT / 'release/cleanup_old_builds.ps1'), '-KeepBuild', str(destination)],
                       cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
