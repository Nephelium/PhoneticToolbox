"""Prepare a new D-drive handoff, exact application source and private metadata."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'output/release-staging/compact-delivery-20261006-R7'
BUILD = ROOT / 'dist/PhoneticToolbox-Preview1-20261006-Compact-R7'
DELIVERY = ROOT / 'dist/PhoneticToolbox-3.0.0-preview.1-Compact-20261006'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    checks = [Path('D:/PTB-Compact-QA-20261006') / folder / report
              for folder, report in [('compact-r7', 'launch-report.json'), ('textgrid-r7', 'launch-report.json'),
                                     ('direct-r7', 'direct-report.json')]]
    checks += [ROOT / 'output/validation' / folder / 'relaunch-report.json'
               for folder in ('compact-portable-update-r7', 'compact-installed-update-r7')]
    if not all(json.loads(path.read_text('utf8'))['success'] is True for path in checks):
        raise ValueError('Final artifact verification is incomplete')
    DELIVERY.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((STAGE / 'package-report.json').read_text('utf8'))
    files = {}
    for name, expected in receipt['files'].items():
        source = STAGE / 'artifacts' / name
        if source.stat().st_size != expected['size'] or digest(source) != expected['sha256']:
            raise ValueError('Final software identity changed')
        target = DELIVERY / name; shutil.copyfile(source, target)
        if digest(target) != expected['sha256']:
            raise ValueError('Delivery copy checksum mismatch')
        files[name] = expected
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(BUILD / 'PhoneticToolbox.exe'))
    source_zip = DELIVERY / 'PhoneticToolbox-3.0.0-preview.1-application-source-snapshot.zip'
    source_files = {}
    with zipfile.ZipFile(source_zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        def add(name, raw):
            if name in source_files:
                raise ValueError('Duplicate source snapshot entry')
            if '\\' in name or any(part in ('', '.', '..') for part in name.split('/')):
                raise ValueError('Invalid source snapshot path')
            source_files[name] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            output.writestr(name, raw)
        for stored in sorted(archive.toc):
            name = stored.replace('\\', '/')
            if name.startswith(('backend/', 'desktop/', 'packages/phonetic_core/', 'frontend/dist/', 'resources/')):
                if name.endswith(('.pyc', '.pyo')) or '__pycache__' in name.split('/'):
                    continue
                add('embedded/' + name, archive.extract(stored))
        for folder in ('frontend/src', 'scripts', 'release', 'third_party'):
            for path in sorted((ROOT / folder).rglob('*')):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(ROOT)
                if any(part.startswith('.') or part in ('__pycache__', 'node_modules') for part in relative.parts):
                    continue
                if path.suffix.lower() not in ('.py', '.ps1', '.mjs', '.vue', '.ts', '.js', '.css', '.html', '.json', '.md', '.txt', '.iss', '.bib'):
                    continue
                add('recipe/' + relative.as_posix(), path.read_bytes())
        for name in ('README.md', 'AGENTS.md', 'pyproject.toml', 'frontend/package.json', 'frontend/package-lock.json',
                     'docs/testing/2026-10-06-compact-release-report.md', 'docs/decisions/ADR-Preview1-compact-packaging.md'):
            path = ROOT / name
            if path.is_file(): add('recipe/' + name, path.read_bytes())
        manifest = dict(schema='ptb-application-source-snapshot/1', version=receipt['version'],
                        exeSha256=receipt['files'][next(n for n in receipt['files'] if n.endswith('portable.exe'))]['sha256'],
                        public=False, includesRestrictedSoftwareManualMedia=True,
                        scope='Exact embedded application source/resources plus build recipe. Not complete native GPL corresponding-source; public release remains pending.',
                        files=source_files)
        output.writestr('snapshot-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(source_zip) as output:
        if output.testzip() is not None: raise ValueError('Source snapshot CRC failure')
    files[source_zip.name] = dict(size=source_zip.stat().st_size, sha256=digest(source_zip))
    for source, name in [(ROOT / 'docs/testing/2026-10-06-compact-release-report.md', '验收报告.md'),
                         (STAGE / 'package-report.json', 'package-report.json'),
                         (BUILD / 'build-info.json', 'build-info.json')]:
        target = DELIVERY / name; shutil.copyfile(source, target)
        files[name] = dict(size=target.stat().st_size, sha256=digest(target))
    (DELIVERY / '使用方法.txt').write_text('''PhoneticToolbox 3.0.0-preview.1，Windows x64

免安装 EXE：双击 portable.exe 直接使用，没有安装或目录选择向导。
安装 EXE：运行 setup.exe，以当前用户方式安装，默认当前用户应用目录，无需管理员权限。
portable.zip：更新使用，也可解压后运行同一 EXE。首次下载任选免安装或安装版即可。
两种 EXE 均小于 500 MB，MFA 环境、模型与词典未包含，其他模块的必要运行时随包。

每次启动需要临时展开并校验必要运行文件，约 2 GB，正常退出后清理。
设置、草稿与研究数据位于原固定用户目录，程序更新继续保留。
未导出内部结果默认保留 30 天，成功导出的内部副本默认 7 天，可调整或关闭。
原始输入、录音、工程、标注与正式导出受保护。旧程序和下载包保留以便回退。

说明书内容与原媒体本轮冻结，由作者后续修改。作者编辑器不进入普通应用。
最终成品验证见验收报告.md。另一台实体电脑、实体声卡/摄像头、听辨/DPI未验。

服务器私有暂存，未发布公共 latest.json。GitHub 未上传。
源码准备快照含受限说明书例音，不得上传公开 GitHub，也不能当作完整原生对应源码。
''', 'utf8')
    instructions = DELIVERY / '使用方法.txt'
    files[instructions.name] = dict(size=instructions.stat().st_size, sha256=digest(instructions))
    staging = dict(schema='ptb-private-staging/1', version=receipt['version'], public=False,
                   status='local-verified-public-license-and-native-source-pending', files=files)
    (DELIVERY / 'staging.json').write_text(json.dumps(staging, ensure_ascii=False, indent=2), 'utf8')
    (DELIVERY / 'SHA256SUMS').write_text(''.join(row['sha256'] + '  ' + name + '\n' for name, row in files.items()), 'utf8')
    print(json.dumps(dict(destination=str(DELIVERY), files=len(files), sourceBytes=source_zip.stat().st_size, public=False)))


if __name__ == '__main__':
    main()
