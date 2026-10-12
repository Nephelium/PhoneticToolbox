"""Build pinned Windows MFA ZIP and accompanying sources, without changing input env."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'PhoneticToolbox-MFA-3.3.8-r1-windows-x64'
URL = 'https://www.phonetictoolbox.com/releases/components/mfa/3.3.8-r1/'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8')


def public(value):
    """Remove developer-machine paths from distributable audit metadata."""
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k not in
                ('local_path', 'link', 'extracted_package_dir', 'package_tarball_full_path', 'requested_spec')}
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, str) and re.match(r'^[A-Za-z]:[/\\]', value):
        return Path(value).name
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', required=True)
    p.add_argument('--sources', required=True)
    p.add_argument('--conda-cache', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    runtime, sources, out = map(lambda s: Path(s).absolute(), (args.runtime, args.sources, args.output))
    out.mkdir(parents=True, exist_ok=False)
    from ptb_worker.mfa.components import no_links, safe_name
    no_links(runtime); no_links(sources); no_links(out)
    targets = json.loads((sources / 'source-status-manifest.json').read_text('utf8'))['targets']
    recipes = json.loads((sources / 'conda-recipe-inventory.json').read_text('utf8'))
    conda = [json.loads(p.read_text('utf8')) for p in sorted((runtime / 'conda-meta').glob('*.json'))]
    recipe_ids = {(r['name'], r['version'], r['build']): r for r in recipes}
    for row in conda:
        match = recipe_ids[(row['name'], row['version'], row['build'])]
        if match['binary_archive_sha256'] != row['sha256']:
            raise ValueError('Exact Conda recipe identity changed: ' + row['name'])
    selected = [r for r in targets if r['target_manifest'] in ('conda-source-targets.json', 'conda-codec-source-targets.json')
                or (r['component'], r.get('version')) in
                (('PyQt6', '6.6.1'), ('PyQt6_sip', '13.10.3'), ('Qt including QtWebEngine/Chromium', '6.6.1'))]
    model_targets = [r for r in targets if r['target_manifest'] == 'model-comparison-targets.json']
    for row in selected + model_targets:
        if row['status'] != 'verified' or sha(row['local_path']) != row['sha256']:
            raise ValueError('Source identity mismatch: ' + row['filename'])
    print(f'Validated {len(conda)} exact Conda recipes and {len(selected)} source bindings', flush=True)

    # Sources are a companion download, not installed into the user's runtime.
    source_path = out / (NAME + '-sources.zip')
    notices = {}
    recipe_files = {}
    for row in recipes:
        for f in row['recipe_files']:
            path = sources / f['path']; no_links(path)
            if sha(path) != f['sha256']:
                raise ValueError('Recipe changed: ' + f['path'])
            if path.name == 'installed-conda-meta.json':
                raw = encoded(public(json.loads(path.read_text('utf8'))))
            else:
                raw = path.read_bytes()
            recipe_files[f['path']] = raw
            if re.search(r'(?i)(license|licence|copying|copyright|notice)', path.name):
                notices['THIRD_PARTY/' + f['path']] = raw
    # Conda does not install info/licenses into the environment. Read originals
    # from the exact binary archives, not just source recipe license markers.
    import zstandard
    cache = Path(args.conda_cache).absolute(); no_links(cache)
    def collect_licenses(archive, row):
        package = row['name'] + '-' + row['version'] + '-' + row['build']
        for member in archive:
            if member.isfile() and member.name.startswith('info/licenses/'):
                name = 'conda-recipes/' + package + '/' + member.name
                safe_name(name)
                raw = archive.extractfile(member).read()
                recipe_files[name] = raw
                notices['THIRD_PARTY/' + name] = raw
    for row in recipes:
        binary = cache / row['binary_archive_fn']; no_links(binary)
        if sha(binary) != row['binary_archive_sha256']:
            raise ValueError('Conda archive identity changed: ' + binary.name)
        if binary.suffix == '.conda':
            with zipfile.ZipFile(binary) as package:
                names = [n for n in package.namelist() if n.endswith('.tar.zst')]
                if not any(n.startswith('info-') for n in names):raise ValueError('Conda info archive missing')
                for name in names:
                    with package.open(name) as packed, zstandard.ZstdDecompressor().stream_reader(packed) as stream:
                        with tarfile.open(fileobj=stream, mode='r|') as info:collect_licenses(info, row)
        else:
            with tarfile.open(binary, 'r:*') as info:collect_licenses(info, row)
    if not any('/ffmpeg-' in p and '/info/licenses/' in p for p in notices):
        raise ValueError('FFmpeg license notice missing')
    source_rows = []
    with zipfile.ZipFile(source_path, 'x', allowZip64=True) as z:
        seen = set()
        for row in selected:
            dest = 'archives/' + row['filename']
            if dest not in seen:
                z.write(row['local_path'], dest); seen.add(dest)
            source_rows.append(public(row))
        for path, raw in recipe_files.items():
            z.writestr(path, raw, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('sources.json', encoded(source_rows), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('README.txt', ('Corresponding source material for PhoneticToolbox MFA 3.3.8-r1.\n'
            'Contains exact upstream source archives and the original Conda build recipes/patches.\n'
            'See sources.json and conda-recipes/*/recipe/ for source identities and build instructions.\n'
            'Retain the upstream notices and licenses. The pre-existing runtime binaries are unmodified.\n'
            'This archive does not claim byte-reproducibility of third-party wheel builds.\n').encode())
    source_id = dict(filename=source_path.name, bytes=source_path.stat().st_size, sha256=sha(source_path),
                     url=URL + source_path.name, bindings=source_rows)
    notices['THIRD_PARTY/sources.json'] = encoded(source_id)
    notices['THIRD_PARTY/conda-packages.json'] = encoded([
        {k: r.get(k) for k in ('name', 'version', 'build', 'url', 'sha256', 'license', 'depends')} for r in conda])
    for role in ('acoustic', 'dictionary'):
        base = sources / f'model-materials/official-git/{role}/mandarin/mfa/v2.0.0a'
        for path in base.rglob('*'):
            if path.is_file():
                notices[f'THIRD_PARTY/models/{role}/' + path.relative_to(base).as_posix()] = path.read_bytes()
    notices['THIRD_PARTY/model-sources.json'] = encoded([public(r) for r in model_targets])
    notices['THIRD_PARTY_NOTICES.txt'] = (
        'PhoneticToolbox MFA 3.3.8-r1 optional Windows x64 component\n\n'
        'Montreal Forced Aligner 3.3.8: Montreal Corpus Tools, MIT.\n'
        'https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/tree/v3.3.8\n'
        'MFA/Kaldi, Python and all dependencies retain their respective licenses.\n'
        'Original notices are in THIRD_PARTY/conda-recipes, env/Lib/site-packages/*dist-info,\n'
        'env/Library/share and env/Library/licenses. See THIRD_PARTY/conda-packages.json.\n'
        'Mandarin MFA acoustic model and dictionary v2.0.0a: Montreal Corpus Tools, CC BY 4.0.\n'
        'The original model/dictionary bytes are unchanged. Attribution and licenses: THIRD_PARTY/models.\n'
        'McAuliffe, Michael, and Morgan Sonderegger (2022). Mandarin MFA acoustic model v2.0.0a.\n'
        'The runtime includes GPL/LGPL components including FFmpeg, codecs, PyQt and Qt.\n'
        'Corresponding source archives and exact Conda build recipes/patches are available alongside this ZIP:\n'
        + URL + source_path.name + '\n'
        'You may replace or modify the external runtime in accordance with its licenses, and use\n'
        'Advanced > Use another existing environment to register a compatible modified runtime.\n'
        'No technical restriction is imposed on inspecting/debugging LGPL library modifications.\n'
        'Redistributors must provide the sources companion alongside this component, including on mirrors.\n'
    ).encode('utf8')
    notices['使用说明.txt'] = (
        'PhoneticToolbox MFA 自动标注组件 3.3.8-r1（Windows 64 位）\n\n'
        '推荐：在新版 PhoneticToolbox 的 MFA 页面展开组件安装，选择本 ZIP，等待自动检查完成。\n'
        '也可以先解压到固定的短目录，例如 D:\\PTB-MFA，再选择含 env、models 的根文件夹。\n'
        '文件夹方式直接使用所选位置，安装后请勿移动或删除该文件夹。\n'
        '程序内置校验信息，无需选择 JSON 或输入 SHA-256。需要支持此组件版本的新版主程序。\n'
        '旧版 v3 可在已有环境入口选择 env，并分别导入 models 内的模型与词典，再检查应用。\n'
        '组件内含 Python、MFA 3.3.8、Kaldi 及配套普通话模型和词典，无需另装 Conda 或 Python。\n'
        '配套模型为 mandarin_mfa.zip，词典为 mandarin_mfa.dict，使用词典所收录的汉字转写。\n'
        '需要拼音或其他语言时，在上方导入相互兼容的声学模型与发音词典。\n'
        '当前对齐使用 CPU。自动边界仍需人工核对。安装完成后离线可用。\n'
        'env 供主程序调用，不需要执行 env 内的 pip、mfa 或旧 Conda 激活脚本。\n'
        '第三方来源、许可证和对应源码获取方式见 THIRD_PARTY_NOTICES.txt。\n'
    ).encode('utf8')

    archive = out / (NAME + '.zip')
    records, exclusions = [], []
    start = time.monotonic()
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        def add(name, source=None, raw=None):
            safe_name(name)
            if raw is None:
                no_links(source)
                before = sha(source)
                z.write(source, name)
                if sha(source) != before:
                    raise ValueError('Input changed during packaging: ' + name)
                size, digest = source.stat().st_size, before
            else:
                z.writestr(name, raw)
                size, digest = len(raw), hashlib.sha256(raw).hexdigest()
            records.append(dict(path=name, bytes=size, sha256=digest))
        for i, path in enumerate(sorted(runtime.rglob('*'))):
            no_links(path)
            if not path.is_file():
                continue
            rel = path.relative_to(runtime).as_posix()
            if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo') or rel in (
                'conda-meta/history', 'Scripts/conda-unpack-script.py', 'Scripts/conda-unpack.exe'):
                exclusions.append(rel); continue
            if rel.startswith('conda-meta/') and path.suffix == '.json':
                add('env/' + rel, raw=encoded(public(json.loads(path.read_text('utf8')))))
            else:
                add('env/' + rel, source=path)
            if i % 4000 == 0:
                print(f'Runtime packed: {len(records)} files, {time.monotonic()-start:.1f}s', flush=True)
        for row in model_targets:
            suffix = '.dict' if row['component'].endswith('dictionary') else '.zip'
            add('models/mandarin_mfa' + suffix, source=Path(row['local_path']))
        for name, raw in sorted(notices.items()):
            add(name, raw=raw)
        manifest = dict(schema='m11-component/1', id='mfa-338-windows-x64-r1', platform='windows', arch='x86_64',
            mfa_version='3.3.8', source='PhoneticToolbox MFA 3.3.8-r1; exact existing runtime and official Mandarin MFA v2.0.0a',
            dependencies=json.loads(notices['THIRD_PARTY/conda-packages.json']), files=records,
            installed_bytes=sum(r['bytes'] for r in records))
        raw = encoded(manifest)
        z.writestr('component-manifest.json', raw)
    pinned = dict(id=manifest['id'], sha256=sha(archive), manifest_sha256=hashlib.sha256(raw).hexdigest(),
        download_bytes=archive.stat().st_size, installed_bytes=manifest['installed_bytes'] + len(raw),
        filename=archive.name, url=URL + archive.name, trusted_host='www.phonetictoolbox.com')
    (out / 'release.json').write_bytes(encoded(dict(component=pinned, sources={k:v for k,v in source_id.items() if k!='bindings'})))
    (out / 'build-report.json').write_bytes(encoded(dict(payload_files=len(records), license_files=sum('/info/licenses/' in n for n in notices), exclusions=exclusions,
        source_bindings=len(selected), conda_packages=len(conda), seconds=time.monotonic()-start)))
    (ROOT / 'backend/src/ptb_worker/mfa/release_catalog.py').write_text(
        '\"\"\"Fixed component identities, generated by scripts/build_m11_delivery.py.\"\"\"\n\nRELEASE = '
        + repr(pinned) + '\n', encoding='utf8')
    (out / '上传说明.txt').write_text(
        '服务器上传目录：/releases/components/mfa/3.3.8-r1/\n'
        '将两个 ZIP 按原文件名上传，并提供 HTTPS 静态文件下载。支持 Range 续传，无需目录索引。\n'
        '组件与对应源码的下载链接应同时提供，网盘也请保留这两个文件。普通用户只需组件 ZIP。\n'
        'release.json 为维护者校验记录，不需要用户手动导入。\n'
        '主程序必须更新至含本次安装入口的版本。现有 EXE 不会因源码修改而自动更新。\n'
        '旧主程序用户可解压组件，选择 env 和 models 内模型/词典，再检查应用。\n'
        '实际上传后核对 HTTP 200/206、Content-Length、下载 SHA-256 与 release.json 一致。\n'
        '本地包已生成不等于服务器已上传。本任务不执行上传。\n', encoding='utf8')
    print(json.dumps(pinned), flush=True)


if __name__ == '__main__':
    main()
