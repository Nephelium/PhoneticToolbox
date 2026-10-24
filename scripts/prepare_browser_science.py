"""Prepare same-origin browser runtime; no global install or application build edits.

Download URLs, upstream SHA256 and actual bytes are retained in the output lock.
Only fixed public Python sources are packed, never data, credentials or settings.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, io, json, urllib.request, zipfile, tomllib, shutil

ROOT = Path(__file__).resolve().parents[1]
PYODIDE = '0.29.3'
BASE = f'https://cdn.jsdelivr.net/pyodide/v{PYODIDE}/full/'


def pack_sources(output, runtime):
    source_records={}
    with zipfile.ZipFile(output/'ptb-browser.zip','w',zipfile.ZIP_DEFLATED) as archive:
        version=tomllib.loads((ROOT/'packages/phonetic_core/pyproject.toml').read_text('utf-8'))['project']['version']
        archive.writestr(f'phonetic_core-{version}.dist-info/METADATA',f'Metadata-Version: 2.1\nName: phonetic-core\nVersion: {version}\n')
        sources=list((ROOT/'packages/phonetic_core/src').rglob('*.py'))
        sources.extend((ROOT/'backend/src/ptb_worker').rglob('*.py'))
        for folder in ('ptb_worker/io','ptb_api'):
            sources.extend((ROOT/'backend/src'/folder).glob('*.py'))
        for source in sorted(set(sources)):
            if not source.exists():continue
            anchor=ROOT/('packages/phonetic_core/src' if source.is_relative_to(ROOT/'packages') else 'backend/src')
            name=source.relative_to(anchor).as_posix();raw=source.read_bytes()
            archive.writestr(name,raw);source_records[name]=hashlib.sha256(raw).hexdigest()
        model=ROOT/'packages/phonetic_core/src/phonetic_core/spec2wav/data/parameters-v5.npz'
        if hashlib.sha256(model.read_bytes()).hexdigest()!='e6edbf5c450fc08f4ff4c39d2828a15e24c4dd7286c619308c671ed4df7dffdc':raise ValueError('Approved model identity mismatch')
        for source in (model,model.with_suffix('.json')):
            name=source.relative_to(ROOT/'packages/phonetic_core/src').as_posix();raw=source.read_bytes();archive.writestr(name,raw);source_records[name]=hashlib.sha256(raw).hexdigest()
        for name,module in (('openpyxl','openpyxl'),('et_xmlfile','et_xmlfile'),('python_docx','docx'),('xlrd','xlrd')):
            wheel=next(runtime.glob(name+'-*.whl'))
            with zipfile.ZipFile(wheel) as package:
                for info in package.infolist():
                    if info.filename.startswith(module+'/') and not info.is_dir():archive.writestr(info.filename,package.read(info))
        for name in ('DoulosSIL-Regular.ttf','Doulos-OFL.txt'):
            source=ROOT/'backend/src/ptb_worker/assets'/name
            raw=source.read_bytes();key='ptb_worker/assets/'+name;archive.writestr(key,raw);source_records[key]=hashlib.sha256(raw).hexdigest()
    return source_records


def prepare(output,render_font_directory=None):
    output.mkdir(parents=True, exist_ok=True)
    if render_font_directory is not None:
        folder=output/'fonts';folder.mkdir(exist_ok=True)
        for name in ('NotoSansSC-VF.ttf','NotoSansSC-OFL.txt'):
            shutil.copyfile(render_font_directory/name,folder/name)
    if not (output/'fonts/NotoSansSC-VF.ttf').is_file() or not (output/'fonts/NotoSansSC-OFL.txt').is_file():
        raise ValueError('Provide the verified Noto Sans SC font and OFL via --render-font-directory')
    runtime = output/'runtime'; runtime.mkdir(exist_ok=True)
    records = {}
    previous=json.loads((output/'manifest.json').read_text('utf-8')) if (output/'manifest.json').exists() else {}
    def fetch(name, url, expected=None):
        target=runtime/name
        if target.exists(): raw=target.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=120) as response: raw=response.read()
        sha=hashlib.sha256(raw).hexdigest()
        if expected and sha!=expected: raise ValueError('Upstream SHA mismatch: '+name)
        target.write_bytes(raw)
        records[name]=dict(url=url,sha256=sha,bytes=len(raw))
        return raw
    lock=json.loads(fetch('pyodide-lock.json',BASE+'pyodide-lock.json'))
    wanted=set()
    def include(name):
        name=name.lower().replace('_','-')
        if name in wanted:return
        wanted.add(name)
        for dependency in lock['packages'][name]['depends']:include(dependency)
    for name in ('numpy','scipy','soundfile','pillow','sqlite3','pydantic','matplotlib','opencv-python','pandas','lxml'):include(name)
    pending=[(name,BASE+name,None) for name in ('pyodide.js','pyodide.asm.js','pyodide.asm.wasm','python_stdlib.zip')]
    pending.extend((lock['packages'][name]['file_name'],BASE+lock['packages'][name]['file_name'],lock['packages'][name]['sha256']) for name in sorted(wanted))
    pure_packages=(('openpyxl','3.1.5'),('et_xmlfile','2.0.0'),('python_docx','1.2.0'),('xlrd','2.0.2'))
    for package,version in pure_packages:
        cached=next(iter(runtime.glob(package+'-'+version+'-*.whl')),None)
        if cached and cached.name in previous.get('files',{}):
            record=previous['files'][cached.name];pending.append((cached.name,record['url'],record['sha256']));continue
        metadata=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',timeout=30))
        wheel=next(v for v in metadata['urls'] if v['filename'].endswith('py3-none-any.whl'))
        pending.append((wheel['filename'],wheel['url'],wheel['digests']['sha256']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda args:fetch(*args),pending))
    source_records=pack_sources(output, runtime)
    result=dict(schema='ptb-browser-runtime/1',pyodide=PYODIDE,packages={k:lock['packages'][k]['version'] for k in sorted(wanted)},files=records,sources=source_records,archive_sha256=hashlib.sha256((output/'ptb-browser.zip').read_bytes()).hexdigest())
    if previous.get('native_praat'):
        for name,record in previous['native_praat'].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Native identity mismatch: '+name)
        result['native_praat']=previous['native_praat']
    native_names=('ptb_praat.js','ptb_praat.wasm','praat-parselmouth-0.4.7-source.tar.gz','praat-browser-binding-source.zip')
    if not result.get('native_praat') and all((output/name).exists() for name in native_names):
        result['native_praat']={name:dict(bytes=(output/name).stat().st_size,sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()) for name in native_names}
    font=output/'fonts/NotoSansSC-VF.ttf'
    if font.exists():
        sha=hashlib.sha256(font.read_bytes()).hexdigest()
        if sha!='763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a':raise ValueError('Render font differs from the verified server')
        result['render_fonts']={font.name:dict(bytes=font.stat().st_size,sha256=sha,source='https://github.com/notofonts/noto-cjk',license='SIL-OFL-1.1')}
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    # Notices travel with this web-only runtime, outside the desktop public tree.
    (output/'THIRD_PARTY_NOTICES.txt').write_bytes((ROOT/'third_party/licenses/browser-science-runtime-20261017.txt').read_bytes())
    print(json.dumps(dict(success=True,files=len(records),sources=len(source_records),bytes=sum(r['bytes'] for r in records.values()))))


def prepare_sources(output, runtime_source, *, update_lock=False):
    """Repack current shared sources, reusing only hash-verified local binaries.

    The output is an overlay: serve it before runtime_source. No downloads,
    global installs, or edits to the caller's existing runtime directory.
    """
    lock_path=ROOT/'resources/manifests/browser-science-runtime.json'
    result=json.loads(lock_path.read_text('utf-8'))
    for prefix,records in [('runtime',result['files']),('',result['native_praat']),('fonts',result['render_fonts'])]:
        for name,record in records.items():
            path=runtime_source/prefix/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Existing browser runtime identity mismatch: '+name)
    output.mkdir(parents=True,exist_ok=True)
    result['sources']=pack_sources(output,runtime_source/'runtime')
    result['archive_sha256']=hashlib.sha256((output/'ptb-browser.zip').read_bytes()).hexdigest()
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    (output/'manifest.json').write_text(text,'utf-8')
    if update_lock:lock_path.write_text(text,'utf-8')
    print(json.dumps(dict(success=True,sources=len(result['sources']),archive_sha256=result['archive_sha256'],runtime_source=str(runtime_source))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--render-font-directory',type=Path)
    parser.add_argument('--sources-only',action='store_true');parser.add_argument('--runtime-source',type=Path);parser.add_argument('--update-lock',action='store_true')
    options=parser.parse_args()
    if options.sources_only:
        if options.runtime_source is None:parser.error('--sources-only requires --runtime-source')
        prepare_sources(options.output.resolve(),options.runtime_source.resolve(),update_lock=options.update_lock)
    else:
        if options.update_lock:parser.error('--update-lock requires --sources-only')
        prepare(options.output.resolve(),options.render_font_directory)
