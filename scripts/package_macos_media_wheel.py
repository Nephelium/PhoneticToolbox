"""Package the verified AVFoundation namespace build as a task-local wheel."""
import base64,csv,hashlib,io,json,platform,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    if sys.platform!='darwin' or platform.machine()!='arm64':raise RuntimeError('macOS arm64 required')
    site=Path(sys.prefix)/'lib/python3.11/site-packages';out=ROOT/'output/macos-media'
    report=json.loads((out/'build-report.json').read_text())
    if not report['success']:raise RuntimeError('Native build not verified')
    library=Path(report['library']);assert hashlib.sha256(library.read_bytes()).hexdigest()==report['library_sha256']
    files={}
    for relative in ('av','av-16.1.0.dist-info'):
        for path in (site/relative).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name not in ('RECORD','INSTALLER','REQUESTED','direct_url.json'):
                files[path.relative_to(site).as_posix()]=path.read_bytes()
    files['av/.dylibs/libavdevice.62.1.100.dylib']=library.read_bytes()
    files['av-16.1.0.dist-info/WHEEL']=b'Wheel-Version: 1.0\nGenerator: PhoneticToolbox macOS media build\nRoot-Is-Purelib: false\nBuild: 1ptbmacos\nTag: cp311-cp311-macosx_14_0_arm64\n'
    files['av-16.1.0.dist-info/PTB_MACOS_MEDIA_BUILD.json']=(json.dumps(report,indent=2)+'\n').encode()
    for name in ('COPYING.LGPLv2.1','COPYING.LGPLv3'):
        files['av-16.1.0.dist-info/licenses/FFmpeg-'+name]=(out/'ffmpeg-8.0'/name).read_bytes()
    modified=out/'ffmpeg-8.0/libavdevice/avfoundation.m'
    files['av-16.1.0.dist-info/licenses/PTB-AVFoundation-namespace.txt']=('FFmpeg 8.0 libavdevice rebuilt from '+report['source_url']+'\nOnly private Objective-C class names use the PTBPyAV8 prefix. Original codec, filter, format and Python extensions retain their upstream bytes.\nSource archive SHA256: '+report['source_archive_sha256']+'\nBuild recipe: scripts/build_macos_media.py\n').encode()
    rows=[]
    for name,raw in files.items():rows.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))])
    record='av-16.1.0.dist-info/RECORD';buffer=io.StringIO(newline='');writer=csv.writer(buffer,lineterminator='\n');writer.writerows([*rows,[record,'','']]);files[record]=buffer.getvalue().encode()
    target=ROOT.parent/'downloads/av-16.1.0-1ptbmacos-cp311-cp311-macosx_14_0_arm64.whl'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as wheel:
        for name,raw in files.items():wheel.writestr(name,raw)
    receipt=dict(file=str(target),wheel=target.name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),native_library_sha256=report['library_sha256'],source_archive_sha256=report['source_archive_sha256'],native_build=report,scope='macOS arm64 task-only PyAV 16.1.0 build 1ptbmacos; original package version and codec binaries preserved')
    (ROOT/'resources/macos-media-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT.parent/'receipts/macos-media-wheel.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
