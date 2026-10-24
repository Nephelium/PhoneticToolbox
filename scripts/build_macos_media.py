"""Build namespaced FFmpeg AVFoundation classes for the task-local PyAV wheel.

Only libavdevice is rebuilt. Existing codec/filter/format binaries stay intact.
The two Objective-C class names are private; prefixing them avoids collisions
with OpenCV's separately bundled FFmpeg without disabling capture devices.
"""
from pathlib import Path
import hashlib,json,os,platform,re,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if sys.platform!='darwin' or platform.machine()!='arm64':raise RuntimeError('macOS arm64 required')
    parent=ROOT.parent;downloads=parent/'downloads';downloads.mkdir(exist_ok=True)
    url='https://ffmpeg.org/releases/ffmpeg-8.0.tar.xz'
    expected='b2751fccb6cc4c77708113cd78b561059b6fa904b24162fa0be2d60273d27b8e'
    archive=downloads/'ffmpeg-8.0.tar.xz'
    if not archive.exists():
        pending=archive.with_suffix('.partial')
        subprocess.run(['/usr/bin/curl','--silent','--show-error','--fail','--location','--connect-timeout','15','--max-time','180',url,'-o',str(pending)],check=True)
        subprocess.run(['/usr/bin/tar','-tf',str(pending)],stdout=subprocess.DEVNULL,check=True)
        if digest(pending)!=expected:raise RuntimeError('FFmpeg source archive SHA mismatch')
        pending.rename(archive)
    if digest(archive)!=expected:raise RuntimeError('FFmpeg source archive SHA mismatch')
    out=ROOT/'output/macos-media';out.mkdir(parents=True,exist_ok=True)
    build=out/'ffmpeg-8.0'
    if not build.exists():subprocess.run(['/usr/bin/tar','-xf',str(archive),'-C',str(out)],check=True)
    avf=build/'libavdevice/avfoundation.m'
    original=subprocess.check_output(['/usr/bin/tar','-xOf',str(archive),'ffmpeg-8.0/libavdevice/avfoundation.m']).decode()
    modified=original
    for name in ('AVFFrameReceiver','AVFAudioReceiver'):
        modified=re.sub(r'\b'+name+r'\b','PTBPyAV8'+name,modified)
    avf.write_text(modified)
    assert '@interface PTBPyAV8AVFFrameReceiver' in modified
    command=[str(build/'configure'),'--prefix='+str(out/'install'),'--disable-programs','--disable-doc','--disable-static','--enable-shared','--disable-everything','--enable-version3','--enable-indev=avfoundation','--enable-indev=lavfi','--enable-outdev=audiotoolbox','--enable-filter=buffer,buffersink,abuffer,abuffersink','--extra-cflags=-mmacosx-version-min=14.0','--extra-ldflags=-mmacosx-version-min=14.0']
    started=time.monotonic()
    with (out/'build.log').open('w') as log:
        subprocess.run(command,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['/usr/bin/make','-j8'],cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
    libraries=[p for p in (build/'libavdevice').glob('libavdevice*.dylib') if not p.is_symlink()]
    assert len(libraries)==1,libraries
    library=out/'libavdevice.62.1.100.dylib';shutil.copy2(libraries[0],library)
    site=Path(sys.prefix)/'lib/python3.11/site-packages';original_device=site/'av/.dylibs'/library.name
    assert original_device.is_file()
    old_deps=subprocess.check_output(['/usr/bin/otool','-L',str(original_device)],text=True)
    new_deps=subprocess.check_output(['/usr/bin/otool','-L',str(library)],text=True)
    for line in new_deps.splitlines()[1:]:
        name=line.strip().split(' (')[0]
        if name.startswith(('/usr/lib/','/System/Library/')):continue
        stem=Path(name).name.split('.')[0]+'.'
        matches=[p.name for p in original_device.parent.glob(stem+'*.dylib')]
        if len(matches)!=1:raise RuntimeError('Ambiguous FFmpeg ABI dependency: '+name)
        subprocess.run(['/usr/bin/install_name_tool','-change',name,'@loader_path/'+matches[0],str(library)],check=True)
    subprocess.run(['/usr/bin/install_name_tool','-id','/DLC/av/.dylibs/'+library.name,str(library)],check=True)
    subprocess.run(['/usr/bin/codesign','--force','--sign','-',str(library)],check=True)
    report=dict(success=True,source_url=url,source_archive_sha256=digest(archive),original_avfoundation_sha256=hashlib.sha256(original.encode()).hexdigest(),modified_avfoundation_sha256=digest(avf),renamed_private_classes=['PTBPyAV8AVFFrameReceiver','PTBPyAV8AVFAudioReceiver'],library=str(library),library_sha256=digest(library),original_library_sha256=digest(original_device),minimum_macos='14.0',command=command,seconds=time.monotonic()-started,scope='Rebuild libavdevice from FFmpeg 8.0 with private Objective-C class namespace; original PyAV codec/format/filter libraries retained')
    (out/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
