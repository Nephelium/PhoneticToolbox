"""Freeze the shared source snapshot into a macOS arm64 development .app."""
import argparse,hashlib,json,os,platform,subprocess,sys,time,plistlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for relative in ('desktop/src','backend/src','packages/phonetic_core/src'):
    sys.path.insert(0,str(ROOT/relative))

def protect_qt_build_input():
    # Source WebEngine runs may leave an empty Versions/Resources directory.
    # PyInstaller can interpret it as a real framework version and synthesize
    # Helpers/Resources links there. Remove only that empty task-env artifact
    # and prevent its recreation while framework collection is in progress.
    import PyQt6
    versions=Path(PyQt6.__file__).parent/'Qt6/lib/QtWebEngineCore.framework/Versions'
    if versions.is_symlink() or not versions.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Qt build input must belong to the current isolated environment')
    bad=versions/'Resources';removed=False
    if bad.exists() or bad.is_symlink():
        if bad.is_symlink() or not bad.is_dir() or any(bad.iterdir()):
            raise RuntimeError('Unexpected nonempty Qt framework input; preserve it for diagnosis')
        bad.rmdir();removed=True
    mode=versions.stat().st_mode&0o777;versions.chmod(0o555)
    return versions,mode,dict(removed_empty_generated_resources=removed,temporary_readonly=True,original_mode=oct(mode))

def protect_qt_framework(bundle):
    # QtWebEngineProcess otherwise creates an empty Versions/Resources directory
    # on a writable volume, invalidating the signed parent bundle after launch.
    versions = bundle / 'Contents/Frameworks/PyQt6/Qt6/lib/QtWebEngineCore.framework/Versions'
    if versions.is_symlink() or not versions.resolve().is_relative_to(bundle.resolve()):
        raise RuntimeError('Unexpected Qt framework location')
    if (versions / 'Resources').exists():
        raise RuntimeError('Unexpected framework version before packaging')
    versions.chmod(0o555)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='PhoneticToolbox');args=parser.parse_args()
    if sys.platform!='darwin' or platform.machine()!='arm64':raise RuntimeError('macOS arm64 required')
    if not args.name or Path(args.name).name!=args.name:raise ValueError('Build name must be one filename')
    from macos_build_inputs import freeze_inputs,data_arguments,verify_snapshot,app_version
    from macos_validation_identity import runtime_hashes,require_identity
    from register_macos_validation import VERSIONS,OPERATIONS
    validation=json.loads((ROOT/'resources/macos-validation.json').read_text('utf8'))
    if validation.get('schema')!='ptb-macos-validation/1' or validation.get('versions')!=VERSIONS or any(validation.get('operations',{}).get(name) is not True for name in OPERATIONS):
        raise RuntimeError('A newly registered complete Mac validation receipt is required')
    require_identity(validation.get('source_sha256'),runtime_hashes(ROOT),'Mac validation receipt')
    parent=ROOT.parent;build=parent/'builds'/time.strftime('%Y%m%d-%H%M%S');build.mkdir(parents=True,exist_ok=False)
    snapshot=build/'snapshot';manifest=freeze_inputs(ROOT,snapshot);version=app_version(snapshot)
    require_identity(json.loads((snapshot/'resources/macos-validation.json').read_text('utf8')).get('source_sha256'),runtime_hashes(snapshot),'Frozen Mac validation receipt')
    (snapshot/'resources/research/reaper').chmod(0o755)
    for relative in ('scripts','backend/src','desktop/src','packages/phonetic_core/src'):sys.path.insert(0,str(snapshot/relative))
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(str(snapshot/r) for r in ('scripts','backend/src','desktop/src','packages/phonetic_core/src'))
    env['PYTHONDONTWRITEBYTECODE']='1';env['QT_API']='pyqt6';env.pop('PYTHONHOME',None)
    command=[sys.executable,'-B','-m','PyInstaller','--noconfirm','--onedir','--windowed','--target-arch','arm64',
             '--name',args.name,'--osx-bundle-identifier','org.nephelium.phonetictoolbox',
             '--distpath',str(build/'dist'),'--workpath',str(build/'work'),'--specpath',str(build)]
    for r in ('scripts','backend/src','desktop/src','packages/phonetic_core/src'):command+=['--paths',str(snapshot/r)]
    for package in ('ptb_worker','ptb_api','ptb_desktop','ptb_documents','phonetic_core','uvicorn','docx'):command+=['--collect-submodules',package]
    for package in ('phonetic-core','ptb-api','ptb-desktop','numpy','scipy','praat-parselmouth','pandas','matplotlib','soundfile','python-docx','openpyxl','xlrd','mediapipe','opencv-contrib-python','av','psutil'):command+=['--copy-metadata',package]
    for package in ('mediapipe','docx','_sounddevice_data'):command+=['--collect-data',package]
    for package in ('torch','torchaudio','tensorflow','IPython','notebook','pytest','jax','jaxlib','ptb_worker.vocoder_child'):command+=['--exclude-module',package]
    command+=data_arguments(snapshot)
    for name in ('macos_science_entry','macos_validation_identity','research_entry','m10_recording_entry','workbench_source','verify_macos_runtime','verify_macos_jobs','verify_macos_media','verify_macos_egg_scroll','verify_macos_tab_isolation','verify_macos_icon_size','verify_macos_performance','verify_macos_ui_hotspots','m16_test_backend','verify_release_lifecycle_ui','verify_geolinguistics_qt','ptb_worker.native.darwin_spectrogram','ptb_worker.native.darwin_preview_cache'):command+=['--hidden-import',name]
    qt_input,qt_input_mode,qt_input_receipt=protect_qt_build_input()
    try:
        # Generate macOS icon from the same currently selected application artwork.
        from PyQt6.QtWidgets import QApplication
        from ptb_desktop.app_icon import builtin_image, macos_icon_images
        import ptb_desktop.app_icon as icon_renderer
        if not Path(icon_renderer.__file__).resolve().is_relative_to(snapshot/'desktop/src'):
            raise RuntimeError('Icon renderer must come from this source snapshot')
        app=QApplication.instance() or QApplication([])
        image=macos_icon_images(builtin_image(snapshot/'frontend/dist'))[-1];icon=build/'app.png';image.save(str(icon))
        command+=['--icon',str(icon),str(snapshot/'scripts/macos_entry.py')]
        started=time.monotonic()
        with (build/'pyinstaller.log').open('w') as log:result=subprocess.run(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT)
    finally:
        qt_input.chmod(qt_input_mode)
    report={'qt_framework_input':qt_input_receipt,'state':'failed' if result.returncode else 'built','platform':'darwin','architecture':'arm64','source_root':str(ROOT),'snapshot':str(snapshot),'version':version,'input_manifest_sha256':hashlib.sha256((snapshot/'macos-input-manifest.json').read_bytes()).hexdigest(),'command':command,'seconds':round(time.monotonic()-started,2),'signing':'ad-hoc development signature','notarized':False,'scope':'packaging only; final application verification recorded separately'}
    if result.returncode:
        (build/'report.json').write_text(json.dumps(report,indent=2)+'\n');print((build/'pyinstaller.log').read_text()[-7000:]);return result.returncode
    verify_snapshot(snapshot,manifest)
    bundle=build/'dist'/(args.name+'.app');info=bundle/'Contents/Info.plist'
    value=plistlib.loads(info.read_bytes());value.update(NSMicrophoneUsageDescription='PhoneticToolbox 使用麦克风录制经你确认的语音实验材料。',NSCameraUsageDescription='PhoneticToolbox 使用摄像头提取经你确认的唇形运动。',CFBundleDisplayName='PhoneticToolbox',CFBundleShortVersionString=version,CFBundleVersion=version,LSMinimumSystemVersion='14.0',NSHighResolutionCapable=True)
    info.write_bytes(plistlib.dumps(value,sort_keys=False))
    subprocess.run(['/usr/bin/codesign','--force','--deep','--sign','-',str(bundle)],check=True)
    protect_qt_framework(bundle)
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict','--verbose=2',str(bundle)],check=True)
    report.update(app=str(bundle),bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file() and not p.is_symlink()))
    (parent/'receipts').mkdir(parents=True,exist_ok=True)
    (build/'report.json').write_text(json.dumps(report,indent=2)+'\n');(parent/'receipts/latest-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
