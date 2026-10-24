"""Apply child-local limits before importing science; fixed macOS protocols."""
import json,math,os
from pathlib import Path
import runpy,sys

def main(arguments=None):
    arguments=sys.argv[1:] if arguments is None else arguments
    entry,request=arguments
    root=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
    for relative in ('backend/src','packages/phonetic_core/src','desktop/src'):sys.path.insert(0,str(root/relative))
    import resource
    cpu=int(os.environ.get('PTB_MACOS_CHILD_CPU','1803'))
    resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.environ['MPLBACKEND']='Agg'
    from ptb_worker.native.darwin import ENTRIES,fingerprint
    if entry not in ENTRIES:raise ValueError('Unsupported fixed macOS entry')
    fingerprint()
    # Register only bundled fonts, avoiding host font availability changes.
    from matplotlib import font_manager
    for directory in (root/'backend/src/ptb_worker/assets',root/'frontend/dist/fonts'):
        for font in directory.glob('*.ttf'):font_manager.fontManager.addfont(str(font))
    module=ENTRIES[entry];sys.argv=[module,request,'/dev/stdout']
    if entry=='egg_persistent':
        from ptb_worker.egg_interactive_child import run
        run()
    elif entry=='fonts':
        from ptb_worker.fonts import check_fonts
        from ptb_api.font_models import FigureFontSnapshot
        print(os.getpid(),flush=True)
        result=check_fonts(FigureFontSnapshot.model_validate_json(sys.stdin.buffer.readline(8192)))
        sys.stdout.buffer.write(json.dumps(result,ensure_ascii=False).encode())
    elif entry in ('spectrogram','spec2wav_preview','egg_interactive','parameter_preview'):
        if entry=='spectrogram':
            if request=='--interactive':from ptb_worker.spectrogram_preview import interactive_child as child
            else:from ptb_worker.spectrogram_preview import child
        elif entry=='spec2wav_preview':from ptb_worker.spec2wav_preview import main as child
        elif entry=='egg_interactive':from ptb_worker.egg_interactive_child import once as child
        else:from ptb_worker.parameter_preview import child
        child()
    else:
        sys.stdout=sys.stderr
        if entry=='lpc':
            from ptb_worker.lpc_child import run
            run()
        else:runpy.run_module(module,run_name='__main__')
    return 0

if __name__=='__main__':raise SystemExit(main())
