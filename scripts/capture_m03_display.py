"""Capture newly exposed display paths in original V2, never product imports.

Public synthetic arrays only. Existing reference files are never overwritten.
"""
import json,os,subprocess,sys,hashlib
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
def worker(out):
    assert sys.dont_write_bytecode
    source=ROOT.parent/'PhoneticToolbox_v2';sys.path.insert(0,str(source))
    import numpy as np
    from scipy.io import wavfile
    from PyQt6.QtWidgets import QApplication
    from phonetic_toolbox.gui.widgets.egg_widget import EGGWidget,InverseFilteringResultDialog
    from phonetic_toolbox.models.config import EGGConfig
    from phonetic_toolbox.services.egg_service import EGGAnalysisService
    with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as a:samples=np.column_stack([a['load.egg_signal_raw'],a['load.audio_signal']])
    wav=out/'input.wav';wavfile.write(wav,44100,samples)
    config=EGGConfig();config.gci_method='slope';config.goi_method='scale'
    service=EGGAnalysisService();result=service.analyze_events(service.load_file(str(wav),config),config)
    app=QApplication(['M03-display-baseline']);widget=EGGWidget();widget.result=result;widget.config=config
    arrays={}
    for raw in (False,True):
        widget.show_filtered_egg=not raw
        for label,center,width in [('middle',.3,50),('start',0.,10),('end',.8,200)]:
            widget.zoom_duration_input.setText(str(width));widget.update_zoom_plots(center)
            line=widget.egg_zoom_ax.lines[0];arrays[f'{label}.{raw}.egg']=line.get_ydata();arrays[f'{label}.{raw}.ms']=line.get_xdata()
    first,last=0,5292;audio=result.audio_signal[first:last];egg=result.egg_signal_processed[first:last]
    import numpy as np
    gci=np.asarray(result.gci_times);gci=gci[(gci>=0)&(gci<.12)]
    filtered=service.apply_simplified_cp_inverse_filtering(audio,44100,gci)
    dialog=InverseFilteringResultDialog(audio,filtered,egg,44100,0,.12)
    axes=dialog.layout().itemAt(0).widget().fig.axes
    for index,axis in enumerate(axes):
        for i,line in enumerate(axis.lines):arrays[f'if.{index}.{i}.x']=line.get_xdata();arrays[f'if.{index}.{i}.y']=line.get_ydata()
    np.savez_compressed(out/'display.npz',**arrays)
    (out/'source.json').write_text(json.dumps({'source':'PENDING-EGG','file':'phonetic_toolbox/gui/widgets/egg_widget.py','source_sha256':hashlib.sha256((source/'phonetic_toolbox/gui/widgets/egg_widget.py').read_bytes()).hexdigest(),'input':'public EGG-SYN-PCM16 normalized samples','keys':sorted(arrays)},indent=2),encoding='utf-8')
    dialog.close();widget.close();app.processEvents()

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':return worker(Path(sys.argv[2]))
    import numpy as np
    import shutil
    output=ROOT/'output/validation/m03-ui'/('v2-display-'+uuid4().hex);output.mkdir(parents=True)
    context=json.loads((ROOT/'output/validation/p03/context-before.json').read_text('utf-8'));prefix=Path(context['v2_environment'])
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',QT_QPA_PLATFORM='offscreen',MPLCONFIGDIR=str(output/'mpl'))
    env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Library/bin'),str(prefix/'Scripts'),env.get('PATH','')])
    for i in (1,2):
        out=output/str(i);out.mkdir()
        with (out/'capture.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(prefix/'python.exe'),'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',str(out)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    with np.load(output/'1/display.npz') as a,np.load(output/'2/display.npz') as b:
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    for name in ('display.npz','source.json'):
        target=ROOT/'tests/fixtures/m03'/('ui-'+name)
        if target.exists():raise RuntimeError('Existing frozen reference retained: '+str(target))
        shutil.copyfile(output/'1'/name,target)
    print(output)

if __name__=='__main__':main()
