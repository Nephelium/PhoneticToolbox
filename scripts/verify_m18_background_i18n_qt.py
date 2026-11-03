"""Owned native M18 background failures relocalize through actual statusChanged."""
import argparse,hashlib,json,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from manual.capture_a_support import Capture
from manual.capture_locale import CaptureLocale
from verification_artifacts import verification_run


def run(args,out,scratch):
    c=Capture(args,out,scratch,'m18','paper-reading')
    from ptb_desktop.papers import PaperError
    bridge=c.window.papers_bridge
    reason={'text':'操作已取消。'}
    def fail_read(*_):raise PaperError(reason['text'])
    bridge.service._read=fail_read
    for name in ('desktop/src/ptb_desktop/papers_bridge.py','frontend/src/platform/papers.ts'):
        path=ROOT/name;c.sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    checks=[]
    try:
        c.until('!!document.querySelector(".paper-reading")');c.prepare();bridge.timer.stop()
        deadline=time.monotonic()+10
        while bridge.auto_cancel is not None and time.monotonic()<deadline:c.pause()
        assert bridge.auto_cancel is None
        for kind,raw in (('known','操作已取消。'),('unknown','未登记诊断-{reason} / private.pdf')):
            reason['text']=raw
            bridge.automatic_check(startup=True)
            c.until('!!document.querySelector(".paper-notice[role=status]")')
            deadline=time.monotonic()+10
            while bridge.auto_cancel is not None and time.monotonic()<deadline:c.pause()
            assert bridge.auto_cancel is None
            snapshot=None
            for code in ('zh-CN','en','ja'):
                c.fill('.language-switch select',code,'change');c.until('document.documentElement.lang==='+json.dumps(code)+'&&document.documentElement.dataset.fontLocale==='+json.dumps(code)+'&&document.fonts.status==="loaded"')
                text=c.js('document.querySelector(".paper-notice[role=status] span").textContent');checks.append({'kind':kind,'locale':code,'text':text})
                if code=='zh-CN':assert text=='自动检查未完成：'+raw
                else:
                    assert not text.startswith('自动检查未完成')
                    if kind=='known':assert '操作已取消' not in text
                    else:assert text.endswith(raw)
                ui=CaptureLocale(code,('paper-reading','workbench','components','common'));image=c.window.view.grab().toImage();raw_size=[image.width(),image.height()];assert raw_size in ([2560,1600],[2561,1601])
                if raw_size!=[2560,1600]:image=image.copy(0,0,2560,1600)
                file=out/(kind+'-'+code+'.png');assert image.save(str(file))
                c.report['captures'].append({'name':file.stem,'file':file.name,'image':[2560,1600],'raw_image':raw_size,'maximized':c.window.isMaximized(),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),**ui.metrics(c.js)})
            c.js('document.querySelector(".paper-notice[role=status] button").click()');c.until('!document.querySelector(".paper-notice[role=status]")')
        c.report.update(success=True,checks=checks,scope='Actual isolated Windows Qt, real native background worker and statusChanged; injected read failure; no external service or personal PDFs')
    finally:c.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--frontend-dist',type=Path,required=True);args=parser.parse_args();args.frontend_dist=args.frontend_dist.resolve(strict=True);args.locale='en'
    with verification_run('m18-background-i18n','qt','Owned native background failure, live three-language switches, no external requests') as (out,scratch):run(args,out,scratch)
