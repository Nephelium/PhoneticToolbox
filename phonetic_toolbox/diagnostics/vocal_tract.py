"""Exercise the real home button and owned worker, including frozen resources.

run.py --vocal-tract-smoke-test <report.json> opens no visible window and emits
no audio. All test state stays beside the supplied report.
"""
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlparse
from urllib.request import urlopen


def main(report_path):
    report=Path(report_path).resolve();report.parent.mkdir(parents=True,exist_ok=True)
    os.environ['QT_QPA_PLATFORM']='offscreen'
    # Match ordinary startup's Qt/OpenCV DLL initialization order.
    import cv2
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication,QPushButton
    from phonetic_toolbox.gui.main_window import MainWindow
    from phonetic_toolbox.services import vocal_tract_service as service_module
    from phonetic_toolbox import __version__
    service=service_module.VocalTractService(profile_dir=report.parent/'profile',silent=True)
    service_module._service=service
    captured=[]
    MainWindow._auto_check_update_once=lambda self:None
    # The real signal is handled, but it must not launch a browser during QA.
    MainWindow._vocal_tract_ready=lambda self,result:captured.append(result)
    app=QApplication.instance() or QApplication([])
    window=MainWindow();window.show()
    result={'version':__version__,'frozen':bool(getattr(sys,'frozen',False))}
    def exercise():
        try:
            button=next(b for b in window.findChildren(QPushButton) if b.text()=='声道工作台')
            button.click()
        except Exception as exc:
            result['error']=str(exc);app.quit()
    start=time.monotonic()
    def inspect():
        if time.monotonic()-start>60:
            result['error']='GUI startup timeout';window.close();app.quit();return
        if not captured:return
        timer.stop()
        try:
            launched=captured[0];assert launched.success,launched.message
            result['url']=launched.url;result['worker_pid']=launched.process_id
            with urlopen(launched.url+'/api/meta',timeout=5) as response:meta=json.load(response)
            assert meta['audio_locked'] and len(meta['parameters'])==19
            with urlopen(launched.url+'/',timeout=5) as response:html=response.read().decode('utf-8')
            assert all(name in html for name in ['desktop.css','tab-motion','sourceMode','pitchCanvas','spectrogramCanvas'])
            assert set(meta['source_presets'])=={'voiced','voiceless','whisper'}
            from urllib.request import Request
            def post(path,body):
                req=Request(launched.url+'/api/'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Session':meta['token']})
                with urlopen(req,timeout=20) as response:return json.load(response)
            source=meta['source_presets']['whisper']
            pose=post('pose',{'params':meta['presets']['a'],'revision':1,'source':source})
            assert pose['source']['vibration']==0
            frame={'params':meta['presets']['a'],'source':source,'duration':.15,'f0':125,'lip_width':1}
            prepared=post('animation/prepare',{'frames':[frame,frame],'pitch_curve':[[0,100],[1,180]]})
            assert prepared['duration']==.3 and prepared['count']>1
            assert post('audio/monitor',{'seconds':1})['waveform']==[]
            result['verified_features']=['source_modes','pitch_trajectory','monitor','bundled_web']
            child=service.process
            window.close()
            assert child.poll() is not None
            try:
                socket.create_connection(('127.0.0.1',urlparse(launched.url).port),timeout=.3).close()
            except OSError:result['port_closed']=True
            else:raise AssertionError('Worker port remains open')
            result['success']=True
        except Exception as exc:result['error']=str(exc)
        finally:service.shutdown();app.quit()
    timer=QTimer();timer.timeout.connect(inspect);timer.start(50);QTimer.singleShot(0,exercise)
    try:app.exec()
    finally:
        service.shutdown()
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if result.get('success') else 1
