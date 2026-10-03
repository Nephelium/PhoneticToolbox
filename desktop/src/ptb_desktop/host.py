"""M01-E shared static workbench, native directory capabilities and owned service."""
import base64
import json
import mimetypes
import os
import threading
from pathlib import Path
from dataclasses import asdict
from urllib.parse import urlsplit,unquote
from PyQt6.QtCore import QObject,QBuffer,QIODevice,QUrl,pyqtSlot,pyqtSignal,QStandardPaths
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication,QMainWindow,QFileDialog,QMessageBox
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import (QWebEnginePage,QWebEngineProfile,QWebEngineSettings,
    QWebEngineUrlRequestJob,QWebEngineUrlScheme,QWebEngineUrlSchemeHandler,QWebEngineUrlRequestInterceptor)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from phonetic_core.textgrid import parse_textgrid
from .file_provider import FileProvider,FileAccessError
from .local_service import LocalService


def register_scheme():
    scheme=QWebEngineUrlScheme(b'ptbapp');scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    scheme.setFlags(QWebEngineUrlScheme.Flag.SecureScheme|QWebEngineUrlScheme.Flag.LocalScheme|
        QWebEngineUrlScheme.Flag.LocalAccessAllowed|QWebEngineUrlScheme.Flag.CorsEnabled|QWebEngineUrlScheme.Flag.FetchApiAllowed)
    QWebEngineUrlScheme.registerScheme(scheme)


def permitted_url(url,service_url):
    part=urlsplit(url)
    return (part.scheme=='ptbapp' and part.netloc=='app') or url=='qrc:///qtwebchannel/qwebchannel.js' or part.scheme in ('data','blob') or (
        part.scheme=='http' and part.netloc==urlsplit(service_url).netloc and part.path.startswith('/api/v1/'))


class Assets(QWebEngineUrlSchemeHandler):
    def __init__(self,root,parent):super().__init__(parent);self.root=root.resolve()
    def requestStarted(self,job):
        if job.requestUrl().host()!='app' or bytes(job.requestMethod())!=b'GET':
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied);return
        try:
            name=unquote(job.requestUrl().path()).lstrip('/') or 'index.html'
            if '\\' in name or ':' in name or '..' in Path(name).parts:raise ValueError()
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():raise ValueError()
            raw=path.read_bytes()
            if name=='index.html':
                raw=raw.replace(b'<head>',b'<head><script src="qrc:///qtwebchannel/qwebchannel.js"></script>')
            buffer=QBuffer(job);buffer.setData(raw);buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            mime={'.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.svg':'image/svg+xml'}.get(path.suffix,mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            job.reply(mime.encode('ascii'),buffer)
        except (OSError,ValueError):job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)


class LocalOnly(QWebEngineUrlRequestInterceptor):
    def __init__(self,url,parent):super().__init__(parent);self.url=url
    def interceptRequest(self,info):
        if not permitted_url(info.requestUrl().toString(),self.url):info.block(True)


class Bridge(QObject):
    vocalReady=pyqtSignal(str,str)
    previewReady=pyqtSignal(str,str)
    taskReady=pyqtSignal(str,str)
    recordingReady=pyqtSignal(str,str)
    def __init__(self,provider,service,window,*,test_dialog=False):
        super().__init__(window);self.provider,self.service,self.window=provider,service,window
        self.test_dialog=test_dialog
        self.preview_lock=threading.Lock()
        self.task_lock=threading.Lock()
        self.recording_lock=threading.Lock()
        self._recording_bridge=None
        from .task_bridge import TaskBridge
        self.tasks=TaskBridge(provider,service)
        self.vocal_slots=threading.BoundedSemaphore(12)
        from .vocal_tract.files import VocalFiles
        self.vocal_files=VocalFiles()

    def recording_bridge(self):
        # Created on the Qt thread, including its native directory picker.
        if self._recording_bridge is None:
            from .m16_bridge import M16Bridge
            self._recording_bridge=M16Bridge(parent=self.window)
        return self._recording_bridge

    def close_recording(self):
        try:return self._recording_bridge is None or self._recording_bridge.close()
        except Exception:return False

    @pyqtSlot(str,result=str)
    def writeClipboard(self,text):
        # QWebChannel invokes this on the GUI thread. Keep clipboard permission
        # narrowly scoped to plain-text writes; never return existing contents.
        try:
            if len(text)>2_000_000:raise ValueError('复制文字过长，请保存为文本文件。')
            clipboard=QApplication.clipboard()
            if clipboard is None:raise RuntimeError('clipboard unavailable')
            clipboard.setText(text)
            if clipboard.text()!=text:raise RuntimeError('clipboard write failed')
            return json.dumps({'ok':True},ensure_ascii=False)
        except ValueError as exc:return json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False)
        except Exception:return json.dumps({'ok':False,'error':'系统剪贴板暂不可用，请重试。'},ensure_ascii=False)

    @pyqtSlot(str,str)
    def recording(self,request_id,raw):
        if len(request_id)>64:return
        try:
            if len(raw)>3_000_000:raise ValueError('录音请求超过大小限制。')
            body=json.loads(raw)
            if not isinstance(body,dict) or not isinstance(body.get('op'),str):raise ValueError('录音请求格式无效。')
            recording=self.recording_bridge()
        except Exception as exc:
            self.recordingReady.emit(request_id,json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False));return
        if not self.recording_lock.acquire(False):
            self.recordingReady.emit(request_id,json.dumps({'ok':False,'error':'录音操作正在收尾，请稍后重试。'},ensure_ascii=False));return
        def work():
            try:result={'ok':True,'value':recording.dispatch(body)}
            except Exception as exc:result={'ok':False,'error':str(exc)}
            finally:self.recording_lock.release()
            try:self.recordingReady.emit(request_id,json.dumps(result,ensure_ascii=False,allow_nan=False))
            except RuntimeError:pass
        threading.Thread(target=work,daemon=True).start()

    @pyqtSlot(str,str)
    def vocal(self,request_id,raw):
        if len(request_id)>64 or len(raw)>8_000_000:return
        if not self.vocal_slots.acquire(False):
            self.vocalReady.emit(request_id,json.dumps({'ok':False,'error':'声道请求繁忙，请稍候。'}));return
        try:
            body=json.loads(raw);op=body['op'];selected=None
            if op in ('document/open','document/save','video/begin'):
                if self.test_dialog and hasattr(self,'test_vocal_picker'):
                    selected=self.test_vocal_picker(op)
                elif op=='document/open':
                    selected=QFileDialog.getOpenFileName(self.window,'导入声道关键帧','','声道关键帧 (*.ptb-vocal.json *.json)')[0]
                else:
                    video=op=='video/begin'
                    selected=QFileDialog.getSaveFileName(self.window,'保存动作视频' if video else '导出声道关键帧','声道动作.webm' if video else '声道关键帧.ptb-vocal.json','WebM 视频 (*.webm)' if video else '声道关键帧 (*.ptb-vocal.json)')[0]
        except Exception as exc:
            self.vocal_slots.release();self.vocalReady.emit(request_id,json.dumps({'ok':False,'error':str(exc)}));return
        def work():
            try:
                if op in ('document/open','document/save') or op.startswith('video/'):
                    value=self.vocal_files.invoke(op,body.get('body',{}),self.window.vocal,selected)
                else:
                    if op=='shutdown':self.vocal_files.close()
                    value=self.window.vocal.invoke(op,body.get('body'))
                result={'ok':True,'value':value}
            except Exception as exc:result={'ok':False,'error':str(exc)}
            finally:self.vocal_slots.release()
            try:self.vocalReady.emit(request_id,json.dumps(result,ensure_ascii=False,allow_nan=False))
            except RuntimeError:pass
        threading.Thread(target=work,daemon=True).start()

    @pyqtSlot(str,str)
    def task(self,request_id,raw):
        if len(request_id)>64:return
        try:
            from .task_requests import decode_task_request
            decoded=decode_task_request(raw)
        except (ValueError,TypeError):
            self.taskReady.emit(request_id,json.dumps({'ok':False,'error':'请求结构或大小不正确。'}));return
        if not self.task_lock.acquire(False):
            self.taskReady.emit(request_id,json.dumps({'ok':False,'error':'任务操作正在进行，请稍候。'}));return
        def work():
            try:result={'ok':True,'value':self.tasks.invoke(decoded)}
            except FileAccessError as exc:result={'ok':False,'error':str(exc)}
            except Exception:result={'ok':False,'error':'任务操作失败，请检查文件关联、目录权限或任务服务。'}
            finally:self.task_lock.release()
            try:self.taskReady.emit(request_id,json.dumps(result,ensure_ascii=False,allow_nan=False))
            except RuntimeError:pass
        threading.Thread(target=work,daemon=True).start()

    @pyqtSlot(str,str)
    def preview(self,request_id,raw):
        if len(request_id)>64 or len(raw)>8192:return
        if not self.preview_lock.acquire(False):
            self.previewReady.emit(request_id,json.dumps({'ok':False,'error':'preview_busy'}));return
        def work():
            try:
                body=json.loads(raw)
                if set(body)!={'id','channel','start','end','width'}:raise ValueError('invalid_preview')
                payload,source_sha=self.provider.spectrogram_payload(body.pop('id'))
                value=self.service.preview(payload,body)
                value['sha256']=source_sha
                result={'ok':True,'value':value}
            except Exception as exc:
                code=str(exc)
                if code not in ('preview_busy','preview_runtime_unavailable','preview_timeout','preview_memory_exceeded','invalid_spectrogram_input'):code='preview_failed'
                result={'ok':False,'error':code}
            finally:self.preview_lock.release()
            try:self.previewReady.emit(request_id,json.dumps(result,ensure_ascii=False,allow_nan=False))
            except RuntimeError:pass  # The owned window may have closed during I/O.
        threading.Thread(target=work,daemon=True).start()

    @pyqtSlot(str,result=str)
    def invoke(self,raw):
        try:
            if len(raw)>8192:raise FileAccessError('请求过长。')
            body=json.loads(raw)
            if not isinstance(body,dict) or set(body)-{'op','id','purpose'}:raise FileAccessError('不支持的请求。')
            op=body.get('op')
            if op=='hello':
                health=self.service.get('/api/v1/health')
                value={'kind':'desktop','session':self.provider.session,'api_version':health['api_version'],'tasks':bool(self.service.local_files_root)}
            elif op=='m05_media':
                if self._recording_bridge is not None and self._recording_bridge.capturing:
                    raise FileAccessError('录音模块正在使用输入设备，请先停止录音或录前检测。')
                value=self.window.m05_media.arm()
            elif op=='m05_media_status':
                value=self.window.m05_media.status()
            elif op=='m16_choose':
                purpose=body.get('purpose')
                if purpose not in ('new','open','export'):raise FileAccessError('不支持的录音目录选择。')
                value=self.recording_bridge().choose(purpose)
            elif op=='fonts':
                from PyQt6.QtGui import QFontDatabase
                value=sorted(QFontDatabase.families())
            elif op=='capture':
                from .m09_capture import capture_spectrogram
                captured=capture_spectrogram(self.window)
                if captured is None:value=None
                else:
                    pixmap,corners=captured;buffer=QBuffer();buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                    if not pixmap.save(buffer,'PNG'):raise FileAccessError('截图失败。')
                    value={**self.provider.capture(bytes(buffer.data())),'corners':corners}
            elif op=='m11_pick':
                purpose=body.get('purpose')
                if purpose in ('runtime','corpus'):
                    path=QFileDialog.getExistingDirectory(self.window,'选择 MFA '+purpose+' 目录','')
                elif purpose in ('model','dictionary','archive','manifest'):
                    filters={'model':'MFA model (*.zip)','dictionary':'Dictionary (*.dict *.txt)','archive':'Component (*.zip)','manifest':'Manifest (*.json)'}
                    path,_=QFileDialog.getOpenFileName(self.window,'选择 MFA '+purpose,'',filters[purpose])
                else:raise FileAccessError('不支持的 MFA 资源。')
                value=self.tasks.m11.grant(purpose,path) if path else None
            elif op=='choose':
                purpose=body.get('purpose')
                def picker():
                    options=QFileDialog.Option.ShowDirsOnly
                    if self.test_dialog:options|=QFileDialog.Option.DontUseNativeDialog
                    return QFileDialog.getExistingDirectory(self.window,'选择'+{'input':'音频','output':'结果','association':'关联文件'}.get(purpose,'')+'目录','',options)
                value=self.provider.choose(purpose,picker)
            elif op=='list':value=self.provider.list(body.get('id'))
            elif op in ('read','textgrid'):
                payload,sha=self.provider.read(body.get('id'))
                if op=='read':value={'base64':base64.b64encode(payload).decode('ascii'),'sha256':sha}
                else:
                    text=payload.decode('utf-16' if payload[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig')
                    value={'sha256':sha,'tiers':[asdict(t) for t in parse_textgrid(text)]}
            else:raise FileAccessError('不支持的操作。')
            return json.dumps({'ok':True,'value':value},ensure_ascii=False,allow_nan=False)
        except FileAccessError as exc:return json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False)
        except Exception:return json.dumps({'ok':False,'error':'读取失败，请检查文件格式或重新选择目录。'},ensure_ascii=False)


class Page(QWebEnginePage):
    def acceptNavigationRequest(self,url,navigation_type,is_main_frame):
        if url.scheme()=='https' and navigation_type==QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url);return False
        return url.scheme()=='ptbapp' and url.host()=='app'
    def javaScriptConfirm(self,origin,message):
        return QMessageBox.question(self.parent(),'未保存的草稿','关闭窗口将放弃未保存的编辑。是否关闭？',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)==QMessageBox.StandardButton.Yes


class Workbench(QMainWindow):
    def __init__(self,dist,*,test=False,jobs_path=None,local_files_root=None,reaper_binary=None,vocal_resources=None,vocal_profile=None,start_module=None):
        super().__init__();self.setWindowTitle('PhoneticToolbox 3.0');self.resize(1440,900)
        from .app_icon import configure_app_icon
        self.setWindowIcon(configure_app_icon(dist))
        self.fit_screen(initial=True)
        self.provider=FileProvider();self.service=LocalService(jobs_path,local_files_root=local_files_root,reaper_binary=reaper_binary);self.service.start();self.closing=False
        from .vocal_tract.client import VocalTractClient
        data=Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
        legacy=Path(os.environ.get('LOCALAPPDATA',Path.home()))/'PhoneticToolbox/vocal_tract'
        self.vocal=VocalTractClient(vocal_resources or dist.parent.parent/'resources/vocal_tract/native',
            vocal_profile or data/'vocal-tract',legacy_profile=None if test else legacy)
        self.profile=QWebEngineProfile(self) if test else QWebEngineProfile('ptb-v3-workbench',self)
        # Local resource filenames may be stable between EXE revisions.
        self.profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
        if not test:
            data=QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
            self.profile.setPersistentStoragePath(str(Path(data)/'workbench'))
        self.assets=Assets(dist,self.profile);self.interceptor=LocalOnly(self.service.url,self.profile)
        self.profile.installUrlSchemeHandler(b'ptbapp',self.assets);self.profile.setUrlRequestInterceptor(self.interceptor)
        self.view=QWebEngineView(self);self.page=Page(self.profile,self.view);self.view.setPage(self.page)
        self.profile.downloadRequested.connect(self.save_download)
        from .m05_permissions import MediaPermission
        self.m05_media=MediaPermission(self)
        self.page.permissionRequested.connect(self.m05_media.requested)
        self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture,True)
        self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows,False)
        self.channel=QWebChannel(self.page);self.bridge=Bridge(self.provider,self.service,self,test_dialog=test)
        self.channel.registerObject('files',self.bridge);self.page.setWebChannel(self.channel)
        self.page.windowCloseRequested.connect(self.accept_close)
        self.setCentralWidget(self.view)
        self.view.load(QUrl('ptbapp://app/index.html'+('#'+start_module if start_module in {'M10','M16','M17'} else '')))

    def fit_screen(self,screen=None,*,initial=False):
        screen=screen or self.screen()
        if screen is None:return
        area=screen.availableGeometry()
        width=max(1,area.width()-32);height=max(1,area.height()-64)
        self.setMinimumSize(min(800,width),min(500,height))
        if not self.isMaximized() and not self.isFullScreen():
            self.resize(min(1440 if initial else self.width(),width),min(900 if initial else self.height(),height))

    def showEvent(self,event):
        super().showEvent(event)
        if not getattr(self,'_screen_connected',False) and self.windowHandle():
            self.windowHandle().screenChanged.connect(self.fit_screen)
            self._screen_connected=True

    def save_download(self,download):
        # Only renderer-generated supported artifacts from this owned page.
        name=Path(download.suggestedFileName()).name
        if download.page()!=self.page or download.url().scheme()!='blob' or not name.lower().endswith(('.svg','.png','.textgrid','.lip.json','.json','.csv','.xlsx','.txt')):
            download.cancel();return
        options=QFileDialog.Option.DontUseNativeDialog if self.bridge.test_dialog else QFileDialog.Option(0)
        title,filter=('保存文本','UTF-8 文本 (*.txt)') if name.lower().endswith('.txt') else ('保存标注','Praat 标注 (*.TextGrid)') if name.lower().endswith('.textgrid') else ('保存安全唇形','安全唇形 (*.lip.json)') if name.lower().endswith('.lip.json') else ('保存实验文件','实验文件 (*.json *.csv *.xlsx)') if name.lower().endswith(('.json','.csv','.xlsx')) else ('保存图像','图像 (*.svg *.png)')
        selected=QFileDialog.getSaveFileName(self,title,name,filter,options=options)[0]
        if not selected:download.cancel();return
        target=Path(selected);download.setDownloadDirectory(str(target.parent));download.setDownloadFileName(target.name);download.accept()

    def accept_close(self):self.closing=True;self.close()
    def closeEvent(self,event):
        if not self.closing:
            event.ignore();self.page.triggerAction(QWebEnginePage.WebAction.RequestClose);return
        if not self.bridge.close_recording():
            self.closing=False;event.ignore()
            QMessageBox.warning(self,'录音尚未安全保存','录音收尾或工程保存失败。窗口继续保留，请回到录音页处理后再关闭。')
            return
        self.bridge.vocal_files.close();self.vocal.close();self.provider.close();self.service.close();event.accept()


def run(dist,**options):
    if not (dist/'index.html').is_file():raise RuntimeError('请先构建共同前端。')
    register_scheme();app=QApplication(['PhoneticToolbox']);app.setApplicationName('PhoneticToolbox-v3')
    window=Workbench(dist,**options)
    app.aboutToQuit.connect(window.service.close);app.aboutToQuit.connect(window.provider.close)
    app.aboutToQuit.connect(window.vocal.close)
    app.aboutToQuit.connect(window.bridge.close_recording)
    window.show();code=app.exec();window.page.deleteLater();app.processEvents();return code
