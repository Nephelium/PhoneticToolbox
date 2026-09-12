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
    def __init__(self,provider,service,window,*,test_dialog=False):
        super().__init__(window);self.provider,self.service,self.window=provider,service,window
        self.test_dialog=test_dialog
        self.preview_lock=threading.Lock()
        self.task_lock=threading.Lock()
        from .task_bridge import TaskBridge
        self.tasks=TaskBridge(provider,service)
        self.vocal_slots=threading.BoundedSemaphore(12)
        from .vocal_tract.files import VocalFiles
        self.vocal_files=VocalFiles()

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
        if len(request_id)>64 or len(raw)>1_000_000:return
        if not self.task_lock.acquire(False):
            self.taskReady.emit(request_id,json.dumps({'ok':False,'error':'任务操作正在进行，请稍候。'}));return
        def work():
            try:result={'ok':True,'value':self.tasks.invoke(json.loads(raw))}
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
                payload,_=self.provider.read(body.pop('id'))
                result={'ok':True,'value':self.service.preview(payload,body)}
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
            elif op=='capture':
                if QMessageBox.question(self.window,'截取语谱图','将暂时隐藏本窗口并截取当前屏幕。请确认屏幕不含敏感内容。截图仅保留在本机内存，开始重建后才加入本机任务。',QMessageBox.StandardButton.Ok|QMessageBox.StandardButton.Cancel)!=QMessageBox.StandardButton.Ok:
                    value=None
                else:
                    from .capture import capture_hidden
                    pixmap=capture_hidden(self.window);buffer=QBuffer();buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                    if not pixmap.save(buffer,'PNG'):raise FileAccessError('截图失败。')
                    value=self.provider.capture(bytes(buffer.data()))
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
        self.page.permissionRequested.connect(lambda request:request.deny())
        self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture,True)
        self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows,False)
        self.channel=QWebChannel(self.page);self.bridge=Bridge(self.provider,self.service,self,test_dialog=test)
        self.channel.registerObject('files',self.bridge);self.page.setWebChannel(self.channel)
        self.page.windowCloseRequested.connect(self.accept_close)
        self.setCentralWidget(self.view)
        self.view.load(QUrl('ptbapp://app/index.html'+('#M10' if start_module=='M10' else '')))

    def save_download(self,download):
        # Only a renderer-generated image from this owned page; never navigate to arbitrary downloads.
        if download.page()!=self.page or download.url().scheme()!='blob' or not download.suggestedFileName().lower().endswith(('.svg','.png')):
            download.cancel();return
        options=QFileDialog.Option.DontUseNativeDialog if self.bridge.test_dialog else QFileDialog.Option(0)
        selected=QFileDialog.getSaveFileName(self,'保存图像',Path(download.suggestedFileName()).name,'图像 (*.svg *.png)',options=options)[0]
        if not selected:download.cancel();return
        target=Path(selected);download.setDownloadDirectory(str(target.parent));download.setDownloadFileName(target.name);download.accept()

    def accept_close(self):self.closing=True;self.close()
    def closeEvent(self,event):
        if not self.closing:
            event.ignore();self.page.triggerAction(QWebEnginePage.WebAction.RequestClose);return
        self.bridge.vocal_files.close();self.vocal.close();self.provider.close();self.service.close();event.accept()


def run(dist,**options):
    if not (dist/'index.html').is_file():raise RuntimeError('请先构建共同前端。')
    register_scheme();app=QApplication(['PhoneticToolbox']);app.setApplicationName('PhoneticToolbox-v3')
    window=Workbench(dist,**options)
    app.aboutToQuit.connect(window.service.close);app.aboutToQuit.connect(window.provider.close)
    app.aboutToQuit.connect(window.vocal.close)
    window.show();code=app.exec();window.page.deleteLater();app.processEvents();return code
