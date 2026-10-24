"""Local static assets shared by desktop windows, without research imports."""
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlsplit,unquote
from PyQt6.QtCore import QBuffer,QIODevice,QObject,QRunnable,QThreadPool,Qt,pyqtSignal,pyqtSlot
from PyQt6.QtWebEngineCore import QWebEngineUrlRequestJob,QWebEngineUrlScheme,QWebEngineUrlSchemeHandler,QWebEngineUrlRequestInterceptor


def register_scheme():
    scheme=QWebEngineUrlScheme(b'ptbapp');scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    scheme.setFlags(QWebEngineUrlScheme.Flag.SecureScheme|QWebEngineUrlScheme.Flag.LocalScheme|
        QWebEngineUrlScheme.Flag.LocalAccessAllowed|QWebEngineUrlScheme.Flag.CorsEnabled|QWebEngineUrlScheme.Flag.FetchApiAllowed)
    QWebEngineUrlScheme.registerScheme(scheme)


def permitted_url(url,service_url,method='GET'):
    part=urlsplit(url)
    return (part.scheme=='ptbapp' and part.netloc=='app') or url=='qrc:///qtwebchannel/qwebchannel.js' or part.scheme in ('data','blob') or (
        part.scheme=='http' and part.netloc==urlsplit(service_url).netloc and part.path.startswith('/api/v1/')) or (
        method=='GET' and part.scheme=='https' and part.netloc=='www.phonetictoolbox.com' and not part.query and not part.fragment and
        part.path.removeprefix('/app/geolinguistics/') in GEO_RUNTIME_FILES and part.path.startswith('/app/geolinguistics/')) or (
        method=='GET' and url=='https://www.phonetictoolbox.com/announcements/v1.json')


GEO_RUNTIME_FILES=frozenset((
    'data-manifest.json','catalog.json','sites.json','observations.json',
    'classification.json','map-index.json','paper.json','main.f64',
    'type-0-0.f64','type-0-1.f64','type-1-0.f64','type-1-1.f64',
    'maps/world.json','maps/territory.json','maps/province.json',
    'maps/city.json','maps/district.json'))


def geolinguistics_directory(dist,executable,*,frozen):
    """Resolve development data or a sibling of the outer .app/Windows EXE."""
    dist,executable=Path(dist),Path(executable)
    candidates=[dist/'geolinguistics']
    if not frozen:candidates.insert(0,dist.parent.parent/'local-data/geolinguistics')
    # A macOS executable lives inside Foo.app/Contents/MacOS; documents stay
    # beside the outer bundle so signing and updates do not modify user data.
    app=next((p for p in executable.parents if p.suffix=='.app' and executable.is_relative_to(p/'Contents/MacOS')),None)
    candidates.append((app.parent if app else executable.parent)/'geolinguistics-data')
    return next((p for p in candidates if p.is_dir()),None)


def _read_asset(root,name):
    """Resolve, validate and read without touching a request or any Qt device."""
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError()
    raw=path.read_bytes()
    if name in ('index.html','feedback-manager.html'):
        raw=raw.replace(b'<head>',b'<head><script src="qrc:///qtwebchannel/qwebchannel.js"></script>')
    mime={'.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.svg':'image/svg+xml'}.get(path.suffix,mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
    return mime.encode('ascii'),raw


class _AssetReplies(QObject):
    completed=pyqtSignal(int,object)


class _AssetRead(QRunnable):
    def __init__(self,root,name,key,replies,cancelled):
        super().__init__()
        self.root,self.name,self.key=root,name,key
        self.replies,self.cancelled=replies,cancelled

    def run(self):
        if self.cancelled.is_set():return
        try:
            result=_read_asset(self.root,self.name)
        except (OSError,ValueError):
            result=QWebEngineUrlRequestJob.Error.UrlNotFound
        except Exception:
            # An exception must not escape a Qt worker's virtual run method.
            result=QWebEngineUrlRequestJob.Error.RequestFailed
        if self.cancelled.is_set():return
        try:
            self.replies.completed.emit(self.key,result)
        except RuntimeError:
            # The owning profile may have destroyed the receiver while reading.
            pass


_asset_reads=None


def _read_pool():
    global _asset_reads
    if _asset_reads is None:
        # Shared by this module's handlers. A profile's destruction must not
        # destroy a pool and synchronously wait for its outstanding disk reads.
        _asset_reads=QThreadPool()
        _asset_reads.setMaxThreadCount(4)
    return _asset_reads


class Assets(QWebEngineUrlSchemeHandler):
    def __init__(self,root,parent,*,geolinguistics_root=None):
        super().__init__(parent);self.root=root.resolve()
        self.geolinguistics_root=Path(geolinguistics_root).resolve() if geolinguistics_root else None
        self._pending={};self._sequence=0
        self._replies=_AssetReplies(self)
        self._replies.completed.connect(self._finish,Qt.ConnectionType.QueuedConnection)
        pending=self._pending
        def discard_all():
            for _,cancelled in pending.values():cancelled.set()
            pending.clear()
        self.destroyed.connect(discard_all)

    def requestStarted(self,job):
        if job.requestUrl().host()!='app' or bytes(job.requestMethod())!=b'GET':
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied);return
        name=unquote(job.requestUrl().path()).lstrip('/') or 'index.html'
        if '\\' in name or ':' in name or '..' in Path(name).parts:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound);return
        self._sequence+=1;key=self._sequence;cancelled=threading.Event()
        self._pending[key]=(job,cancelled)
        pending=self._pending
        def discard():
            cancelled.set();pending.pop(key,None)
        job.destroyed.connect(discard)
        root=self.root
        if name.startswith('geolinguistics/') and self.geolinguistics_root is not None:
            name=name.removeprefix('geolinguistics/')
            if name not in GEO_RUNTIME_FILES:
                self._pending.pop(key,None);job.fail(QWebEngineUrlRequestJob.Error.RequestDenied);return
            root=self.geolinguistics_root
        _read_pool().start(_AssetRead(root,name,key,self._replies,cancelled))

    @pyqtSlot(int,object)
    def _finish(self,key,result):
        entry=self._pending.pop(key,None)
        if entry is None:return
        job,cancelled=entry
        if cancelled.is_set():return
        try:
            if not isinstance(result,tuple):job.fail(result);return
            mime,raw=result
            buffer=QBuffer(job);buffer.setData(raw);buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            job.reply(mime,buffer)
        except RuntimeError:
            # WebEngine owns the job and can cancel it before this queued reply.
            pass


class LocalOnly(QWebEngineUrlRequestInterceptor):
    def __init__(self,url,parent):super().__init__(parent);self.url=url
    def interceptRequest(self,info):
        if not permitted_url(info.requestUrl().toString(),self.url,bytes(info.requestMethod()).decode('ascii','replace')):info.block(True)
