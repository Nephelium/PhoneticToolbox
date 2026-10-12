"""Local static assets shared by desktop windows, without research imports."""
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit,unquote
from PyQt6.QtCore import QBuffer,QIODevice
from PyQt6.QtWebEngineCore import QWebEngineUrlRequestJob,QWebEngineUrlScheme,QWebEngineUrlSchemeHandler,QWebEngineUrlRequestInterceptor


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
            if name in ('index.html','feedback-manager.html'):
                raw=raw.replace(b'<head>',b'<head><script src="qrc:///qtwebchannel/qwebchannel.js"></script>')
            buffer=QBuffer(job);buffer.setData(raw);buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            mime={'.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.svg':'image/svg+xml'}.get(path.suffix,mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            job.reply(mime.encode('ascii'),buffer)
        except (OSError,ValueError):job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)


class LocalOnly(QWebEngineUrlRequestInterceptor):
    def __init__(self,url,parent):super().__init__(parent);self.url=url
    def interceptRequest(self,info):
        if not permitted_url(info.requestUrl().toString(),self.url):info.block(True)
