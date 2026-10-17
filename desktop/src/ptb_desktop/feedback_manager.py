"""Standalone administrator window, using the shared static Vue bundle."""
import json
from pathlib import Path
import sys
import hashlib
import getpass
from PyQt6.QtCore import QUrl,Qt,QTimer,pyqtSlot
from PyQt6.QtNetwork import QLocalServer,QLocalSocket
from PyQt6.QtGui import QAction,QIcon
from PyQt6.QtWidgets import QApplication,QMainWindow,QSystemTrayIcon,QMenu
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineProfile,QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView
from .web_assets import Assets,LocalOnly,register_scheme
from .feedback_bridge import FeedbackBridge
from .platform_paths import user_data_root


class FeedbackManager(QMainWindow):
    def __init__(self, dist, *, root=None, client=None, tray=True):
        super().__init__()
        self.setWindowTitle('PhoneticToolbox 反馈管理')
        self.resize(1180,800)
        self.quitting = False
        self.root = Path(root) if root else user_data_root()/'feedback-manager'
        self.root.mkdir(parents=True,exist_ok=True)
        self.profile = QWebEngineProfile('ptb-feedback-manager',self)
        self.profile.setPersistentStoragePath(str(self.root/'web-storage'))
        self.profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
        self.assets = Assets(Path(dist),self.profile)
        self.interceptor = LocalOnly('http://127.0.0.1:1',self.profile)
        self.profile.installUrlSchemeHandler(b'ptbapp',self.assets)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.view = QWebEngineView(self)
        self.page = QWebEnginePage(self.profile,self.view)
        self.view.setPage(self.page)
        self.channel = QWebChannel(self.page)
        self.bridge = FeedbackBridge(self.root,self,admin=True,client=client)
        self.channel.registerObject('feedback',self.bridge)
        self.page.setWebChannel(self.channel)
        self.setCentralWidget(self.view)
        self.tray = None
        if tray and QSystemTrayIcon.isSystemTrayAvailable():
            icon = QIcon(str(Path(dist)/'favicon.ico'))
            if icon.isNull():
                from PyQt6.QtWidgets import QStyle
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
            self.setWindowIcon(icon)
            self.tray = QSystemTrayIcon(icon,self)
            self.tray.setToolTip('PhoneticToolbox 反馈管理')
            menu = QMenu(self)
            show = QAction('打开反馈管理',self)
            show.triggered.connect(self.show_window)
            quit_action = QAction('退出',self)
            quit_action.triggered.connect(self.quit)
            menu.addAction(show)
            menu.addAction(quit_action)
            self.tray.setContextMenu(menu)
            self.tray.activated.connect(lambda reason:self.show_window() if reason==QSystemTrayIcon.ActivationReason.Trigger else None)
            self.tray.messageClicked.connect(self.show_window)
            self.tray.show()
        self.bridge.notification.connect(self.notify)
        self.bridge.inboxState.connect(self.update_inbox_title,Qt.ConnectionType.QueuedConnection)
        self.bridge.windowAction.connect(self.window_action,Qt.ConnectionType.QueuedConnection)
        self.view.load(QUrl('ptbapp://app/feedback-manager.html'))

    def show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    @pyqtSlot(str)
    def window_action(self,action):
        if action=='quit':
            QTimer.singleShot(100,self.quit)

    @pyqtSlot(int,int)
    def update_inbox_title(self, count, conversations):
        title = 'PhoneticToolbox 反馈管理'
        if count > 0:
            title += f' · {conversations} 个会话有 {count} 条未读留言'
        self.setWindowTitle(title)
        if self.tray:
            self.tray.setToolTip(title)

    def notify(self,payload):
        try:
            event = json.loads(payload)
            if event.get('kind') in ('daily_feedback','new_feedback') and event.get('count',0)>0:
                text = f"{event['conversations']} 个会话有 {event['count']} 条新留言"
                if self.tray:
                    self.tray.showMessage('新的用户反馈',text,QSystemTrayIcon.MessageIcon.Information,10000)
        except (ValueError,KeyError,TypeError):
            pass

    def closeEvent(self,event):
        if self.tray and not self.quitting:
            self.hide()
            event.ignore()
            return
        self.bridge.close()
        if self.tray:
            self.tray.hide()
        event.accept()

    def quit(self):
        self.quitting = True
        self.close()
        QApplication.instance().quit()


def run(dist):
    if not (Path(dist)/'feedback-manager.html').is_file():
        raise RuntimeError('请先构建包含反馈管理工具的共同前端。')
    register_scheme()
    app = QApplication(['PhoneticToolbox 反馈管理'])
    app.setApplicationName('PhoneticToolbox-Feedback-Manager')
    identity=getpass.getuser()+'|'+str(user_data_root().resolve())
    name='ptb-feedback-'+hashlib.sha256(identity.encode()).hexdigest()[:16]
    socket=QLocalSocket()
    socket.connectToServer(name)
    if socket.waitForConnected(800):
        socket.write(b'activate')
        socket.waitForBytesWritten(800)
        return 0
    server=QLocalServer(app)
    if not server.listen(name):
        raise RuntimeError('反馈管理工具已经运行，请从托盘打开。')
    window = FeedbackManager(dist)
    def activate():
        connection=server.nextPendingConnection()
        if connection:
            connection.close()
            connection.deleteLater()
        window.show_window()
    server.newConnection.connect(activate)
    app.aboutToQuit.connect(window.bridge.close)
    window.show()
    return app.exec()
