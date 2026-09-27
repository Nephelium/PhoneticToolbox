"""Per-window media permission, armed by an explicit M05 start action."""
import time
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWebEngineCore import QWebEnginePermission

class MediaPermission:
    def __init__(self,window):self.window=window;self.until=0.;self.events=[];self.requests={}
    def arm(self):
        # Non-persistent media decisions still survive within this page. Reset
        # only our own previous requests on a new explicit capture action.
        for request in self.requests.values():
            if request.isValid():request.reset()
        self.requests.clear()
        self.until=time.monotonic()+30;self.events=[];return True
    def status(self):return self.events[-1] if self.events else dict(decision='not_requested')
    def requested(self,request):
        origin=request.origin()
        allowed=request.permissionType().name in ('MediaAudioCapture','MediaVideoCapture','MediaAudioVideoCapture')
        event=dict(type=request.permissionType().name,origin=origin.toString(),armed=time.monotonic()<=self.until)
        self.events.append(event);self.events=self.events[-20:]
        if not allowed or origin.scheme()!='ptbapp' or origin.host()!='app':
            event['decision']='denied_gate';request.deny();return
        self.requests[request.permissionType().name]=QWebEnginePermission(request)
        if time.monotonic()>self.until:
            event['decision']='denied_gate';request.deny();return
        self.until=0.
        answer=QMessageBox.question(self.window,'唇形采集权限','允许本次唇形采集使用所选摄像头和麦克风？实时视频在本机处理。',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        event['decision']='granted' if answer==QMessageBox.StandardButton.Yes else 'denied_user'
        if answer==QMessageBox.StandardButton.Yes:request.grant()
        else:request.deny()
