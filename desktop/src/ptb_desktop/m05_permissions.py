"""Per-window media permission, armed by explicit capture or device discovery."""
import time
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWebEngineCore import QWebEnginePermission

class MediaPermission:
    def __init__(self,window):self.window=window;self.until=0.;self.events=[];self.requests={};self.purpose='capture'
    def arm(self,purpose='capture'):
        if purpose not in ('capture','devices'):raise ValueError('不支持的媒体权限用途')
        # Non-persistent media decisions still survive within this page. Reset
        # only our own previous requests on a new explicit capture action.
        for request in self.requests.values():
            if request.isValid():request.reset()
        self.requests.clear()
        self.purpose=purpose;self.until=time.monotonic()+30;self.events=[];return True
    def status(self):return self.events[-1] if self.events else dict(decision='not_requested')
    def requested(self,request):
        origin=request.origin()
        allowed=request.permissionType().name in (('MediaAudioCapture',) if self.purpose=='devices' else ('MediaAudioCapture','MediaVideoCapture','MediaAudioVideoCapture'))
        event=dict(type=request.permissionType().name,origin=origin.toString(),armed=time.monotonic()<=self.until)
        self.events.append(event);self.events=self.events[-20:]
        if not allowed or origin.scheme()!='ptbapp' or origin.host()!='app':
            event['decision']='denied_gate';request.deny();return
        self.requests[request.permissionType().name]=QWebEnginePermission(request)
        if time.monotonic()>self.until:
            event['decision']='denied_gate';request.deny();return
        self.until=0.
        message='允许短暂使用麦克风以读取完整音频设备列表？读取后立即释放，不录制音频。' if self.purpose=='devices' else '允许本次唇形采集使用所选摄像头和麦克风？实时视频在本机处理。'
        answer=QMessageBox.question(self.window,'音频设备权限' if self.purpose=='devices' else '唇形采集权限',message,QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        event['decision']='granted' if answer==QMessageBox.StandardButton.Yes else 'denied_user'
        if answer==QMessageBox.StandardButton.Yes:request.grant()
        else:request.deny()
