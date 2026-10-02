"""Qt-facing M16 capability, separate from server jobs and existing databases."""
from .recording.service import RecordingService


class M16Bridge:
    def __init__(self,parent=None,data_root=None):
        self.parent=parent;self.service=RecordingService();self.last_error=''

    @property
    def recording(self):return self.service.recording

    @property
    def capturing(self):return bool(self.service.capture)

    def choose(self,purpose):
        if purpose not in ('new','open','export'):raise ValueError('不支持的目录选择用途')
        from PyQt6.QtWidgets import QFileDialog
        title={'new':'选择空目录建立本地录音工程','open':'打开本地录音工程目录','export':'选择录音导出目录'}[purpose]
        selected=QFileDialog.getExistingDirectory(self.parent,title,'')
        return self.service.grant(selected,purpose) if selected else None

    def dispatch(self,body):
        if not isinstance(body,dict):raise ValueError('M16 请求须为对象')
        try:return self.service.dispatch(body)
        except Exception as exc:self.last_error=str(exc);raise

    def close(self):return self.service.close()
