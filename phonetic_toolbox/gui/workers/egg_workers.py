from PyQt6.QtCore import QObject, pyqtSignal
import threading
from phonetic_toolbox.services.egg_service import EGGAnalysisService
from phonetic_toolbox.models.config import EGGConfig
from phonetic_toolbox.models.egg_models import EGGAnalysisResult

class LoadWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(EGGAnalysisResult)
    error = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(self, service: EGGAnalysisService, filepath: str, config: EGGConfig, flip_channels: bool, cancel_event: threading.Event):
        super().__init__()
        self.service = service
        self.filepath = filepath
        self.config = config
        self.flip_channels = flip_channels
        self.cancel_event = cancel_event

    def run(self):
        try:
            if self.cancel_event.is_set():
                self.canceled.emit(); return

            self.progress.emit(10, "正在读取与预处理文件...")
            result = self.service.load_file(self.filepath, self.config, self.flip_channels, self.cancel_event)
            
            if self.cancel_event.is_set():
                self.canceled.emit(); return
            
            if result:
                self.progress.emit(60, "正在分析 GCI/GOI 事件...")
                result = self.service.analyze_events(result, self.config, self.cancel_event)
            
            if self.cancel_event.is_set():
                self.canceled.emit(); return

            if result:
                self.progress.emit(100, "完成")
                self.finished.emit(result)
            else:
                self.canceled.emit() # Should not happen unless cancelled inside service

        except Exception as e:
            self.error.emit(str(e))

class EventsWorker(QObject):
    finished = pyqtSignal(EGGAnalysisResult)
    error = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(self, service: EGGAnalysisService, result: EGGAnalysisResult, config: EGGConfig, cancel_event: threading.Event):
        super().__init__()
        self.service = service
        self.result = result
        self.config = config
        self.cancel_event = cancel_event

    def run(self):
        try:
            if self.cancel_event.is_set(): self.canceled.emit(); return
            
            updated_result = self.service.analyze_events(self.result, self.config, self.cancel_event)
            
            if self.cancel_event.is_set(): self.canceled.emit(); return
            
            self.finished.emit(updated_result)
        except Exception as e:
            self.error.emit(str(e))

class F0Worker(QObject):
    finished = pyqtSignal(EGGAnalysisResult)
    error = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(self, service: EGGAnalysisService, result: EGGAnalysisResult, cancel_event: threading.Event):
        super().__init__()
        self.service = service
        self.result = result
        self.cancel_event = cancel_event

    def run(self):
        try:
            if self.cancel_event.is_set(): self.canceled.emit(); return
            
            self.service.calculate_praat_f0(self.result, self.cancel_event)
            
            if self.cancel_event.is_set(): self.canceled.emit(); return
            
            self.finished.emit(self.result)
        except Exception as e:
            self.error.emit(str(e))
