import os
import glob
from PyQt6.QtCore import QThread, pyqtSignal
from phonetic_toolbox.services.manipulation_service import ManipulationService

class BatchProcessorWorker(QThread):
    """
    Background worker for batch processing audio files (speed and pitch changes).
    """
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, folder, speed, pitch_ratio, pitch_hz):
        super().__init__()
        self.folder = folder
        self.speed = speed
        self.pitch_ratio = pitch_ratio
        self.pitch_hz = pitch_hz
        self.is_running = True
        self.service = ManipulationService()

    def run(self):
        try:
            files = glob.glob(os.path.join(self.folder, "*.wav")) + \
                    glob.glob(os.path.join(self.folder, "*.mp3")) + \
                    glob.glob(os.path.join(self.folder, "*.flac"))
            
            total = len(files)
            if total == 0:
                self.finished.emit("未找到音频文件")
                return

            # Create output directory
            out_folder = os.path.join(self.folder, f"processed_s{self.speed}_pr{self.pitch_ratio}_ph{self.pitch_hz}")
            os.makedirs(out_folder, exist_ok=True)

            for i, fpath in enumerate(files):
                if not self.is_running:
                    break
                
                fname = os.path.basename(fpath)
                self.progress.emit(i + 1, total, f"正在处理: {fname}")
                
                try:
                    self.service.process_single_file(fpath, self.speed, self.pitch_ratio, self.pitch_hz, out_folder)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")

            self.finished.emit(f"处理完成！输出目录: {out_folder}")
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False
