"""Keep native-engine startup off the Qt event loop."""
from PyQt6.QtCore import QThread, pyqtSignal
from phonetic_toolbox.api import launch_vocal_tract


class VocalTractLaunchWorker(QThread):
    resultReady=pyqtSignal(object)

    def run(self):
        self.resultReady.emit(launch_vocal_tract())
