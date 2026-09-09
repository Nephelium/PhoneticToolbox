from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from phonetic_toolbox.models.phonation_synthesis_models import (
    ContinuumType,
    PhonationAnalysisConfig,
    PhonationAnalysisResult,
    PhonationGenerationConfig,
)
from phonetic_toolbox.services.phonation_synthesis_service import (
    PhonationSynthesisService,
)


class PhonationAnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(object, object)
    failed = pyqtSignal(str, str)
    canceled = pyqtSignal()

    def __init__(
        self,
        service: PhonationSynthesisService,
        source_path: Path,
        target_path: Path,
        config: PhonationAnalysisConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._source_path = source_path
        self._target_path = target_path
        self._config = config
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            source, target = self._service.analyze_file_pair(
                self._source_path,
                self._target_path,
                self._config,
                progress=self.progress.emit,
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                self.canceled.emit()
                return
            self.succeeded.emit(source, target)
        except InterruptedError:
            self.canceled.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


class PhonationGenerationWorker(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    canceled = pyqtSignal()

    def __init__(
        self,
        service: PhonationSynthesisService,
        source: PhonationAnalysisResult,
        target: PhonationAnalysisResult,
        generation: PhonationGenerationConfig,
        output_root: Path,
        continuum_type: ContinuumType | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._source = source
        self._target = target
        self._generation = generation
        self._output_root = output_root
        self._continuum_type = continuum_type
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if self._continuum_type is None:
                result = self._service.generate_all(
                    self._source,
                    self._target,
                    self._generation,
                    self._output_root,
                    progress=self.progress.emit,
                    cancel_event=self._cancel_event,
                )
            else:
                result = self._service.generate_selected(
                    self._source,
                    self._target,
                    self._continuum_type,
                    self._generation,
                    self._output_root,
                    progress=self.progress.emit,
                    cancel_event=self._cancel_event,
                )
            if self._cancel_event.is_set():
                self.canceled.emit()
                return
            self.succeeded.emit(result)
        except InterruptedError:
            self.canceled.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())

