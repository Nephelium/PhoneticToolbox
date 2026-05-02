import os

from PyQt6.QtCore import QThread, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from phonetic_toolbox.models.mfa_models import MFAAlignmentConfig
from phonetic_toolbox.services.mfa_alignment_service import (
    MFAAutoAlignmentService,
)
from phonetic_toolbox.utils import get_resource_path


class MFAAlignmentWorker(QThread):
    finished_sig = pyqtSignal(bool, str, str)

    def __init__(
        self,
        audio_path: str,
        dict_path: str,
        acoustic_path: str,
        output_path: str,
        beam: int,
        retry_beam: int,
    ):
        super().__init__()
        self.audio_path = audio_path
        self.dict_path = dict_path
        self.acoustic_path = acoustic_path
        self.output_path = output_path
        self.beam = beam
        self.retry_beam = retry_beam

    def run(self):
        service = MFAAutoAlignmentService()
        result = service.run_alignment(
            audio_path=self.audio_path,
            dict_path=self.dict_path,
            acoustic_path=self.acoustic_path,
            output_path=self.output_path,
            beam=self.beam,
            retry_beam=self.retry_beam,
        )
        self.finished_sig.emit(result.success, result.message, result.detail)


class MFAAutoAlignmentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MFA自动标注")
        self.resize(600, 400)
        self.worker = None
        self.progress = None
        self.is_dark = True
        self.alignment_config = MFAAlignmentConfig()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        form_layout = QFormLayout()
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(10)
        form_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.acoustic_input = QLineEdit()
        self.btn_acoustic = QPushButton("选择文件")
        self.btn_acoustic.setMinimumWidth(120)
        self.btn_acoustic.setMinimumHeight(42)
        self.btn_acoustic.clicked.connect(self._browse_acoustic)
        h1 = QHBoxLayout()
        h1.setSpacing(10)
        h1.addWidget(self.acoustic_input)
        h1.addWidget(self.btn_acoustic)
        form_layout.addRow("MFA 声学模型路径:", h1)

        self.dict_input = QLineEdit()
        self.btn_dict = QPushButton("选择文件")
        self.btn_dict.setMinimumWidth(120)
        self.btn_dict.setMinimumHeight(42)
        self.btn_dict.clicked.connect(self._browse_dict)
        h2 = QHBoxLayout()
        h2.setSpacing(10)
        h2.addWidget(self.dict_input)
        h2.addWidget(self.btn_dict)
        form_layout.addRow("MFA 字典路径:", h2)

        self.audio_input = QLineEdit()
        self.audio_input.textChanged.connect(self._sync_output_path)
        self.btn_audio = QPushButton("选择文件夹")
        self.btn_audio.setMinimumWidth(120)
        self.btn_audio.setMinimumHeight(42)
        self.btn_audio.clicked.connect(self._browse_audio)
        h3 = QHBoxLayout()
        h3.setSpacing(10)
        h3.addWidget(self.audio_input)
        h3.addWidget(self.btn_audio)
        form_layout.addRow("音频路径:", h3)

        self.output_input = QLineEdit()
        self.btn_output = QPushButton("选择文件夹")
        self.btn_output.setMinimumWidth(120)
        self.btn_output.setMinimumHeight(42)
        self.btn_output.clicked.connect(self._browse_output)
        h4 = QHBoxLayout()
        h4.setSpacing(10)
        h4.addWidget(self.output_input)
        h4.addWidget(self.btn_output)
        form_layout.addRow("输出路径:", h4)

        self.beam_input = QSpinBox()
        self.beam_input.setRange(1, 10000)
        self.beam_input.setValue(self.alignment_config.beam)
        self.beam_input.valueChanged.connect(self._sync_retry_beam)
        form_layout.addRow("Beam:", self.beam_input)

        self.retry_beam_input = QSpinBox()
        self.retry_beam_input.setRange(1, 40000)
        self.retry_beam_input.setValue(self.alignment_config.retry_beam)
        form_layout.addRow("Retry beam:", self.retry_beam_input)

        for line_edit in [
            self.acoustic_input,
            self.dict_input,
            self.audio_input,
            self.output_input,
        ]:
            line_edit.setMinimumHeight(40)

        layout.addLayout(form_layout)

        self.btn_start = QPushButton("开始对齐")
        self.btn_start.setFixedHeight(48)
        self.btn_start.clicked.connect(self._start_alignment)

        self.btn_help = QPushButton("帮助")
        self.btn_help.setFixedSize(72, 24)
        self.btn_help.clicked.connect(self._open_help)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.btn_start)
        action_layout.addWidget(
            self.btn_help,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        layout.addLayout(action_layout)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("运行日志将显示在这里...")
        layout.addWidget(self.log_output)

        self._apply_dialog_styles()

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark
        self._apply_dialog_styles()

    def _apply_dialog_styles(self):
        if self.is_dark:
            start_style = """
            QPushButton {
                background-color: #35b74a;
                color: white;
                font-size: 14px;
                padding: 0 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #43c85a;
            }
            QPushButton:disabled {
                background-color: #5c6f60;
                color: #d0d0d0;
            }
            """
            log_style = (
                "background-color: #1e1e1e; color: #d4d4d4; "
                "font-family: Consolas;"
            )
        else:
            start_style = """
            QPushButton {
                background-color: #35b74a;
                color: white;
                font-size: 14px;
                padding: 0 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #43c85a;
            }
            QPushButton:disabled {
                background-color: #b7d7be;
                color: #f8f8f8;
            }
            """
            log_style = (
                "background-color: #ffffff; color: #333333; "
                "font-family: Consolas;"
            )
        self.btn_start.setStyleSheet(start_style)
        self.btn_help.setStyleSheet(
            "background-color: #28a745; color: white; "
            "font-size: 14px; font-weight: bold; border-radius: 4px;"
        )
        self.log_output.setStyleSheet(log_style)

    def _browse_acoustic(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 MFA 声学模型",
            "",
            "Zip Files (*.zip);;All Files (*)",
        )
        if path:
            self.acoustic_input.setText(path)

    def _browse_dict(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 MFA 字典",
            "",
            "Dictionary Files (*.dict *.txt);;All Files (*)",
        )
        if path:
            self.dict_input.setText(path)

    def _browse_audio(self):
        path = QFileDialog.getExistingDirectory(self, "选择音频文件夹")
        if path:
            self.audio_input.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_input.setText(path)

    def _sync_output_path(self, text: str):
        self.output_input.setText(text)

    def _open_help(self):
        help_path = get_resource_path(r"Phonetic_Export\index.html")
        if not os.path.exists(help_path):
            QMessageBox.warning(self, "帮助", f"找不到帮助文件:\n{help_path}")
            return
        url = QUrl.fromLocalFile(help_path)
        url.setFragment("s1765795935692")
        opened = QDesktopServices.openUrl(url)
        if not opened:
            QMessageBox.warning(self, "帮助", "帮助页面打开失败。")

    def _start_alignment(self):
        audio_path = self.audio_input.text().strip()
        dict_path = self.dict_input.text().strip()
        acoustic_path = self.acoustic_input.text().strip()
        output_path = self.output_input.text().strip()
        beam = self.beam_input.value()
        retry_beam = self.retry_beam_input.value()

        if not all([audio_path, dict_path, acoustic_path, output_path]):
            QMessageBox.warning(self, "错误", "请填写所有路径。")
            return

        self.btn_start.setEnabled(False)
        self.log_output.appendPlainText(">>> 开始 MFA 对齐任务...")
        self.log_output.appendPlainText(f"音频路径: {audio_path}")
        self.log_output.appendPlainText(f"输出路径: {output_path}")
        self.log_output.appendPlainText(f"Beam: {beam}")
        self.log_output.appendPlainText(f"Retry beam: {retry_beam}")

        self.progress = QProgressDialog("正在运行 MFA 对齐...", "取消", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setCancelButton(None)
        self.progress.show()

        self.worker = MFAAlignmentWorker(
            audio_path=audio_path,
            dict_path=dict_path,
            acoustic_path=acoustic_path,
            output_path=output_path,
            beam=beam,
            retry_beam=retry_beam,
        )
        self.worker.finished_sig.connect(self._on_alignment_finished)
        self.worker.start()

    def _sync_retry_beam(self, beam: int):
        if self.retry_beam_input.value() < beam * 4:
            self.retry_beam_input.setValue(beam * 4)

    def _on_alignment_finished(self, success: bool, message: str, detail: str):
        if self.progress is not None:
            self.progress.close()
        self.btn_start.setEnabled(True)
        if success:
            self.log_output.appendPlainText("\n>>> 任务完成")
            self.log_output.appendPlainText(message)
            QMessageBox.information(self, "成功", message)
            return
        self.log_output.appendPlainText("\n>>> 任务失败")
        self.log_output.appendPlainText(message)
        if detail:
            self.log_output.appendPlainText(detail)
        QMessageBox.critical(self, "失败", message)
