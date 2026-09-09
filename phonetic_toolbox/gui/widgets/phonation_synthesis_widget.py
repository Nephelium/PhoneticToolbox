from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from phonetic_toolbox.core.manipulation.phonation_synthesis import (
    build_f0_control_axis,
    interpolate_f0_control_points,
    sample_f0_on_axis,
    voiced_range,
)
from phonetic_toolbox.gui.workers.phonation_synthesis_workers import (
    PhonationAnalysisWorker,
    PhonationGenerationWorker,
)
from phonetic_toolbox.models.phonation_synthesis_models import (
    ContinuumType,
    F0AlignmentMode,
    F0Backend,
    PhonationAnalysisConfig,
    PhonationAnalysisResult,
    PhonationGenerationConfig,
)
from phonetic_toolbox.services.phonation_synthesis_service import (
    PhonationSynthesisService,
)


class PhonationF0Canvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        windows_directory = Path(os.environ.get("WINDIR", r"C:\Windows"))
        font_candidates = (
            windows_directory / "Fonts" / "msyh.ttc",
            windows_directory / "Fonts" / "simsun.ttc",
        )
        font_path = next((path for path in font_candidates if path.exists()), None)
        self._chinese_font = (
            FontProperties(fname=str(font_path)) if font_path is not None else None
        )
        self._is_dark = True
        self._source: PhonationAnalysisResult | None = None
        self._target: PhonationAnalysisResult | None = None
        self._source_f0: np.ndarray | None = None
        self._target_f0: np.ndarray | None = None
        self._alignment = F0AlignmentMode.NORMALIZED

        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.set_theme(True)

    def set_theme(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.figure.set_facecolor("#1e1e1e" if self._is_dark else "#ffffff")
        self.plot(
            self._source,
            self._target,
            self._source_f0,
            self._target_f0,
            self._alignment,
        )

    def _style_axis(self, axis) -> None:
        foreground = "#e0e0e0" if self._is_dark else "#202020"
        background = "#252526" if self._is_dark else "#ffffff"
        grid_color = "#555555" if self._is_dark else "#d7d7d7"
        axis.set_facecolor(background)
        axis.tick_params(colors=foreground, which="both")
        axis.xaxis.label.set_color(foreground)
        axis.yaxis.label.set_color(foreground)
        axis.grid(True, color=grid_color, alpha=0.28, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_edgecolor(foreground)

    @staticmethod
    def _aligned_f0(
        f0_hz: np.ndarray,
        alignment: F0AlignmentMode,
    ) -> tuple[np.ndarray, np.ndarray]:
        bounds = voiced_range(f0_hz)
        if bounds is None:
            return np.array([]), np.array([])
        start, end = bounds
        segment = np.asarray(f0_hz[start : end + 1], dtype=np.float64)
        y_values = np.where(segment > 0, segment, np.nan)
        if alignment == F0AlignmentMode.NORMALIZED:
            x_values = np.linspace(0.0, 100.0, len(segment))
        else:
            x_values = np.arange(len(segment), dtype=np.float64)
        return x_values, y_values

    def plot(
        self,
        source: PhonationAnalysisResult | None,
        target: PhonationAnalysisResult | None,
        source_f0: np.ndarray | None,
        target_f0: np.ndarray | None,
        alignment: F0AlignmentMode,
    ) -> None:
        self._source = source
        self._target = target
        self._source_f0 = source_f0
        self._target_f0 = target_f0
        self._alignment = alignment
        self.figure.clear()
        waveform_axis = self.figure.add_subplot(2, 1, 1)
        f0_axis = self.figure.add_subplot(2, 1, 2)
        self._style_axis(waveform_axis)
        self._style_axis(f0_axis)

        if source is not None:
            time = np.arange(len(source.signal)) / source.sample_rate
            waveform_axis.plot(
                time,
                source.signal,
                color="#4aa3ff",
                linewidth=0.8,
                label="源音频",
            )
        if target is not None:
            time = np.arange(len(target.signal)) / target.sample_rate
            waveform_axis.plot(
                time,
                target.signal,
                color="#ff6b6b",
                linewidth=0.8,
                alpha=0.78,
                label="目标音频",
            )
        waveform_axis.set_ylabel("振幅", fontproperties=self._chinese_font)
        if source is not None or target is not None:
            waveform_axis.legend(
                loc="upper right",
                prop=self._chinese_font,
            )

        if source_f0 is not None:
            x_values, y_values = self._aligned_f0(source_f0, alignment)
            f0_axis.plot(
                x_values,
                y_values,
                color="#4aa3ff",
                linewidth=1.35,
                label="源 F0",
            )
        if target_f0 is not None:
            x_values, y_values = self._aligned_f0(target_f0, alignment)
            f0_axis.plot(
                x_values,
                y_values,
                color="#ff6b6b",
                linewidth=1.35,
                label="目标 F0",
            )
        x_label = (
            "归一化有声时间 (%)"
            if alignment == F0AlignmentMode.NORMALIZED
            else "从有声起点计时 (ms)"
        )
        f0_axis.set_xlabel(x_label, fontproperties=self._chinese_font)
        f0_axis.set_ylabel("F0 (Hz)")
        if source_f0 is not None or target_f0 is not None:
            f0_axis.legend(loc="upper right", prop=self._chinese_font)
        self.canvas.draw_idle()


class PhonationSynthesisWidget(QWidget):
    def __init__(
        self,
        parent=None,
        service: PhonationSynthesisService | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("发声类型合成 - Phonetic Toolbox v2")
        self.resize(1480, 900)
        self.setMinimumSize(1100, 700)

        self.service = service or PhonationSynthesisService()
        self.source_path: Path | None = None
        self.target_path: Path | None = None
        self.output_directory = (
            Path.home() / "Documents" / "PhoneticToolbox" / "发声类型合成输出"
        )
        self._output_selected_by_user = False
        self.source_analysis: PhonationAnalysisResult | None = None
        self.target_analysis: PhonationAnalysisResult | None = None
        self.source_f0_edit: np.ndarray | None = None
        self.target_f0_edit: np.ndarray | None = None
        self.last_analysis_config: PhonationAnalysisConfig | None = None
        self._active_worker: PhonationAnalysisWorker | PhonationGenerationWorker | None = None
        self._is_dark = True

        self._create_controls()
        self._build_layout()
        self._connect_analysis_signals()
        self.set_theme(True)

    def _create_controls(self) -> None:
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("parselmouth", F0Backend.PARSELMOUTH.value)
        self.backend_combo.addItem("reaper", F0Backend.REAPER.value)

        self.min_f0_spin = QDoubleSpinBox()
        self.min_f0_spin.setRange(20.0, 500.0)
        self.min_f0_spin.setValue(50.0)
        self.min_f0_spin.setSuffix(" Hz")

        self.max_f0_spin = QDoubleSpinBox()
        self.max_f0_spin.setRange(50.0, 1000.0)
        self.max_f0_spin.setValue(300.0)
        self.max_f0_spin.setSuffix(" Hz")

        self.f0_interval_spin = QDoubleSpinBox()
        self.f0_interval_spin.setRange(0.5, 20.0)
        self.f0_interval_spin.setDecimals(2)
        self.f0_interval_spin.setSingleStep(0.5)
        self.f0_interval_spin.setValue(1.0)
        self.f0_interval_spin.setSuffix(" ms")

        self.step_count_spin = QSpinBox()
        self.step_count_spin.setRange(2, 50)
        self.step_count_spin.setValue(9)

        self.point_count_spin = QSpinBox()
        self.point_count_spin.setRange(20, 200)
        self.point_count_spin.setValue(21)
        self.point_count_spin.setSuffix(" 点")

        self.continuum_combo = QComboBox()
        self.continuum_combo.addItem("仅改变 F0", int(ContinuumType.F0_ONLY))
        self.continuum_combo.addItem(
            "仅改变发声类型",
            int(ContinuumType.PHONATION_ONLY),
        )
        self.continuum_combo.addItem(
            "同时改变 F0 和发声类型",
            int(ContinuumType.F0_AND_PHONATION),
        )

        self.alignment_combo = QComboBox()
        self.alignment_combo.addItem(
            "时长归一化对齐",
            F0AlignmentMode.NORMALIZED.value,
        )
        self.alignment_combo.addItem(
            "不归一化，仅有声起点对齐",
            F0AlignmentMode.ONSET.value,
        )

        self.lpc_order_spin = QSpinBox()
        self.lpc_order_spin.setRange(4, 60)
        self.lpc_order_spin.setValue(20)

        self.frame_length_spin = QSpinBox()
        self.frame_length_spin.setRange(32, 1024)
        self.frame_length_spin.setSingleStep(16)
        self.frame_length_spin.setValue(128)
        self.frame_length_spin.setSuffix(" 点")

        self.frame_shift_spin = QSpinBox()
        self.frame_shift_spin.setRange(8, 512)
        self.frame_shift_spin.setSingleStep(8)
        self.frame_shift_spin.setValue(32)
        self.frame_shift_spin.setSuffix(" 点")

        self.window_combo = QComboBox()
        self.window_combo.addItems(("hamming", "hann", "blackman", "rectangular"))

        self.preemphasis_spin = QDoubleSpinBox()
        self.preemphasis_spin.setRange(0.0, 0.999)
        self.preemphasis_spin.setDecimals(3)
        self.preemphasis_spin.setSingleStep(0.01)
        self.preemphasis_spin.setValue(0.98)

        self.negative_peak_spin = QDoubleSpinBox()
        self.negative_peak_spin.setRange(-0.2, 0.0)
        self.negative_peak_spin.setDecimals(4)
        self.negative_peak_spin.setSingleStep(0.001)
        self.negative_peak_spin.setValue(-0.005)

        self.pulse_inner_spin = QDoubleSpinBox()
        self.pulse_inner_spin.setRange(0.1, 1.2)
        self.pulse_inner_spin.setDecimals(2)
        self.pulse_inner_spin.setSingleStep(0.05)
        self.pulse_inner_spin.setValue(0.5)

        self.pulse_outer_spin = QDoubleSpinBox()
        self.pulse_outer_spin.setRange(0.6, 3.0)
        self.pulse_outer_spin.setDecimals(2)
        self.pulse_outer_spin.setSingleStep(0.05)
        self.pulse_outer_spin.setValue(1.5)

        self.silence_threshold_spin = QDoubleSpinBox()
        self.silence_threshold_spin.setRange(-80.0, -10.0)
        self.silence_threshold_spin.setValue(-45.0)
        self.silence_threshold_spin.setSuffix(" dB")

        self.silence_padding_spin = QDoubleSpinBox()
        self.silence_padding_spin.setRange(0.0, 80.0)
        self.silence_padding_spin.setValue(8.0)
        self.silence_padding_spin.setSuffix(" ms")

        self.voiced_margin_spin = QDoubleSpinBox()
        self.voiced_margin_spin.setRange(0.0, 120.0)
        self.voiced_margin_spin.setValue(30.0)
        self.voiced_margin_spin.setSuffix(" ms")

        self.peak_limit_spin = QDoubleSpinBox()
        self.peak_limit_spin.setRange(0.1, 1.0)
        self.peak_limit_spin.setDecimals(2)
        self.peak_limit_spin.setSingleStep(0.01)
        self.peak_limit_spin.setValue(0.98)

        self.energy_match_check = QCheckBox("周期能量匹配")
        self.energy_match_check.setChecked(True)
        self.normalize_output_check = QCheckBox("输出响度匹配源音频")
        self.normalize_output_check.setChecked(True)

        self.source_label = QLabel("未选择")
        self.target_label = QLabel("未选择")
        self.output_label = QLabel(str(self.output_directory))
        for label in (self.source_label, self.target_label, self.output_label):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(("对齐时间", "源 F0", "目标 F0"))
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.plot = PhonationF0Canvas()
        self.status_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    @staticmethod
    def _form_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return label

    def _make_button(self, text: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(slot)
        return button

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.source_button = self._make_button(
            "源音频",
            lambda: self.choose_wav("source"),
        )
        self.target_button = self._make_button(
            "目标音频",
            lambda: self.choose_wav("target"),
        )
        self.output_button = self._make_button("输出目录", self.choose_output)
        self.help_button = self._make_button("参数说明", self.show_parameter_help)
        self.extract_button = self._make_button("提取 F0", self.start_analysis)
        self.apply_button = self._make_button("应用编辑", self.apply_table_edits)
        self.save_csv_button = self._make_button("保存 F0 CSV", self.save_f0_csv)
        self.generate_selected_button = self._make_button(
            "生成当前",
            self.generate_selected,
        )
        self.generate_all_button = self._make_button("生成全部", self.generate_all)
        self.cancel_button = self._make_button("取消任务", self.cancel_active_task)
        self.cancel_button.setEnabled(False)
        self._mutating_buttons = (
            self.source_button,
            self.target_button,
            self.output_button,
            self.extract_button,
            self.apply_button,
            self.save_csv_button,
            self.generate_selected_button,
            self.generate_all_button,
        )
        for button in (
            self.source_button,
            self.target_button,
            self.output_button,
            self.help_button,
            self.extract_button,
            self.apply_button,
            self.save_csv_button,
            self.generate_selected_button,
            self.generate_all_button,
            self.cancel_button,
        ):
            actions.addWidget(button)
        root.addLayout(actions)

        self.config_group = QGroupBox("参数设置")
        grid = QGridLayout(self.config_group)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        rows = (
            (
                "F0 算法",
                self.backend_combo,
                "最低 F0",
                self.min_f0_spin,
                "最高 F0",
                self.max_f0_spin,
                "F0 帧距",
                self.f0_interval_spin,
            ),
            (
                "合成步数",
                self.step_count_spin,
                "F0 编辑点数",
                self.point_count_spin,
                "连续统",
                self.continuum_combo,
                "F0 对齐",
                self.alignment_combo,
            ),
            (
                "LPC 阶数",
                self.lpc_order_spin,
                "窗长",
                self.frame_length_spin,
                "帧移",
                self.frame_shift_spin,
                "窗函数",
                self.window_combo,
            ),
            (
                "预加重",
                self.preemphasis_spin,
                "负峰阈值",
                self.negative_peak_spin,
                "脉冲内窗",
                self.pulse_inner_spin,
                "脉冲外窗",
                self.pulse_outer_spin,
            ),
            (
                "静音阈值",
                self.silence_threshold_spin,
                "静音保留",
                self.silence_padding_spin,
                "有声边界",
                self.voiced_margin_spin,
                "峰值限制",
                self.peak_limit_spin,
            ),
        )
        for row_index, row in enumerate(rows):
            for group_index in range(4):
                label_text = row[group_index * 2]
                editor = row[group_index * 2 + 1]
                grid.addWidget(self._form_label(label_text), row_index, group_index * 2)
                grid.addWidget(editor, row_index, group_index * 2 + 1)
        grid.addWidget(self.energy_match_check, 5, 0, 1, 2)
        grid.addWidget(self.normalize_output_check, 5, 2, 1, 2)
        grid.addWidget(self._form_label("源"), 5, 4)
        grid.addWidget(self.source_label, 5, 5)
        grid.addWidget(self._form_label("目标"), 5, 6)
        grid.addWidget(self.target_label, 5, 7)
        grid.addWidget(self._form_label("输出"), 6, 0)
        grid.addWidget(self.output_label, 6, 1, 1, 7)
        root.addWidget(self.config_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((380, 1100))
        root.addWidget(splitter, 1)

        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, 1)
        self.progress_bar.setFixedWidth(260)
        status_row.addWidget(self.progress_bar)
        root.addLayout(status_row)

    def _connect_analysis_signals(self) -> None:
        analysis_widgets = (
            self.backend_combo,
            self.min_f0_spin,
            self.max_f0_spin,
            self.f0_interval_spin,
            self.frame_length_spin,
            self.frame_shift_spin,
            self.lpc_order_spin,
            self.window_combo,
            self.preemphasis_spin,
            self.negative_peak_spin,
            self.pulse_inner_spin,
            self.pulse_outer_spin,
            self.silence_threshold_spin,
            self.silence_padding_spin,
            self.voiced_margin_spin,
        )
        for widget in analysis_widgets:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.mark_analysis_stale)
            else:
                widget.valueChanged.connect(self.mark_analysis_stale)
        self.point_count_spin.valueChanged.connect(self.populate_table)
        self.alignment_combo.currentIndexChanged.connect(self.on_alignment_changed)

    def analysis_config(self) -> PhonationAnalysisConfig:
        config = PhonationAnalysisConfig(
            f0_backend=F0Backend(str(self.backend_combo.currentData())),
            min_f0_hz=self.min_f0_spin.value(),
            max_f0_hz=self.max_f0_spin.value(),
            f0_frame_interval_ms=self.f0_interval_spin.value(),
            frame_length=self.frame_length_spin.value(),
            frame_shift=self.frame_shift_spin.value(),
            lpc_order=self.lpc_order_spin.value(),
            preemphasis=self.preemphasis_spin.value(),
            window_name=self.window_combo.currentText(),
            negative_peak_threshold=self.negative_peak_spin.value(),
            pulse_inner_periods=self.pulse_inner_spin.value(),
            pulse_outer_periods=self.pulse_outer_spin.value(),
            silence_threshold_db=self.silence_threshold_spin.value(),
            silence_padding_ms=self.silence_padding_spin.value(),
            voiced_margin_ms=self.voiced_margin_spin.value(),
        )
        config.validate()
        return config

    def generation_config(self) -> PhonationGenerationConfig:
        config = PhonationGenerationConfig(
            step_count=self.step_count_spin.value(),
            energy_match=self.energy_match_check.isChecked(),
            normalize_to_source=self.normalize_output_check.isChecked(),
            output_peak_limit=self.peak_limit_spin.value(),
        )
        config.validate()
        return config

    def alignment_mode(self) -> F0AlignmentMode:
        return F0AlignmentMode(str(self.alignment_combo.currentData()))

    def continuum_type(self) -> ContinuumType:
        return ContinuumType(int(self.continuum_combo.currentData()))

    def set_theme(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self.plot.set_theme(self._is_dark)

    def set_busy(self, busy: bool, message: str = "") -> None:
        for button in self._mutating_buttons:
            button.setEnabled(not busy)
        self.config_group.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress_bar.setVisible(busy)
        if not busy:
            self.progress_bar.setValue(0)
        if message:
            self.status_label.setText(message)

    def mark_analysis_stale(self, *_args) -> None:
        if self.source_analysis is not None or self.target_analysis is not None:
            self.status_label.setText("分析参数已改变，请重新提取 F0。")

    def _display_path(self, path: Path | None) -> str:
        return "未选择" if path is None else path.name

    def choose_wav(self, role: str) -> None:
        initial = self.source_path or self.target_path or Path.home()
        initial_directory = initial.parent if initial.is_file() else initial
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择 WAV 音频",
            str(initial_directory),
            "WAV 文件 (*.wav);;所有文件 (*)",
        )
        if not selected:
            return
        path = Path(selected)
        if role == "source":
            self.source_path = path
            self.source_label.setText(self._display_path(path))
            self.source_label.setToolTip(str(path))
            if not self._output_selected_by_user:
                self.output_directory = path.parent / "发声类型合成输出"
                self.output_label.setText(str(self.output_directory))
                self.output_label.setToolTip(str(self.output_directory))
        else:
            self.target_path = path
            self.target_label.setText(self._display_path(path))
            self.target_label.setToolTip(str(path))
        self.clear_analysis()
        if self.source_path is not None and self.target_path is not None:
            self.start_analysis()
        else:
            missing = "目标音频" if self.target_path is None else "源音频"
            self.status_label.setText(f"请选择{missing}。")

    def choose_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            str(self.output_directory),
        )
        if selected:
            self.output_directory = Path(selected)
            self._output_selected_by_user = True
            self.output_label.setText(str(self.output_directory))
            self.output_label.setToolTip(str(self.output_directory))

    def clear_analysis(self) -> None:
        self.source_analysis = None
        self.target_analysis = None
        self.source_f0_edit = None
        self.target_f0_edit = None
        self.last_analysis_config = None
        self.table.setRowCount(0)
        self.refresh_plot()

    def start_analysis(self) -> None:
        if self._active_worker is not None and self._active_worker.isRunning():
            return
        if self.source_path is None or self.target_path is None:
            self.show_error("请先选择源音频和目标音频。")
            return
        try:
            config = self.analysis_config()
        except Exception as exc:
            self.show_error(str(exc))
            return
        worker = PhonationAnalysisWorker(
            self.service,
            self.source_path,
            self.target_path,
            config,
            self,
        )
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._on_analysis_succeeded)
        worker.failed.connect(self._on_task_failed)
        worker.canceled.connect(self._on_task_canceled)
        worker.finished.connect(self._on_worker_finished)
        self._active_worker = worker
        self.set_busy(True, "正在分析源音频和目标音频...")
        worker.start()

    def _on_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(int(np.clip(value, 0, 100)))
        self.status_label.setText(message)

    def _on_analysis_succeeded(
        self,
        source: PhonationAnalysisResult,
        target: PhonationAnalysisResult,
    ) -> None:
        self.source_analysis = source
        self.target_analysis = target
        self.source_f0_edit = source.f0_hz.copy()
        self.target_f0_edit = target.f0_hz.copy()
        self.last_analysis_config = source.config
        self.populate_table()
        self.refresh_plot()
        self.set_busy(False, f"已使用 {source.config.f0_backend.value} 完成分析。")

    def _on_task_failed(self, message: str, detail: str) -> None:
        self.set_busy(False, message)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("发声类型合成")
        dialog.setText(message)
        dialog.setDetailedText(detail)
        dialog.exec()

    def _on_task_canceled(self) -> None:
        self.set_busy(False, "任务已取消。")

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._active_worker:
            self._active_worker = None
        if worker is not None:
            worker.deleteLater()

    def cancel_active_task(self) -> None:
        worker = self._active_worker
        if worker is None or not worker.isRunning():
            return
        worker.cancel()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("正在取消任务...")

    def populate_table(self, *_args) -> None:
        if self.source_f0_edit is None or self.target_f0_edit is None:
            return
        mode = self.alignment_mode()
        axis = build_f0_control_axis(
            self.source_f0_edit,
            self.target_f0_edit,
            self.point_count_spin.value(),
            mode,
        )
        source_values = sample_f0_on_axis(self.source_f0_edit, axis, mode)
        target_values = sample_f0_on_axis(self.target_f0_edit, axis, mode)
        axis_label = (
            "归一化有声时间 (%)"
            if mode == F0AlignmentMode.NORMALIZED
            else "有声起点后时间 (ms)"
        )
        self.table.setHorizontalHeaderLabels((axis_label, "源 F0", "目标 F0"))
        self.table.setRowCount(len(axis))
        for row, axis_value in enumerate(axis):
            time_item = QTableWidgetItem(f"{axis_value:g}")
            time_item.setFlags(time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, time_item)
            source_value = source_values[row]
            target_value = target_values[row]
            self.table.setItem(
                row,
                1,
                QTableWidgetItem("" if source_value <= 0 else f"{source_value:.3f}"),
            )
            self.table.setItem(
                row,
                2,
                QTableWidgetItem("" if target_value <= 0 else f"{target_value:.3f}"),
            )

    def apply_table_edits(self) -> bool:
        if self.source_f0_edit is None or self.target_f0_edit is None:
            self.show_error("请先提取 F0。")
            return False
        try:
            axis_values: list[float] = []
            source_values: list[float] = []
            target_values: list[float] = []
            for row in range(self.table.rowCount()):
                axis_item = self.table.item(row, 0)
                source_item = self.table.item(row, 1)
                target_item = self.table.item(row, 2)
                if axis_item is None:
                    continue
                axis_values.append(float(axis_item.text()))
                source_values.append(
                    float(source_item.text())
                    if source_item is not None and source_item.text().strip()
                    else 0.0
                )
                target_values.append(
                    float(target_item.text())
                    if target_item is not None and target_item.text().strip()
                    else 0.0
                )
            axis = np.asarray(axis_values, dtype=np.float64)
            source = np.asarray(source_values, dtype=np.float64)
            target = np.asarray(target_values, dtype=np.float64)
            mode = self.alignment_mode()
            self.source_f0_edit = interpolate_f0_control_points(
                axis,
                source,
                self.source_f0_edit,
                mode,
            )
            self.target_f0_edit = interpolate_f0_control_points(
                axis,
                target,
                self.target_f0_edit,
                mode,
            )
            if self.source_analysis is not None:
                self.source_analysis = self.source_analysis.copy_with_f0(
                    self.source_f0_edit
                )
            if self.target_analysis is not None:
                self.target_analysis = self.target_analysis.copy_with_f0(
                    self.target_f0_edit
                )
            self.refresh_plot()
            self.status_label.setText("F0 编辑已应用。")
            return True
        except Exception as exc:
            self.show_error(f"F0 编辑无效：{exc}")
            return False

    def on_alignment_changed(self, *_args) -> None:
        self.populate_table()
        self.refresh_plot()

    def refresh_plot(self) -> None:
        self.plot.plot(
            self.source_analysis,
            self.target_analysis,
            self.source_f0_edit,
            self.target_f0_edit,
            self.alignment_mode(),
        )

    def save_f0_csv(self) -> None:
        if self.source_f0_edit is None or self.target_f0_edit is None:
            self.show_error("请先提取 F0。")
            return
        if not self.apply_table_edits():
            return
        try:
            path = self.service.save_f0_csv(
                self.output_directory,
                self.source_f0_edit,
                self.target_f0_edit,
            )
            self.status_label.setText(f"已保存：{path}")
        except Exception as exc:
            self.show_error(f"保存 F0 CSV 失败：{exc}")

    def _analysis_ready(self) -> bool:
        if self.source_analysis is None or self.target_analysis is None:
            self.show_error("请先提取 F0。")
            return False
        try:
            if self.last_analysis_config != self.analysis_config():
                self.show_error("分析参数已改变，请重新提取 F0。")
                return False
        except Exception as exc:
            self.show_error(str(exc))
            return False
        return self.apply_table_edits()

    def generate_selected(self) -> None:
        if not self._analysis_ready():
            return
        self._start_generation(self.continuum_type())

    def generate_all(self) -> None:
        if not self._analysis_ready():
            return
        self._start_generation(None)

    def _start_generation(self, continuum_type: ContinuumType | None) -> None:
        if self.source_analysis is None or self.target_analysis is None:
            return
        try:
            generation = self.generation_config()
        except Exception as exc:
            self.show_error(str(exc))
            return
        worker = PhonationGenerationWorker(
            self.service,
            self.source_analysis,
            self.target_analysis,
            generation,
            self.output_directory,
            continuum_type,
            self,
        )
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._on_generation_succeeded)
        worker.failed.connect(self._on_task_failed)
        worker.canceled.connect(self._on_task_canceled)
        worker.finished.connect(self._on_worker_finished)
        self._active_worker = worker
        task_name = "全部连续统" if continuum_type is None else continuum_type.display_name
        self.set_busy(True, f"正在生成{task_name}...")
        worker.start()

    def _on_generation_succeeded(self, result) -> None:
        if isinstance(result, tuple):
            message = f"全部连续统已生成到：{result[0].output_directory.parent}"
        else:
            message = f"已生成：{result.output_directory}"
        self.set_busy(False, message)
        QMessageBox.information(self, "发声类型合成", message)

    def show_error(self, message: str) -> None:
        self.status_label.setText(message)
        QMessageBox.warning(self, "发声类型合成", message)

    def show_parameter_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("发声类型合成参数说明")
        dialog.resize(760, 650)
        layout = QVBoxLayout(dialog)
        text = QTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(
            "功能说明\n\n"
            "本模块以源音频的时长和声道滤波器为基础，通过 LPC 残差信号近似迁移目标音频的发声类型。"
            "它可以仅改变 F0、仅改变发声类型，或同时改变两者。结果是信号处理意义上的实验刺激，"
            "不等同于对声门生理机制的直接还原。\n\n"
            "F0 算法：parselmouth 使用 Praat；reaper 使用项目内置 REAPER，并保留纯 Python 回退。\n"
            "最低/最高 F0：基频搜索范围。范围太窄会断裂，太宽可能出现倍频或半频。\n"
            "F0 帧距：提取时间步长，默认 1 ms。\n"
            "合成步数：连续统包含的刺激数量。\n"
            "F0 编辑点数：表格中的控制点数量，修改后点击“应用编辑”。\n"
            "F0 对齐：归一化模式把两段有声区映射到 0–100%；起点模式保留各自时长。\n\n"
            "LPC 阶数：源声道滤波器阶数，常用 12–30。\n"
            "窗长/帧移：LPC 分帧参数，帧移必须小于窗长。\n"
            "窗函数：hamming 默认；hann、blackman 可降低不同类型的频谱泄漏。\n"
            "预加重：LPC 前增强高频，常用 0.95–0.99。\n"
            "负峰阈值：声门脉冲负峰检测阈值；越接近 0 越宽松。\n"
            "脉冲内窗/外窗：按当前 F0 周期搜索相邻脉冲，外窗必须大于内窗。\n\n"
            "静音阈值/静音保留：控制首尾静音裁剪。\n"
            "有声边界：在 F0 有声区两侧保留的 LPC 分析范围。\n"
            "峰值限制：限制输出峰值，避免削波。\n"
            "周期能量匹配：把目标残差周期能量匹配到源周期，通常建议开启。\n"
            "输出响度匹配源音频：使各 step 平均响度接近源音频。\n\n"
            "“生成当前”输出所选方向和类型；“生成全部”输出源到目标、目标到源各三类，共六组。"
        )
        layout.addWidget(text)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    def closeEvent(self, event: QCloseEvent) -> None:
        worker = self._active_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            self.status_label.setText("正在停止后台任务，请稍候...")
            if not worker.wait(5000):
                event.ignore()
                QTimer.singleShot(250, self.close)
                return
        event.accept()
