import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QSpinBox, QDoubleSpinBox, QTableWidget, 
    QTableWidgetItem, QMessageBox, QGroupBox, QFormLayout,
    QProgressBar, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ...services.acoustic_service import AcousticAnalysisService

class AnalysisWorker(QThread):
    finished = pyqtSignal(object)  # Emits DataFrame on success
    error = pyqtSignal(str)        # Emits error message on failure

    def __init__(self, service, wav_path, config):
        super().__init__()
        self.service = service
        self.wav_path = wav_path
        self.config = config

    def run(self):
        try:
            df = self.service.analyze_file(self.wav_path, self.config)
            self.finished.emit(df)
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            self.error.emit(str(e))

class AcousticWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.service = AcousticAnalysisService()
        self.df_result = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- File Selection ---
        file_group = QGroupBox("音频文件")
        file_layout = QHBoxLayout()
        self.path_label = QLabel("未选择文件")
        self.btn_select = QPushButton("选择文件...")
        self.btn_select.clicked.connect(self.select_file)
        file_layout.addWidget(self.path_label)
        file_layout.addWidget(self.btn_select)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # --- Parameters ---
        param_group = QGroupBox("参数设置")
        param_layout = QFormLayout()
        
        self.spin_min_f0 = QDoubleSpinBox()
        self.spin_min_f0.setRange(20, 500)
        self.spin_min_f0.setValue(50)
        
        self.spin_max_f0 = QDoubleSpinBox()
        self.spin_max_f0.setRange(100, 1000)
        self.spin_max_f0.setValue(500)
        
        self.spin_frameshift = QDoubleSpinBox()
        self.spin_frameshift.setRange(1, 100)
        self.spin_frameshift.setValue(10)
        
        param_layout.addRow("最小 F0 (Hz):", self.spin_min_f0)
        param_layout.addRow("最大 F0 (Hz):", self.spin_max_f0)
        param_layout.addRow("帧移 (ms):", self.spin_frameshift)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- Actions ---
        action_layout = QHBoxLayout()
        self.btn_run = QPushButton("运行分析")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        
        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_save)
        layout.addLayout(action_layout)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Results Table ---
        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.setLayout(layout)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "WAV Files (*.wav);;All Files (*)"
        )
        if path:
            self.path_label.setText(path)
            self.btn_run.setEnabled(True)

    def run_analysis(self):
        wav_path = self.path_label.text()
        if not wav_path or wav_path == "未选择文件":
            return

        config = {
            "min_f0": self.spin_min_f0.value(),
            "max_f0": self.spin_max_f0.value(),
            "frameshift_ms": self.spin_frameshift.value(),
            "f0_method": "praat_cc", # default
            "n_periods": 4
        }

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.btn_run.setEnabled(False)
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        self.worker = AnalysisWorker(self.service, wav_path, config)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_finished(self, df):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.df_result = df
        self.btn_save.setEnabled(True)
        
        # Display in table (first 100 rows)
        if df is not None and not df.empty:
            rows = min(100, len(df))
            cols = len(df.columns)
            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            self.table.setHorizontalHeaderLabels(df.columns)
            
            for r in range(rows):
                for c in range(cols):
                    val = df.iloc[r, c]
                    item = QTableWidgetItem(f"{val:.4f}")
                    self.table.setItem(r, c, item)
            
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            QMessageBox.information(self, "完成", "分析完成！仅显示前100行。")
        else:
            QMessageBox.warning(self, "警告", "分析结果为空。")

    def on_analysis_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "错误", f"分析失败:\n{msg}")

    def save_results(self):
        if self.df_result is None:
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if path:
            try:
                self.service.save_results(self.df_result, path)
                QMessageBox.information(self, "成功", f"结果已保存到:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败:\n{e}")
