import os
import re
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QPushButton, QLineEdit, QLabel, QGroupBox, 
                             QProgressBar, QTextEdit, QMessageBox, QFileDialog,
                             QTableWidget, QTableWidgetItem, QComboBox)
from PyQt6.QtCore import Qt

from phonetic_toolbox.gui.workers.manipulation_workers import BatchProcessorWorker
from phonetic_toolbox.utils import parse_float_list

class BatchProcessorDialog(QDialog):
    """
    Dialog for batch processing audio files (speed and pitch changes).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量变速变调处理")
        self.resize(500, 400)
        self.layout = QVBoxLayout(self)
        
        # 文件夹选择
        h1 = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择包含音频的文件夹")
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(self.browse_folder)
        h1.addWidget(self.path_edit)
        h1.addWidget(btn_browse)
        self.layout.addLayout(h1)
        
        # 参数设置
        gbox = QGroupBox("处理参数")
        gl = QGridLayout()
        
        self.speed_edit = QLineEdit("1.0")
        gl.addWidget(QLabel("语速倍率 (1.0为原速):"), 0, 0)
        gl.addWidget(self.speed_edit, 0, 1)
        
        self.pitch_ratio_edit = QLineEdit("1.0")
        gl.addWidget(QLabel("音高倍率 (1.0为原调):"), 1, 0)
        gl.addWidget(self.pitch_ratio_edit, 1, 1)
        
        self.pitch_hz_edit = QLineEdit("0.0")
        gl.addWidget(QLabel("音高偏移 (Hz):"), 2, 0)
        gl.addWidget(self.pitch_hz_edit, 2, 1)
        
        gbox.setLayout(gl)
        self.layout.addWidget(gbox)
        
        # 进度显示
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.layout.addWidget(self.log_view)
        
        # 按钮
        h2 = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        h2.addWidget(self.btn_start)
        h2.addWidget(self.btn_close)
        self.layout.addLayout(h2)
        
        self.worker = None

    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if d:
            self.path_edit.setText(d)

    def log(self, msg):
        self.log_view.append(msg)

    def start_processing(self):
        folder = self.path_edit.text().strip()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "提示", "请选择有效的文件夹")
            return
            
        try:
            speed = float(self.speed_edit.text())
            pr = float(self.pitch_ratio_edit.text())
            ph = float(self.pitch_hz_edit.text())
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效的数字参数")
            return
            
        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("开始批量处理...")
        
        self.worker = BatchProcessorWorker(folder, speed, pr, ph)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, current, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.log(msg)

    def on_finished(self, msg):
        self.log(msg)
        QMessageBox.information(self, "完成", msg)
        self.btn_start.setEnabled(True)

    def on_error(self, msg):
        self.log(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
        self.btn_start.setEnabled(True)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)


class ImportF0Dialog(QDialog):
    """
    Dialog for importing F0 sequence from text.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入基频序列")
        self.resize(600, 400)
        self.layout = QVBoxLayout(self)
        
        self.label = QLabel("请在下方粘贴基频序列（支持空格、Tab、逗号、换行分隔）：\n如果粘贴两列，第一列为时间（将忽略），第二列为基频值。")
        self.layout.addWidget(self.label)
        
        self.text_edit = QTextEdit()
        self.layout.addWidget(self.text_edit)
        
        btns = QHBoxLayout()
        self.btn_ok = QPushButton("导入")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        self.layout.addLayout(btns)
        
    def get_data(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            return None
            
        # 预处理：替换常见分隔符为统一空格
        # 将逗号、分号等替换为空格
        text = re.sub(r'[,，;；]', ' ', text)
        
        lines = text.strip().split('\n')
        f0_values = []
        
        try:
            for line in lines:
                line = line.strip()
                if not line: continue
                
                parts = re.split(r'\s+', line)
                # 过滤空字符串
                parts = [p for p in parts if p]
                
                if not parts: continue
                
                val = 0.0
                if len(parts) == 1:
                    # 一列：基频
                    val = float(parts[0])
                else:
                    # 多列：取最后一列作为基频？或者按题目要求“第一列时间，第二列基频”
                    if len(parts) >= 2:
                        val = float(parts[1])
                    else:
                        val = float(parts[0])
                
                if np.isnan(val):
                    raise ValueError("包含NaN值")
                f0_values.append(val)
                
            if not f0_values:
                return None
                
            return np.array(f0_values)
            
        except Exception as e:
            QMessageBox.warning(self, "解析错误", f"无法解析数据: {e}")
            return None

class KnotEditorDialog(QDialog):
    """
    Dialog for editing knot points and their connections.
    """
    def __init__(self, parent=None, t1=0.0, f1_text="", start_mode="order",
                 t2=1.0, f2_text="", end_mode="order",
                 knot_points=None, knot_modes=None):
        super().__init__(parent)
        self.setWindowTitle("编辑拐点")
        self.resize(600, 500)
        
        self.t1 = t1
        self.f1_text = f1_text
        self.start_mode = start_mode
        self.t2 = t2
        self.f2_text = f2_text
        self.end_mode = end_mode
        self.knot_points = knot_points or []
        self.knot_modes = knot_modes or []
        
        self.result_data = None # Will store the result dict on success
        
        self.init_ui()
        
    def init_ui(self):
        vbox = QVBoxLayout(self)
        
        # 表格：起点 + 拐点们 + 终点
        total_rows = 1 + len(self.knot_points) + 1
        self.table = QTableWidget(total_rows, 3, self)
        self.table.setHorizontalHeaderLabels(["时间(s)", "频率列表(Hz,逗号)", "连接方式"])
        
        # 起点行
        self.table.setItem(0, 0, QTableWidgetItem(f"{self.t1:.3f}"))
        self.table.setItem(0, 1, QTableWidgetItem(self.f1_text))
        cb0 = self._create_mode_combo(self.start_mode)
        self.table.setCellWidget(0, 2, cb0)
        
        # 拐点行
        for i, kp in enumerate(self.knot_points):
            row = 1 + i
            self.table.setItem(row, 0, QTableWidgetItem(f"{kp['time']:.3f}"))
            self.table.setItem(row, 1, QTableWidgetItem(",".join([str(x) for x in kp['freqs']])))
            
            mode = self.knot_modes[i] if i < len(self.knot_modes) else "order"
            cb = self._create_mode_combo(mode)
            self.table.setCellWidget(row, 2, cb)
            
        # 终点行
        last_row = total_rows - 1
        self.table.setItem(last_row, 0, QTableWidgetItem(f"{self.t2:.3f}"))
        self.table.setItem(last_row, 1, QTableWidgetItem(self.f2_text))
        cbn = self._create_mode_combo(self.end_mode)
        self.table.setCellWidget(last_row, 2, cbn)
        
        # Buttons
        hbox = QHBoxLayout()
        btn_row_add = QPushButton("添加行", self)
        btn_row_del = QPushButton("删除选中", self)
        btn_save = QPushButton("保存更改", self)
        btn_close = QPushButton("关闭", self)
        
        btn_row_add.clicked.connect(self.add_row)
        btn_row_del.clicked.connect(self.del_rows)
        btn_save.clicked.connect(self.save_changes)
        btn_close.clicked.connect(self.reject)
        
        hbox.addWidget(btn_row_add)
        hbox.addWidget(btn_row_del)
        hbox.addWidget(btn_save)
        hbox.addWidget(btn_close)
        
        vbox.addWidget(self.table)
        vbox.addLayout(hbox)

    def _create_mode_combo(self, current_mode):
        cb = QComboBox()
        cb.addItems(["全连接", "顺序", "逆序", "常量"])
        idx = {"full":0, "order":1, "reverse":2, "constant":3}.get(current_mode, 1)
        cb.setCurrentIndex(idx)
        return cb

    def add_row(self):
        # 在终点前插入为新的拐点
        end_row = self.table.rowCount() - 1
        self.table.insertRow(end_row)
        self.table.setItem(end_row, 0, QTableWidgetItem(""))
        self.table.setItem(end_row, 1, QTableWidgetItem(""))
        cb = self._create_mode_combo("order")
        self.table.setCellWidget(end_row, 2, cb)

    def del_rows(self):
        rows = set([idx.row() for idx in self.table.selectedIndexes()])
        # 不允许删除起点与终点
        rows = [r for r in rows if r not in (0, self.table.rowCount()-1)]
        for r in sorted(rows, reverse=True):
            self.table.removeRow(r)

    def save_changes(self):
        # 起点
        t1_text = self.table.item(0,0).text().strip() if self.table.item(0,0) else ""
        f1_text = self.table.item(0,1).text().strip() if self.table.item(0,1) else ""
        mode0 = self._get_mode_from_combo(self.table.cellWidget(0,2))
        
        # 终点
        last = self.table.rowCount()-1
        t2_text = self.table.item(last,0).text().strip() if self.table.item(last,0) else ""
        f2_text = self.table.item(last,1).text().strip() if self.table.item(last,1) else ""
        moden = self._get_mode_from_combo(self.table.cellWidget(last,2))

        try:
            t1v = float(t1_text)
            t2v = float(t2_text)
            f1v = parse_float_list(f1_text)
            f2v = parse_float_list(f2_text)
            if not f1v or not f2v:
                raise ValueError("起止频率列表不能为空")
            if t2v <= t1v:
                raise ValueError("终止时间必须大于起始时间")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"起止点错误: {e}")
            return

        new_knots = []
        new_modes = []
        for r in range(1, last):
            t_item = self.table.item(r, 0)
            f_item = self.table.item(r, 1)
            mode_widget = self.table.cellWidget(r, 2)
            t_text = t_item.text().strip() if t_item else ""
            f_text = f_item.text().strip() if f_item else ""
            try:
                kt = float(t_text)
                fl = parse_float_list(f_text)
                if not fl:
                    raise ValueError("频率列表为空")
                new_knots.append({"time": kt, "freqs": fl})
                new_modes.append({"mode": self._get_mode_from_combo(mode_widget)})
            except Exception as e:
                QMessageBox.warning(self, "提示", f"第{r+1}行错误: {e}")
                return
        
        # 排序并应用
        order = np.argsort([k['time'] for k in new_knots])
        new_knots = [new_knots[i] for i in order]
        new_modes = [new_modes[i]["mode"] for i in order]
        
        # 检查时间重复且在(t1,t2)内
        times = [k['time'] for k in new_knots]
        if len(set(times)) != len(times):
            QMessageBox.warning(self, "提示", "拐点时间不能重复")
            return
        for kt in times:
            if kt <= t1v or kt >= t2v:
                QMessageBox.warning(self, "提示", "拐点时间需在起止时间之间")
                return

        self.result_data = {
            "t1": t1v,
            "t2": t2v,
            "f1": f1v,
            "f2": f2v,
            "knots": new_knots,
            "knot_modes": new_modes,
            "start_mode": mode0,
            "end_mode": moden
        }
        self.accept()

    def _get_mode_from_combo(self, combo):
        mapping = {"全连接":"full","顺序":"order","逆序":"reverse","常量":"constant"}
        return mapping.get(combo.currentText(), "order")
