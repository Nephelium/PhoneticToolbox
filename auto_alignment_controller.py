from __future__ import annotations

import os
import subprocess
import shutil
import json
from dataclasses import dataclass
from typing import Optional

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QThread, pyqtSignal

# Import the rename tool relative to this file
# Assuming rename_tool.py is in the root project directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rename_tool

@dataclass
class AutoAlignmentController:
    widget: QtWidgets.QWidget
    
    def __init__(self, widget: QtWidgets.QWidget):
        self.widget = widget
        self.rename_log_path: Optional[str] = None
        self.is_renamed_temporarily = False

    def init(self):
        # Connect Browse Buttons
        self.widget.buttonBrowseModel.clicked.connect(self.browse_model)
        self.widget.buttonBrowseDict.clicked.connect(self.browse_dict)
        self.widget.buttonBrowseAudio.clicked.connect(self.browse_audio)
        self.widget.buttonBrowseOutput.clicked.connect(self.browse_output)
        
        # Connect Action Buttons
        self.widget.buttonBatchRename.clicked.connect(self.batch_rename)
        self.widget.buttonStartAlignment.clicked.connect(self.start_alignment)
        
        # Initial State
        self.widget.editOutputPath.setPlaceholderText("默认同音频路径 (Default: Same as Audio Path)")

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.widget, "选择MFA声学模型", "", "Zip Files (*.zip);;All Files (*)"
        )
        if path:
            self.widget.editModelPath.setText(path)

    def browse_dict(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.widget, "选择MFA字典", "", "Dictionary Files (*.dict *.txt);;All Files (*)"
        )
        if path:
            self.widget.editDictPath.setText(path)

    def browse_audio(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self.widget, "选择音频文件夹"
        )
        if path:
            self.widget.editAudioPath.setText(path)
            # Auto-set output path if empty
            if not self.widget.editOutputPath.text():
                self.widget.editOutputPath.setText(path)

    def browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self.widget, "选择输出文件夹"
        )
        if path:
            self.widget.editOutputPath.setText(path)

    def batch_rename(self):
        audio_path = self.widget.editAudioPath.text().strip()
        if not audio_path or not os.path.exists(audio_path):
            QtWidgets.QMessageBox.warning(self.widget, "警告", "请先选择有效的音频路径！")
            return

        # Ask for mode
        msg_box = QtWidgets.QMessageBox(self.widget)
        msg_box.setWindowTitle("选择重命名模式")
        msg_box.setText("请选择如何处理中英文转换：")
        overwrite_btn = msg_box.addButton("覆盖原文件夹 (Overwrite)", QtWidgets.QMessageBox.ButtonRole.YesRole)
        new_folder_btn = msg_box.addButton("另存为新文件夹 (Save as New)", QtWidgets.QMessageBox.ButtonRole.NoRole)
        cancel_btn = msg_box.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        
        msg_box.exec()
        
        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_btn:
            return
            
        mode = "overwrite" if clicked_button == overwrite_btn else "new_folder"
        
        # If "new_folder", we need to handle the copying logic here or rely on rename_tool
        # rename_tool.process_renaming assumes working in place, so if "new_folder", 
        # we should copy first, then rename the copy.
        
        work_dir = audio_path
        if mode == "new_folder":
            parent_dir = os.path.dirname(audio_path)
            dir_name = os.path.basename(audio_path)
            new_dir_name = dir_name + "_en"
            work_dir = os.path.join(parent_dir, new_dir_name)
            
            if os.path.exists(work_dir):
                QtWidgets.QMessageBox.critical(self.widget, "错误", f"目标文件夹已存在: {work_dir}\n请先删除或重命名。")
                return
                
            try:
                shutil.copytree(audio_path, work_dir)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.widget, "错误", f"复制文件夹失败: {e}")
                return
        
        # Perform renaming
        success, log_path, final_dir = rename_tool.process_renaming(work_dir)
        
        if success:
            QtWidgets.QMessageBox.information(self.widget, "成功", f"重命名完成！\n日志路径: {log_path}")
            self.rename_log_path = log_path
            
            # If user chose "Save as New", update the paths
            if mode == "new_folder":
                self.widget.editAudioPath.setText(final_dir)
                self.widget.editOutputPath.setText(final_dir)
                QtWidgets.QMessageBox.information(self.widget, "提示", "音频路径和输出路径已自动更新为新文件夹。")
                
            # Set flag to indicate we should restore later (only if we overwrote? Or always? 
            # User said: "If user clicked batch convert... after alignment... convert back to Chinese. Overwrite the original English folder.")
            # So yes, we mark it for restoration.
            self.is_renamed_temporarily = True
            
        else:
            QtWidgets.QMessageBox.critical(self.widget, "错误", "重命名过程中出现错误，请检查控制台输出。")

    def start_alignment(self):
        # 1. Gather Inputs
        model_path = self.widget.editModelPath.text().strip()
        dict_path = self.widget.editDictPath.text().strip()
        audio_path = self.widget.editAudioPath.text().strip()
        output_path = self.widget.editOutputPath.text().strip()
        
        if not output_path:
            output_path = audio_path

        # 2. Validation
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self.widget, "错误", "请选择有效的声学模型文件！")
            return
        if not dict_path or not os.path.exists(dict_path):
            QtWidgets.QMessageBox.warning(self.widget, "错误", "请选择有效的字典文件！")
            return
        if not audio_path or not os.path.exists(audio_path):
            QtWidgets.QMessageBox.warning(self.widget, "错误", "请选择有效的音频文件夹！")
            return

        # Check for Chinese characters in Audio Path
        # Simple check: see if any character is outside ASCII or use the helper from rename_tool
        if rename_tool.has_chinese(audio_path):
            QtWidgets.QMessageBox.warning(
                self.widget, 
                "路径包含中文", 
                "音频路径包含中文字符，MFA 可能无法处理。\n请先点击“批量转换中英文名”按钮进行转换。"
            )
            return

        # 3. Construct Command
        # mfa align --clean <audio> <dict> <model> <output>
        cmd = [
            "mfa", "align", "--clean",
            audio_path,
            dict_path,
            model_path,
            output_path
        ]
        
        # 4. Run Command
        # Using a QProgressDialog to block UI but keep it responsive-ish
        progress = QtWidgets.QProgressDialog("正在执行 MFA 对齐...", "取消", 0, 0, self.widget)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        
        try:
            # We use subprocess.run, but we need to process events to keep UI alive?
            # Or just let it freeze for now as per "blocking" is acceptable if simple.
            # A better way is to use QProcess, but for simplicity in this snippet:
            
            # NOTE: subprocess.run will freeze the UI. 
            # If the task is long, this is bad UX. But for this task, I'll stick to simple implementation first.
            # If I want to be better, I'd use a thread. Let's use a thread to be safe.
            
            self.worker = AlignmentWorker(cmd)
            self.worker.finished.connect(lambda success, msg: self.on_alignment_finished(success, msg, progress))
            self.worker.start()
            
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(self.widget, "执行失败", str(e))

    def on_alignment_finished(self, success, msg, progress_dialog):
        progress_dialog.close()
        if success:
            QtWidgets.QMessageBox.information(self.widget, "完成", "MFA 对齐执行完成！")
            
            # 5. Auto Restore if needed
            if self.is_renamed_temporarily and self.rename_log_path:
                self.restore_original_names()
        else:
            QtWidgets.QMessageBox.critical(self.widget, "MFA 错误", f"执行出错:\n{msg}")

    def restore_original_names(self):
        # "If user clicked batch convert... convert back to Chinese. Overwrite the original English folder."
        # This means we just run the restore logic.
        
        if not self.rename_log_path or not os.path.exists(self.rename_log_path):
            return

        progress = QtWidgets.QProgressDialog("正在恢复文件名...", None, 0, 0, self.widget)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        QtCore.QCoreApplication.processEvents()
        
        success, msg = rename_tool.restore_names_from_log(self.rename_log_path)
        
        progress.close()
        
        if success:
            QtWidgets.QMessageBox.information(self.widget, "恢复完成", f"已自动将文件名恢复为中文。\n{msg}")
            # Reset flag
            self.is_renamed_temporarily = False
            self.rename_log_path = None
        else:
            QtWidgets.QMessageBox.warning(self.widget, "恢复失败", f"无法自动恢复文件名: {msg}")


class AlignmentWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        try:
            # shell=True on Windows might be needed if mfa is a batch file, 
            # but usually it's an exe or in path. 
            # Using shell=True for command line tools often helps resolving PATH issues on Windows.
            result = subprocess.run(
                self.cmd, 
                capture_output=True, 
                text=True, 
                shell=True,
                encoding='utf-8',  # Ensure we handle encoding right
                errors='replace'
            )
            
            if result.returncode == 0:
                self.finished.emit(True, result.stdout)
            else:
                self.finished.emit(False, result.stderr + "\n" + result.stdout)
        except Exception as e:
            self.finished.emit(False, str(e))
