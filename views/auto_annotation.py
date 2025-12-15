import sys
import os
import shutil
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QHBoxLayout, QLabel, QInputDialog,
                             QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Add project root to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Fix for PyInstaller bundled app
if getattr(sys, 'frozen', False):
    project_root = sys._MEIPASS

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Try importing from root (bundled) or relative
    import rename_tool
except ImportError as e:
    try:
        # Try importing as package
        from PhoneticToolbox import rename_tool
    except ImportError as e2:
        # Fallback or error handling
        print(f"Error: Could not import rename_tool. \nTrace 1: {e}\nTrace 2: {e2}")
        rename_tool = None

# Worker Thread for MFA Alignment
class MFAAlignmentWorker(QThread):
    finished_sig = pyqtSignal(bool, str) # success, message
    
    def __init__(self, audio_path, dict_path, acoustic_path, output_path):
        super().__init__()
        self.audio_path = audio_path
        self.dict_path = dict_path
        self.acoustic_path = acoustic_path
        self.output_path = output_path
        
    def run(self):
        try:
            import os
            import sys
            import tempfile
            
            # --- ENV SETUP for Frozen App ---
            if getattr(sys, 'frozen', False):
                # 1. Path to binaries (Ensure they are found)
                if hasattr(sys, '_MEIPASS'):
                    mfa_bin = os.path.join(sys._MEIPASS, 'mfa_bin')
                    if os.path.exists(mfa_bin):
                        os.environ["PATH"] = mfa_bin + os.pathsep + os.environ["PATH"]
                        # Also add to DLL search path for Windows
                        if hasattr(os, 'add_dll_directory'):
                            try:
                                os.add_dll_directory(mfa_bin)
                            except:
                                pass
            
            # TEST SOUNDFILE
            try:
                # Pre-load sndfile.dll for frozen app
                if getattr(sys, 'frozen', False):
                    import ctypes
                    try:
                        # Try to load from PATH (mfa_bin or root)
                        ctypes.CDLL('sndfile.dll')
                        print("Successfully pre-loaded sndfile.dll from PATH")
                    except Exception as e1:
                        print(f"Failed to pre-load sndfile.dll from PATH: {e1}")
                        # Try explicit paths
                        base_dir = os.path.dirname(sys.executable)
                        candidates = [
                            os.path.join(base_dir, 'sndfile.dll'),
                            os.path.join(base_dir, '_internal', 'sndfile.dll'),
                        ]
                        if hasattr(sys, '_MEIPASS'):
                            candidates.insert(0, os.path.join(sys._MEIPASS, 'sndfile.dll'))
                            
                        for c in candidates:
                            if os.path.exists(c):
                                try:
                                    ctypes.CDLL(c)
                                    print(f"Successfully pre-loaded sndfile.dll from {c}")
                                    break
                                except Exception as e2:
                                    print(f"Failed to load {c}: {e2}")

                import soundfile as sf
                print(f"SoundFile imported: {sf.__file__}")
                
                # Diagnostic: Try to read one file from the corpus directory
                if os.path.isdir(self.audio_path):
                    print(f"Scanning {self.audio_path} for test audio file...")
                    test_file = None
                    for root, _, files in os.walk(self.audio_path):
                        for f in files:
                            if f.lower().endswith(('.wav', '.flac', '.ogg')):
                                test_file = os.path.join(root, f)
                                break
                        if test_file: break
                    
                    if test_file:
                        print(f"Attempting to read test file: {test_file}")
                        try:
                            info = sf.info(test_file)
                            print(f"Read success: {info}")
                        except Exception as e:
                            print(f"Read FAILED: {e}")
                            # If read fails here, MFA will also fail. 
                            # We should probably warn the user, but MFA will throw CorpusError anyway.
                    else:
                        print("No common audio files found in directory during diagnostic scan.")

            except Exception as e:
                print(f"CRITICAL: Failed to import soundfile: {e}")
                # We can't really emit here easily without breaking flow, but it will show in log if captured

            # 3. Force OpenBLAS/MKL to single thread to prevent deadlocks in multiprocessing
            # Even when using multiprocessing, the individual worker processes should be single-threaded
            # for linear algebra operations to avoid thread oversubscription.
            os.environ["BLAS_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            os.environ["NUMBA_DISABLE_JIT"] = "1"

            # 4. FIX LLVMLITE DLL LOADING
            # llvmlite.binding.ffi tries to load "llvmlite.dll". 
            # In frozen app, it might not be in a place where ctypes.CDLL finds it by name only.
            # We explicitly help it by loading it or adding directory.
            if getattr(sys, 'frozen', False):
                try:
                    base_dir = os.path.dirname(sys.executable)
                    internal_dir = os.path.join(base_dir, "_internal")
                    llvmlite_dll = None
                    for root, dirs, files in os.walk(internal_dir):
                        for f in files:
                            if f.lower() == "llvmlite.dll":
                                llvmlite_dll = os.path.join(root, f)
                                break
                        if llvmlite_dll:
                            break
                    if llvmlite_dll:
                        import ctypes
                        try:
                            ctypes.CDLL(llvmlite_dll)
                        except OSError:
                            if hasattr(os, "add_dll_directory"):
                                os.add_dll_directory(os.path.dirname(llvmlite_dll))
                                ctypes.CDLL(llvmlite_dll)
                except Exception:
                    pass

            # Import MFA API here to avoid heavy load at startup
            from montreal_forced_aligner.alignment import PretrainedAligner
            from montreal_forced_aligner.config import GLOBAL_CONFIG
            
            # Configure Aligner
            # Equivalent to: mfa align --clean
            
            # Use a safe temp dir in user's temp folder
            temp_dir = os.path.join(tempfile.gettempdir(), "MFA_PhoneticToolbox")
            # Explicitly clean up previous temp dir to avoid stale data
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up previous temp dir: {temp_dir}")
                except Exception as e:
                    print(f"Warning: Failed to clean temp dir {temp_dir}: {e}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Default configuration for multiprocessing
            num_jobs = 1 # Start with safe default
            use_mp = False
            
            # If frozen, we try to keep multiprocessing but disable rich progress bars 
            # as they can cause IO deadlocks in frozen apps without console attached properly.
            verbose = True
            
            if getattr(sys, 'frozen', False):
                 print("Frozen mode detected: ENFORCING Single-Process Mode.")
                 num_jobs = 1
                 use_mp = False
                 
                 # 1. Set Environment Variables (MFA checks these)
                 os.environ["MFA_NUM_JOBS"] = "1"
                 os.environ["CALC_JOBS"] = "1"
                 os.environ["NUM_JOBS"] = "1"
                 os.environ["JOBLIB_MULTIPROCESSING"] = "0"
            
            print(f"Initializing PretrainedAligner with num_jobs={num_jobs}, use_mp={use_mp}")

            aligner = PretrainedAligner(
                corpus_directory=self.audio_path,
                dictionary_path=self.dict_path,
                acoustic_model_path=self.acoustic_path,
                output_directory=self.output_path,
                temporary_directory=temp_dir,
                clean=True, # --clean
                verbose=verbose,
                num_jobs=num_jobs,
                use_mp=use_mp
            )
            
            # CRITICAL: Force override configuration to ensure single process
            if getattr(sys, 'frozen', False):
                try:
                    aligner.num_jobs = 1
                    aligner.use_mp = False
                    if hasattr(aligner, 'config'):
                        aligner.config.num_jobs = 1
                        aligner.config.use_mp = False
                    # Also try to modify the internal corpus config if it exists
                    if hasattr(aligner, 'corpus_configuration'):
                        aligner.corpus_configuration.num_jobs = 1
                    
                    print("Successfully enforced single-process mode on aligner instance.")
                except Exception as e:
                    print(f"Warning: Failed to set aligner properties: {e}")
            
            # Run alignment
            aligner.align()
            aligner.export_files(self.output_path)
            
            self.finished_sig.emit(True, f"对齐完成!\n输出路径: {self.output_path}")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.finished_sig.emit(False, f"执行出错: {e}\n{tb}")

class AutoAnnotationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自动标注 (Auto Annotation)")
        self.resize(600, 300)
        self.init_ui()
        self.restore_log_path = None # To store log path for auto-restore
        self.renamed_in_place = False # Track if we did an in-place rename
        self.original_audio_path = None # Track original path if we did "Save as new"

    def init_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # 1. MFA Acoustic Model
        self.acoustic_input = QLineEdit()
        self.btn_acoustic = QPushButton("选择文件")
        self.btn_acoustic.clicked.connect(self.browse_acoustic)
        h1 = QHBoxLayout()
        h1.addWidget(self.acoustic_input)
        h1.addWidget(self.btn_acoustic)
        form_layout.addRow("MFA 声学模型路径:", h1)

        # 2. MFA Dictionary
        self.dict_input = QLineEdit()
        self.btn_dict = QPushButton("选择文件")
        self.btn_dict.clicked.connect(self.browse_dict)
        h2 = QHBoxLayout()
        h2.addWidget(self.dict_input)
        h2.addWidget(self.btn_dict)
        form_layout.addRow("MFA 字典路径:", h2)

        # 3. Audio Path
        self.audio_input = QLineEdit()
        self.audio_input.textChanged.connect(self.sync_output_path)
        self.btn_audio = QPushButton("选择文件夹")
        self.btn_audio.clicked.connect(self.browse_audio)
        h3 = QHBoxLayout()
        h3.addWidget(self.audio_input)
        h3.addWidget(self.btn_audio)
        form_layout.addRow("音频路径:", h3)

        # 4. Output Path
        self.output_input = QLineEdit()
        self.btn_output = QPushButton("选择文件夹")
        self.btn_output.clicked.connect(self.browse_output)
        h4 = QHBoxLayout()
        h4.addWidget(self.output_input)
        h4.addWidget(self.btn_output)
        form_layout.addRow("输出路径:", h4)

        layout.addLayout(form_layout)

        # Batch Rename Button
        self.btn_rename = QPushButton("批量转换中英文名 (Batch Rename)")
        self.btn_rename.clicked.connect(self.batch_rename)
        
        # Restore Button
        self.btn_restore = QPushButton("还原为中文名 (Restore Chinese Names)")
        self.btn_restore.clicked.connect(self.restore_names)
        
        h_rename = QHBoxLayout()
        h_rename.addWidget(self.btn_rename)
        h_rename.addWidget(self.btn_restore)
        layout.addLayout(h_rename)

        # Start Alignment Button
        self.btn_start = QPushButton("开始对齐 (Start Alignment)")
        self.btn_start.clicked.connect(self.start_alignment)
        # Style it to look prominent
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-size: 16px; 
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
        """)
        layout.addWidget(self.btn_start)

        # Log Output Area
        from PyQt6.QtWidgets import QPlainTextEdit
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("运行日志将显示在这里...")
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        layout.addWidget(self.log_output)

    def browse_acoustic(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 MFA 声学模型", "", "Zip Files (*.zip);;All Files (*)")
        if path:
            self.acoustic_input.setText(path)

    def browse_dict(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 MFA 字典", "", "Dictionary Files (*.dict *.txt);;All Files (*)")
        if path:
            self.dict_input.setText(path)

    def browse_audio(self):
        path = QFileDialog.getExistingDirectory(self, "选择音频文件夹")
        if path:
            self.audio_input.setText(path)
            # Output path syncs automatically via textChanged

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_input.setText(path)
            # Disconnect sync if user manually sets output? 
            # For simplicity, we just set it. The sync only happens when audio changes.

    def sync_output_path(self, text):
        # Default output path to same as audio path
        self.output_input.setText(text)

    def batch_rename(self):
        target_dir = self.audio_input.text().strip()
        if not target_dir:
            QMessageBox.warning(self, "提示", "请先选择音频路径")
            return

        if not rename_tool:
            QMessageBox.critical(self, "错误", "无法加载重命名工具")
            return

        # Ask for mode
        items = ["另存为新文件夹 (Save as new)", "覆盖原文件夹 (Overwrite)"]
        item, ok = QInputDialog.getItem(self, "选择模式", "请选择转换模式:", items, 0, False)
        if not ok or not item:
            return

        mode = '2' if "另存为" in item else '1'
        
        reply = QMessageBox.question(self, "确认", f"确定要对以下目录执行转换吗?\n{target_dir}\n模式: {item}",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        success, new_work_dir, log_path, msg = rename_tool.batch_rename_process(target_dir, mode)
        
        QMessageBox.information(self, "结果", msg)
        
        if success:
            self.restore_log_path = log_path
            self.audio_input.setText(new_work_dir) # This triggers sync_output_path
            self.output_input.setText(new_work_dir)
            
            if mode == '2':
                self.renamed_in_place = False
                self.original_audio_path = target_dir
                QMessageBox.information(self, "提示", f"音频路径和输出路径已更新为新文件夹:\n{new_work_dir}\n\n后续生成的TextGrid也会在恢复时被自动重命名回中文。")
            else:
                self.renamed_in_place = True
                self.original_audio_path = target_dir # Same as new_work_dir
                QMessageBox.information(self, "提示", f"已在原文件夹重命名。\n后续生成的TextGrid也会在恢复时被自动重命名回中文。")

    def restore_names(self):
        # Try to find log file
        log_path = None
        
        # 1. Check if we have it in memory
        if self.restore_log_path and os.path.exists(self.restore_log_path):
            log_path = self.restore_log_path
            
        # 2. Check current audio directory
        if not log_path:
            current_dir = self.audio_input.text().strip()
            if current_dir and os.path.isdir(current_dir):
                possible_log = os.path.join(current_dir, "rename_log.json")
                if os.path.exists(possible_log):
                    log_path = possible_log
        
        # 3. Ask user
        if not log_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择重命名日志 (Select Rename Log)", "", "JSON Files (*.json);;All Files (*)")
            if file_path:
                log_path = file_path
        
        if not log_path:
            return
            
        if not rename_tool:
            QMessageBox.critical(self, "错误", "无法加载重命名工具")
            return
            
        reply = QMessageBox.question(self, "确认还原", f"确定要根据日志还原文件名吗?\n日志: {log_path}\n注意: 这也会将生成的 TextGrid 文件重命名回中文。",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        success, msg = rename_tool.restore_process(log_path)
        QMessageBox.information(self, "结果", msg)

    def start_alignment(self):
        audio_path = self.audio_input.text().strip()
        dict_path = self.dict_input.text().strip()
        acoustic_path = self.acoustic_input.text().strip()
        output_path = self.output_input.text().strip()

        if not all([audio_path, dict_path, acoustic_path, output_path]):
            QMessageBox.warning(self, "错误", "请填写所有路径")
            return

        # Check Chinese in audio path
        # Using rename_tool.has_chinese if available, or just implement it
        try:
            has_zh = rename_tool.has_chinese(audio_path)
        except:
            has_zh = any('\u4e00' <= char <= '\u9fff' for char in audio_path)

        if has_zh:
            QMessageBox.warning(self, "路径错误", "音频路径中包含中文字符，MFA 可能无法处理。\n请使用全部英文字符的路径，或先点击“批量转换中英文名”按钮。")
            return

        # Start alignment in background thread using Python API
        # This handles both Dev mode (using installed MFA) and Frozen mode (using bundled binaries)
        
        self.btn_start.setEnabled(False)
        self.log_output.appendPlainText(">>> 开始 MFA 对齐任务...")
        self.log_output.appendPlainText(f"音频路径: {audio_path}")
        self.log_output.appendPlainText(f"输出路径: {output_path}")

        # Restore Progress Dialog as requested
        self.progress = QProgressDialog("正在运行 MFA 对齐...", "取消", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0) # Show immediately
        self.progress.show()
        
        self.worker = MFAAlignmentWorker(audio_path, dict_path, acoustic_path, output_path)
        self.worker.finished_sig.connect(self.on_alignment_finished)
        self.worker.start()

    def on_alignment_finished(self, success, msg):
        self.progress.close()
        self.btn_start.setEnabled(True)
        
        if success:
            self.log_output.appendPlainText("\n>>> 任务完成!")
            self.log_output.appendPlainText(msg)
            QMessageBox.information(self, "成功", msg)
            if self.restore_log_path and os.path.exists(self.restore_log_path):
                self.auto_restore()
        else:
            self.log_output.appendPlainText("\n>>> 任务失败!")
            self.log_output.appendPlainText(msg)
            QMessageBox.critical(self, "失败", msg)

    def auto_restore(self):
        if not rename_tool: return
        
        # "If user clicked batch convert ... then after alignment finishes, automatically convert back to Chinese. 
        # Directly overwrite the original English folder."
        
        # If we did "Save as new" (mode 2), we have `folder` and `folder_en`.
        # The user says "Directly overwrite the original English folder". 
        # This part is slightly ambiguous.
        # If we have `_en` folder, and we "convert back to Chinese", do we rename `_en` back to Chinese names?
        # Yes, `restore_process` uses the log to rename files back.
        # If we are in `_en` folder, the files are currently English (pinyin).
        # The log maps `Original Chinese Path` -> `New English Path`.
        # `restore_process` reverses this: `New English Path` -> `Original Chinese Path`.
        
        # BUT: `Original Chinese Path` in the log points to the *source* folder if we did `shutil.copytree`?
        # Wait, let's check `rename_tool.py` logic.
        # In mode 2: `shutil.copytree(target_dir, work_dir)`. `work_dir` is `_en`.
        # Then we walk `work_dir`.
        # `old_path` is in `work_dir`. `new_full_path` is in `work_dir`.
        # So the log contains paths inside `_en` folder.
        # So restoring will rename files INSIDE `_en` folder back to Chinese.
        # This effectively makes the `_en` folder contain Chinese files again.
        
        # "Directly overwrite the original English folder." -> This seems to mean "modify the English folder in place".
        # Which is exactly what `restore_process` does if the paths in log match the current files.
        
        reply = QMessageBox.question(self, "自动恢复", "对齐已完成。是否将文件名恢复为中文?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            success, msg = rename_tool.restore_process(self.restore_log_path)
            QMessageBox.information(self, "恢复结果", msg)
