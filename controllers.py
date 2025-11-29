from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6 import QtWidgets, QtCore, uic, QtGui
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import numpy as np

from .models.state import AppState
from .services.praat_service import compute_praat_f0_formants
from .utils.csv_io import save_csv, load_csv_any
from .utils.textgrid_parser import parse_textgrid


log = logging.getLogger(__name__)

ASCII_TITLE = (
    " ██████╗ ██╗  ██╗ ██████╗ ███╗   ██╗███████╗████████╗██╗ ██████╗████████╗ ██████╗  ██████╗ ██╗     ██████╗  ██████╗ ██╗  ██╗\n"
    " ██╔══██╗██║  ██║██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██║██╔════╝╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔══██╗██╔═══██╗╚██╗██╔╝\n"
    "██████╔╝███████║██║   ██║██╔██╗ ██║█████╗     ██║   ██║██║        ██║   ██║   ██║██║   ██║██║     ██████╔╝██║   ██║ ╚███╔╝ \n"
    "██╔═══╝ ██╔══██║██║   ██║██║╚██╗██║██╔══╝     ██║   ██║██║        ██║   ██║   ██║██║   ██║██║     ██╔══██╗██║   ██║ ██╔██╗ \n"
    " ██║     ██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   ██║╚██████╗   ██║   ╚██████╔╝╚██████╔╝███████╗██████╔╝╚██████╔╝██╔╝ ██╗\n"
    " ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝"
)


def resource_path(relative: str) -> Path:
    import sys
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    direct = base / relative
    if direct.exists():
        return direct
    pkg_base = base / "PhoneticToolbox"
    cand = pkg_base / relative
    return cand if cand.exists() else Path(__file__).parent / relative


def load_ui(name: str) -> QtWidgets.QWidget:
    ui_file = resource_path(f"views/{name}.ui")
    return uic.loadUi(str(ui_file))


@dataclass
class MainController:
    widget: QtWidgets.QWidget
    state: AppState

    def init(self) -> None:
        self.widget.setWindowTitle("PhoneticToolbox (PyQt6)")
        self._children: list[tuple[QtWidgets.QWidget, object]] = []
        
        # ASCII Art Title
        ascii_title = ASCII_TITLE
        try:
            self.widget.labelTitle.setText(ascii_title)
            # 设置字体为等宽字体，确保对齐
            font = QtGui.QFont("Consolas", 8)  # 调整字体大小以适应宽度
            font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
            self.widget.labelTitle.setFont(font)
        except Exception:
            pass

        # 绑定按钮
        self.widget.buttonParameterEstimation.clicked.connect(self.open_parameter_estimation)
        self.widget.buttonOutputToText.clicked.connect(self.open_output_text)
        self.widget.buttonOutputToEMU.clicked.connect(self.open_egg_analysis)
        self.widget.buttonParameterDisplay.clicked.connect(self.open_parameter_display)
        self.widget.buttonSettings.clicked.connect(self.open_settings)
        self.widget.buttonManualData.clicked.connect(self.open_manual_data)
        self.widget.buttonAbout.clicked.connect(self.open_about)
        try:
            self.widget.buttonSynthesisAudio.clicked.connect(self.open_synthesis_app)
            self.widget.buttonChangeF0.clicked.connect(self.open_change_f0_app)
        except Exception:
            pass
        self.widget.buttonExit.clicked.connect(self.open_help)

    def open_parameter_estimation(self) -> None:
        self._show_child("ui_parameter_estimation", ParameterEstimationController)

    def open_settings(self) -> None:
        self._show_child("ui_settings", SettingsController)

    def open_parameter_display(self) -> None:
        self._show_child("ui_parameter_display", ParameterDisplayController)

    def open_output_text(self) -> None:
        try:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            html_path = resource_path("perception_experiment/perception_experiment.html")
            if not html_path.exists():
                QtWidgets.QMessageBox.warning(self.widget, "感知实验", f"未找到页面: {html_path}")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "感知实验", f"打开失败: {e}")

    def open_help(self) -> None:
        try:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            candidates = [
                Path(r"c:\Users\13680\Desktop\project\【中山大学】\PhoneticToolbox\PhoneticToolboxDoc.html"),
                resource_path("PhoneticToolboxDoc.html"),
            ]
            target = None
            for p in candidates:
                try:
                    if Path(p).exists():
                        target = Path(p)
                        break
                except Exception:
                    pass
            if not target:
                QtWidgets.QMessageBox.warning(self.widget, "使用说明", "未找到使用说明页面")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "使用说明", f"打开失败: {e}")

    def open_egg_analysis(self) -> None:
        try:
            import subprocess
            exe_rel = "EGG/EGG信号分析.exe"
            exe_path = resource_path(exe_rel)
            if not exe_path.exists():
                cand = Path(__file__).resolve().parent / "EGG" / "dist" / "EGG信号分析.exe"
                if cand.exists():
                    exe_path = cand
            if not exe_path.exists():
                QtWidgets.QMessageBox.warning(self.widget, "EGG信号分析", f"未找到可执行文件: {exe_rel}")
                return
            subprocess.Popen([str(exe_path)])
            self._integrate_egg_results()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "EGG信号分析", f"启动失败: {e}")

    def _integrate_egg_results(self) -> None:
        pass

    def open_synthesis_app(self) -> None:
        try:
            import subprocess
            exe_rel = "klatt/klatt合成器.exe"
            exe_path = resource_path(exe_rel)
            if not exe_path.exists():
                alt_rel = "klatt/Klatt合成器.exe"
                exe_path = resource_path(alt_rel) if resource_path(alt_rel).exists() else exe_path
            if not exe_path.exists():
                cand = Path(__file__).resolve().parent / "klatt" / "dist" / "Klatt合成器.exe"
                if cand.exists():
                    exe_path = cand
                else:
                    cand2 = Path(__file__).resolve().parent / "klatt" / "dist" / "klatt合成器.exe"
                    if cand2.exists():
                        exe_path = cand2
            if not exe_path.exists():
                QtWidgets.QMessageBox.warning(self.widget, "合成音频", f"未找到可执行文件: {exe_rel}")
                return
            subprocess.Popen([str(exe_path)])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "合成音频", f"启动失败: {e}")

    def open_change_f0_app(self) -> None:
        try:
            import subprocess
            exe_rel = "changeF0/changeF0.exe"
            exe_path = resource_path(exe_rel)
            if not exe_path.exists():
                cand = Path(__file__).resolve().parent / "changeF0" / "dist" / "changeF0.exe"
                if cand.exists():
                    exe_path = cand
            if not exe_path.exists():
                QtWidgets.QMessageBox.warning(self.widget, "修改基频", f"未找到可执行文件: {exe_rel}")
                return
            subprocess.Popen([str(exe_path)])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "修改基频", f"启动失败: {e}")

    def open_manual_data(self) -> None:
        self._show_child("ui_manual_data", ManualDataController)

    def open_about(self) -> None:
        self._show_child("ui_about", AboutController)

    def _show_child(self, ui_name: str, controller_cls) -> None:
        dlg = load_ui(ui_name)
        # 构造控制器（AboutController 仅需 widget 参数）
        try:
            ctrl = controller_cls(dlg, self.state)
        except TypeError:
            ctrl = controller_cls(dlg)
        ctrl.init()
        dlg.destroyed.connect(lambda _=None, d=dlg, c=ctrl: self._on_child_closed(d, c))
        self._children.append((dlg, ctrl))
        dlg.show()

    def _on_child_closed(self, dlg: QtWidgets.QWidget, ctrl: object) -> None:
        self._children = [(d, c) for (d, c) in self._children if d is not dlg]


@dataclass
class SettingsController:
    widget: QtWidgets.QWidget
    state: AppState

    def init(self) -> None:
        try:
            self.widget.editWavDir.setText(self.state.wavdir)
            self.widget.editMatDir.setText(self.state.matdir)
        except Exception:
            pass
        try:
            self.widget.spinEnergyRatio.setValue(float(self.state.voicing_energy_threshold_ratio))
            self.widget.spinEnergyWin.setValue(int(self.state.voicing_energy_window_ms))
            self.widget.spinFrameShift.setValue(int(self.state.frameshift))
            self.widget.spinWindowSize.setValue(int(self.state.windowsize))
            self.widget.spinSmoothWin.setValue(int(self.state.O_smoothwinsize))
            # F0 method
            pass
            self.widget.spinNPeriodsH.setValue(int(self.state.Nperiods))
            self.widget.spinNPeriodsEC.setValue(int(self.state.Nperiods_EC))
        except Exception:
            pass

        # REAPER 参数
        try:
            # Check default path and update if needed
            if not self.state.F0ReaperBin or self.state.F0ReaperBin == "reaper":
                self.state.F0ReaperBin = str(Path("reaper/reaper.exe"))

            self.widget.editReaperBin.setText(self.state.F0ReaperBin)
            self.widget.spinReaperMinF0.setValue(int(self.state.F0ReaperMinF0))
            self.widget.spinReaperMaxF0.setValue(int(self.state.F0ReaperMaxF0))
            self.widget.checkReaperHilbert.setChecked(bool(self.state.F0ReaperHilbert))
            self.widget.checkReaperNoHighpass.setChecked(bool(self.state.F0ReaperNoHighpass))

            # Rename labels
            for label in self.widget.findChildren(QtWidgets.QLabel):
                txt = label.text().lower()
                if "min-f0" in txt:
                    label.setText("最低基频")
                elif "max-f0" in txt:
                    label.setText("最高基频")
        except Exception:
            pass

        try:
            self.widget.checkRecurse.setChecked(bool(self.state.recursedir))
            self.widget.checkLinkMat.setChecked(bool(self.state.linkmatdir))
            self.widget.checkLinkWav.setChecked(bool(self.state.linkwavdir))
        except Exception:
            pass

        try:
            self.widget.buttonBrowseWavDir.clicked.connect(self._browse_wav)
            self.widget.buttonBrowseMatDir.clicked.connect(self._browse_mat)
        except Exception:
            pass
        self.widget.buttonSave.clicked.connect(self._save)
        self.widget.buttonClose.clicked.connect(self.widget.close)

        try:
            self.widget.checkRecurse.toggled.connect(lambda v: setattr(self.state, "recursedir", 1 if v else 0))
            self.widget.checkLinkMat.toggled.connect(lambda v: setattr(self.state, "linkmatdir", 1 if v else 0))
            self.widget.checkLinkWav.toggled.connect(lambda v: setattr(self.state, "linkwavdir", 1 if v else 0))
        except Exception:
            pass

    def _browse_wav(self) -> None:
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择WAV目录", self.state.wavdir)
        if new_dir:
            self.state.wavdir = new_dir
            self.widget.editWavDir.setText(new_dir)
            if self.state.linkmatdir:
                self.state.matdir = new_dir
                self.widget.editMatDir.setText(new_dir)

    def _browse_mat(self) -> None:
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择CSV目录", self.state.matdir)
        if new_dir:
            self.state.matdir = new_dir
            self.widget.editMatDir.setText(new_dir)
            if self.state.linkwavdir:
                self.state.wavdir = new_dir
                self.widget.editWavDir.setText(new_dir)

    def _save(self) -> None:
        try:
            self.state.voicing_energy_threshold_ratio = float(self.widget.spinEnergyRatio.value())
            self.state.voicing_energy_window_ms = int(self.widget.spinEnergyWin.value())
            self.state.frameshift = int(self.widget.spinFrameShift.value())
            self.state.windowsize = int(self.widget.spinWindowSize.value())
            self.state.O_smoothwinsize = int(self.widget.spinSmoothWin.value())
            # 已默认使用两种 F0 方法并行，不再提供选择
            self.state.Nperiods = int(self.widget.spinNPeriodsH.value())
            self.state.Nperiods_EC = int(self.widget.spinNPeriodsEC.value())
            # 保存 REAPER 参数
            try:
                self.state.F0ReaperBin = self.widget.editReaperBin.text().strip()
                try:
                    # Default use frame shift for reaper frame interval
                    self.state.F0ReaperFrameIntervalSec = float(self.widget.spinFrameShift.value()) / 1000.0
                except Exception:
                    self.state.F0ReaperFrameIntervalSec = float(self.state.frameshift) / 1000.0
                self.state.F0ReaperMinF0 = int(self.widget.spinReaperMinF0.value())
                self.state.F0ReaperMaxF0 = int(self.widget.spinReaperMaxF0.value())
                self.state.F0ReaperHilbert = 1 if self.widget.checkReaperHilbert.isChecked() else 0
                self.state.F0ReaperNoHighpass = 1 if self.widget.checkReaperNoHighpass.isChecked() else 0
            except Exception:
                pass
            try:
                self.state.recursedir = 1 if self.widget.checkRecurse.isChecked() else 0
                self.state.linkmatdir = 1 if self.widget.checkLinkMat.isChecked() else 0
                self.state.linkwavdir = 1 if self.widget.checkLinkWav.isChecked() else 0
            except Exception:
                pass
        except Exception:
            pass
        QtWidgets.QMessageBox.information(self.widget, "设置", "设置已保存")


@dataclass
class ParameterEstimationController:
    widget: QtWidgets.QWidget
    state: AppState

    def init(self) -> None:
        self.widget.editInputDir.setText(self.state.wavdir)
        self.widget.editOutputDir.setText(self.state.matdir)
        self.widget.checkboxSaveMatWithWav.setChecked(bool(self.state.PE_savematwithwav))
        self.widget.buttonBrowseInput.clicked.connect(self._browse_input)
        self.widget.buttonBrowseOutput.clicked.connect(self._browse_output)
        self.widget.buttonRefreshFiles.clicked.connect(self._refresh_files)
        self.widget.buttonStart.clicked.connect(self._start)
        try:
            self.widget.listFiles.itemSelectionChanged.connect(self._show_waveform)
        except Exception:
            pass
        try:
            self._setup_waveplot()
        except Exception:
            pass
        try:
            self.widget.buttonPlaySelected.clicked.connect(self._play_selected)
            self.widget.buttonStopPlayback.clicked.connect(self._stop_playback)
            self._player = QMediaPlayer(self.widget)
            self._audio = QAudioOutput(self.widget)
            self._player.setAudioOutput(self._audio)
        except Exception:
            pass
        try:
            self.widget.buttonReadTextGrid.clicked.connect(self._read_textgrid)
            self.widget.buttonTextGridSegmentation.clicked.connect(self._toggle_segmentation)
            self.widget.buttonSaveSegmentedAudio.clicked.connect(self._save_segmented_audio)
        except Exception:
            pass
        self._textgrid_cache = {}
        self._selected_tier_name = None
        try:
            self.widget.listFiles.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        except Exception:
            pass
        self._refresh_files()

    def _read_textgrid(self) -> None:
        self._refresh_files()
        QtWidgets.QMessageBox.information(self.widget, "TextGrid", "已重新读取目录下的TextGrid文件")

    def _toggle_segmentation(self) -> None:
        items = self.widget.listFiles.selectedItems()
        if not items:
            QtWidgets.QMessageBox.warning(self.widget, "TextGrid切分", "请先在文件列表中选中一个文件，以读取其TextGrid层级信息")
            return
        name = items[0].text()
        tg = self._textgrid_cache.get(name)
        if not tg or not tg.tiers:
            QtWidgets.QMessageBox.warning(self.widget, "TextGrid切分", f"未找到文件 {name} 的TextGrid数据或层级为空")
            return
        
        current_idx = -1
        if self._selected_tier_name:
            for i, t in enumerate(tg.tiers):
                if t.name == self._selected_tier_name:
                    current_idx = i
                    break
        
        next_idx = current_idx + 1
        if next_idx >= len(tg.tiers):
            self._selected_tier_name = None
            self.widget.buttonTextGridSegmentation.setText("TextGrid切分")
        else:
            self._selected_tier_name = tg.tiers[next_idx].name
            self.widget.buttonTextGridSegmentation.setText(f"item [{next_idx+1}]：{self._selected_tier_name}")

    def _save_segmented_audio(self) -> None:
        items = self.widget.listFiles.selectedItems()
        if not items:
            QtWidgets.QMessageBox.warning(self.widget, "保存", "请先选择要保存的文件")
            return
        
        if not self._selected_tier_name:
            QtWidgets.QMessageBox.warning(self.widget, "保存", "请先点击'TextGrid切分'选择一个层级")
            return

        self.widget.buttonSaveSegmentedAudio.setDown(True)
        QtWidgets.QApplication.processEvents()

        try:
            input_dir = Path(self.widget.editInputDir.text())
            output_dir = Path(self.widget.editOutputDir.text())
            output_dir.mkdir(parents=True, exist_ok=True)
            from scipy.io import wavfile
            
            count = 0
            for item in items:
                name = item.text()
                tg = self._textgrid_cache.get(name)
                if not tg:
                    continue
                
                tier = next((t for t in tg.tiers if t.name == self._selected_tier_name), None)
                if not tier:
                    continue
                
                wav_path = input_dir / name
                try:
                    fs, data = wavfile.read(wav_path)
                except Exception:
                    continue

                original_stem = Path(name).stem
                csv_path = output_dir / f"{original_stem}.csv"
                # Load CSV if it exists, else None
                csv_data = load_csv_any(csv_path) if csv_path.exists() else None

                for interval in tier.intervals:

                    txt = interval.text.strip()
                    if not txt or txt.lower() in ["sil", "eps", "<sil>", "<eps>"]:
                        continue
                    
                    start_sample = int(interval.xmin * fs)
                    end_sample = int(interval.xmax * fs)
                    
                    if start_sample >= data.shape[0] or end_sample <= start_sample:
                        continue
                    
                    sliced_audio = data[start_sample:end_sample]
                    
                    new_stem = f"{original_stem}_{tier.name}_{txt}_{interval.xmin:.3f}_{interval.xmax:.3f}"
                    new_stem = "".join([c for c in new_stem if c.isalnum() or c in "._-"])
                    
                    new_wav_path = output_dir / f"{new_stem}.wav"
                    wavfile.write(new_wav_path, fs, sliced_audio)
                    
                    if csv_data:
                        frameshift_ms = float(csv_data.get("frameshift", self.state.frameshift))
                        sliced_csv = {"frameshift": frameshift_ms}
                        
                        n_frames = 0
                        for k, v in csv_data.items():
                            if isinstance(v, np.ndarray) and v.ndim == 1:
                                n_frames = max(n_frames, len(v))
                        
                        times = np.arange(n_frames) * frameshift_ms / 1000.0
                        mask = (times >= interval.xmin) & (times <= interval.xmax)
                        
                        for k, v in csv_data.items():
                            if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n_frames:
                                sliced_csv[k] = v[mask]
                            else:
                                sliced_csv[k] = v
                        
                        if mask.any():
                            # Add just this tier's text
                            sliced_csv[f"textgrid_{tier.name}"] = np.full(np.sum(mask), txt, dtype=object)
                            save_csv(output_dir / f"{new_stem}.csv", sliced_csv)
                    
                    count += 1
            
            QtWidgets.QMessageBox.information(self.widget, "保存", f"已保存 {count} 个片段")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "保存", f"保存失败: {e}")
        finally:
            self.widget.buttonSaveSegmentedAudio.setDown(False)

    def _browse_input(self) -> None:
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择输入目录", self.state.wavdir)
        if new_dir:
            self.state.wavdir = new_dir
            self.widget.editInputDir.setText(new_dir)
            if self.widget.checkboxSaveMatWithWav.isChecked():
                self.state.matdir = new_dir
                self.widget.editOutputDir.setText(new_dir)
            self._refresh_files()

    def _browse_output(self) -> None:
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择输出目录", self.state.matdir)
        if new_dir:
            self.state.matdir = new_dir
            self.widget.editOutputDir.setText(new_dir)

    def _refresh_files(self) -> None:
        self.widget.listFiles.clear()
        self._textgrid_cache = {}
        p = Path(self.widget.editInputDir.text())
        if not p.exists():
            return
        
        from .utils.textgrid_parser import parse_textgrid
        
        for wav in p.rglob("*.wav") if self.state.recursedir else p.glob("*.wav"):
            self.widget.listFiles.addItem(str(wav.name))
            tg_path = wav.with_suffix(".TextGrid")
            if tg_path.exists():
                tg = parse_textgrid(tg_path)
                if tg:
                    self._textgrid_cache[wav.name] = tg

    def _start(self) -> None:
        items = [self.widget.listFiles.item(i).text() for i in range(self.widget.listFiles.count())]
        if not items:
            QtWidgets.QMessageBox.warning(self.widget, "参数估计", "没有输入文件")
            return
        input_dir = Path(self.widget.editInputDir.text())
        output_dir = Path(self.widget.editOutputDir.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        self.widget.buttonStart.setEnabled(False)
        progress = QtWidgets.QProgressDialog("处理中...", "取消", 0, len(items) * 20, self.widget)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        
        # User requested to ignore segmentation options for main processing
        tier_name = None

        self._worker = PEWorker(items, input_dir, output_dir, self.state, tier_name)
        def _on_progress(val: int, name: str) -> None:
            try:
                progress.setValue(val)
                progress.setLabelText(f"处理中: {name}")
            except Exception:
                pass
        self._worker.progress_sig.connect(_on_progress)
        self._worker.error_sig.connect(lambda msg: QtWidgets.QMessageBox.critical(self.widget, "参数估计错误", msg))
        def _on_finished():
            progress.close()
            self.widget.buttonStart.setEnabled(True)
            QtWidgets.QMessageBox.information(self.widget, "参数估计", "处理完成")
        self._worker.finished_sig.connect(_on_finished)
        def _on_cancel():
            if self._worker.isRunning():
                self._worker.requestInterruption()
        progress.canceled.connect(_on_cancel)
        self._worker.start()

    def _show_waveform(self) -> None:
        items = [self.widget.listFiles.item(i).text() for i in range(self.widget.listFiles.count())]
        sel_items = self.widget.listFiles.selectedItems()
        if not items:
            QtWidgets.QMessageBox.warning(self.widget, "波形显示", "没有输入文件")
            return
        if not sel_items:
            QtWidgets.QMessageBox.warning(self.widget, "波形显示", "请先在列表中选中一个音频文件")
            return
        name = sel_items[0].text()
        wav_path = Path(self.widget.editInputDir.text()) / name
        try:
            from .services.praat_service import read_wav_mono_float
            fs, y = read_wav_mono_float(wav_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "波形显示", f"读取失败: {e}")
            return
        self._current_wav_path = wav_path
        self._current_wave_fs = fs
        self._current_wave_y = y
        t_ms = np.arange(y.size, dtype=float) * 1000.0 / float(fs)
        try:
            self._ax_wave.clear()
            self._ax_wave.plot(t_ms, y, color="black")
            self._ax_wave.set_xlabel("Time (ms)")
            self._ax_wave.set_ylabel("Amplitude")
            self._wave_canvas.draw()
        except Exception:
            pass

    def _setup_waveplot(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
        from matplotlib.figure import Figure
        cont = getattr(self.widget, "wavePlotContainer", None)
        if cont is None:
            return
        try:
            lay = QtWidgets.QVBoxLayout(cont)
        except Exception:
            lay = QtWidgets.QVBoxLayout()
            cont.setLayout(lay)
        self._wave_fig = Figure(figsize=(6, 3), dpi=100)
        self._ax_wave = self._wave_fig.add_subplot(1, 1, 1)
        self._wave_canvas = Canvas(self._wave_fig)
        lay.addWidget(self._wave_canvas)

    def _play_selected(self) -> None:
        try:
            from PyQt6.QtCore import QUrl
            items = self.widget.listFiles.selectedItems()
            if not items:
                QtWidgets.QMessageBox.warning(self.widget, "播放", "请先选中一个音频")
                return
            name = items[0].text()
            wav_path = Path(self.widget.editInputDir.text()) / name
            self._current_wav_path = wav_path
            url = QUrl.fromLocalFile(str(wav_path))
            self._player.setSource(url)
            self._audio.setVolume(0.9)
            self._player.play()
        except Exception:
            pass

    def _stop_playback(self) -> None:
        try:
            self._player.stop()
        except Exception:
            pass


@dataclass
class ParameterDisplayController:
    widget: QtWidgets.QWidget
    state: AppState

    def init(self) -> None:
        self.widget.editWavDir.setText(self.state.PD_wavdir)
        try:
            if not self.state.PD_matdir or self.state.PD_matdir == f".{self.state.dirdelimiter}":
                self.state.PD_matdir = self.state.PD_wavdir
        except Exception:
            pass
        self.widget.editMatDir.setText(self.state.PD_matdir)
        self.widget.buttonBrowseWavDir.clicked.connect(self._browse_wav)
        self.widget.buttonBrowseMatDir.clicked.connect(self._browse_mat)
        self.widget.buttonRefresh.clicked.connect(self._refresh)
        self.widget.listWavFiles.itemSelectionChanged.connect(self._on_wav_selected)
        self.widget.listParams.itemSelectionChanged.connect(self._plot)
        self.widget.buttonClose.clicked.connect(self.widget.close)
        self._setup_plot()
        try:
            self.widget.gridLayout.setColumnStretch(0, 1)
            self.widget.gridLayout.setColumnStretch(1, 1)
            self.widget.gridLayout.setColumnStretch(2, 2)
        except Exception:
            pass
        try:
            self.widget.listParams.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        except Exception:
            pass
        try:
            self.widget.editFilterParams.textChanged.connect(self._filter_params)
        except Exception:
            pass
        try:
            self.widget.editFilterFiles.textChanged.connect(self._filter_files)
        except Exception:
            pass
        try:
            self.widget.buttonPlay.clicked.connect(self._play_current)
            self._player = QMediaPlayer(self.widget)
            self._audio = QAudioOutput(self.widget)
            self._player.setAudioOutput(self._audio)
        except Exception:
            pass
        try:
            self.widget.buttonSaveFigure.clicked.connect(self._save_figure_png)
        except Exception:
            pass
        self._refresh()

    def _setup_plot(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
        from matplotlib.figure import Figure
        self._fig = Figure(figsize=(5, 4), dpi=100)
        self._ax_wave = self._fig.add_subplot(2, 1, 1)
        self._ax_params = self._fig.add_subplot(2, 1, 2)
        self._canvas = Canvas(self._fig)
        layout = QtWidgets.QVBoxLayout(self.widget.plotContainer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._pan_active = False
        self._press_x = None
        self._xlim_wave = None
        self._xlim_params = None
        self._visible_xlim = None
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    def _browse_wav(self) -> None:
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择WAV目录", self.state.PD_wavdir)
        if new_dir:
            self.state.PD_wavdir = new_dir
            self.widget.editWavDir.setText(new_dir)
            try:
                if int(self.state.linkmatdir) == 1:
                    self.state.PD_matdir = new_dir
                    self.widget.editMatDir.setText(new_dir)
                if int(self.state.linkwavdir) == 1:
                    self.state.wavdir = new_dir
                    self.state.matdir = self.state.PD_matdir
            except Exception:
                pass
            self._refresh()

    def _browse_mat(self) -> None:
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择CSV目录", self.state.PD_matdir)
        if new_dir:
            self.state.PD_matdir = new_dir
            self.widget.editMatDir.setText(new_dir)
            self._refresh()

    def _refresh(self) -> None:
        self.widget.listWavFiles.clear()
        self._all_wav_files = []
        p = Path(self.widget.editWavDir.text())
        if p.exists():
            for wav in p.rglob("*.wav") if self.state.recursedir else p.glob("*.wav"):
                self._all_wav_files.append(str(wav.name))
        
        # Apply filter
        try:
            self._filter_files(self.widget.editFilterFiles.text())
        except Exception:
            # Fallback if widget not ready or attribute missing
            for n in self._all_wav_files:
                self.widget.listWavFiles.addItem(n)

        self.widget.listParams.clear()
        self._clear_plot()

    def _filter_files(self, text: str) -> None:
        try:
            base = getattr(self, "_all_wav_files", [])
            q = str(text).strip().lower()
            items = base if q == "" else [t for t in base if q in t.lower()]
            self.widget.listWavFiles.clear()
            for n in items:
                self.widget.listWavFiles.addItem(n)
        except Exception:
            pass

    def _on_wav_selected(self) -> None:
        items = self.widget.listWavFiles.selectedItems()
        if not items:
            return
        name = items[0].text()
        wav_path = Path(self.widget.editWavDir.text()) / name
        self._current_wav_path = wav_path
        mat_path = Path(self.widget.editMatDir.text()) / f"{Path(name).stem}.csv"
        from .services.praat_service import read_wav_mono_float
        try:
            fs, y = read_wav_mono_float(wav_path)
        except Exception:
            fs, y = 16000, np.zeros(1000)
        self._current_wave_fs = fs
        self._current_wave_y = y
        t_wave = np.arange(len(y), dtype=float) * (1000.0 / float(fs))
        self._ax_wave.clear()
        self._ax_wave.plot(t_wave, y, color="black")
        self._ax_wave.set_ylabel("Amplitude")
        self._ax_wave.set_xlabel("") # Remove xlabel as requested
        self._ax_wave.set_xlim(0, t_wave[-1] if len(t_wave) else 1)
        self._visible_xlim = (0.0, float(t_wave[-1]) if len(t_wave) else 1.0)

        data = load_csv_any(mat_path)
        self._data = data
        frameshift = float(data.get("frameshift", self.state.frameshift))
        try:
            max_len = max((np.array(v).squeeze().shape[0] for v in data.values() if isinstance(v, np.ndarray)), default=1)
        except Exception:
            max_len = 1
        self._t = np.arange(max_len, dtype=float) * frameshift
        param_names = self._available_params(data)
        
        # Check for TextGrid
        tg_path = self._current_wav_path.with_suffix(".TextGrid")
        has_textgrid = tg_path.exists()
        if has_textgrid:
            from .utils.textgrid_parser import parse_textgrid
            tg = parse_textgrid(tg_path)
            if tg:
                for tier in tg.tiers:
                    # Display name: TextGrid - TierName
                    title = f"TextGrid - {tier.name}"
                    param_names.append(title)
                    # Internal key: TG:TierName
                    self._param_rev_map[title] = f"TG:{tier.name}"

        self._all_param_titles = list(param_names)
        self.widget.listParams.clear()
        for n in self._all_param_titles:
            item = QtWidgets.QListWidgetItem(n)
            self.widget.listParams.addItem(item)
            # Default hide TextGrid (do not select)
        
        
        self._plot()

    def _filter_params(self, text: str) -> None:
        try:
            base = getattr(self, "_all_param_titles", [])
            q = str(text).strip().lower()
            items = base if q == "" else [t for t in base if q in t.lower()]
            self.widget.listParams.clear()
            for n in items:
                self.widget.listParams.addItem(n)
        except Exception:
            pass

    def _available_params(self, data: dict) -> list[str]:
        mapping = {
                "pF0": "F0 - Praat",
                "rF0": "F0 - REAPER",
                "sF0": "F0 - Snack",
                "shrF0": "F0 - SHR",
            "pF1": "F1 - Praat",
            "pF2": "F2 - Praat",
            "pF3": "F3 - Praat",
            "pF4": "F4 - Praat",
            "pB1": "B1 - Praat",
            "pB2": "B2 - Praat",
            "pB3": "B3 - Praat",
            "pB4": "B4 - Praat",
            "H1_pF0": "H1 (pF0)",
            "H1_rF0": "H1 (rF0)",
            "H2_pF0": "H2 (pF0)",
            "H2_rF0": "H2 (rF0)",
            "H4_pF0": "H4 (pF0)",
            "H4_rF0": "H4 (rF0)",
            "A1_pF0": "A1 (pF0)",
            "A1_rF0": "A1 (rF0)",
            "A2_pF0": "A2 (pF0)",
            "A2_rF0": "A2 (rF0)",
            "A3_pF0": "A3 (pF0)",
            "A3_rF0": "A3 (rF0)",
            "H1H2u_pF0": "H1-H2 (pF0)",
            "H1H2u_rF0": "H1-H2 (rF0)",
            "H2H4u_pF0": "H2-H4 (pF0)",
            "H2H4u_rF0": "H2-H4 (rF0)",
            "H1A1u_pF0": "H1-A1 (pF0)",
            "H1A1u_rF0": "H1-A1 (rF0)",
            "H1A2u_pF0": "H1-A2 (pF0)",
            "H1A2u_rF0": "H1-A2 (rF0)",
            "H1A3u_pF0": "H1-A3 (pF0)",
            "H1A3u_rF0": "H1-A3 (rF0)",
            "H1A1c_pF0": "H1*-A1* (pF0)",
            "H1A1c_rF0": "H1*-A1* (rF0)",
            "H1A2c_pF0": "H1*-A2* (pF0)",
            "H1A2c_rF0": "H1*-A2* (rF0)",
            "H1A3c_pF0": "H1*-A3* (pF0)",
            "H1A3c_rF0": "H1*-A3* (rF0)",
            "H1H2c_pF0": "H1*-H2* (pF0)",
            "H1H2c_rF0": "H1*-H2* (rF0)",
            "H2H4c_pF0": "H2*-H4* (pF0)",
            "H2H4c_rF0": "H2*-H4* (rF0)",
            "H2K_pF0": "2K (pF0)",
            "H2K_rF0": "2K (rF0)",
            "H5K_pF0": "5K (pF0)",
            "H5K_rF0": "5K (rF0)",
            "H42Ku_pF0": "H4-2K (pF0)",
            "H42Ku_rF0": "H4-2K (rF0)",
            "H2KH5Ku_pF0": "2K-5K (pF0)",
            "H2KH5Ku_rF0": "2K-5K (rF0)",
            "H42Kc_pF0": "H4*-2K* (pF0)",
            "H42Kc_rF0": "H4*-2K* (rF0)",
            "H2KH5Kc_pF0": "2K*-5K* (pF0)",
            "H2KH5Kc_rF0": "2K*-5K* (rF0)",
            "CPP_pF0": "CPP (pF0)",
            "CPP_rF0": "CPP (rF0)",
            "Energy": "Energy",
            "HNR05_pF0": "HNR05 (pF0)",
            "HNR15_pF0": "HNR15 (pF0)",
            "HNR25_pF0": "HNR25 (pF0)",
            "HNR35_pF0": "HNR35 (pF0)",
            "HNR05_rF0": "HNR05 (rF0)",
            "HNR15_rF0": "HNR15 (rF0)",
            "HNR25_rF0": "HNR25 (rF0)",
            "HNR35_rF0": "HNR35 (rF0)",
                "SHR_pF0": "SHR (pF0)",
                "SHR_rF0": "SHR (rF0)",
                "SHR_shrF0": "SHR (shrF0)",
            "SpectralSlope_pF0": "频谱倾斜 (pF0)",
            "SpectralSlope_rF0": "频谱倾斜 (rF0)",
            "Jitter": "基频抖动",
            "Shimmer": "振幅抖动",
        }
        out = []
        rev = {}
        for k, title in mapping.items():
            v = data.get(k)
            if isinstance(v, np.ndarray) and v.size > 0:
                out.append(title)
                rev[title] = k
        self._param_rev_map = rev
        return out

    def _plot(self) -> None:
        if not hasattr(self, "_data"):
            return
        
        # Clear both axes to ensure no leftover TextGrid lines/text on waveform
        self._ax_params.clear()
        self._ax_wave.clear()
        
        # Re-plot waveform
        if hasattr(self, "_current_wave_y") and hasattr(self, "_current_wave_fs"):
            y = self._current_wave_y
            fs = self._current_wave_fs
            t_wave = np.arange(len(y), dtype=float) * (1000.0 / float(fs))
            self._ax_wave.plot(t_wave, y, color="black")
            self._ax_wave.set_ylabel("Amplitude")
            self._ax_wave.set_xlabel("")
            # Restore xlim if available
            if self._visible_xlim:
                 self._ax_wave.set_xlim(*self._visible_xlim)
            else:
                 self._ax_wave.set_xlim(0, t_wave[-1] if len(t_wave) else 1)

        selected = [i.text() for i in self.widget.listParams.selectedItems()]
        legend = []
        max_t = 0.0
        
        tg_tiers_to_plot = []

        for title in selected:
            key = getattr(self, "_param_rev_map", {}).get(title)
            if not key:
                continue
            
            if key.startswith("TG:"):
                tier_name = key[3:]
                tg_tiers_to_plot.append(tier_name)
                continue

            v = self._data.get(key)
            if isinstance(v, np.ndarray):
                arr = np.array(v).squeeze().astype(float)
                fs_ms = float(self._data.get("frameshift", self.state.frameshift))
                if key == "rF0" and isinstance(self._data.get("rTimes"), np.ndarray):
                    t = np.array(self._data.get("rTimes"), dtype=float) * 1000.0
                    if t.shape[0] != arr.shape[0]:
                        t = np.arange(arr.shape[0], dtype=float) * fs_ms
                else:
                    t = np.arange(arr.shape[0], dtype=float) * fs_ms
                self._ax_params.plot(t, arr)
                legend.append(key)
                if t.size:
                    max_t = max(max_t, float(t[-1]))
        
        if legend:
            self._ax_params.legend(legend)
        
        # Plot TextGrid tiers (only selected ones)
        if tg_tiers_to_plot:
            tg_path = self._current_wav_path.with_suffix(".TextGrid")
            if tg_path.exists():
                from .utils.textgrid_parser import parse_textgrid
                tg = parse_textgrid(tg_path)
                if tg:
                    for idx, tier_name in enumerate(tg_tiers_to_plot):
                        target_tier = next((t for t in tg.tiers if t.name == tier_name), None)
                        if not target_tier:
                            continue
                        
                        # Stack text: 1.02, 1.15, 1.28...
                        text_y = 1.02 + (idx * 0.12)
                        
                        for interval in target_tier.intervals:
                            if not interval.text.strip():
                                continue
                            # Vertical lines (ms) on BOTH axes
                            xmin_ms = interval.xmin * 1000.0
                            xmax_ms = interval.xmax * 1000.0
                            
                            for ax in [self._ax_wave, self._ax_params]:
                                ax.axvline(xmin_ms, color='red', linestyle='--', alpha=0.5)
                                ax.axvline(xmax_ms, color='red', linestyle='--', alpha=0.5)
                            
                            # Text on Waveform axis
                            mid_ms = (xmin_ms + xmax_ms) / 2.0
                            self._ax_wave.text(mid_ms, text_y, interval.text,
                                               transform=self._ax_wave.get_xaxis_transform(),
                                               ha='center', va='bottom', color='black', fontsize=10,
                                               fontfamily=['Doulos SIL', 'Times New Roman', 'SimSun', 'SimHei', 'sans-serif'])

        self._ax_params.set_xlabel("Time (ms)")
        self._ax_params.set_ylabel("Value")
        if self._visible_xlim is not None:
            self._ax_wave.set_xlim(*self._visible_xlim)
            self._ax_params.set_xlim(*self._visible_xlim)
        else:
            wl = self._ax_wave.get_xlim()
            end = wl[1]
            if max_t > 0:
                end = max(end, max_t)
            self._ax_wave.set_xlim(0, end)
            self._ax_params.set_xlim(0, end)
        self._canvas.draw_idle()

    def _clear_plot(self) -> None:
        self._ax_wave.clear()
        self._ax_params.clear()
        self._canvas.draw_idle()

    def _save_figure_png(self) -> None:
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self.widget, "保存图片", "plot.png", "PNG Files (*.png)")
            if not path:
                return
            self._fig.savefig(path, dpi=300)
            QtWidgets.QMessageBox.information(self.widget, "保存图片", f"已保存到: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "保存图片", f"失败: {e}")


    def _apply_param_set(self) -> None:
        try:
            name = self.widget.comboParamSet.currentText().strip()
        except Exception:
            return
        keys = self.state.PD_param_combos.get(name)
        if not keys:
            return
        # 遍历列表并选择匹配项
        self.widget.listParams.clearSelection()
        for i in range(self.widget.listParams.count()):
            item = self.widget.listParams.item(i)
            key = item.text().split("(")[-1].strip(")")
            if key in keys:
                item.setSelected(True)
        self._plot()

    def _compute_energy(self, y: np.ndarray, fs: int, F0: np.ndarray) -> np.ndarray:
        N_periods = self.state.Nperiods_EC
        sampleshift = int(round(fs / 1000.0 * self.state.frameshift))
        E = np.full(F0.shape[0], np.nan, dtype=float)
        for k in range(F0.shape[0]):
            ks = int(round((k + 1) * sampleshift))
            f0c = float(F0[k])
            if np.isnan(f0c) or f0c <= 0:
                continue
            N0 = fs / f0c
            ystart = int(round(ks - N_periods / 2.0 * N0))
            yend = int(round(ks + N_periods / 2.0 * N0)) - 1
            if ystart <= 0 or yend >= y.size:
                continue
            seg = y[ystart:yend]
            E[k] = float(np.sum(seg.astype(float) ** 2))
        return E

    def _on_scroll(self, event) -> None:
        try:
            ax = event.inaxes
            if ax not in [self._ax_wave, self._ax_params]:
                return
            cur_xlim = ax.get_xlim()
            x = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2.0
            scale_factor = 0.9 if event.button == "up" else 1.1
            new_left = x + (cur_xlim[0] - x) * scale_factor
            new_right = x + (cur_xlim[1] - x) * scale_factor
            self._ax_wave.set_xlim(new_left, new_right)
            self._ax_params.set_xlim(new_left, new_right)
            self._visible_xlim = (float(new_left), float(new_right))
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_press(self, event) -> None:
        if event.button != 1:
            return
        if event.inaxes not in [self._ax_wave, self._ax_params]:
            return
        self._pan_active = True
        self._press_x = event.xdata
        self._xlim_wave = self._ax_wave.get_xlim()
        self._xlim_params = self._ax_params.get_xlim()

    def _on_motion(self, event) -> None:
        if not self._pan_active or self._press_x is None:
            return
        if event.xdata is None:
            return
        dx = event.xdata - self._press_x
        wl0, wl1 = self._xlim_wave
        pl0, pl1 = self._xlim_params
        self._ax_wave.set_xlim(wl0 - dx, wl1 - dx)
        self._ax_params.set_xlim(pl0 - dx, pl1 - dx)
        self._visible_xlim = (float(wl0 - dx), float(wl1 - dx))
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        self._pan_active = False
        self._press_x = None
        self._xlim_wave = None
        self._xlim_params = None

    def _play_current(self) -> None:
        try:
            from PyQt6.QtCore import QUrl
            import tempfile
            from scipy.io import wavfile as wavwrite
            if not hasattr(self, "_current_wav_path"):
                return
            # 截取当前可见范围播放
            if not hasattr(self, "_current_wave_fs") or not hasattr(self, "_current_wave_y"):
                url = QUrl.fromLocalFile(str(self._current_wav_path))
                self._player.setSource(url)
                self._audio.setVolume(0.9)
                self._player.play()
                return
            fs = int(self._current_wave_fs)
            y = np.array(self._current_wave_y, dtype=float)
            xlim = self._ax_wave.get_xlim()
            # 毫秒到样本索引
            s = int(max(0, min(len(y) - 1, round(xlim[0] / 1000.0 * fs))))
            e = int(max(s + 1, min(len(y), round(xlim[1] / 1000.0 * fs))))
            seg = y[s:e]
            # 写入临时 WAV（16-bit PCM）
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            wavwrite.write(tmp.name, fs, np.int16(np.clip(seg, -1.0, 1.0) * 32767))
            url = QUrl.fromLocalFile(tmp.name)
            self._player.setSource(url)
            self._audio.setVolume(0.9)
            self._player.play()
        except Exception:
            pass


@dataclass
class OutputTextController:
    widget: QtWidgets.QWidget
    state: AppState

    def init(self) -> None:
        try:
            self.widget.edit_matdir.setText(self.state.OT_matdir)
            self.widget.edit_outputdir.setText(self.state.OT_outputdir)
        except Exception:
            pass
        try:
            self.widget.button_matdir_browse.clicked.connect(self._browse_matdir)
            self.widget.button_outputdir_browse.clicked.connect(self._browse_outputdir)
            self.widget.button_start.clicked.connect(self._start_export)
        except Exception:
            pass
        try:
            self.widget.buttonClose.clicked.connect(self.widget.close)
        except Exception:
            pass
        self._populate_paramlist()
        self._refresh_files()

    def _browse_matdir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择CSV目录", self.state.OT_matdir)
        if d:
            self.state.OT_matdir = d
            self.widget.edit_matdir.setText(d)
            self._refresh_files()

    def _browse_outputdir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择输出目录", self.state.OT_outputdir)
        if d:
            self.state.OT_outputdir = d
            self.widget.edit_outputdir.setText(d)

    def _standard_param_titles(self) -> dict[str, str]:
        return {
            "pF0": "F0 - Praat",
            "rF0": "F0 - REAPER",
            "sF0": "F0 - Snack",
            "shrF0": "F0 - SHR",
            "pF1": "F1 - Praat",
            "pF2": "F2 - Praat",
            "pF3": "F3 - Praat",
            "pF4": "F4 - Praat",
            "pB1": "B1 - Praat",
            "pB2": "B2 - Praat",
            "pB3": "B3 - Praat",
            "pB4": "B4 - Praat",
            "H1_pF0": "H1 (pF0)",
            "H1_rF0": "H1 (rF0)",
            "H2_pF0": "H2 (pF0)",
            "H2_rF0": "H2 (rF0)",
            "H4_pF0": "H4 (pF0)",
            "H4_rF0": "H4 (rF0)",
            "A1_pF0": "A1 (pF0)",
            "A1_rF0": "A1 (rF0)",
            "A2_pF0": "A2 (pF0)",
            "A2_rF0": "A2 (rF0)",
            "A3_pF0": "A3 (pF0)",
            "A3_rF0": "A3 (rF0)",
            "H1H2u_pF0": "H1-H2 (pF0)",
            "H1H2u_rF0": "H1-H2 (rF0)",
            "H2H4u_pF0": "H2-H4 (pF0)",
            "H2H4u_rF0": "H2-H4 (rF0)",
            "H1A1u_pF0": "H1-A1 (pF0)",
            "H1A1u_rF0": "H1-A1 (rF0)",
            "H1A2u_pF0": "H1-A2 (pF0)",
            "H1A2u_rF0": "H1-A2 (rF0)",
            "H1A3u_pF0": "H1-A3 (pF0)",
            "H1A3u_rF0": "H1-A3 (rF0)",
            "H1A1c_pF0": "H1*-A1* (pF0)",
            "H1A1c_rF0": "H1*-A1* (rF0)",
            "H1A2c_pF0": "H1*-A2* (pF0)",
            "H1A2c_rF0": "H1*-A2* (rF0)",
            "H1A3c_pF0": "H1*-A3* (pF0)",
            "H1A3c_rF0": "H1*-A3* (rF0)",
            "H1H2c_pF0": "H1*-H2* (pF0)",
            "H1H2c_rF0": "H1*-H2* (rF0)",
            "H2H4c_pF0": "H2*-H4* (pF0)",
            "H2H4c_rF0": "H2*-H4* (rF0)",
            "H2K_pF0": "2K (pF0)",
            "H2K_rF0": "2K (rF0)",
            "H5K_pF0": "5K (pF0)",
            "H5K_rF0": "5K (rF0)",
            "H42Ku_pF0": "H4-2K (pF0)",
            "H42Ku_rF0": "H4-2K (rF0)",
            "H2KH5Ku_pF0": "2K-5K (pF0)",
            "H2KH5Ku_rF0": "2K-5K (rF0)",
            "H42Kc_pF0": "H4*-2K* (pF0)",
            "H42Kc_rF0": "H4*-2K* (rF0)",
            "H2KH5Kc_pF0": "2K*-5K* (pF0)",
            "H2KH5Kc_rF0": "2K*-5K* (rF0)",
            "CPP": "CPP",
            "CPP_pF0": "CPP (pF0)",
            "CPP_rF0": "CPP (rF0)",
            "Energy": "Energy",
            "HNR05_pF0": "HNR05 (pF0)",
            "HNR15_pF0": "HNR15 (pF0)",
            "HNR25_pF0": "HNR25 (pF0)",
            "HNR35_pF0": "HNR35 (pF0)",
            "HNR05_rF0": "HNR05 (rF0)",
            "HNR15_rF0": "HNR15 (rF0)",
            "HNR25_rF0": "HNR25 (rF0)",
            "HNR35_rF0": "HNR35 (rF0)",
            "SHR_pF0": "SHR (pF0)",
            "SHR_rF0": "SHR (rF0)",
            "SpectralSlope_pF0": "频谱倾斜 (pF0)",
            "SpectralSlope_rF0": "频谱倾斜 (rF0)",
            "Jitter": "基频抖动",
            "Shimmer": "振幅抖动",
        }

    def _populate_paramlist(self) -> None:
        try:
            self.widget.listbox_Parameters_paramlist.clear()
            for title in self._standard_param_titles().values():
                self.widget.listbox_Parameters_paramlist.addItem(title)
        except Exception:
            pass

    def _refresh_files(self) -> None:
        try:
            self.widget.listMatFiles.clear()
            p = Path(self.widget.edit_matdir.text())
            if not p.exists():
                return
            items = p.rglob("*.csv") if self.state.recursedir else p.glob("*.csv")
            for f in items:
                self.widget.listMatFiles.addItem(str(Path(f).name))
        except Exception:
            pass

    def _selected_params(self) -> list[str]:
        items = [i.text() for i in self.widget.listbox_Parameters_paramlist.selectedItems()]
        keys = []
        mapping = self._standard_param_titles()
        rev = {v: k for k, v in mapping.items()}
        for t in items:
            k = rev.get(t)
            if k:
                keys.append(k)
        return keys

    def _start_export(self) -> None:
        mat_dir = Path(self.widget.edit_matdir.text())
        out_dir = Path(self.widget.edit_outputdir.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        params = self._selected_params()
        if not params:
            QtWidgets.QMessageBox.warning(self.widget, "输出到文本", "未选择参数")
            return
        files = [self.widget.listMatFiles.item(i).text() for i in range(self.widget.listMatFiles.count())]
        if not files:
            QtWidgets.QMessageBox.warning(self.widget, "输出到文本", "没有 csv 文件")
            return
        progress = QtWidgets.QProgressDialog("导出中...", "取消", 0, len(files), self.widget)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        self.widget.listbox_messages.clear()
        for idx, name in enumerate(files, start=1):
            progress.setValue(idx - 1)
            QtWidgets.QApplication.processEvents()
            mat_path = mat_dir / name
            data = load_csv_any(mat_path)
            frameshift = float(data.get("frameshift", self.state.frameshift))
            cols = {k: np.array(data.get(k)).squeeze().astype(float) if isinstance(data.get(k), np.ndarray) else np.array([], dtype=float) for k in params}
            max_len = max([v.shape[0] for v in cols.values()]) if cols else 0
            if max_len == 0:
                self.widget.listbox_messages.addItem(f"{name}: 无数据")
                continue
            T = np.arange(max_len, dtype=float) * frameshift
            out_rows = []
            header = ["Time_ms"] + params
            for i in range(max_len):
                row = [T[i]]
                for k in params:
                    v = cols[k]
                    row.append(float(v[i]) if i < v.shape[0] else np.nan)
                out_rows.append(row)
            outfile = out_dir / f"{Path(name).stem}_params.txt"
            try:
                with outfile.open("w", encoding="utf-8") as w:
                    w.write("\t".join(header) + "\n")
                    for r in out_rows:
                        w.write("\t".join([str(x) for x in r]) + "\n")
                self.widget.listbox_messages.addItem(f"{name}: 导出 {outfile.name}")
            except Exception as e:
                self.widget.listbox_messages.addItem(f"{name}: 错误 {e}")
        progress.setValue(len(files))
        QtWidgets.QMessageBox.information(self.widget, "输出到文本", "导出完成")




@dataclass
class ManualDataController:
    widget: QtWidgets.QWidget
    state: AppState

    def init(self) -> None:
        try:
            self.widget.edit_matdir.setText(self.state.MD_matdir)
            self.widget.edit_outputdir.setText(self.state.OT_outputdir)
        except Exception:
            pass
        try:
            self.widget.buttonBrowseMatDir.clicked.connect(self._browse_matdir)
            self.widget.buttonBrowseOutputDir.clicked.connect(self._browse_outputdir)
            self.widget.buttonAddRow.clicked.connect(self._add_row)
            self.widget.buttonRemoveRow.clicked.connect(self._remove_row)
            self.widget.buttonSave.clicked.connect(self._save)
            self.widget.listMatFiles.itemSelectionChanged.connect(self._load_selected_manual)
            try:
                self.widget.spinFreezeCols.valueChanged.connect(lambda _=None: self._load_selected_manual())
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.widget.buttonClose.clicked.connect(self.widget.close)
        except Exception:
            pass
        try:
            for tbl in [getattr(self.widget, "tableFrozen", None), self.widget.tableManualData]:
                if not tbl:
                    continue
                tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
                tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
                tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
                try:
                    tbl.horizontalHeader().setSectionsClickable(True)
                    tbl.horizontalHeader().sectionClicked.connect(lambda col, t=tbl: self._select_entire_column(t, col))
                except Exception:
                    pass
            self._copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Copy, self.widget)
            self._copy_shortcut.activated.connect(self._copy_selection_to_clipboard)
            try:
                f = getattr(self.widget, "tableFrozen", None)
                if f:
                    f.verticalScrollBar().valueChanged.connect(lambda v: self.widget.tableManualData.verticalScrollBar().setValue(v))
                    self.widget.tableManualData.verticalScrollBar().valueChanged.connect(lambda v: f.verticalScrollBar().setValue(v))
                    f.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            except Exception:
                pass
        except Exception:
            pass
        self._refresh_files()

    def _browse_matdir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择CSV目录", self.state.MD_matdir)
        if d:
            self.state.MD_matdir = d
            self.widget.edit_matdir.setText(d)
            self._refresh_files()

    def _browse_outputdir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self.widget, "选择输出目录", self.state.OT_outputdir)
        if d:
            self.state.OT_outputdir = d
            self.widget.edit_outputdir.setText(d)

    def _refresh_files(self) -> None:
        try:
            self.widget.listMatFiles.clear()
            p = Path(self.widget.edit_matdir.text())
            if not p.exists():
                return
            items = p.rglob("*.csv") if self.state.recursedir else p.glob("*.csv")
            for f in items:
                self.widget.listMatFiles.addItem(str(Path(f).name))
        except Exception:
            pass

    def _add_row(self) -> None:
        r = self.widget.tableManualData.rowCount()
        self.widget.tableManualData.insertRow(r)
        # Default time logic if needed? 
        # Maybe guess from previous row + frameshift?
        if r > 0:
            prev_t_item = self.widget.tableManualData.item(r-1, 0)
            if prev_t_item:
                 try:
                     prev_t = float(prev_t_item.text())
                     # Guess frameshift from row 0 and 1 if possible
                     fs = 1.0 # Default
                     if r > 1:
                         t0 = self.widget.tableManualData.item(0, 0)
                         t1 = self.widget.tableManualData.item(1, 0)
                         if t0 and t1:
                             fs = float(t1.text()) - float(t0.text())
                     self.widget.tableManualData.setItem(r, 0, QtWidgets.QTableWidgetItem(str(prev_t + fs)))
                 except:
                     pass

    def _remove_row(self) -> None:
        r = self.widget.tableManualData.currentRow()
        if r >= 0:
            self.widget.tableManualData.removeRow(r)

    def _load_selected_manual(self) -> None:
        items = self.widget.listMatFiles.selectedItems()
        if not items:
            return
        name = items[0].text()
        mat_dir = Path(self.widget.edit_matdir.text())
        data = load_csv_any(mat_dir / name)
        
        self.widget.tableManualData.setRowCount(0)
        self.widget.tableManualData.setColumnCount(0)

        # 1. Prepare Columns (Keys)
        # 保持 CSV 原始列顺序
        try:
            keys = list(data.keys())
        except Exception:
            keys = sorted(list(data.keys()))
        priority_keys = ["time_ms", "time", "Time", "timestamp", "filename", "file", "File"]
        
        # Filter keys to be displayed as columns
        # If it's legacy manual data, handle separately?
        # The user wants "CSV original rows/cols". 
        # If it's legacy Long Format, we probably should convert to Wide for display as requested?
        # But wait, load_csv_any handles legacy formats and returns a dict of arrays.
        # So we effectively have "Wide" data in memory (Dict[str, Array]).
        
        # Let's just display the Dict as a Table.
        
        # Determine Row Count (Time dimension)
        row_count = 0
        
        # Find the time array to determine row count
        time_col = None
        for k in priority_keys:
            if k in data:
                v = data[k]
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    row_count = len(v)
                    time_col = k
                    break
        
        if row_count == 0:
            # Maybe scalar or empty?
            # Check max length of any array
            for v in data.values():
                if isinstance(v, (list, np.ndarray)):
                    row_count = max(row_count, len(v))
            if row_count == 0 and data:
                 row_count = 1 # Scalar case

        self.widget.tableManualData.setRowCount(row_count)
        
        # Determine Column Order (preserve original order)
        display_keys = [k for k in keys if not isinstance(data.get(k), dict)]
        
        freeze_count = 0
        try:
            freeze_count = int(self.widget.spinFreezeCols.value())
        except Exception:
            freeze_count = 0
        freeze_count = max(0, min(freeze_count, len(display_keys)))

        left_keys = display_keys[:freeze_count]
        right_keys = display_keys[freeze_count:]

        f_tbl = getattr(self.widget, "tableFrozen", None)
        if f_tbl:
            f_tbl.clear()
            f_tbl.setRowCount(row_count)
            f_tbl.setColumnCount(len(left_keys))
            if left_keys:
                f_tbl.setHorizontalHeaderLabels(left_keys)
            f_tbl.setVisible(bool(left_keys))
        self.widget.tableManualData.clear()
        self.widget.tableManualData.setRowCount(row_count)
        self.widget.tableManualData.setColumnCount(len(right_keys))
        if right_keys:
            self.widget.tableManualData.setHorizontalHeaderLabels(right_keys)
        
        # Populate Data (frozen and main)
        if f_tbl and left_keys:
            for col_idx, k in enumerate(left_keys):
                v = data[k]
                if isinstance(v, (list, np.ndarray)):
                    arr = v
                    for r_idx in range(row_count):
                        val = ""
                        if r_idx < len(arr):
                            val = str(arr[r_idx])
                        f_tbl.setItem(r_idx, col_idx, QtWidgets.QTableWidgetItem(val))
                else:
                    val = str(v)
                    for r_idx in range(row_count):
                        f_tbl.setItem(r_idx, col_idx, QtWidgets.QTableWidgetItem(val))
        for col_idx, k in enumerate(right_keys):
            v = data[k]
            if isinstance(v, (list, np.ndarray)):
                arr = v
                for r_idx in range(row_count):
                    val = ""
                    if r_idx < len(arr):
                        val = str(arr[r_idx])
                    self.widget.tableManualData.setItem(r_idx, col_idx, QtWidgets.QTableWidgetItem(val))
            else:
                val = str(v)
                for r_idx in range(row_count):
                    self.widget.tableManualData.setItem(r_idx, col_idx, QtWidgets.QTableWidgetItem(val))

        # "Freeze" Time Column by setting vertical header?
        # If we found a time column, use it for vertical header labels?
        # User asked to "freeze row/col". 
        # Setting vertical header labels to Time makes it frozen on the left.
        if time_col:
            labels = []
            v = data[time_col]
            if isinstance(v, (list, np.ndarray)):
                for x in v:
                    labels.append(str(x))
            else:
                labels = [str(v)] * row_count
            try:
                f_tbl = getattr(self.widget, "tableFrozen", None)
                if f_tbl and f_tbl.columnCount() > 0:
                    f_tbl.setVerticalHeaderLabels(labels)
            except Exception:
                pass
            self.widget.tableManualData.setVerticalHeaderLabels(labels)

    def _save(self) -> None:
        out_dir = Path(self.widget.edit_outputdir.text())
        out_dir.mkdir(parents=True, exist_ok=True)

        f_tbl = getattr(self.widget, "tableFrozen", None)
        freeze_cols = f_tbl.columnCount() if f_tbl else 0
        rows = max(self.widget.tableManualData.rowCount(), f_tbl.rowCount() if f_tbl else 0)
        main_cols = self.widget.tableManualData.columnCount()
        total_cols = freeze_cols + main_cols

        if rows == 0 or total_cols == 0:
            return

        # Collect headers from both tables
        out_data: Dict[str, Any] = {}
        headers: list[str] = []
        for c in range(freeze_cols):
            h_item = f_tbl.horizontalHeaderItem(c) if f_tbl else None
            headers.append(h_item.text() if h_item else f"Col{c}")
        for c in range(main_cols):
            h_item = self.widget.tableManualData.horizontalHeaderItem(c)
            headers.append(h_item.text() if h_item else f"Col{freeze_cols+c}")

        for c, key in enumerate(headers):
            col_vals = []
            for r in range(rows):
                if c < freeze_cols:
                    item = f_tbl.item(r, c) if f_tbl else None
                else:
                    mc = c - freeze_cols
                    item = self.widget.tableManualData.item(r, mc)
                txt = item.text() if item else ""
                col_vals.append(txt)

            try:
                floats = []
                is_num = True
                for x in col_vals:
                    if x == "" or x.lower() == "nan":
                        floats.append(np.nan)
                    else:
                        try:
                            floats.append(float(x))
                        except ValueError:
                            is_num = False
                            break
                if is_num:
                    out_data[key] = np.array(floats, dtype=float)
                else:
                    out_data[key] = np.array(col_vals, dtype=object)
            except:
                out_data[key] = np.array(col_vals, dtype=object)

        sel = self.widget.listMatFiles.selectedItems()
        if not sel:
            QtWidgets.QMessageBox.warning(self.widget, "保存", "请先在左侧选择一个 CSV 文件")
            return
        name = sel[0].text()
        base = Path(name).stem
        csv_path = out_dir / f"{base}.csv"
        
        if csv_path.exists():
            ret = QtWidgets.QMessageBox.question(
                self.widget,
                "保存",
                f"{csv_path.name} 已存在，是否覆盖？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                csv_path = out_dir / f"{base}_manual.csv"
            
        try:
            save_csv(csv_path, out_data)
            QtWidgets.QMessageBox.information(self.widget, "手动数据", "保存完成")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.widget, "手动数据", f"保存失败: {e}")

    def _copy_selection_to_clipboard(self) -> None:
        try:
            f_tbl = getattr(self.widget, "tableFrozen", None)
            sel_main = self.widget.tableManualData.selectedIndexes()
            sel_frozen = f_tbl.selectedIndexes() if f_tbl else []
            if not sel_main and not sel_frozen:
                return
            freeze_cols = f_tbl.columnCount() if f_tbl else 0
            total_cols = freeze_cols + self.widget.tableManualData.columnCount()
            by_row: dict[int, dict[int, str]] = {}
            for idx in sel_frozen:
                r = idx.row()
                c = idx.column()
                item = f_tbl.item(r, c)
                by_row.setdefault(r, {})[c] = item.text() if item else ""
            for idx in sel_main:
                r = idx.row()
                c = freeze_cols + idx.column()
                item = self.widget.tableManualData.item(r, idx.column())
                by_row.setdefault(r, {})[c] = item.text() if item else ""
            lines: list[str] = []
            for r in sorted(by_row.keys()):
                cols = by_row[r]
                line = "\t".join([cols.get(c, "") for c in range(total_cols)])
                lines.append(line)
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        except Exception:
            pass

    def _select_entire_column(self, tbl: QtWidgets.QTableWidget, col: int) -> None:
        try:
            rng = QtWidgets.QTableWidgetSelectionRange(0, col, max(0, tbl.rowCount()-1), col)
            tbl.clearSelection()
            tbl.setRangeSelected(rng, True)
        except Exception:
            pass


@dataclass
class AboutController:
    widget: QtWidgets.QWidget

    def init(self) -> None:
        self.widget.buttonOK.clicked.connect(self.widget.close)
        
        # Beautify Interface
        # Dark theme
        self.widget.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        
        # ASCII Art Title
        ascii_title = ASCII_TITLE
        try:
            self.widget.labelTitle.setText(ascii_title)
            font = QtGui.QFont("Consolas", 8)
            font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
            self.widget.labelTitle.setFont(font)
        except Exception:
            pass
class PEWorker(QThread):
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)
    finished_sig = pyqtSignal()

    def __init__(self, items: list[str], input_dir: Path, output_dir: Path, state: AppState, tier_name: Optional[str] = None):
        super().__init__()
        self.items = items
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.state = state
        self.tier_name = tier_name

    def run(self) -> None:
        for i, name in enumerate(self.items, start=1):
            if self.isInterruptionRequested():
                break
            wav_path = self.input_dir / name
            try:
                base = (i - 1) * 20
                self.progress_sig.emit(base + 1, name)
                result = compute_praat_f0_formants(
                wav_path=wav_path,
                frameshift_ms=self.state.frameshift,
                min_f0=self.state.F0Praatmin,
                max_f0=self.state.F0Praatmax,
                method=self.state.F0Praatmethod,
                )
                try:
                    from .services.praat_service import compute_reaper_f0
                    r_res = compute_reaper_f0(
                        wav_path=wav_path,
                        frame_interval_sec=float(self.state.F0ReaperFrameIntervalSec),
                        min_f0=float(self.state.F0ReaperMinF0),
                        max_f0=float(self.state.F0ReaperMaxF0),
                        hilbert=bool(self.state.F0ReaperHilbert),
                        no_highpass=bool(self.state.F0ReaperNoHighpass),
                        reaper_bin=self.state.F0ReaperBin or None,
                    )
                    rF0 = np.array(r_res.get("rF0", []), dtype=float)
                    rTimes = np.array(r_res.get("rTimes", []), dtype=float)
                    rF0Raw = np.array(r_res.get("rF0Raw", []), dtype=float)
                    rVoiced = np.array(r_res.get("rVoiced", []), dtype=int)
                    result["rF0"] = rF0
                    if rTimes.size > 0:
                        result["rTimes"] = rTimes
                    if rF0Raw.size > 0:
                        result["rF0Raw"] = rF0Raw
                    if rVoiced.size > 0:
                        result["rVoiced"] = rVoiced
                except Exception as e:
                    print("REAPER F0 计算失败:", e)
                
                self.progress_sig.emit(base + 2, name)
                from .services.praat_service import read_wav_mono_float
                fs, y = read_wav_mono_float(wav_path)
                try:
                    from .services.praat_service import compute_shrp_f0
                    shrp_res = compute_shrp_f0(
                        wav_path=wav_path,
                        frameshift_ms=self.state.frameshift,
                        min_f0=self.state.F0Praatmin,
                        max_f0=self.state.F0Praatmax,
                        shr_threshold=float(self.state.SHRThreshold),
                    )
                    shrF0 = np.array(shrp_res.get("shrF0", []), dtype=float)
                    if shrF0.size > 0:
                        result["shrF0"] = shrF0
                        result["SHR_shrF0"] = np.array(shrp_res.get("SHR", []), dtype=float)
                except Exception:
                    pass
                try:
                    if ("rF0" in result):
                        duration_sec = float(y.size) / float(fs)
                        tgt_len = max(1, int(round(duration_sec / (float(self.state.frameshift) / 1000.0))))
                        rF0 = np.array(result.get("rF0", []), dtype=float)
                        rTimes = np.array(result.get("rTimes", []), dtype=float)
                        if tgt_len > 0 and rF0.size > 0:
                            if rTimes.size == rF0.size and rTimes.size > 1:
                                target_t = np.arange(1, tgt_len + 1, dtype=float) * (float(self.state.frameshift) / 1000.0)
                                src = np.nan_to_num(rF0, nan=-1.0)
                                from .services.praat_service import round_half_away_from_zero
                                times_sec = round_half_away_from_zero(rTimes * 1000.0).astype(float) / 1000.0
                                rF0i = np.interp(target_t, times_sec, src, left=-1.0, right=-1.0)
                            else:
                                xp = np.linspace(0.0, 1.0, num=rF0.size)
                                x = np.linspace(0.0, 1.0, num=tgt_len)
                                rF0i = np.interp(x, xp, np.nan_to_num(rF0, nan=-1.0))
                            rF0i[rF0i <= 0.0] = np.nan
                            result["rF0Uniform"] = rF0i
                except Exception:
                    pass
                pF0 = np.array(result.get("pF0", []), dtype=float)
                F1 = np.array(result.get("pF1", []), dtype=float)
                F2 = np.array(result.get("pF2", []), dtype=float)
                F3 = np.array(result.get("pF3", []), dtype=float)
                
                sampleshift = int(round(fs / 1000.0 * self.state.frameshift))
                win_ms = max(1, self.state.voicing_energy_window_ms)
                win = int(round(fs / 1000.0 * win_ms))
                nf = pF0.shape[0]
                Ewin = np.full(nf, 0.0, dtype=float)
                for k in range(nf):
                    s = k * sampleshift
                    e = min(s + win, y.size)
                    seg = y[s:e]
                    if seg.size > 0:
                        Ewin[k] = float(np.sum(seg * seg))
                th = float(np.nanmax(Ewin)) * float(self.state.voicing_energy_threshold_ratio)
                voiced_mask = Ewin > th
                
                if np.any(np.isnan(pF0) & voiced_mask):
                    from .services.praat_service import compute_creaky_f0
                    creaky = compute_creaky_f0(y, fs, self.state.frameshift)
                    n = min(pF0.shape[0], creaky.shape[0])
                    for kk in range(n):
                        if voiced_mask[kk] and (np.isnan(pF0[kk]) or pF0[kk] <= 0):
                            pF0[kk] = creaky[kk]
                    result["pF0"] = pF0
                from .services.praat_service import (
                    compute_harmonics_H1H2H4,
                    compute_A1A2A3,
                    compute_H1A1A2A3_corrected,
                    compute_H1H2_H2H4_corrected,
                    compute_CPP,
                    compute_HNR,
                    compute_SHR,
                    compute_spectral_slope,
                    compute_jitter_shimmer,
                    compute_harmonic_at_fixed_freq,
                    compute_harmonic_at_fixed_freq_with_freq,
                    compute_H42K_corrected,
                    compute_2K5K_corrected,
                )
                rF0u = np.array(result.get("rF0Uniform", []), dtype=float)
                if rF0u.size == 0:
                    rF0u = np.array(result.get("rF0", []), dtype=float)
                def _compute_with(label: str, F0x: np.ndarray) -> None:
                    if F0x.size == 0:
                        return
                    hh = compute_harmonics_H1H2H4(y, fs, self.state.frameshift, F0x, self.state.Nperiods, voiced_mask=voiced_mask)
                    result[f"H1_{label}"] = hh.get("H1")
                    result[f"H2_{label}"] = hh.get("H2")
                    result[f"H4_{label}"] = hh.get("H4")
                    aa = compute_A1A2A3(y, fs, self.state.frameshift, F0x, F1, F2, F3, self.state.Nperiods, voiced_mask=voiced_mask)
                    result[f"A1_{label}"] = aa.get("A1")
                    result[f"A2_{label}"] = aa.get("A2")
                    result[f"A3_{label}"] = aa.get("A3")
                    # 差值改为在平滑/掩蔽之后统一计算
                    H2K, F2K = compute_harmonic_at_fixed_freq_with_freq(y, fs, self.state.frameshift, F0x, self.state.Nperiods, 2000.0, voiced_mask=voiced_mask)
                    result[f"H2K_{label}"] = H2K
                    result[f"F2K_{label}"] = F2K
                    H5K = compute_harmonic_at_fixed_freq(y, fs, self.state.frameshift, F0x, self.state.Nperiods, 5000.0, voiced_mask=voiced_mask)
                    result[f"H5K_{label}"] = H5K
                    if f"H4_{label}" in result and f"H2K_{label}" in result:
                        result[f"H42Ku_{label}"] = result[f"H4_{label}"] - result[f"H2K_{label}"]
                    if f"H2K_{label}" in result and f"H5K_{label}" in result:
                        result[f"H2KH5Ku_{label}"] = result[f"H2K_{label}"] - result[f"H5K_{label}"]
                    result[f"CPP_{label}"] = compute_CPP(y, fs, self.state.frameshift, F0x, self.state.Nperiods_EC, voiced_mask=voiced_mask)
                    hnr = compute_HNR(y, fs, self.state.frameshift, F0x, self.state.Nperiods_EC, voiced_mask=voiced_mask)
                    for k, v in hnr.items():
                        result[f"{k}_{label}"] = v
                    def _shr_cb(frac: float):
                        self.progress_sig.emit(base + 16 + int(max(0, min(1.0, frac)) * 3), name)
                    result[f"SHR_{label}"] = compute_SHR(y, fs, self.state.frameshift, F0x, float(self.state.SHRmin), float(self.state.SHRmax), progress_cb=_shr_cb, voiced_mask=voiced_mask, shr_threshold=float(self.state.SHRThreshold))
                    B1 = np.array(result.get("pB1", []), dtype=float)
                    B2 = np.array(result.get("pB2", []), dtype=float)
                    B3 = np.array(result.get("pB3", []), dtype=float)
                    use_formula = (self.state.BandwidthMethod == "Use formula values")
                    corr = compute_H1A1A2A3_corrected(
                        result[f"H1_{label}"],
                        result[f"A1_{label}"],
                        result[f"A2_{label}"],
                        result[f"A3_{label}"],
                        fs,
                        F0x,
                        F1,
                        F2,
                        F3,
                        None if use_formula else B1,
                        None if use_formula else B2,
                        None if use_formula else B3,
                    )
                    result[f"H1A1c_{label}"] = corr.get("H1A1c")
                    result[f"H1A2c_{label}"] = corr.get("H1A2c")
                    result[f"H1A3c_{label}"] = corr.get("H1A3c")
                    hcorr = compute_H1H2_H2H4_corrected(
                        result.get(f"H1_{label}", np.array([], dtype=float)),
                        result.get(f"H2_{label}", np.array([], dtype=float)),
                        result.get(f"H4_{label}", np.array([], dtype=float)),
                        fs,
                        F0x,
                        F1,
                        F2,
                        None if use_formula else B1,
                        None if use_formula else B2,
                    )
                    result[f"H1H2c_{label}"] = hcorr.get("H1H2c")
                    result[f"H2H4c_{label}"] = hcorr.get("H2H4c")
                    if f"H4_{label}" in result and f"H2K_{label}" in result and f"F2K_{label}" in result:
                        h42kc = compute_H42K_corrected(
                            result[f"H4_{label}"], result[f"H2K_{label}"], result[f"F2K_{label}"], fs, F0x, F1, F2, F3,
                            None if use_formula else B1,
                            None if use_formula else B2,
                            None if use_formula else B3,
                        )
                        result[f"H42Kc_{label}"] = h42kc
                    if f"H2K_{label}" in result and f"F2K_{label}" in result and f"H5K_{label}" in result:
                        h2kh5kc = compute_2K5K_corrected(
                            result[f"H2K_{label}"], result[f"F2K_{label}"], result[f"H5K_{label}"], fs, F0x, F1, F2, F3,
                            None if use_formula else B1,
                            None if use_formula else B2,
                            None if use_formula else B3,
                        )
                        result[f"H2KH5Kc_{label}"] = h2kh5kc
                    slope = compute_spectral_slope(y, fs, self.state.frameshift, F0x, voiced_mask=voiced_mask)
                    result[f"SpectralSlope_{label}"] = slope
                _compute_with("pF0", pF0)
                _compute_with("rF0", rF0u)
                self.progress_sig.emit(base + 19, name)
                jit, shim = compute_jitter_shimmer(y, fs, self.state.frameshift, self.state.windowsize, voiced_mask=voiced_mask)
                result["Jitter"] = jit
                result["Shimmer"] = shim
                
                def _smooth_points(arr: np.ndarray, points: int) -> np.ndarray:
                    try:
                        x = np.array(arr, dtype=float)
                        p = int(points)
                        if p <= 0:
                            return x
                        m = ~np.isnan(x)
                        if np.count_nonzero(m) == 0:
                            return x
                        out = x.copy()
                        idx = np.where(m)[0]
                        if idx.size == 0:
                            return x
                        cuts = np.where(np.diff(idx) > 1)[0]
                        starts = np.concatenate(([0], cuts + 1))
                        ends = np.concatenate((cuts, [idx.size - 1]))
                        hl = p // 2
                        hr = p - hl
                        for si, ei in zip(starts, ends):
                            s = int(idx[si]); e = int(idx[ei]); L = e - s + 1
                            left_ok = (s > 0 and np.isnan(x[s - 1]))
                            right_ok = (e < x.size - 1 and np.isnan(x[e + 1]))
                            if L < p and left_ok and right_ok:
                                out[s:e+1] = np.nan
                                continue
                            seg = x[s:e+1]
                            if p <= 1 or L <= 1:
                                out[s:e+1] = seg
                                continue
                            vals = np.empty_like(seg)
                            for t in range(L):
                                l = max(0, t - hl)
                                r = min(L, t + hr)
                                w = seg[l:r]
                                vals[t] = float(np.mean(w)) if w.size > 0 else np.nan
                            out[s:e+1] = vals
                        return out
                    except Exception:
                        return arr
                win = int(self.state.O_smoothwinsize)
                keys_for_smooth = [
                    "pF0","pF1","pF2","pF3","pF4","pB1","pB2","pB3","pB4",
                    "H1_pF0","H2_pF0","H4_pF0","A1_pF0","A2_pF0","A3_pF0",
                    "H1A1c_pF0","H1A2c_pF0","H1A3c_pF0","H2K_pF0","H5K_pF0","H42Ku_pF0","H2KH5Ku_pF0","H42Kc_pF0","H2KH5Kc_pF0","CPP_pF0","HNR05_pF0","HNR15_pF0","HNR25_pF0","HNR35_pF0","SHR_pF0","SpectralSlope_pF0",
                    "H1_rF0","H2_rF0","H4_rF0","A1_rF0","A2_rF0","A3_rF0",
                    "H1A1c_rF0","H1A2c_rF0","H1A3c_rF0","H2K_rF0","H5K_rF0","H42Ku_rF0","H2KH5Ku_rF0","H42Kc_rF0","H2KH5Kc_rF0","CPP_rF0","HNR05_rF0","HNR15_rF0","HNR25_rF0","HNR35_rF0","SHR_rF0","SpectralSlope_rF0",
                    "Energy","Jitter","Shimmer"
                ]
                for key in keys_for_smooth:
                    if key in result:
                        result[key] = _smooth_points(result[key], win)
                # 平滑/离群之后再计算未校正的差值，确保与最终H/A值一致
                for label in ["pF0", "rF0"]:
                    try:
                        if f"H1_{label}" in result and f"H2_{label}" in result:
                            result[f"H1H2u_{label}"] = np.array(result[f"H1_{label}"], dtype=float) - np.array(result[f"H2_{label}"], dtype=float)
                        if f"H2_{label}" in result and f"H4_{label}" in result:
                            result[f"H2H4u_{label}"] = np.array(result[f"H2_{label}"], dtype=float) - np.array(result[f"H4_{label}"], dtype=float)
                        if f"H1_{label}" in result and f"A1_{label}" in result:
                            result[f"H1A1u_{label}"] = np.array(result[f"H1_{label}"], dtype=float) - np.array(result[f"A1_{label}"], dtype=float)
                        if f"H1_{label}" in result and f"A2_{label}" in result:
                            result[f"H1A2u_{label}"] = np.array(result[f"H1_{label}"], dtype=float) - np.array(result[f"A2_{label}"], dtype=float)
                        if f"H1_{label}" in result and f"A3_{label}" in result:
                            result[f"H1A3u_{label}"] = np.array(result[f"H1_{label}"], dtype=float) - np.array(result[f"A3_{label}"], dtype=float)
                    except Exception:
                        pass
                # 无声段统一置 NaN（或 SHR 置 0）
                def _apply_mask(arr: np.ndarray) -> np.ndarray:
                    if arr is None:
                        return arr
                    try:
                        x = np.array(arr, dtype=float)
                        n = min(x.shape[0], voiced_mask.shape[0])
                        if n <= 0:
                            return x
                        mask = ~voiced_mask[:n]
                        x[:n][mask] = np.nan
                        return x
                    except Exception:
                        return arr
                for key in [
                    "pF0", "pF1", "pF2", "pF3", "pF4", "pB1", "pB2", "pB3", "pB4",
                    "H1_pF0","H2_pF0","H4_pF0","A1_pF0","A2_pF0","A3_pF0","H1H2u_pF0","H2H4u_pF0","H1A1u_pF0","H1A2u_pF0","H1A3u_pF0",
                    "H1A1c_pF0","H1A2c_pF0","H1A3c_pF0","H1H2c_pF0","H2H4c_pF0","H2K_pF0","H5K_pF0","H42Ku_pF0","H2KH5Ku_pF0","H42Kc_pF0","H2KH5Kc_pF0","CPP_pF0","HNR05_pF0","HNR15_pF0","HNR25_pF0","HNR35_pF0","SHR_pF0","SpectralSlope_pF0",
                    "H1_rF0","H2_rF0","H4_rF0","A1_rF0","A2_rF0","A3_rF0","H1H2u_rF0","H2H4u_rF0","H1A1u_rF0","H1A2u_rF0","H1A3u_rF0",
                    "H1A1c_rF0","H1A2c_rF0","H1A3c_rF0","H1H2c_rF0","H2H4c_rF0","H2K_rF0","H5K_rF0","H42Ku_rF0","H2KH5Ku_rF0","H42Kc_rF0","H2KH5Kc_rF0","CPP_rF0","HNR05_rF0","HNR15_rF0","HNR25_rF0","HNR35_rF0","SHR_rF0","SpectralSlope_rF0",
                    "Energy", "Jitter", "Shimmer"
                ]:
                    if key in result:
                        result[key] = _apply_mask(result[key])
                for key in ["SHR_pF0","SHR_rF0"]:
                    if key in result:
                        try:
                            shr = np.array(result[key], dtype=float)
                            for k in range(min(voiced_mask.shape[0], shr.shape[0])):
                                if not voiced_mask[k]:
                                    shr[k] = np.nan
                            result[key] = shr
                        except Exception:
                            pass
            except Exception as e:
                self.error_sig.emit(f"{wav_path.name}: {e}")
                break

            # TextGrid Processing (Process ALL tiers)
            try:
                tg_path = wav_path.with_suffix(".TextGrid")
                if tg_path.exists():
                    from .utils.textgrid_parser import parse_textgrid
                    tg = parse_textgrid(tg_path)
                    if tg:
                        # Determine frame count
                        n_frames = 0
                        if "pF0" in result and len(result["pF0"]) > 0:
                            n_frames = len(result["pF0"])
                        elif "rF0" in result and len(result["rF0"]) > 0:
                            n_frames = len(result["rF0"])
                        else:
                            try:
                                duration_ms = float(y.size) * 1000.0 / float(fs)
                                n_frames = int(duration_ms / self.state.frameshift)
                            except Exception:
                                n_frames = 0

                        if n_frames > 0:
                            times_sec = np.arange(n_frames, dtype=float) * self.state.frameshift / 1000.0
                            
                            for tier in tg.tiers:
                                text_col = np.full(n_frames, "", dtype=object)
                                current_int_idx = 0
                                intervals = tier.intervals
                                n_intervals = len(intervals)
                                
                                for f_idx, t_sec in enumerate(times_sec):
                                    while current_int_idx < n_intervals and intervals[current_int_idx].xmax < t_sec:
                                        current_int_idx += 1
                                    if current_int_idx >= n_intervals:
                                        break
                                    interval = intervals[current_int_idx]
                                    if interval.xmin <= t_sec <= interval.xmax:
                                        text_col[f_idx] = interval.text
                                
                                # Save as separate column: textgrid_TierName
                                result[f"textgrid_{tier.name}"] = text_col

            except Exception as e:
                print(f"TextGrid processing failed for {wav_path.name}: {e}")

            result.update(
                {
                    "HF0algorithm": "Dual (Praat + REAPER)",
                    "AFMTalgorithm": "Formants (Praat)",
                    "frameshift": self.state.frameshift,
                    "preemphasis": self.state.preemphasis,
                    "windowsize": self.state.windowsize,
                }
            )
            mat_path = self.output_dir / f"{wav_path.stem}.csv"
            save_csv(mat_path, result)
            self.progress_sig.emit(base + 20, name)
        self.finished_sig.emit()
def run_parameter_estimation_once(wav_dir: Optional[str] = None, mat_dir: Optional[str] = None) -> None:
    state = AppState()
    input_dir = Path(wav_dir or state.wavdir)
    output_dir = Path(mat_dir or state.matdir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = [p.name for p in (input_dir.rglob(state.I_searchstring) if state.recursedir else input_dir.glob(state.I_searchstring))]
    if not items:
        print("[PE] No wav files found.")
        return
    for i, name in enumerate(items, start=1):
        wav_path = input_dir / name
        print(f"[PE] ({i}/{len(items)}) {name}")
        try:
            result = compute_praat_f0_formants(
                wav_path=wav_path,
                frameshift_ms=state.frameshift,
                min_f0=state.F0Praatmin,
                max_f0=state.F0Praatmax,
                method=state.F0Praatmethod,
            )
            
            from .services.praat_service import read_wav_mono_float
            fs, y = read_wav_mono_float(wav_path)
            try:
                from .services.praat_service import compute_shrp_f0
                shrp_res = compute_shrp_f0(
                    wav_path=wav_path,
                    frameshift_ms=state.frameshift,
                    min_f0=state.F0Praatmin,
                    max_f0=state.F0Praatmax,
                    shr_threshold=float(state.SHRThreshold),
                )
                shrF0 = np.array(shrp_res.get("shrF0", []), dtype=float)
                if shrF0.size > 0:
                    result["shrF0"] = shrF0
                    result["SHR_shrF0"] = np.array(shrp_res.get("SHR", []), dtype=float)
            except Exception:
                pass
            pF0 = np.array(result.get("pF0", []), dtype=float)
            F1 = np.array(result.get("pF1", []), dtype=float)
            F2 = np.array(result.get("pF2", []), dtype=float)
            F3 = np.array(result.get("pF3", []), dtype=float)
            sampleshift = int(round(fs / 1000.0 * state.frameshift))
            win_ms = max(1, state.voicing_energy_window_ms)
            win = int(round(fs / 1000.0 * win_ms))
            nf = pF0.shape[0]
            Ewin = np.full(nf, 0.0, dtype=float)
            for k in range(nf):
                ks = int(round((k + 1) * sampleshift))
                f0c = float(pF0[k])
                if np.isnan(f0c) or f0c <= 0:
                    continue
                N0 = fs / f0c
                ystart = int(round(ks - state.Nperiods_EC / 2.0 * N0))
                yend = int(round(ks + state.Nperiods_EC / 2.0 * N0)) - 1
                if ystart <= 0 or yend >= y.size:
                    continue
                seg = y[ystart:yend]
                Ewin[k] = float(np.sum(seg.astype(float) ** 2))
            th = float(np.max(Ewin)) * float(state.voicing_energy_threshold_ratio)
            voiced_mask = Ewin > th
            from .services.praat_service import (
                compute_harmonics_H1H2H4,
                compute_A1A2A3,
                compute_H1A1A2A3_corrected,
                compute_CPP,
                compute_HNR,
                compute_SHR,
                compute_spectral_slope,
                compute_jitter_shimmer,
                compute_harmonic_at_fixed_freq,
                compute_harmonic_at_fixed_freq_with_freq,
                compute_H42K_corrected,
                compute_2K5K_corrected,
            )
            # Dual-run: 使用 pF0 和 rF0 计算两套 F0 相关参数
            rF0u = np.array(result.get("rF0Uniform", []), dtype=float)
            if rF0u.size == 0:
                rF0u = np.array(result.get("rF0", []), dtype=float)
            def _compute_with(label: str, F0x: np.ndarray) -> None:
                if F0x.size == 0:
                    return
                hh = compute_harmonics_H1H2H4(y, fs, state.frameshift, F0x, state.Nperiods, voiced_mask=voiced_mask)
                result[f"H1_{label}"] = hh.get("H1")
                result[f"H2_{label}"] = hh.get("H2")
                result[f"H4_{label}"] = hh.get("H4")
                aa = compute_A1A2A3(y, fs, state.frameshift, F0x, F1, F2, F3, state.Nperiods, voiced_mask=voiced_mask)
                result[f"A1_{label}"] = aa.get("A1")
                result[f"A2_{label}"] = aa.get("A2")
                result[f"A3_{label}"] = aa.get("A3")
                # 差值改为在平滑/掩蔽之后统一计算
                H2K, F2K = compute_harmonic_at_fixed_freq_with_freq(y, fs, state.frameshift, F0x, state.Nperiods, 2000.0, voiced_mask=voiced_mask)
                result[f"H2K_{label}"] = H2K
                result[f"F2K_{label}"] = F2K
                H5K = compute_harmonic_at_fixed_freq(y, fs, state.frameshift, F0x, state.Nperiods, 5000.0, voiced_mask=voiced_mask)
                result[f"H5K_{label}"] = H5K
                if f"H4_{label}" in result and f"H2K_{label}" in result:
                    result[f"H42Ku_{label}"] = result[f"H4_{label}"] - result[f"H2K_{label}"]
                if f"H2K_{label}" in result and f"H5K_{label}" in result:
                    result[f"H2KH5Ku_{label}"] = result[f"H2K_{label}"] - result[f"H5K_{label}"]
                result[f"CPP_{label}"] = compute_CPP(y, fs, state.frameshift, F0x, state.Nperiods_EC, voiced_mask=voiced_mask)
                hnr = compute_HNR(y, fs, state.frameshift, F0x, state.Nperiods_EC, voiced_mask=voiced_mask)
                for k, v in hnr.items():
                    result[f"{k}_{label}"] = v
                result[f"SHR_{label}"] = compute_SHR(y, fs, state.frameshift, F0x, float(state.SHRmin), float(state.SHRmax), progress_cb=None, voiced_mask=voiced_mask, shr_threshold=float(state.SHRThreshold))
                B1 = np.array(result.get("pB1", []), dtype=float)
                B2 = np.array(result.get("pB2", []), dtype=float)
                B3 = np.array(result.get("pB3", []), dtype=float)
                use_formula = (state.BandwidthMethod == "Use formula values")
                corr = compute_H1A1A2A3_corrected(
                    result[f"H1_{label}"],
                    result[f"A1_{label}"],
                    result[f"A2_{label}"],
                    result[f"A3_{label}"],
                    fs,
                    F0x,
                    F1,
                    F2,
                    F3,
                    None if use_formula else B1,
                    None if use_formula else B2,
                    None if use_formula else B3,
                )
                result[f"H1A1c_{label}"] = corr.get("H1A1c")
                result[f"H1A2c_{label}"] = corr.get("H1A2c")
                result[f"H1A3c_{label}"] = corr.get("H1A3c")
                hcorr = compute_H1H2_H2H4_corrected(
                    result.get(f"H1_{label}", np.array([], dtype=float)),
                    result.get(f"H2_{label}", np.array([], dtype=float)),
                    result.get(f"H4_{label}", np.array([], dtype=float)),
                    fs,
                    F0x,
                    F1,
                    F2,
                    None if use_formula else B1,
                    None if use_formula else B2,
                )
                result[f"H1H2c_{label}"] = hcorr.get("H1H2c")
                result[f"H2H4c_{label}"] = hcorr.get("H2H4c")
                if f"H4_{label}" in result and f"H2K_{label}" in result and f"F2K_{label}" in result:
                    h42kc = compute_H42K_corrected(
                        result[f"H4_{label}"], result[f"H2K_{label}"], result[f"F2K_{label}"], fs, F0x, F1, F2, F3,
                        None if use_formula else B1,
                        None if use_formula else B2,
                        None if use_formula else B3,
                    )
                    result[f"H42Kc_{label}"] = h42kc
                if f"H2K_{label}" in result and f"F2K_{label}" in result and f"H5K_{label}" in result:
                    h2kh5kc = compute_2K5K_corrected(
                        result[f"H2K_{label}"], result[f"F2K_{label}"], result[f"H5K_{label}"], fs, F0x, F1, F2, F3,
                        None if use_formula else B1,
                        None if use_formula else B2,
                        None if use_formula else B3,
                    )
                    result[f"H2KH5Kc_{label}"] = h2kh5kc
                slope = compute_spectral_slope(y, fs, state.frameshift, F0x, voiced_mask=voiced_mask)
                result[f"SpectralSlope_{label}"] = slope
            _compute_with("pF0", pF0)
            _compute_with("rF0", rF0u)
            jit, shim = compute_jitter_shimmer(y, fs, state.frameshift, state.windowsize, voiced_mask=voiced_mask)
            result["Jitter"] = jit
            result["Shimmer"] = shim
            # 平滑与离群剔除
            def _smooth_points(arr: np.ndarray, points: int) -> np.ndarray:
                try:
                    x = np.array(arr, dtype=float)
                    p = int(points)
                    if p <= 0:
                        return x
                    m = ~np.isnan(x)
                    if np.count_nonzero(m) == 0:
                        return x
                    out = x.copy()
                    idx = np.where(m)[0]
                    if idx.size == 0:
                        return x
                    cuts = np.where(np.diff(idx) > 1)[0]
                    starts = np.concatenate(([0], cuts + 1))
                    ends = np.concatenate((cuts, [idx.size - 1]))
                    hl = p // 2
                    hr = p - hl
                    for si, ei in zip(starts, ends):
                        s = int(idx[si]); e = int(idx[ei]); L = e - s + 1
                        left_ok = (s > 0 and np.isnan(x[s - 1]))
                        right_ok = (e < x.size - 1 and np.isnan(x[e + 1]))
                        if L < p and left_ok and right_ok:
                            out[s:e+1] = np.nan
                            continue
                        seg = x[s:e+1]
                        if p <= 1 or L <= 1:
                            out[s:e+1] = seg
                            continue
                        vals = np.empty_like(seg)
                        for t in range(L):
                            l = max(0, t - hl)
                            r = min(L, t + hr)
                            w = seg[l:r]
                            vals[t] = float(np.mean(w)) if w.size > 0 else np.nan
                        out[s:e+1] = vals
                    return out
                except Exception:
                    return arr
            win = int(state.O_smoothwinsize)
            keys_for_smooth = [
                "pF0","strF0","pF1","pF2","pF3","pF4","pB1","pB2","pB3","pB4",
                "H1","H2","H4","A1","A2","A3",
                "H1A1c","H1A2c","H1A3c","H2K","H5K","H42Ku","H2KH5Ku","H42Kc","H2KH5Kc","CPP","Energy","HNR05","HNR15","HNR25","HNR35","SHR","SpectralSlope","Jitter","Shimmer"
            ]
            for key in keys_for_smooth:
                if key in result:
                    result[key] = _smooth_points(result[key], win)
            # 平滑/离群之后再计算未校正差值
            try:
                if "H1" in result and "H2" in result:
                    result["H1H2u"] = np.array(result["H1"], dtype=float) - np.array(result["H2"], dtype=float)
                if "H2" in result and "H4" in result:
                    result["H2H4u"] = np.array(result["H2"], dtype=float) - np.array(result["H4"], dtype=float)
                if "H1" in result and "A1" in result:
                    result["H1A1u"] = np.array(result["H1"], dtype=float) - np.array(result["A1"], dtype=float)
                if "H1" in result and "A2" in result:
                    result["H1A2u"] = np.array(result["H1"], dtype=float) - np.array(result["A2"], dtype=float)
                if "H1" in result and "A3" in result:
                    result["H1A3u"] = np.array(result["H1"], dtype=float) - np.array(result["A3"], dtype=float)
            except Exception:
                pass
            # 掩蔽无声段
            for key in ["pF0", "strF0", "pF1", "pF2", "pF3", "pF4", "pB1", "pB2", "pB3", "pB4", "H1", "H2", "H4", "A1", "A2", "A3", "H1H2u", "H2H4u", "H1A1u", "H1A2u", "H1A3u", "H1A1c", "H1A2c", "H1A3c", "H2K", "H5K", "H42Ku", "H2KH5Ku", "CPP", "Energy", "HNR05", "HNR15", "HNR25", "HNR35", "SpectralSlope", "Jitter", "Shimmer"]:
                if key in result:
                    x = np.array(result[key], dtype=float)
                    n = min(x.shape[0], voiced_mask.shape[0])
                    if n > 0:
                        mask = ~voiced_mask[:n]
                        x[:n][mask] = np.nan
                    result[key] = x
            if "SHR" in result:
                shr = np.array(result["SHR"], dtype=float)
                for k in range(min(voiced_mask.shape[0], shr.shape[0])):
                    if not voiced_mask[k]:
                        shr[k] = np.nan
                result["SHR"] = shr
            # 元数据
            result.update({
                "HF0algorithm": state.F0algorithm,
                "AFMTalgorithm": "Formants (Praat)",
                "frameshift": state.frameshift,
                "preemphasis": state.preemphasis,
                "windowsize": state.windowsize,
                "O_smoothwinsize": state.O_smoothwinsize,
            })
            mat_path = output_dir / f"{wav_path.stem}.csv"
            save_csv(mat_path, result)
            print(f"[PE] wrote {mat_path.name}")
        except Exception as e:
            print(f"[PE] error on {name}: {e}")
