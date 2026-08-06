from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGridLayout, QPushButton, 
    QLabel, QStackedWidget, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6 import QtCore
from PyQt6.QtGui import QDesktopServices, QIcon
import os
import webbrowser
import urllib.request
import json
import threading
import re
from pathlib import Path

from .styles import DARK_MAIN_STYLESHEET, GLOBAL_DARK_STYLESHEET, LIGHT_MAIN_STYLESHEET, GLOBAL_LIGHT_STYLESHEET
from .widgets.parameter_estimation_widget import ParameterEstimationWidget
from .widgets.parameter_display_widget import ParameterDisplayWidget
from .widgets.pitch_manipulation_widget import PitchManipulationWidget
from .widgets.egg_widget import EGGWidget
from .widgets.spec2wav_widget import Spec2WavWidget
from .widgets.speech_synthesis_widget import SpeechSynthesisWidget
from .widgets.lpc_spectrum_widget import LPCSpectrumWidget
from .widgets.phonology_induction_widget import PhonologyInductionWidget
from .widgets.lip_gui import launch_lip_gui
from .dialogs.about_dialog import AboutDialog
from phonetic_toolbox import __version__
from phonetic_toolbox.utils import get_resource_path
from phonetic_toolbox.api import (
    launch_articulatory_synth,
    launch_ipa_trans,
    launch_perception_experiment,
)

ASCII_TITLE = (
    " ██████╗ ██╗  ██╗ ██████╗ ███╗   ██╗███████╗████████╗██╗ ██████╗████████╗ ██████╗  ██████╗ ██╗     ██████╗  ██████╗ ██╗  ██╗\n"
    " ██╔══██╗██║  ██║██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██║██╔════╝╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔══██╗██╔═══██╗╚██╗██╔╝\n"
    "██████╔╝███████║██║   ██║██╔██╗ ██║█████╗     ██║   ██║██║        ██║   ██║   ██║██║   ██║██║     ██████╔╝██║   ██║ ╚███╔╝ \n"
    "██╔═══╝ ██╔══██║██║   ██║██║╚██╗██║██╔══╝     ██║   ██║██║        ██║   ██║   ██║██║   ██║██║     ██╔══██╗██║   ██║ ██╔██╗ \n"
    " ██║     ██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   ██║╚██████╗   ██║   ╚██████╔╝╚██████╔╝███████╗██████╔╝╚██████╔╝██╔╝ ██╗\n"
    " ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝"
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phonetic Toolbox v2")
        self.resize(700, 460)
        self.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        
        # Central Widget is a StackedWidget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # Page 0: Home (Grid of Buttons)
        self.home_page = QWidget()
        self.home_page.setObjectName("home_page") # Set ObjectName for scoped styling
        self.init_home_ui()
        self.stack.addWidget(self.home_page)
        
        # Sub-windows storage to prevent garbage collection
        self.sub_windows = {}
        
        # Apply Theme
        self.is_dark = True
        self.apply_theme()

        # 启动时仅自动检查一次更新（网络允许时）：已是最新则完全静默，仅有新版本时弹窗提醒
        QtCore.QTimer.singleShot(1500, self._auto_check_update_once)

    def init_home_ui(self):
        layout = QVBoxLayout(self.home_page)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(10)
        
        # Title (ASCII Art)
        title = QLabel(ASCII_TITLE)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use Monospace font for ASCII art to align correctly
        # Adjust size as needed, 8px might be good for large ASCII block
        title.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 9px; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        layout.addSpacing(2)
        
        # Grid
        grid = QGridLayout()
        grid.setSpacing(10)
        
        buttons = [
            ("唇形提取", self.on_lip_extraction),
            ("参数估计", self.on_parameter_estimation),
            ("参数显示", self.on_parameter_display),
            ("EGG信号分析", self.on_egg_analysis),
            ("语音合成", self.on_speech_synthesis),
            ("发音物理模拟", self.on_articulatory_synth),
            ("变速变调", self.on_pitch_manipulation),
            ("感知实验", self.on_perception_experiment),
            ("MFA自动标注", self.on_mfa_auto_alignment),
            ("语谱图转音频", self.on_spec2wav),
            ("普通话转IPA", self.on_ipa_trans),
            ("LPC谱图", self.on_lpc_spectrum),
            ("音系归纳", self.on_phonology_induction),
            ("检查更新", self.on_check_update),
            ("关于", self.on_about),
            ("使用说明", self.on_open_help),
            ("☀/🌙", self.toggle_theme)
        ]
        
        row = 0
        col = 0
        for text, slot in buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(62)
            btn.clicked.connect(slot)
            grid.addWidget(btn, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1
                
        layout.addLayout(grid)

        # Helper method removed as it is no longer needed
    
    def apply_theme(self):
        app = QApplication.instance()
        if self.is_dark:
            app.setStyleSheet(GLOBAL_DARK_STYLESHEET)
            self.setStyleSheet(DARK_MAIN_STYLESHEET)
            # Update ASCII Title color for Dark Mode
            title_label = self.home_page.findChild(QLabel)
            if title_label:
                title_label.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 9px; font-weight: bold; color: white;")
        else:
            app.setStyleSheet(GLOBAL_LIGHT_STYLESHEET)
            self.setStyleSheet(LIGHT_MAIN_STYLESHEET)
            # Update ASCII Title color for Light Mode
            title_label = self.home_page.findChild(QLabel)
            if title_label:
                title_label.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 9px; font-weight: bold; color: black;")
        
        # Propagate theme to sub-windows
        if "pe" in self.sub_windows and self.sub_windows["pe"].isVisible():
            if hasattr(self.sub_windows["pe"], "set_theme"):
                self.sub_windows["pe"].set_theme(self.is_dark)
        if "pd" in self.sub_windows and self.sub_windows["pd"].isVisible():
            if hasattr(self.sub_windows["pd"], "set_theme"):
                self.sub_windows["pd"].set_theme(self.is_dark)
        if "pm" in self.sub_windows and self.sub_windows["pm"].isVisible():
            # Update PitchManipulationWidget theme
            # It has apply_theme method and checks self.is_dark internally?
            # We need to set its internal state first
            self.sub_windows["pm"].is_dark = self.is_dark
            if hasattr(self.sub_windows["pm"], "apply_theme"):
                self.sub_windows["pm"].apply_theme()

        if "egg" in self.sub_windows and self.sub_windows["egg"].isVisible():
            if hasattr(self.sub_windows["egg"], "set_theme"):
                self.sub_windows["egg"].set_theme(self.is_dark)

        if "s2w" in self.sub_windows and self.sub_windows["s2w"].isVisible():
            if hasattr(self.sub_windows["s2w"], "set_theme"):
                self.sub_windows["s2w"].set_theme(self.is_dark)
        if "ss" in self.sub_windows and self.sub_windows["ss"].isVisible():
            if hasattr(self.sub_windows["ss"], "set_theme"):
                self.sub_windows["ss"].set_theme(self.is_dark)
        if "lpc" in self.sub_windows and self.sub_windows["lpc"].isVisible():
            if hasattr(self.sub_windows["lpc"], "set_theme"):
                self.sub_windows["lpc"].set_theme(self.is_dark)
        if "pi" in self.sub_windows and self.sub_windows["pi"].isVisible():
            if hasattr(self.sub_windows["pi"], "set_theme"):
                self.sub_windows["pi"].set_theme(self.is_dark)
        if "lip" in self.sub_windows and self.sub_windows["lip"].isVisible():
            if hasattr(self.sub_windows["lip"], "set_theme"):
                self.sub_windows["lip"].set_theme(self.is_dark)

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.apply_theme()

    def on_parameter_estimation(self):
        # Check if window already open
        if "pe" in self.sub_windows and self.sub_windows["pe"].isVisible():
            self.sub_windows["pe"].activateWindow()
            return
            
        w = ParameterEstimationWidget()
        w.setWindowTitle("参数估计 - Phonetic Toolbox v2")
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        w.resize(900, 700)
        
        # Apply current theme to new window
        # We can pass stylesheets directly or rely on global application stylesheet
        # Since we use scoped styles for home page and general for others, it should be fine.
        # But we need to ensure the widget itself doesn't need specific styling that was in MainWindow
        
        self.sub_windows["pe"] = w
        
        # Apply current theme
        if hasattr(w, "set_theme"):
            w.set_theme(self.is_dark)
            
        w.show()

    def on_parameter_display(self):
        if "pd" in self.sub_windows and self.sub_windows["pd"].isVisible():
            self.sub_windows["pd"].activateWindow()
            return
            
        w = ParameterDisplayWidget()
        w.setWindowTitle("参数显示 - Phonetic Toolbox v2")
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        w.resize(1000, 700)
        self.sub_windows["pd"] = w
        
        if hasattr(w, "set_theme"):
            w.set_theme(self.is_dark)
            
        w.show()

    def on_pitch_manipulation(self):
        if "pm" in self.sub_windows and self.sub_windows["pm"].isVisible():
            self.sub_windows["pm"].activateWindow()
            return
            
        w = PitchManipulationWidget()
        w.setWindowTitle("基频实验室 (Pitch Lab) - Phonetic Toolbox v2")
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        w.resize(1400, 900)
        self.sub_windows["pm"] = w
        
        # Apply initial theme
        w.is_dark = self.is_dark
        if hasattr(w, "apply_theme"):
            w.apply_theme()
            
        w.show()

    def on_egg_analysis(self):
        if "egg" in self.sub_windows and self.sub_windows["egg"].isVisible():
            self.sub_windows["egg"].activateWindow()
            return
            
        w = EGGWidget()
        w.setWindowTitle("EGG 信号分析 - Phonetic Toolbox v2")
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        w.resize(1200, 800)
        self.sub_windows["egg"] = w
        
        # Apply theme if supported
        if hasattr(w, "set_theme"):
            w.set_theme(self.is_dark)
            
        w.show()

    def on_spec2wav(self):
        if "s2w" in self.sub_windows and self.sub_windows["s2w"].isVisible():
            self.sub_windows["s2w"].activateWindow()
            return
            
        w = Spec2WavWidget()
        # Window title is already set in __init__ but we can override or ensure consistency
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        self.sub_windows["s2w"] = w
        
        if hasattr(w, "set_theme"):
            w.set_theme(self.is_dark)
            
        w.show()

    def on_speech_synthesis(self):
        if "ss" in self.sub_windows and self.sub_windows["ss"].isVisible():
            self.sub_windows["ss"].activateWindow()
            return
        w = SpeechSynthesisWidget()
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        self.sub_windows["ss"] = w
        if hasattr(w, "set_theme"):
            w.set_theme(self.is_dark)
        w.show()

    def on_lpc_spectrum(self):
        if "lpc" in self.sub_windows and self.sub_windows["lpc"].isVisible():
            self.sub_windows["lpc"].activateWindow()
            return
        w = LPCSpectrumWidget()
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        w.resize(1200, 700)
        self.sub_windows["lpc"] = w
        if hasattr(w, "set_theme"):
            w.set_theme(self.is_dark)
        w.show()

    def on_phonology_induction(self):
        if "pi" in self.sub_windows and self.sub_windows["pi"].isVisible():
            self.sub_windows["pi"].activateWindow()
            return
        w = PhonologyInductionWidget()
        w.setWindowIcon(QIcon(get_resource_path("PhoneticToolbox.ico")))
        w.resize(360, 240)
        self.sub_windows["pi"] = w
        if hasattr(w, "set_theme"):
            w.set_theme(self.is_dark)
        w.show()

    def on_lip_extraction(self):
        if "lip" in self.sub_windows and self.sub_windows["lip"].isVisible():
            self.sub_windows["lip"].activateWindow()
            return
        w = launch_lip_gui(is_dark=self.is_dark)
        self.sub_windows["lip"] = w

    def on_ipa_trans(self):
        result = launch_ipa_trans()
        if result.success:
            return
        else:
            message = result.message
            if result.generator_output:
                message = f"{message}\n\n{result.generator_output}"
            QMessageBox.critical(self, "普通话转IPA", message)

    def on_perception_experiment(self):
        result = launch_perception_experiment()
        if result.success:
            return
        else:
            QMessageBox.critical(self, "感知实验", result.message)

    def on_articulatory_synth(self):
        result = launch_articulatory_synth()
        if result.success:
            return
        QMessageBox.critical(self, "发音物理模拟", result.message)

    def on_mfa_auto_alignment(self):
        from .dialogs.mfa_auto_alignment_dialog import MFAAutoAlignmentDialog

        dialog = MFAAutoAlignmentDialog(self)
        dialog.setWindowIcon(self.windowIcon())
        if hasattr(dialog, "set_theme"):
            dialog.set_theme(self.is_dark)
        dialog.exec()

    def on_open_help(self):
        # 使用 get_resource_path 获取兼容打包环境的路径
        help_path = get_resource_path(r"Phonetic_Export\index.html")
        if os.path.exists(help_path):
            webbrowser.open(f"file:///{help_path}")
        else:
            QMessageBox.warning(self, "错误", f"找不到使用说明文件:\n{help_path}\n\n请确保 Phonetic_Export 文件夹在程序运行目录中。")

    def _open_help_with_fragment(self, fragment: str):
        help_path = get_resource_path(r"Phonetic_Export\index.html")
        if os.path.exists(help_path):
            url = QUrl.fromLocalFile(help_path)
            url.setFragment(fragment)
            opened = QDesktopServices.openUrl(url)
            if not opened:
                QMessageBox.warning(self, "帮助", "帮助页面打开失败。")
        else:
            QMessageBox.warning(self, "帮助", f"找不到帮助文件:\n{help_path}")

    def on_check_update(self):
        current_version = self._load_current_version()

        self.update_msg_box = QMessageBox(self)
        self.update_msg_box.setWindowTitle("检查更新")
        self.update_msg_box.setWindowIcon(self.windowIcon())
        self.update_msg_box.setIcon(QMessageBox.Icon.NoIcon)
        self.update_msg_box.setText("正在检查 GitHub 最新版本")
        self.update_msg_box.setInformativeText("请稍候，正在获取发布信息…")
        self.update_msg_box.setStandardButtons(QMessageBox.StandardButton.NoButton)
        self._style_update_message_box(self.update_msg_box)
        self.update_msg_box.show()

        threading.Thread(target=self._fetch_latest_version, args=(current_version,), daemon=True).start()

    def _auto_check_update_once(self):
        """启动时自动检查一次更新：无加载弹窗、无结果弹窗、网络失败静默，仅发现新版本时提醒。"""
        if getattr(self, "_auto_update_checked", False):
            return
        self._auto_update_checked = True
        current_version = self._load_current_version()
        threading.Thread(target=self._fetch_latest_version_silent, args=(current_version,), daemon=True).start()

    def _fetch_latest_version_silent(self, current_version):
        url = "https://api.github.com/repos/Nephelium/PhoneticToolbox/releases/latest"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                latest_version = data.get("tag_name", "")
                release_url = data.get("html_url", "https://github.com/Nephelium/PhoneticToolbox/releases")
                release_notes = data.get("body", "")

                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(self, "_show_auto_update_result",
                                         Qt.ConnectionType.QueuedConnection,
                                         Q_ARG(str, current_version),
                                         Q_ARG(str, latest_version),
                                         Q_ARG(str, release_url),
                                         Q_ARG(str, release_notes))
        except Exception:
            # 网络不可用或请求受限时静默跳过，不打扰用户
            pass

    @QtCore.pyqtSlot(str, str, str, str)
    def _show_auto_update_result(self, current_version, latest_version, release_url, release_notes):
        # 仅在当前版本低于 GitHub 最新版本时弹窗；一致/更高/无法比较均保持静默
        if self._compare_versions(current_version, latest_version) != -1:
            return
        notes = self._summarize_release_notes(release_notes)
        msg = QMessageBox(self)
        msg.setWindowTitle("发现新版本")
        msg.setWindowIcon(self.windowIcon())
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setText("发现新版本，建议更新")
        msg.setInformativeText(
            f"当前版本：{current_version}\nGitHub 最新：{latest_version}\n\n更新说明：\n{notes}"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.button(QMessageBox.StandardButton.Yes).setText("立即下载")
        msg.button(QMessageBox.StandardButton.No).setText("稍后")
        self._style_update_message_box(msg)
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Yes:
            webbrowser.open(release_url)

    def _fetch_latest_version(self, current_version):
        url = "https://api.github.com/repos/Nephelium/PhoneticToolbox/releases/latest"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                latest_version = data.get("tag_name", "未知")
                release_url = data.get("html_url", "https://github.com/Nephelium/PhoneticToolbox/releases")
                release_notes = data.get("body", "")
                
                # Update GUI safely
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(self, "_show_update_result", 
                                         Qt.ConnectionType.QueuedConnection, 
                                         Q_ARG(str, current_version), 
                                         Q_ARG(str, latest_version), 
                                         Q_ARG(str, release_url),
                                         Q_ARG(str, release_notes))
        except Exception as e:
            from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(self, "_show_update_error", 
                                     Qt.ConnectionType.QueuedConnection, 
                                     Q_ARG(str, str(e)))

    @QtCore.pyqtSlot(str, str, str, str)
    def _show_update_result(self, current_version, latest_version, release_url, release_notes):
        msg = getattr(self, "update_msg_box", QMessageBox(self))
        msg.setWindowTitle("检查更新")
        msg.setWindowIcon(self.windowIcon())
        msg.setIcon(QMessageBox.Icon.NoIcon)
        relation = self._compare_versions(current_version, latest_version)

        if relation == 0:
            msg.setText("版本结论：当前版本与 GitHub 最新版本一致")
            msg.setInformativeText(
                f"当前版本：{current_version}\nGitHub 最新：{latest_version}\n\n无需更新。"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        elif relation == -1:
            notes = self._summarize_release_notes(release_notes)
            msg.setText("版本结论：当前版本低于 GitHub 最新版本")
            msg.setInformativeText(
                f"当前版本：{current_version}\nGitHub 最新：{latest_version}\n\n建议更新到最新版本。"
                f"\n\n更新说明：\n{notes}"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.button(QMessageBox.StandardButton.Yes).setText("立即下载")
            msg.button(QMessageBox.StandardButton.No).setText("稍后")
        elif relation == 1:
            msg.setText("版本结论：当前版本高于 GitHub 最新版本")
            msg.setInformativeText(
                f"当前版本：{current_version}\nGitHub 最新：{latest_version}\n\n当前为更高版本，无需更新。"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        else:
            msg.setText("版本结论：无法精确比较版本号")
            msg.setInformativeText(
                f"当前版本：{current_version}\nGitHub 最新：{latest_version}\n\n可手动前往发布页确认是否需要更新。"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.button(QMessageBox.StandardButton.Yes).setText("打开发布页")
            msg.button(QMessageBox.StandardButton.No).setText("取消")

        self._style_update_message_box(msg)
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Yes:
            webbrowser.open(release_url)

    @QtCore.pyqtSlot(str)
    def _show_update_error(self, error_msg):
        msg = getattr(self, "update_msg_box", QMessageBox(self))
        msg.setWindowTitle("检查更新")
        msg.setWindowIcon(self.windowIcon())
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setText("无法获取 GitHub 最新版本")
        msg.setInformativeText(
            f"可能是网络问题或访问受限。\n\n错误信息：{error_msg}\n\n是否手动打开 Releases 页面？"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.button(QMessageBox.StandardButton.Yes).setText("手动打开")
        msg.button(QMessageBox.StandardButton.No).setText("取消")

        self._style_update_message_box(msg)
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Yes:
            webbrowser.open("https://github.com/Nephelium/PhoneticToolbox/releases")

    def _style_update_message_box(self, msg):
        msg.setMinimumWidth(560)
        msg.setStyleSheet("""
            QMessageBox {
                font-family: 'Times New Roman', 'KaiTi', 'Microsoft YaHei', sans-serif;
                font-size: 14px;
            }
            QLabel {
                font-family: 'Times New Roman', 'KaiTi', 'Microsoft YaHei', sans-serif;
                font-size: 14px;
                min-width: 420px;
                line-height: 1.45;
            }
            QPushButton {
                min-width: 120px;
                min-height: 38px;
                border-radius: 5px;
                font-family: 'Times New Roman', 'KaiTi', 'Microsoft YaHei', sans-serif;
                font-size: 14px;
                font-weight: bold;
                padding: 6px 14px;
            }
        """)
        if self.is_dark:
            msg.setStyleSheet(msg.styleSheet() + """
                QMessageBox {
                    background-color: #202124;
                    color: #e8eaed;
                }
                QLabel {
                    color: #e8eaed;
                }
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1abc9c;
                }
            """)
        else:
            msg.setStyleSheet(msg.styleSheet() + """
                QMessageBox {
                    background-color: #ffffff;
                    color: #333333;
                }
                QLabel {
                    color: #333333;
                }
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1abc9c;
                }
            """)

    def _normalize_version(self, version):
        if not version:
            return None
        normalized = version.strip().lower().lstrip("v")
        parts = []
        for part in normalized.split("."):
            match = re.match(r"(\d+)", part)
            if not match:
                return None
            parts.append(int(match.group(1)))
        if not parts:
            return None
        return tuple(parts)

    def _compare_versions(self, current_version, latest_version):
        current = self._normalize_version(current_version)
        latest = self._normalize_version(latest_version)
        if current is None or latest is None:
            return None

        max_len = max(len(current), len(latest))
        current = current + (0,) * (max_len - len(current))
        latest = latest + (0,) * (max_len - len(latest))

        if current < latest:
            return -1
        if current > latest:
            return 1
        return 0

    def _summarize_release_notes(self, text):
        if not text:
            return "GitHub 未提供本次发布说明。"
        normalized = text.replace("\r\n", "\n").strip()
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        lines = normalized.split("\n")
        if len(lines) > 18:
            normalized = "\n".join(lines[:18]).rstrip() + "\n..."
        if len(normalized) > 1600:
            normalized = normalized[:1600].rstrip() + "..."
        return normalized

    def _load_current_version(self):
        candidates = [
            Path(get_resource_path("pyproject.toml")),
            Path(__file__).resolve().parents[2] / "pyproject.toml",
            Path.cwd() / "pyproject.toml",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            match = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if value:
                    return f"v{value}"
        return f"v{__version__}"

    def on_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def on_placeholder(self):
        QMessageBox.information(self, "Info", "该功能尚未实现或正在迁移中")
