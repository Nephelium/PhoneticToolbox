from pathlib import Path
from typing import List, Tuple, Set
from PyQt6 import QtWidgets, QtCore, QtGui

from phonetic_toolbox.utils import get_resource_path


class ParameterSelectionDialog(QtWidgets.QDialog):
    def __init__(self, items: List[Tuple[str, str]], selected_keys: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("参数选择")
        self.resize(520, 700)
        self._items = items
        self._selected_keys: Set[str] = set(selected_keys)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget()
        for key, label in self._items:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if key in self._selected_keys else QtCore.Qt.CheckState.Unchecked
            )
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        quick_layout = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("全选")
        btn_none = QtWidgets.QPushButton("全不选")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        quick_layout.addWidget(btn_all)
        quick_layout.addWidget(btn_none)
        quick_layout.addStretch()
        layout.addLayout(quick_layout)

        bottom = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("确定")
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        bottom.addStretch()
        bottom.addWidget(btn_ok)
        bottom.addWidget(btn_cancel)
        layout.addLayout(bottom)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.CheckState.Checked)

    def _select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)

    def selected_keys(self) -> List[str]:
        out = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return out


class ParameterHelpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("声学参数说明")
        self.resize(800, 900)
        icon_path = get_resource_path("PhoneticToolbox.ico")
        if Path(icon_path).exists():
            self.setWindowIcon(QtGui.QIcon(icon_path))
        self.is_dark = False
        if parent and hasattr(parent, "is_dark"):
            self.is_dark = parent.is_dark
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        if self.is_dark:
            h3_color = "#8cc4ff"
            code_bg = "#3e3e3e"
            code_color = "#e0e0e0"
            text_color = "#f0f0f0"
        else:
            h3_color = "#2c3e50"
            code_bg = "#f0f0f0"
            code_color = "#000000"
            text_color = "#000000"
        html = f"""
        <style>
            body {{ color: {text_color}; }}
            h3 {{ color: {h3_color}; margin-top: 20px; }}
            li {{ margin-bottom: 5px; }}
            code {{ background-color: {code_bg}; color: {code_color}; padding: 2px 4px; border-radius: 3px; font-family: Consolas, monospace; }}
        </style>
        <h2>声学参数详细说明</h2>
        <p>本工具提供的参数包含基于 <b>Praat F0 (pF0)</b> 和 <b>REAPER F0 (rF0)</b> 的两套版本（例如 <code>H1_pF0</code> 和 <code>H1_rF0</code>），分别对应两种不同的基频提取算法。</p>

        <h3>1. 基频 (Fundamental Frequency)</h3>
        <ul>
            <li><b>F0</b>: 声带振动的基频，对应听感上的“音高”。</li>
            <li><code>F0 - Praat (pF0)</code>: 使用 Praat (Auto-correlation/Cross-correlation) 算法提取的基频。</li>
            <li><code>F0 - REAPER (rF0)</code>: 使用 REAPER (Robust Epoch And Pitch Estimator) 算法提取的基频。通常在嘎裂 (Creak) 相关音质中表现更稳健。</li>
        </ul>

        <h3>2. 共振峰 (Formants)</h3>
        <ul>
            <li><code>F1, F2, F3, F4</code>: 声道的前四个共振峰频率。
                <ul>
                    <li><b>F1</b>: 与开口度有关（舌位越高，F1越低）。</li>
                    <li><b>F2</b>: 与舌位前后有关（舌位越前，F2越高）。</li>
                    <li><b>F3</b>: 与圆唇、卷舌有关（圆唇和卷舌时F3较低，靠近F2）。</li>
                </ul>
            </li>
            <li><code>B1, B2, B3, B4</code>: 对应共振峰的带宽 (Bandwidth)。带宽越宽，共振峰越不显著（衰减越快）。</li>
        </ul>

        <h3>3. 谐波与幅度 (Harmonics & Amplitudes)</h3>
        <p>所有幅度单位均为 dB。</p>
        <ul>
            <li><code>H1</code>: 第一谐波（即基频 F0）的幅度。与声带闭合程度有关。</li>
            <li><code>H2</code>: 第二谐波（2*F0）的幅度。</li>
            <li><code>H4</code>: 第四谐波 (4*F0) 的幅度。</li>
            <li><code>A1, A2, A3</code>: 分别为最接近 F1, F2, F3 频率处的谐波幅度。代表了共振峰处的能量。</li>
            <li><code>H2K</code>: 频率最接近 2000Hz 的谐波幅度。</li>
            <li><code>H5K</code>: 频率最接近 5000Hz 的谐波幅度。</li>
        </ul>

        <h3>4. 频谱倾斜 (Spectral Tilt) - 未校正</h3>
        <p>反映嗓音音质（如气声、挤喉声）的重要指标。未校正参数受元音共振峰影响较大。</p>
        <ul>
            <li><code>H1-H2</code>: H1 与 H2 的幅度差。正值通常与气声 (Breathy Voice) 有关，负值常与嘎裂 (Creak)、嘎裂声 (Creaky Voice) 有关。</li>
            <li><code>H2-H4</code>: H2 与 H4 的幅度差。</li>
            <li><code>H1-A1</code>: H1 与第一共振峰处幅度的差。反映低频段的频谱倾斜。</li>
            <li><code>H1-A2</code>: H1 与第二共振峰处幅度的差。</li>
            <li><code>H1-A3</code>: H1 与第三共振峰处幅度的差。反映整体频谱倾斜。</li>
            <li><code>H4-2K</code>: H4 与 2000Hz 处幅度的差。</li>
            <li><code>2K-5K</code>: 2000Hz 与 5000Hz 处幅度的差。</li>
        </ul>

        <h3>5. 参数校正 (*)</h3>
        <p>带有 <code>*</code> 号或 <code>c</code> 后缀的参数。使用 Iseli & Alwan 算法，根据共振峰频率 (F1-F4) 和带宽 (B1-B4) 对谐波幅度进行了校正，<b>消除了声道共振对声源频谱的影响</b>，更能准确反映声源特征。</p>
        <ul>
            <li><code>H1*-H2*</code>: 校正后的 H1-H2。</li>
            <li><code>H1*-A1*, H1*-A2*, H1*-A3*</code>: 校正后的 H1-Ax 差值。</li>
            <li><code>H4*-2K*, 2K*-5K*</code>: 校正后的高频倾斜指标。</li>
        </ul>

        <h3>6. 噪声与微扰 (Noise & Perturbation)</h3>
        <ul>
            <li><code>CPP</code> (Cepstral Peak Prominence): 倒谱峰值显著度。反映嗓音的周期性程度，值越大嗓音质量越好（越清晰）。对发声障碍非常敏感。</li>
            <li><code>HNR</code> (Harmonics-to-Noise Ratio): 谐波噪声比。
                <ul>
                    <li><code>HNR05, HNR15, HNR25, HNR35</code>: 分别对应 0-500Hz, 0-1500Hz, 0-2500Hz, 0-3500Hz 频带的 HNR。</li>
                </ul>
            </li>
            <li><code>SHR</code> (Subharmonic-to-Harmonic Ratio): 次谐波与谐波比。用于检测次谐波。嘎裂 (Creak)、嘎裂声 (Creaky Voice)相关音质的SHR通常接近1。</li>
            <li><code>Jitter</code> (频率微扰): 周期之间的时间长度变化。反映频率的稳定性。结果为比率 (0-1)，本项目中计算jitter和shimmer的算法参考了https://github.com/Mak-Sim/Troparion/tree/master这一仓库。
                <ul>
                    <li><code>Local</code>: 相邻周期差的平均值。</li>
                    <li><code>RAP</code>: 3点平均相对微扰。</li>
                    <li><code>PPQ5</code>: 5点平均相对微扰。</li>
                </ul>
            </li>
            <li><code>Shimmer</code> (振幅微扰): 周期之间的振幅变化。反映振幅的稳定性。结果为比率 (0-1)。
                <ul>
                    <li><code>Local</code>: 相邻周期振幅差的平均值。</li>
                    <li><code>APQ3</code>: 3点平均振幅微扰。</li>
                    <li><code>APQ5</code>: 5点平均振幅微扰。</li>
                    <li><code>APQ11</code>: 11点平均振幅微扰。</li>
                </ul>
            </li>
        </ul>

        <h3>7. 其他参数</h3>
        <ul>
            <li><code>Intensity</code>: 信号强度 (dB)。</li>
            <li><code>SOE</code> (Strength of Excitation): 激励强度。基于零频率滤波 (ZFF) 计算，反映声门闭合瞬间的急剧程度。该算法参考了VoiceSauce中的函数</li>
        </ul>
        """

        label = QtWidgets.QLabel(html)
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        content_layout.addWidget(label)
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
