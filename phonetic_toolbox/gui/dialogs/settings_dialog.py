
from PyQt6 import QtWidgets, QtCore
from phonetic_toolbox.services.settings_service import SettingsService

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(300, 400)
        self.settings = SettingsService()
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- General Settings Group ---
        grp_general = QtWidgets.QGroupBox("常用设置")
        form_general = QtWidgets.QFormLayout()
        
        self.spin_silence_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_silence_threshold.setRange(0.0, 1.0)
        self.spin_silence_threshold.setSingleStep(0.01)
        self.spin_silence_threshold.setDecimals(3)
        
        self.spin_energy_win = QtWidgets.QSpinBox()
        self.spin_energy_win.setRange(1, 1000)
        self.spin_energy_win.setSuffix(" ms")
        
        self.spin_frameshift = QtWidgets.QDoubleSpinBox()
        self.spin_frameshift.setRange(0.1, 1000.0)
        self.spin_frameshift.setSuffix(" ms")
        
        self.spin_windowsize = QtWidgets.QSpinBox()
        self.spin_windowsize.setRange(1, 1000)
        self.spin_windowsize.setSuffix(" ms")
        
        self.spin_smooth_win = QtWidgets.QSpinBox()
        self.spin_smooth_win.setRange(1, 100)
        
        self.chk_only_voiced = QtWidgets.QCheckBox("仅保留浊音 (ZCR判定)")
        
        self.spin_n_periods = QtWidgets.QSpinBox()
        self.spin_n_periods.setRange(1, 100)

        self.spin_num_formants = QtWidgets.QSpinBox()
        self.spin_num_formants.setRange(3, 10)
        
        self.spin_max_formant = QtWidgets.QDoubleSpinBox()
        self.spin_max_formant.setRange(0.0, 10000.0)
        self.spin_max_formant.setSuffix(" Hz")

        form_general.addRow("静音阈值 (0-1):", self.spin_silence_threshold)
        form_general.addRow("能量窗口 (毫秒):", self.spin_energy_win)
        form_general.addRow("帧移 (毫秒):", self.spin_frameshift)
        form_general.addRow("窗宽 (毫秒):", self.spin_windowsize)
        form_general.addRow("平滑点数:", self.spin_smooth_win)
        form_general.addRow("输出过滤:", self.chk_only_voiced)
        form_general.addRow("谐波估计周期数:", self.spin_n_periods)
        form_general.addRow("共振峰数量:", self.spin_num_formants)
        form_general.addRow("共振峰上限 (Hz):", self.spin_max_formant)
        
        grp_general.setLayout(form_general)
        layout.addWidget(grp_general)
        
        # --- REAPER Settings Group ---
        grp_reaper = QtWidgets.QGroupBox("REAPER 参数")
        form_reaper = QtWidgets.QFormLayout()
        
        self.spin_min_f0 = QtWidgets.QDoubleSpinBox()
        self.spin_min_f0.setRange(10.0, 1000.0)
        self.spin_min_f0.setSuffix(" Hz")
        
        self.spin_max_f0 = QtWidgets.QDoubleSpinBox()
        self.spin_max_f0.setRange(50.0, 2000.0)
        self.spin_max_f0.setSuffix(" Hz")
        
        self.chk_hilbert = QtWidgets.QCheckBox("hilbert")
        self.chk_no_highpass = QtWidgets.QCheckBox("no-highpass")
        
        form_reaper.addRow("最低基频:", self.spin_min_f0)
        form_reaper.addRow("最高基频:", self.spin_max_f0)
        
        hbox_chk = QtWidgets.QHBoxLayout()
        hbox_chk.addWidget(self.chk_hilbert)
        hbox_chk.addWidget(self.chk_no_highpass)
        form_reaper.addRow("", hbox_chk)
        
        grp_reaper.setLayout(form_reaper)
        layout.addWidget(grp_reaper)
        
        # --- Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_save = QtWidgets.QPushButton("保存")
        self.btn_cancel = QtWidgets.QPushButton("关闭")
        
        self.btn_save.clicked.connect(self.save_settings)
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addStretch()
        layout.addLayout(btn_layout)

    def load_settings(self):
        # Use get_config_object to ensure we have defaults and types
        cfg_obj = self.settings.get_config_object()
        
        self.spin_silence_threshold.setValue(cfg_obj.silence_threshold)
        self.spin_energy_win.setValue(int(cfg_obj.energy_window_ms))
        self.spin_frameshift.setValue(cfg_obj.frameshift_ms)
        self.spin_windowsize.setValue(int(cfg_obj.windowsize_ms))
        
        self.spin_smooth_win.setValue(cfg_obj.smooth_win_size)
        self.chk_only_voiced.setChecked(cfg_obj.only_voiced)
        self.spin_n_periods.setValue(cfg_obj.n_periods)
        
        self.spin_num_formants.setValue(cfg_obj.num_formants)
        self.spin_max_formant.setValue(cfg_obj.max_formant)
        
        self.spin_min_f0.setValue(cfg_obj.min_f0)
        self.spin_max_f0.setValue(cfg_obj.max_f0)
        
        self.chk_hilbert.setChecked(cfg_obj.reaper_hilbert)
        self.chk_no_highpass.setChecked(cfg_obj.reaper_no_highpass)

    def save_settings(self):
        self.settings.set("silence_threshold", self.spin_silence_threshold.value())
        self.settings.set("energy_window_ms", self.spin_energy_win.value())
        self.settings.set("frameshift_ms", self.spin_frameshift.value())
        self.settings.set("windowsize_ms", self.spin_windowsize.value())
        self.settings.set("smooth_win_size", self.spin_smooth_win.value())
        self.settings.set("only_voiced", self.chk_only_voiced.isChecked())
        self.settings.set("n_periods", self.spin_n_periods.value())
        
        self.settings.set("num_formants", self.spin_num_formants.value())
        self.settings.set("max_formant", self.spin_max_formant.value())
        
        self.settings.set("min_f0", self.spin_min_f0.value())
        self.settings.set("max_f0", self.spin_max_f0.value())
        
        self.settings.set("reaper_hilbert", self.chk_hilbert.isChecked())
        self.settings.set("reaper_no_highpass", self.chk_no_highpass.isChecked())
        
        self.settings.save()
        self.accept()
