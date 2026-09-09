from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton

from phonetic_toolbox.gui.main_window import MainWindow
from phonetic_toolbox.gui.widgets.phonation_synthesis_widget import (
    PhonationSynthesisWidget,
)
from phonetic_toolbox.models.phonation_synthesis_models import (
    F0Backend,
)


_QT_APP = QApplication.instance() or QApplication([])


def _app() -> QApplication:
    return _QT_APP


def test_widget_constructs_with_expected_defaults():
    _app()
    widget = PhonationSynthesisWidget()
    try:
        analysis = widget.analysis_config()
        generation = widget.generation_config()

        assert widget.windowTitle() == "发声类型合成 - Phonetic Toolbox v2"
        assert analysis.f0_backend == F0Backend.PARSELMOUTH
        assert analysis.target_sample_rate == 11025
        assert analysis.frame_length == 128
        assert analysis.frame_shift == 32
        assert analysis.lpc_order == 20
        assert generation.step_count == 9
        assert generation.energy_match is True
        assert generation.normalize_to_source is True
    finally:
        widget.close()


def test_widget_exposes_full_user_workflow_buttons():
    _app()
    widget = PhonationSynthesisWidget()
    try:
        texts = {button.text() for button in widget.findChildren(QPushButton)}
        assert {
            "源音频",
            "目标音频",
            "输出目录",
            "参数说明",
            "提取 F0",
            "应用编辑",
            "保存 F0 CSV",
            "生成当前",
            "生成全部",
            "取消任务",
        } <= texts
    finally:
        widget.close()


def test_busy_state_disables_mutating_actions_and_enables_cancel():
    _app()
    widget = PhonationSynthesisWidget()
    try:
        widget.set_busy(True, "测试任务")
        assert widget.extract_button.isEnabled() is False
        assert widget.generate_selected_button.isEnabled() is False
        assert widget.generate_all_button.isEnabled() is False
        assert widget.cancel_button.isEnabled() is True

        widget.set_busy(False, "完成")
        assert widget.extract_button.isEnabled() is True
        assert widget.generate_selected_button.isEnabled() is True
        assert widget.generate_all_button.isEnabled() is True
        assert widget.cancel_button.isEnabled() is False
    finally:
        widget.close()


def test_theme_updates_plot_background():
    _app()
    widget = PhonationSynthesisWidget()
    try:
        widget.set_theme(True)
        assert widget.plot.figure.get_facecolor()[:3] == (0.11764705882352941,) * 3
        widget.set_theme(False)
        assert widget.plot.figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    finally:
        widget.close()


def test_main_window_contains_phonation_synthesis_entry():
    _app()
    window = MainWindow()
    try:
        button_texts = {button.text() for button in window.home_page.findChildren(QPushButton)}
        assert "发声类型合成" in button_texts
        assert hasattr(window, "on_phonation_synthesis")
    finally:
        window.close()
