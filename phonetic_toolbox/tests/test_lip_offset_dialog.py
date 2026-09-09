from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from phonetic_toolbox.gui.widgets.lip_gui import LipOffsetAdjustDialog


_QT_APP = QApplication.instance() or QApplication([])


def _dialog() -> LipOffsetAdjustDialog:
    sample_rate = 8000
    audio = np.zeros(sample_rate, dtype=np.int16)
    lip_times = np.linspace(0.0, 1.0, 31)
    lip_open = 0.2 + 0.1 * np.sin(2 * np.pi * lip_times)
    return LipOffsetAdjustDialog(
        audio_array=audio,
        sample_rate=sample_rate,
        lip_times=lip_times,
        lip_metrics={"open": lip_open},
        is_dark=False,
    )


def test_dialog_labels_distinguish_apply_and_zero_offset_save():
    dialog = _dialog()
    try:
        assert dialog.btn_apply.text() == "应用偏移并保存"
        assert dialog.btn_no.text() == "保存但不应用偏移"
        assert "同步校正" in dialog.lbl_limit.text()
    finally:
        dialog.close()


def test_apply_uses_manual_offset():
    dialog = _dialog()
    try:
        dialog.manual_check.setChecked(True)
        dialog.offset_spin.setValue(0.125)
        dialog._on_apply()
        assert dialog.selected_offset_seconds == pytest.approx(0.125)
    finally:
        dialog.close()


def test_save_without_offset_uses_zero():
    dialog = _dialog()
    try:
        dialog.selected_offset_seconds = 0.125
        dialog._on_skip()
        assert dialog.selected_offset_seconds == pytest.approx(0.0)
    finally:
        dialog.close()
