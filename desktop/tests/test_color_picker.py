from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QPushButton
from ptb_desktop.color_picker import color_dialog, pick_color


def test_native_picker_has_screen_sampling_and_preserves_cancel():
    app = QApplication.instance() or QApplication([])
    dialog = color_dialog(None, '#123456')
    assert dialog.currentColor().name() == '#123456'
    assert any(button.text() == '屏幕取色' for button in dialog.findChildren(QPushButton))
    dialog.deleteLater()
    QTimer.singleShot(20, lambda: app.activeModalWidget().reject())
    assert pick_color(None, '#123456') is None


def test_native_picker_returns_the_selected_hex():
    app = QApplication.instance() or QApplication([])
    def select():
        dialog = app.activeModalWidget()
        dialog.setCurrentColor(QColor('#abcdef'))
        dialog.accept()
    QTimer.singleShot(20, select)
    assert pick_color(None, 'invalid') == '#abcdef'
