from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QPushButton,QWidget,QLabel
from ptb_desktop.color_picker import color_dialog, pick_color
from ptb_desktop.native_i18n import NativeLocale


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


def test_picker_captions_follow_the_owner_language_without_changing_color_values():
    app=QApplication.instance() or QApplication([])
    owner=QWidget();owner.native_locale=NativeLocale()
    for code,title,screen,red in [('zh-CN','吸管取色','屏幕取色','红色：'),('en','Eyedropper','Pick screen color','Red:'),('ja','スポイト','画面から色を取得','赤：')]:
        owner.native_locale.select(code);dialog=color_dialog(owner,'#123456')
        assert dialog.windowTitle()==title
        assert any(button.text()==screen for button in dialog.findChildren(QPushButton))
        assert any(label.text()==red for label in dialog.findChildren(QLabel))
        assert dialog.currentColor().name()=='#123456'
        dialog.deleteLater()
    owner.deleteLater()
