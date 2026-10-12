"""Explicitly requested native colour selection. No screen image is persisted."""
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QColorDialog, QPushButton


def color_dialog(parent, initial):
    color = QColor(initial) if isinstance(initial, str) and len(initial) == 7 and initial.startswith('#') else QColor('#657b35')
    if not color.isValid():
        color = QColor('#657b35')
    dialog = QColorDialog(color, parent)
    dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
    dialog.setWindowTitle('吸管取色')
    for button in dialog.findChildren(QPushButton):
        text = button.text().replace('&', '')
        if text == 'Pick Screen Color':
            button.setText('屏幕取色')
        elif text == 'Add to Custom Colors':
            button.setText('保存到自定义颜色')
        elif text == 'OK':
            button.setText('使用颜色')
        elif text == 'Cancel':
            button.setText('取消')
    return dialog


def pick_color(parent, initial):
    dialog = color_dialog(parent, initial)
    try:
        if dialog.exec() == QColorDialog.DialogCode.Accepted:
            return dialog.selectedColor().name(QColor.NameFormat.HexRgb)
        return None
    finally:
        dialog.deleteLater()
