"""Explicitly requested native colour selection. No screen image is persisted."""
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QColorDialog, QPushButton,QLabel
from .native_i18n import native_text


def color_dialog(parent, initial):
    color = QColor(initial) if isinstance(initial, str) and len(initial) == 7 and initial.startswith('#') else QColor('#657b35')
    if not color.isValid():
        color = QColor('#657b35')
    dialog = QColorDialog(color, parent)
    dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
    dialog.setWindowTitle(native_text(parent,'color.title'))
    # Explicit Qt QColorDialog captions, verified against the bundled Qt build.
    # Numeric inputs, color values and user custom swatches are never translated.
    keys={'Pick Screen Color':'color.screen','Add to Custom Colors':'color.addCustom','OK':'color.use','Cancel':'color.cancel',
          'Basic colors':'color.basic','Custom colors':'color.custom','Red:':'color.red','Green:':'color.green','Blue:':'color.blue',
          'Alpha channel:':'color.alpha','Hue:':'color.hue','Sat:':'color.saturation','Val:':'color.value','HTML:':'color.html'}
    for item in [*dialog.findChildren(QPushButton),*dialog.findChildren(QLabel)]:
        key=keys.get(item.text().replace('&',''))
        if key:item.setText(native_text(parent,key))
    return dialog


def pick_color(parent, initial):
    dialog = color_dialog(parent, initial)
    try:
        if dialog.exec() == QColorDialog.DialogCode.Accepted:
            return dialog.selectedColor().name(QColor.NameFormat.HexRgb)
        return None
    finally:
        dialog.deleteLater()
