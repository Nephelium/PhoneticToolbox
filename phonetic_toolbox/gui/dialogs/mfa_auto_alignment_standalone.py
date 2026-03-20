import multiprocessing
import os
import sys

from PyQt6.QtWidgets import QApplication


def _prepare_import_path():
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(
        os.path.join(current_file, "..", "..", "..", "..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main():
    _prepare_import_path()
    from phonetic_toolbox.gui.dialogs.mfa_auto_alignment_dialog import (
        MFAAutoAlignmentDialog,
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dialog = MFAAutoAlignmentDialog()
    dialog.set_theme(True)
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
