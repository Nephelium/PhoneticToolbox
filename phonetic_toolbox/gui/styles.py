# --- Theme Definitions ---

# Dark Theme for Main Window (黑底白字) - 主页专用
# 注意：这里只针对 #home_page 下的按钮进行特殊大号样式定义，避免影响子页面
DARK_MAIN_STYLESHEET = """
QMainWindow {
    background-color: #121212;
    color: #ffffff;
}
QLabel {
    color: #ffffff;
    background-color: transparent;
}
#home_page QPushButton {
    background-color: #2d2d2d;
    border: 2px solid #404040;
    border-radius: 10px;
    color: #ffffff;
    font-size: 16px;
    font-weight: bold;
    min-height: 50px;
    padding: 10px 18px;
}
#home_page QPushButton:hover {
    background-color: #3d3d3d;
    border-color: #606060;
}
#home_page QPushButton:pressed {
    background-color: #1a1a1a;
    border-color: #505050;
}
"""

# Global Dark Theme for Dialogs (黑底白字) - 对话框和子窗口，灰白配色
# 这里的按钮样式更加精致小巧
GLOBAL_DARK_STYLESHEET = """
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 12px;
}
QMainWindow, QDialog {
    background-color: #1e1e1e;
}
QLabel {
    color: #e0e0e0;
    background-color: transparent;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
    border-radius: 3px;
    padding: 2px 4px;
    min-height: 20px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #007fd4;
}
QTextEdit, QPlainTextEdit, QTextBrowser {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
    border-radius: 3px;
    padding: 4px;
}
QListWidget, QTreeWidget, QTableWidget {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
    border-radius: 3px;
    alternate-background-color: #2d2d2d;
}
QHeaderView::section {
    background-color: #2d2d2d;
    color: #e0e0e0;
    padding: 4px;
    border: 1px solid #3e3e3e;
}
/* 精致小巧的按钮风格 */
QPushButton {
    background-color: #333333;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
    border-radius: 3px;
    padding: 4px 12px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #3e3e3e;
    border-color: #505050;
}
QPushButton:pressed {
    background-color: #1e1e1e;
    border-color: #007fd4;
}
QPushButton:disabled {
    background-color: #252526;
    color: #666666;
    border-color: #2d2d2d;
}
QMenuBar {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QMenu {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
}
QMenu::item:selected {
    background-color: #094771;
    color: #ffffff;
}
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #424242;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

# Light Theme for Main Window (白底黑字) - 主页专用
LIGHT_MAIN_STYLESHEET = """
QMainWindow {
    background-color: #f5f5f5;
    color: #000000;
}
QLabel {
    color: #000000;
    background-color: transparent;
}
#home_page QPushButton {
    background-color: #ffffff;
    border: 2px solid #dcdcdc;
    border-radius: 10px;
    color: #000000;
    font-size: 16px;
    font-weight: bold;
    min-height: 50px;
    padding: 10px 18px;
}
#home_page QPushButton:hover {
    background-color: #f0f0f0;
    border-color: #c0c0c0;
}
#home_page QPushButton:pressed {
    background-color: #e0e0e0;
    border-color: #a0a0a0;
}
"""

# Global Light Theme for Dialogs (白底黑字)
GLOBAL_LIGHT_STYLESHEET = """
QWidget {
    background-color: #ffffff;
    color: #333333;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 12px;
}
QMainWindow, QDialog {
    background-color: #f5f5f5;
}
QLabel {
    color: #333333;
    background-color: transparent;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #cccccc;
    border-radius: 3px;
    padding: 2px 4px;
    min-height: 20px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0078d4;
}
QTextEdit, QPlainTextEdit, QTextBrowser {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #cccccc;
    border-radius: 3px;
    padding: 4px;
}
QListWidget, QTreeWidget, QTableWidget {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #cccccc;
    border-radius: 3px;
}
/* 精致小巧的按钮风格 */
QPushButton {
    background-color: #f8f8f8;
    color: #333333;
    border: 1px solid #cccccc;
    border-radius: 3px;
    padding: 4px 12px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #e8e8e8;
    border-color: #bbbbbb;
}
QPushButton:pressed {
    background-color: #d0d0d0;
    border-color: #0078d4;
}
QPushButton:disabled {
    background-color: #f0f0f0;
    color: #999999;
    border-color: #e0e0e0;
}
QMenuBar {
    background-color: #f5f5f5;
    color: #333333;
}
QMenu {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #cccccc;
}
QMenu::item:selected {
    background-color: #e8e8e8;
}
"""
