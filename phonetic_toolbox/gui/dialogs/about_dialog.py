from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextBrowser, QPushButton, QWidget
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon

ABOUT_ASCII_TITLE = (
    " ██████╗ ██╗  ██╗ ██████╗ ███╗   ██╗███████╗████████╗██╗ ██████╗████████╗ ██████╗  ██████╗ ██╗     ██████╗  ██████╗ ██╗  ██╗\n"
    " ██╔══██╗██║  ██║██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██║██╔════╝╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔══██╗██╔═══██╗╚██╗██╔╝\n"
    "██████╔╝███████║██║   ██║██╔██╗ ██║█████╗     ██║   ██║██║        ██║   ██║   ██║██║   ██║██║     ██████╔╝██║   ██║ ╚███╔╝ \n"
    "██╔═══╝ ██╔══██║██║   ██║██║╚██╗██║██╔══╝     ██║   ██║██║        ██║   ██║   ██║██║   ██║██║     ██╔══██╗██║   ██║ ██╔██╗ \n"
    " ██║     ██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   ██║╚██████╗   ██║   ╚██████╔╝╚██████╔╝███████╗██████╔╝╚██████╔╝██╔╝ ██╗\n"
    " ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝"
)

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于 PhoneticToolbox")
        self.resize(560, 680)
        
        # Determine theme
        is_dark = False
        if parent and hasattr(parent, 'is_dark'):
            is_dark = parent.is_dark
            
        # Colors
        bg_color = "#202124" if is_dark else "#ffffff"
        text_color = "#e8eaed" if is_dark else "#333333"
        heading_color = "#ffffff" if is_dark else "#2c3e50"
        highlight_color = "#f39c12" if is_dark else "#e67e22"
        link_color = "#5dade2" if is_dark else "#3498db"
        label_color = "#9aa0a6" if is_dark else "#7f8c8d"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
            }}
            QLabel {{
                color: {heading_color};
            }}
        """)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title Label
        title_label = QLabel(ABOUT_ASCII_TITLE)
        title_label.setTextFormat(Qt.TextFormat.PlainText)
        title_label.setWordWrap(False)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            f"font-family: 'Consolas', 'Courier New', monospace; "
            f"font-size: 9px; font-weight: bold; color: {heading_color};"
        )
        layout.addWidget(title_label)
        
        # HTML Content
        content = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8"/>
    <style>
      body {{ font-family: 'Times New Roman', 'KaiTi', 'Microsoft YaHei', sans-serif; font-size: 14pt; line-height: 1.6; color: {text_color}; }}
      p {{ margin-bottom: 10px; text-align: justify; text-indent: 2em; }}
      .highlight {{ color: {highlight_color}; font-weight: bold; }}
      .footer {{ margin-top: 20px; font-size: 12pt; padding: 0; border: none; background-color: transparent; }}
      .info-row {{ margin-bottom: 8px; display: flex; align-items: center; }}
      .label {{ font-weight: bold; color: {label_color}; margin-right: 10px; min-width: 80px; display: inline-block; }}
      a {{ color: {link_color}; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
    </style>
  </head>
<body>
<div>
    <p>
      本项目脱胎于经典的 MATLAB 版 <b>VoiceSauce</b>。在深度整合两者的基础上，我们希望为语音学的日常研究打造一个既严谨高效又贴心易用的专属工具箱。
    </p>
    <p>
      除了悉数保留原版的经典功能，工具箱还特别引入了 <span class="highlight">REAPER 基频检测算法</span>以及经过优化的 <span class="highlight">SHR 逻辑</span>。面对嘎裂等特殊发声类型时，它能更加敏锐地捕捉极低基频的细节，让声学分析的精度得到切实的提升。
    </p>
    <p>
      为了进一步解放研究者的精力，工具箱还拓展了多项实用的自动化处理模块，涵盖唇形提取、自动标注、EGG 信号分析、语音合成、变速变调、感知实验、语谱图转音频以及普通话转 IPA 等功能。希望这些丰富的拓展件，能陪伴大家顺利开展更加多元的学术探索。
    </p>
  </div>

  <div class="footer">
    <div class="info-row">
      <span class="label">单位</span> 中山大学中国语言文学系（珠海）
    </div>
    <div class="info-row">
      <span class="label">作者</span> 井立文（知乎 @井韶子）
    </div>
    <div class="info-row">
      <span class="label">邮箱</span> <a href="mailto:jinglw3@mail2.sysu.edu.cn">jinglw3@mail2.sysu.edu.cn</a>
    </div>
    <div class="info-row">
      <span class="label">个人主页</span> <a href="https://www.zhihu.com/people/jingshaozi">https://www.zhihu.com/people/jingshaozi</a>
    </div>
  </div>
</body>
</html>"""

        text_browser = QTextBrowser()
        text_browser.setHtml(content)
        text_browser.setOpenExternalLinks(True)
        # Remove frame for cleaner look
        text_browser.setFrameShape(QTextBrowser.Shape.NoFrame)
        # Set transparent background to blend with dialog
        text_browser.setStyleSheet("QTextBrowser { background-color: transparent; border: none; }")
        layout.addWidget(text_browser)
        
        # OK Button
        btn_ok = QPushButton("确定")
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.clicked.connect(self.accept)
        # Style the button
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1abc9c;
            }
        """)
        
        # Button layout to center it
        from PyQt6.QtWidgets import QHBoxLayout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
