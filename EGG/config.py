# config.py

# --- Default Values ---
DEFAULT_PEAK_PROMINENCE = 0.01
DEFAULT_VALLEY_PROMINENCE = 0.01
DEFAULT_ZOOM_WINDOW_MS = 50.0
DEFAULT_ROI_START = 0.0
DEFAULT_ROI_DURATION = 2.0

# --- Filter Parameters ---
DEFAULT_HIGHPASS_CUTOFF = 25
DEFAULT_LOWPASS_CUTOFF = 1000

# --- Spectrogram Parameters ---
DEFAULT_SPEC_WINDOW_MS = 20.0 # 默认窗宽，单位毫秒
SPEC_WINDOW_OVERLAP_RATIO = 0.75 # 窗口重叠比例 (保持不变或也设为可调)
# --- NEW: Spectrogram color defaults ---
DEFAULT_SPEC_VMIN = -70.0
DEFAULT_SPEC_VMAX = -10.0


# --- Styling ---
# 使用大写常量名
DARK_STYLESHEET = """
    QMainWindow { background-color: #2E2E2E; } QWidget { background-color: #2E2E2E; color: #E0E0E0; font-size: 10pt; }
    QLabel { background-color: transparent; color: #C0C0C0; }
    QLineEdit { background-color: #3C3C3C; border: 1px solid #5A5A5A; padding: 3px; color: #E0E0E0; border-radius: 3px; }
    QPushButton { background-color: #4A4A4A; border: 1px solid #5A5A5A; padding: 5px 10px; min-width: 80px; border-radius: 3px; }
    QPushButton:hover { background-color: #5A5A5A; border: 1px solid #6A6A6A; } QPushButton:pressed { background-color: #3A3A3A; }
    QPushButton:disabled { background-color: #3A3A3A; color: #777777; border: 1px solid #4A4A4A;}
    QCheckBox { spacing: 5px; color: #E0E0E0; }
    QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #5A5A5A; border-radius: 3px; background-color: #3C3C3C; }
    QCheckBox::indicator:checked { background-color: #6366f1; border: 1px solid #6366f1; image: url(none); }
    QCheckBox::indicator:checked:hover { background-color: #4f46e5; }
    QCheckBox::indicator:unchecked:hover { border: 1px solid #7A7A7A; }
    QMenuBar { background-color: #383838; color: #E0E0E0; } QMenuBar::item { background-color: transparent; padding: 4px 8px; }
    QMenuBar::item:selected { background-color: #5A5A5A; } QMenu { background-color: #383838; border: 1px solid #5A5A5A; color: #E0E0E0; }
    QMenu::item:selected { background-color: #5A5A5A; } MplCanvas { background-color: transparent; }
"""

MATPLOTLIB_STYLE_SETTINGS = {
    'axes.edgecolor': '#AAAAAA',
    'axes.labelcolor': '#DDDDDD',
    'xtick.color': '#DDDDDD',
    'ytick.color': '#DDDDDD',
    'grid.color': '#555555',
    'figure.facecolor': '#2E2E2E',
    'axes.facecolor': '#2E2E2E',
    'savefig.facecolor': '#2E2E2E',
    'text.color': '#DDDDDD',
    'figure.subplot.bottom': 0.12,
    'figure.subplot.top': 0.92,
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.96,
    'figure.subplot.hspace': 0.4,
    'figure.subplot.wspace': 0.3
}

# 注意：移除了 plt.style.use('dark_background')
