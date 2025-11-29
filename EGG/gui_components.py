from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy
from matplotlib.patches import Rectangle # For timeline ROI visualization
import matplotlib.lines as mlines # For custom legend in zoom plot if needed

# --- Matplotlib Canvas Widget (Unchanged) ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.patch.set_alpha(0.0)
        self.axes.patch.set_alpha(0.0)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
    def plot(self, *args, **kwargs):
        self.axes.plot(*args, **kwargs)
        self.draw()
    def clear(self):
        self.axes.cla()
    def get_axes(self):
        return self.axes
