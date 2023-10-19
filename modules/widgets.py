from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

class MplCanvas(FigureCanvasQTAgg):
    """Single plot MatPlotLib widget."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure()
        self.axes = fig.add_subplot()
        self.xdata = []
        self.ydata = []
        self._plot_ref: Line2D| None = None
        super().__init__(fig)