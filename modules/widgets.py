from typing import Iterable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pint.facets.plain.quantity import PlainQuantity
from PySide6.QtCore import (
    QRunnable
)

class MplCanvas(FigureCanvasQTAgg):
    """Single plot MatPlotLib widget."""
    def __init__(self, parent = None):
        fig = Figure()
        self.axes = fig.add_subplot()
        self._xdata: Iterable|None = None
        self._ydata: Iterable|None = None
        self.xlabel: str|None = None
        self.ylabel: str|None = None
        self._plot_ref: Line2D|None = None
        super().__init__(fig)

    @property
    def xdata(self) -> Iterable|None:
        return self._xdata
    
    @xdata.setter
    def xdata(self, data: Iterable) -> None:
        if isinstance(data, PlainQuantity):
            self.xlabel = data.u
            self._xdata = data.m
        else:
            self.xlabel = None
            self._xdata = data

    @property
    def ydata(self) -> Iterable|None:
        return self._ydata
    
    @ydata.setter
    def ydata(self, data: Iterable) -> None:
        if isinstance(data, PlainQuantity):
            self.ylabel = data.u
            self._ydata = data.m
        else:
            self.ylabel = None
            self._ydata = data
            