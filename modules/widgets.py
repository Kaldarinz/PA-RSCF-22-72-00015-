from typing import Iterable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pint.facets.plain.quantity import PlainQuantity
from PySide6.QtCore import (
    QRunnable
)
from PySide6.QtWidgets import (
    QSpinBox,
    QDoubleSpinBox
)

from modules import ureg, Q_

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
        self._sp_ref: Line2D|None = None
        self._sp: float|None = None
        super().__init__(fig)

    @property
    def xdata(self) -> Iterable|None:
        return self._xdata
    
    @xdata.setter
    def xdata(self, data: Iterable) -> None:
        if isinstance(data, PlainQuantity):
            if self.xlabel is not None:
                data = data.to(self.xlabel)
            self.xlabel = f'{data.u:~.2gP}'
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
            if self.ylabel is not None:
                data = data.to(self.ylabel)
            self.ylabel = f'{data.u:~.2gP}'
            self._ydata = data.m
        else:
            self.ylabel = None
            self._ydata = data

    @property
    def sp(self) -> float|None:
        return self._sp
    
    @sp.setter
    def sp(self, value: float) -> None:
        if self.ylabel is not None:
            value = value.to(self.ylabel)
        else:
            self.ylabel = f'{value.u:~.2gP}'
        self._sp = value.m
            
class QuantSpinBox(QDoubleSpinBox):

    @property
    def quantity(self) -> PlainQuantity:
        units = self.suffix().strip()
        value = self.value()
        result = Q_(value, units)
        return result
    
    @quantity.setter
    def quantity(self, val: PlainQuantity) -> None:

        # For editable it leads to problems with units
        if self.isReadOnly():
            val = val.to_compact()
        self.setSuffix(' ' + f'{val.u:~.2gP}')
        self.setValue(float(val.m))
