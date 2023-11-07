from typing import Iterable
import os

import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pint.facets.plain.quantity import PlainQuantity
from PySide6.QtCore import (
    QRunnable
)
from PySide6.QtWidgets import (
    QSpinBox,
    QDoubleSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QFileDialog
)
from PySide6.QtGui import (
    QStandardItem,
    QStandardItemModel,
    QAction,
    QContextMenuEvent
)

from modules import ureg, Q_

class MplCanvas(FigureCanvasQTAgg):
    """Single plot MatPlotLib widget."""

    def __init__(self, parent = None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot()
        self._xdata: Iterable|None = None
        self._ydata: Iterable|None = None
        self.xlabel: str|None = None
        self.ylabel: str|None = None
        self._plot_ref: Line2D|None = None
        self._sp_ref: Line2D|None = None
        self._sp: float|None = None
        self._marker_ref: Line2D|None = None
        super().__init__(self.fig)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Context menu."""

        context = QMenu(self)
        save_data = QAction('Export data to *.txt', self)
        context.addAction(save_data)
        save_data.triggered.connect(self._save_data)
        context.exec(event.globalPos())

    def _save_data(self) -> None:
        """Save data from current plot."""

        path = os.getcwd()
        initial_dir = os.path.join(path, 'measuring results')
        filename = QFileDialog.getSaveFileName(
            self,
            caption='Choose a file',
            dir=initial_dir,
            filter='text file (*.txt)'
        )[0]
        if filename:
            data = np.column_stack((self.xdata,self.ydata))
            header = ''
            if self.xlabel is not None:
                header = self.xlabel
            if self.ylabel is not None:
                header = header + '\n' + self.ylabel
            np.savetxt(
                fname = filename,
                X = data,
                header = header
            )


    @property
    def xdata(self) -> npt.NDArray:
        return np.array(self._xdata)
    
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
    def ydata(self) -> npt.NDArray:
        return np.array(self._ydata)
    
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
        """SetPoint property."""
        return self._sp
    
    @sp.setter
    def sp(self, value: PlainQuantity) -> None:
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

class StandartItem(QStandardItem):
    """Element of TreeView."""

    def __init__(self, text: str='') -> None:
        super().__init__()

        self.setText(text)

class TreeInfoWidget(QTreeWidget):
    """Have additional attributes to store metadata."""

    def __init__(self, parent = None) -> None:
        super().__init__(parent)

        self.fmd: QTreeWidgetItem
        "File MetaData"
        self.mmd: QTreeWidgetItem
        "Measurement MetaData"
        self.pmd: QTreeWidgetItem|None = None
        "PointMetaData"