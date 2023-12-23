from typing import Iterable, Literal, cast
import os
import logging

import numpy as np
import numpy.typing as npt
from pint.facets.plain.quantity import PlainQuantity
from pint.errors import DimensionalityError
from PySide6.QtCore import (
    QRunnable,
    Signal
)
from PySide6.QtWidgets import (
    QSpinBox,
    QDoubleSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QFileDialog,
    QWidget,
    QVBoxLayout
)
from PySide6.QtGui import (
    QStandardItem,
    QStandardItemModel,
    QAction,
    QContextMenuEvent
)
from PySide6.QtCore import (
    Signal
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle, Patch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT # type: ignore
from matplotlib.backend_bases import MouseEvent

from modules import ureg, Q_

logger = logging.getLogger(__name__)

class MplCanvas(FigureCanvasQTAgg):
    """Single plot MatPlotLib widget."""

    def __init__(self, parent = None):
        self.fig = Figure()
        self.fig.set_tight_layout(True) # type: ignore
        self.axes = self.fig.add_subplot()
        self._xdata: Iterable|None = None
        self._ydata: Iterable|None = None
        self.xlabel: str|None = None
        self.ylabel: str|None = None
        self._plot_ref: Line2D|None = None
        self._sp_ref: Line2D|None = None
        self._sp: float|None = None
        self._marker_ref: Line2D|None = None
        self.fixed_scales: bool = False
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
            if self.xdata is not None and self.ydata is not None:
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
            else:
                logger.warning('Attempt to export empty data!')


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
            #self.xlabel = None
            self._xdata = np.array(data)

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
            #self.ylabel = None
            self._ydata = np.array(data)

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

class MplNavCanvas(QWidget):
    """MPL plot with navigation toolbar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

class MplMap(FigureCanvasQTAgg):
    """Single plot MatPlotLib widget."""

    selection_changed = Signal()

    def __init__(self, parent = None):
        self.fig = Figure()
        self.fig.set_tight_layout(True) # type: ignore
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot()

        # Axes formatting
        self.axes.set_aspect('equal')
        self.axes.tick_params(
            top = True,
            labeltop = True,
            right = True,
            labelright = True
        )

        self.data: np.ndarray|None = None
        self.xlabel: str = 'X, []'
        self.ylabel: str = 'Y, []'
        self._xunits: str = ''
        self._yunits: str = ''
        self.xstep: float|None = None
        self.ystep: float|None = None
        self.xmax: float|None = None
        self.ymax: float|None = None
        self._plot_ref: AxesImage|None = None
        self._sel_ref: Rectangle|None = None

        # events
        self.press_event: MouseEvent|None = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def set_scanrange(
            self,
            x: PlainQuantity|None = None,
            y: PlainQuantity|None = None
        ) -> None:
        """Set scan range."""

        if x is not None:
            # if label already set
            if len(self.xunits):
                try:
                    x = x.to(self.xunits)
                except DimensionalityError:
                    logger.warning('Wrong dimension in scan range.')
                else:
                    self.xmax = float(x.m)
            # if label was not set, then set it
            else:
                self.xmax = float(x.m)
                self.xunits = f'{x.u:~.2gP}'

            # update plot
            if self.xmax is not None:
                self.axes.set_xlim(xmax=self.xmax)

        if y is not None:
            # if label already set
            if len(self.yunits):
                try:
                    y = y.to(self.yunits)
                except DimensionalityError:
                    logger.warning('Wrong dimension in scan range.')
                else:
                    self.ymax = float(y.m)
            # if label was not set, then set it
            else:
                self.ymax = float(y.m)
                self.yunits = f'{y.u:~.2gP}'

            # update plot
            if self.ymax is not None:
                self.axes.set_ylim(ymax=self.ymax)

    def set_selarea(
            self,
            x0: PlainQuantity|None = None,
            y0: PlainQuantity|None = None,
            width: PlainQuantity|None = None,
            height: PlainQuantity|None = None
        ) -> tuple[float,...]|None:
        """
        Set selected area on plot.
        
        If selection was changed, emit ``selectionChanged`` signal.
        If set was successfull return tuple with ``x0``, ``y0``, 
        ``width``, ``height`` as floats.
        """

        # dict for tracking which values were provided by call
        updated = dict(
            x0 = False,
            y0 = False,
            width = False,
            height = False
        )
        # Selection is already on plot
        if self._sel_ref is not None:
            # Get x
            if x0 is not None:
                try:
                    x = float(x0.to(self.xunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if x < 0:
                    x = 0.
                if x != self._sel_ref.get_x(): 
                    updated['x0'] = True
            else:
                x = self._sel_ref.get_x()

            # Get y
            if y0 is not None:
                try:
                    y = float(y0.to(self.yunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if y < 0:
                    y = 0.
                if y != self._sel_ref.get_y():
                    updated['y0'] = True
            else:
                y = self._sel_ref.get_y()

            # Get width_m
            if width is not None:
                try:
                    width_m = float(width.to(self.xunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if width_m != self._sel_ref.get_width():
                    updated['width'] = True
            else:
                width_m = self._sel_ref.get_width()

            # Get hight_m
            if height is not None:
                try:
                    height_m = float(height.to(self.yunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if height_m != self._sel_ref.get_height():
                    updated['height'] = True
            else:
                height_m = self._sel_ref.get_height()

            # check if new vals are correct
            if (x + width_m) > self.xmax: # type: ignore
                # First try to correct position if it was changed
                if updated['x0']:
                    x = self.xmax - width_m # type: ignore
                # Correct size, if position was not changed
                else:
                    width_m = self.xmax - x # type: ignore

            if (y + height_m) > self.ymax: # type: ignore
                # First try to correct position if it was changed
                if updated['y0']:
                    y = self.ymax - height_m # type: ignore
                # Correct size, if position was not changed
                else:
                    height_m = self.ymax - y # type: ignore

            # update patch
            if updated['x0'] or updated['y0']:
                self._sel_ref.set_xy((x, y))
            if updated['width']:
                self._sel_ref.set_width(width_m)
            if updated['height']:
                self._sel_ref.set_height(height_m)
            
            # draw and emit signal only if something was actualy changed
            if True in updated.values():
                self.fig.canvas.draw()
                self.selection_changed.emit()
            return (x, y, width_m, height_m)
        
        # No selection is already present
        else:
            # All values should be provided for new sel area
            if None in [x0, y0, width, height]:
                logger.error('Selected area cannot be created.')
                return
            # Convert to map units
            try:
                x0 = x0.to(self.xunits).m # type: ignore
                y0 = y0.to(self.yunits).m # type: ignore
                width = width.to(self.xunits).m # type: ignore
                height = height.to(self.yunits).m # type: ignore
            except DimensionalityError:
                logger.error('Selected area cannot be set. Wrong units.')
                return
            rect = Rectangle((x0,y0), width, height, alpha = 0.7) # type: ignore
            self._sel_ref = self.axes.add_patch(rect) # type: ignore
            self._sel_ref = cast(Rectangle, self._sel_ref)
            # Emit signal
            self.selection_changed.emit()
            return (x0,y0,width,height) # type: ignore

    def on_press(self, event: MouseEvent) -> None:
        """Callback method for press event."""

        if (event.inaxes == self.axes 
            and self._sel_ref is not None 
            and self._sel_ref.contains(event)[0]
        ):
            self.press_event = event
            self.dx = event.xdata - self._sel_ref.get_x() # type: ignore
            self.dy = event.ydata - self._sel_ref.get_y() # type: ignore

    def on_release(self, event: MouseEvent) -> None:
        """Callback method for mouse btn release."""

        self.press_event = None

    def on_move(self, event: MouseEvent) -> None:
        """Callback method for mouse move."""

        if self.press_event is not None and event.inaxes == self.axes:
            self.set_selarea(
                Q_(event.xdata - self.dx, self.xunits),
                Q_(event.ydata - self.dy, self.yunits)
            )

    @property
    def xunits(self) -> str:
        return self._xunits
    
    @xunits.setter
    def xunits(self, units: str) -> None:
        """Also update label on plot."""
        self._xunits = units
        self._update_label('x')
        self.axes.set_xlabel(self.xlabel)

    @property
    def yunits(self) -> str:
        return self._yunits
    
    @yunits.setter
    def yunits(self, units: str) -> None:
        """Also update label on plot."""
        self._yunits = units
        self._update_label('y')
        self.axes.set_ylabel(self.ylabel)

    @property
    def selected_area(self) -> tuple[PlainQuantity,...]|None:
        """
        Currently selected area.
        
        Contains: ``x0``, ``y0``, ``width``, ``height``.
        """

        if self._sel_ref is None:
            return None
        x0 = Q_(self._sel_ref.get_x(), self.xunits)
        y0 = Q_(self._sel_ref.get_y(), self.yunits)
        width = Q_(self._sel_ref.get_width(), self.xunits)
        height = Q_(self._sel_ref.get_height(), self.yunits)

        return (x0, y0, width, height)

    def _update_label(self, axis: Literal['x','y']) -> None:

        if axis == 'x':
            title = self.xlabel.split('[')[0]
            self.xlabel = title + '[' + self.xunits + ']'
        elif axis == 'y':
            title = self.ylabel.split('[')[0]
            self.ylabel = title + '[' + self.yunits + ']'

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
            logger.warning('Save for map data not implemented!')
            return
            if self.xdata is not None and self.ydata is not None:
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
            else:
                logger.warning('Attempt to export empty data!')

class QuantSpinBox(QDoubleSpinBox):
    stepChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.quantity = ureg(self.text())

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

    def stepBy(self, step):
        value = self.value()
        super().stepBy(step)
        if self.value() != value:
            self.stepChanged.emit()

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