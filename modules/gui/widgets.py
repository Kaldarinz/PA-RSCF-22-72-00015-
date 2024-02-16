"""
Base widgets, which are used as parts in other widgets.
"""

from typing import Iterable, Literal, cast
import os
import logging
from dataclasses import fields

import numpy as np
import numpy.typing as npt
from pint.facets.plain.quantity import PlainQuantity
from pint.errors import DimensionalityError
from PySide6.QtCore import (
    QRunnable,
    Signal,
    QRectF,
    QPointF
)
from PySide6.QtWidgets import (
    QSpinBox,
    QDoubleSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QGraphicsRectItem
)
from PySide6.QtGui import (
    QStandardItem,
    QStandardItemModel,
    QAction,
    QContextMenuEvent
)
from PySide6.QtCore import (
    Signal,
    Slot
)
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import (
    MouseClickEvent
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle, Patch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT # type: ignore
from matplotlib.backend_bases import MouseEvent

from ..data_classes import (
    MapData,
    ScanLine,
    Position,
    MeasuredPoint,
    QuantRect
)
from modules import ureg, Q_
rng = np.random.default_rng()
logger = logging.getLogger(__name__)

class MplCanvas(FigureCanvasQTAgg):
    """Single plot MatPlotLib widget."""

    def __init__(self, parent=None):
        self.fig = Figure()
        self.fig.set_tight_layout(True) # type: ignore
        self.axes = self.fig.add_subplot()
        self._xdata: np.ndarray | None = None
        self._ydata: np.ndarray | None = None
        self.xunits: str = ''
        self.yunits: str = ''
        self._xlabel: str = ''
        self._ylabel: str = ''
        self._plot_ref: Line2D|None = None
        self._sp_ref: Line2D|None = None
        self._sp: float|None = None
        self._marker_ref: Line2D|None = None
        self.fixed_scales: bool = False
        self.enable_pick: bool = False
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

    def plot(
            self,
            ydata: Iterable,
            xdata: Iterable | None=None,
            ylabel: str | None=None,
            xlabel: str | None=None,
            fmt: str | None=None
        ) -> None:
        """
        Update a plot.
        
        Attributes
        ----------
        ``xdata`` - Iterable containing data for X axes. Should not
        be shorter than ``ydata``. If None, then enumeration of 
        ``ydata`` will be used.\n
        ``ydata`` - Iterable containing data for Y axes.\n
        ``fmt`` - Format string.\n
        ``xlabel`` - Optional label for x data.\n
        ``ylabel`` - Optional label for y data.
        """

        # Update data
        self.ydata = ydata
        if xdata is None:
            self.xdata = np.arange(self.ydata.size) # type: ignore
        else:
            self.xdata = xdata
        if xlabel is not None:
            self.xlabel = xlabel
        if ylabel is not None:
            self.ylabel = ylabel

        # Plot data
        # If plot is empty, create reference to 2D line
        if self._plot_ref is None:
            # Set format
            if fmt is None:
                fmt = 'r'
            # Actual plot
            self._plot_ref = self.axes.plot(
                self.xdata,
                self.ydata,
                fmt,
                picker = self.enable_pick,
                pickradius = 10
            )[0]
        # otherwise just update data
        else:
            self._plot_ref.set_xdata(self.xdata)
            self._plot_ref.set_ydata(self.ydata)
        # Update SetPoint if necessary
        if self.sp is not None:
            spdata = np.ones_like(self.xdata)*self.sp.m
            if self._sp_ref is None:
                self._sp_ref = self.axes.plot(
                    self.xdata,
                    spdata,
                    'b'
                )[0]
            else:
                self._sp_ref.set_xdata(self.xdata)
                self._sp_ref.set_ydata(spdata)

        # Set labels
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        # Update scale limits
        if not self.fixed_scales:
            self.axes.relim()
            self.axes.autoscale(True, 'both')
        self.draw()

    def set_marker(self, markers: list[int]) -> None:
        """
        Add markers to the plot.
        
        Atrributes
        ----------
        `markers` - list of data indexes.
        """

        if self._plot_ref is None:
            return
        
        if self._marker_ref is None:
            # Marker Style
            marker_style = {
                'marker': 'o',
                'alpha': 0.4,
                'ms': 12,
                'color': 'yellow',
                'linewidth': 0
            }
            self._marker_ref, = self.axes.plot(
                self.xdata[markers],
                self.ydata[markers],
                **marker_style
            )
        else:
            self._marker_ref.set_data(
                [self.xdata[markers]],
                [self.ydata[markers]]
            )
        self.draw()

    def clear_plot(self) -> None:
        """Restore plot to dafault state."""

        self._xdata = None
        self._ydata = None
        self.xlabel = ''
        self.ylabel = ''
        if self._plot_ref is not None:
            self._plot_ref.remove()
            self._plot_ref = None
        if self._sp_ref is not None:
            self._sp_ref.remove()
            self._sp_ref = None
        self._sp = None
        if self._marker_ref is not None:
            self._marker_ref.remove()
            self._marker_ref = None

    @property
    def xdata(self) -> npt.NDArray:
        return np.array(self._xdata)  
    @xdata.setter
    def xdata(self, data: Iterable) -> None:
        try:
            iter(data)
        except TypeError:
            logger.warning('Attempt to plot non-terable X data.')
            return
        if isinstance(data, PlainQuantity):
            try:
                data = data.to(self.xunits)
            except:
                self.xunits = f'{data.u:~.2gP}'
            self._xdata = data.m
        else:
            self.xunits = ''
            self._xdata = np.array(data)
    
    @property
    def ydata(self) -> np.ndarray:
        return np.array(self._ydata)   
    @ydata.setter
    def ydata(self, data: Iterable) -> None:
        try:
            iter(data)
        except TypeError:
            logger.warning('Attempt to plot non-terable Y data.')
            return
        if isinstance(data, PlainQuantity):
            try:
                data = data.to(self.yunits)
            except:
                self.yunits = f'{data.u:~.2gP}'
            self._ydata = data.m
        else:
            self.yunits = ''
            self._ydata = np.array(data)

    @property
    def xlabel(self) -> str:
        if not len(self._xlabel):
            if not len(self.xunits):
                return ''
            else:
                return f'[{self.xunits}]'
        else:
            if not len(self.xunits):
                return self._xlabel
            else:
                return f'{self._xlabel}, [{self.xunits}]'
    @xlabel.setter
    def xlabel(self, new_lbl: str) -> None:
        self._xlabel = new_lbl

    @property
    def ylabel(self) -> str:
        if not len(self._ylabel):
            if not len(self.yunits):
                return ''
            else:
                return f'[{self.yunits}]'
        else:
            if not len(self.yunits):
                return self._ylabel
            else:
                return f'{self._ylabel}, [{self.yunits}]'
    @ylabel.setter
    def ylabel(self, new_lbl: str) -> None:
        self._ylabel = new_lbl

    @property
    def sp(self) -> PlainQuantity | None:
        """SetPoint property."""
        if self._sp is not None:
            return Q_(self._sp, self.yunits)
        else:
            return None
    @sp.setter
    def sp(self, value: PlainQuantity) -> None:
        try:
            value = value.to(self.ylabel)
        except:
            logger.error(f' Trying to set wrong SetPoint {value}')
            return
        self._sp = value.m

class MplNavCanvas(QWidget):
    """MPL plot with navigation toolbar."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)

        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def plot(
            self,
            ydata: Iterable,
            xdata: Iterable | None=None,
            ylabel: str | None=None,
            xlabel: str | None=None,
            fmt: str | None=None
        ) -> None:
        """
        Update a plot.
        
        Attributes
        ----------
        ``xdata`` - Iterable containing data for X axes. Should not
        be shorter than ``ydata``. If None, then enumeration of 
        ``ydata`` will be used.\n
        ``ydata`` - Iterable containing data for Y axes.\n
        ``fmt`` - Format string.\n
        ``xlabel`` - Optional label for x data.\n
        ``ylabel`` - Optional label for y data.
        """

        self.canvas.plot(ydata, xdata, ylabel, xlabel, fmt)
        # Reset history of navigation toolbar
        self.toolbar.update()

    def set_marker(self, markers: list[int]) -> None:
        """
        Add markers to the plot.
        
        Atrributes
        ----------
        `markers` - list of data indexes.
        """
        self.canvas.set_marker(markers)

    def clear_plot(self) -> None:
        """Restore plot to dafault state."""
        self.canvas.clear_plot()

class MplMap(FigureCanvasQTAgg):
    """Single plot MatPlotLib widget."""

    selection_changed = Signal()

    def __init__(self, parent=None):
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
            x: PlainQuantity | None=None,
            y: PlainQuantity | None=None
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
            x0: PlainQuantity | None=None,
            y0: PlainQuantity | None=None,
            width: PlainQuantity | None=None,
            height: PlainQuantity | None=None
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
    _stepChanged = Signal()
    quantSet = Signal(PlainQuantity)
    "Emitted when enter button or arrows is pressed. Or sb lost focus."

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)

        self.quantity = ureg(self.text())

        # Emit signal with quantity when editing is finished
        self.editingFinished.connect(
            lambda: self.quantSet.emit(self.quantity)
        )
        self._stepChanged.connect(
            lambda: self.quantSet.emit(self.quantity)
        )

    def set_quantity(self, val: PlainQuantity) -> None:
        """Setter method for quantity."""
        if val.m is not np.nan:
            self.quantity = val
        # else:
        #     self.quantity = Q_(-1, self.suffix().strip())

    @property
    def quantity(self) -> PlainQuantity:
        units = self.suffix().strip()
        value = self.value()
        result = Q_(value, units)
        return result
    
    @quantity.setter
    def quantity(self, val: PlainQuantity) -> None:

        # For editable it leads to problems with units
        if self.isReadOnly() and val.m is not np.nan:
            val = val.to_compact()
        self.setSuffix(' ' + f'{val.u:~.2gP}')
        self.setValue(float(val.m))

    def stepBy(self, step):
        value = self.value()
        super().stepBy(step)
        if self.value() != value:
            self._stepChanged.emit()

class StandartItem(QStandardItem):
    """Element of TreeView."""

    def __init__(self, text: str='') -> None:
        super().__init__()

        self.setText(text)

class TreeInfoWidget(QTreeWidget):
    """Have additional attributes to store metadata."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.fmd: QTreeWidgetItem | None = None
        "File MetaData"
        self.mmd: QTreeWidgetItem | None = None
        "Measurement MetaData"
        self.gpmd: QTreeWidgetItem | None = None
        "General point MetaData"
        self.spmd: QTreeWidgetItem | None = None
        "Signal specific point MetaData"
    
    def clear_md(self) -> None:
        """Clear all metadata."""

        while self.topLevelItemCount():
            self.takeTopLevelItem(0)
        self.fmd = None
        self.mmd = None
        self.gpmd = None
        self.spmd = None
        logger.debug('All md cleared')

    def set_file_md(self, attrs) -> None:
        """Set file metadata from a dataclass."""

        logger.debug('Starting file metadata update.')
        # clear existing information
        if self.fmd is not None:
            while self.topLevelItemCount():
                self.takeTopLevelItem(0)
            self.mmd = None
            self.gpmd = None
            self.spmd = None
            logger.debug('All metadata cleared.')
        # Set top level item as "File attributes" string
        self.fmd = QTreeWidgetItem(self, ['File attributes'])
        self.fmd.setExpanded(True)
        # Add file information
        self._set_items(self.fmd, attrs)
        logger.debug(f'fmd set at {self.indexOfTopLevelItem(self.fmd)}')

    def set_msmnt_md(self, attrs) -> None:
        """Set measurement metadata from a dataclass."""

        # clear existing information
        if self.mmd is not None:
            index = self.indexOfTopLevelItem(self.mmd)
            while self.topLevelItemCount() > index:
                self.takeTopLevelItem(index)
            self.gpmd = None
            self.spmd = None
        # Set top level item as "Measurement attributes" string
        self.mmd = QTreeWidgetItem(self, ['Measurement attributes']) # type: ignore
        self.mmd.setExpanded(True)
        # Add file information
        self._set_items(self.mmd, attrs)

    def set_gen_point_md(self, attrs) -> None:
        """Set general point metadata from a dataclass."""

        # clear existing information
        if self.gpmd is not None:
            index = self.indexOfTopLevelItem(self.gpmd)
            while self.topLevelItemCount() > index:
                self.takeTopLevelItem(index)
            self.spmd = None
        # Set top level item as "Measurement attributes" string
        self.gpmd = QTreeWidgetItem(self, ['General point attributes'])
        self.gpmd.setExpanded(True)
        # Add file information
        self._set_items(self.gpmd, attrs)

    def set_signal_md(self, attrs) -> None:
        """Set signal specific point metadata from a dataclass."""

        # clear existing information
        if self.spmd is not None:
            index = self.indexOfTopLevelItem(self.spmd)
            while self.topLevelItemCount() > index:
                self.takeTopLevelItem(index)
        # Set top level item as "Measurement attributes" string
        self.spmd = QTreeWidgetItem(self, ['Signal attributes'])
        self.spmd.setExpanded(True)
        # Add file information
        self._set_items(self.spmd, attrs)

    def _set_items(self, group: QTreeWidgetItem, attrs) -> None:
        """Fill group with info from a dataclass."""

        for field in fields(attrs):
            # For PointMetadata we should not include data iself
            # in the description
            if field.name in ['data', 'data_raw']:
                continue
            val = getattr(attrs, field.name)
            # Use str rather than repr for list items
            if type(val) == list:
                val = [str(x) for x in val]
            QTreeWidgetItem(group, [field.name, str(val)])

class PgMap(pg.GraphicsLayoutWidget):
    """2d map pyqtgraph widget."""

    selection_changed = Signal(object)

    def __init__(self, parent=None):
        
        super().__init__(parent = parent)
        
        self.plot_item = self.addPlot(0, 0, title="Scan area")
        # Set square area
        self.plot_item.setAspectLocked() # type: ignore
        # Disabe axes movements
        self.plot_item.setMouseEnabled(x=False, y=False)
        # Hide autoscale button
        self.plot_item.hideButtons()

        self.data: MapData|None = None
        """Data to display."""
        self._abs_coords: bool = False

        self.hcoords: npt.NDArray | None = None
        """2D array with horizontal axis coords of scanned points."""
        self.vcoords: npt.NDArray | None = None
        """2D array with vertical axis coords of scanned points."""
        self.signal: npt.NDArray | None = None
        """2D array with values of scanned points."""
        self.width: PlainQuantity = Q_(np.nan, 'mm')
        "Width of the displayed map area."
        self.height: PlainQuantity = Q_(np.nan, 'mm')
        "Height of the displayed map area."
        self._plot_ref: pg.PColorMeshItem | None = None
        "Reference to plotted data, which could be irregular map."
        self._sel_ref: pg.RectROI|None = None
        "Selected area for scanning."
        self._bar: pg.ColorBarItem | None = None
        "Reference to color bar."
        self._boundary_ref: QGraphicsRectItem | None = None
        "Reference to boundary."
        self.pick_enabled: bool = False
        self.scene().sigMouseClicked.connect(self.pick_point)

    def pick_point(self, event: MouseClickEvent) -> None:
        # This return correct coordinates
        if self.data is None:
            return
        click_pos = self._plot_ref.getViewBox().mapSceneToView(event.scenePos()) # type: ignore
        click_pos = cast(QPointF, click_pos)
        pos = Position.from_tuples([
            (self.data.haxis, Q_(click_pos.x(), self.hunits)),
            (self.data.vaxis, Q_(click_pos.y(), self.vunits))
        ])
        point = self.data.point_from_pos(pos)
        print(self.data.point_index(point))
        
    # def point_from_pos(self, pos: Position) -> MeasuredPoint | None:
    #     """
    #     Get Datapoint for a given position.
        
    #     Attributes
    #     ----------
    #     `pos` - Position in relative coordinates.

    #     Return
    #     ------
    #     A datapoint, which is represented as a pixel on plot
    #     and the `pos` is located within this pixel.\n
    #     Relative coordinates are used.
    #     """

    #     if self.data is None:
    #         return
        
    #     sstep = self.data.sstep
    #     # Scan starting point in relative coords
    #     scan_startp = self.data.startp - self.data.blp
    #     # Vector from scan starting point to clicked_pos
    #     scan_rel_pos = pos - scan_startp
    #     # Distance to the pos from start point along slow axis as number of sstep's
    #     pos_sstep = (sstep).dotprod(scan_rel_pos)/sstep.value()**2
    #     line_no = int(pos_sstep)
    #     line = self.data.data[line_no]
    #     # The needed datapoint is the closest to the clicked pos in the line.
    #     point = min(line.raw_data, key = lambda x: (x.pos - self.data.blp - pos).value())
    #     return point

    def select_point(self, point: MeasuredPoint) -> None:
        """Select given point."""

    def get_point_rect(self, point: MeasuredPoint) -> None:
        """Get QuantRect for the point."""

    def set_scanrange(
            self,
            width: PlainQuantity | None=None,
            height: PlainQuantity | None=None
        ) -> None:
        """Set displayed map area."""

        if width is not None:
            # if label already set
            if len(self.hunits):
                try:
                    self.width = width.to(self.hunits)
                except DimensionalityError:
                    logger.warning('Wrong dimension in scan range.')
            # if label was not set, then set it
            else:
                self.width = width
                self.hunits = f'{width.u:~.2gP}'

            # update plot
            if self.width.m is not np.nan:
                self.plot_item.setXRange(0, self.width.m, padding = 0.)

        if height is not None:
            # if label already set
            if len(self.vunits):
                try:
                    self.height = height.to(self.vunits)
                except DimensionalityError:
                    logger.warning('Wrong dimension in scan range.')
            # if label was not set, then set it
            else:
                self.height = height
                self.vunits = f'{height.u:~.2gP}'

            # update plot
            if self.height.m is not np.nan:
                self.plot_item.setYRange(0, self.height.m, padding = 0.)

        logger.debug(f'Plot size set to ({self.width}:{self.height})')
        self._plot_boundary()

    def set_selarea(
            self,
            left: PlainQuantity | None=None,
            bottom: PlainQuantity | None=None,
            width: PlainQuantity | None=None,
            height: PlainQuantity | None=None
        ) -> tuple[PlainQuantity,...]|None:
        """
        Set selected area on plot.
        
        If selection was changed, emit ``selection_Changed`` signal.
        If set was successfull return tuple with (``left``, ``bottom``, 
        ``width``, ``height``), otherwise return ``None``.
        """
        logger.debug(
            f'set_selarea called with:{left=}; {bottom=}; {width=}; {height=}'
        )
        # dict for tracking which value was updated
        updated = dict(
            left = False,
            bottom = False,
            width = False,
            height = False
        )

        scan_width = self.width.to(self.hunits).m
        scan_height = self.height.to(self.vunits).m
        # Selection is already on plot
        if self._sel_ref is not None:
            # Calculate x if left is provided
            if left is not None:
                try:
                    x = float(left.to(self.hunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if x < 0:
                    x = 0.
                if x != self._sel_ref.pos()[0]:
                    updated['left'] = True
            # Otherwise get current x
            else:
                x = self._sel_ref.pos()[0]

            # Calculate y if bottom is provided
            if bottom is not None:
                try:
                    y = float(bottom.to(self.vunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if y < 0:
                    y = 0.
                if y != self._sel_ref.pos()[1]:
                    updated['bottom'] = True
             # Otherwise get current y
            else:
                y = self._sel_ref.pos()[1]

            # Calculate width_m if width is provided
            if width is not None:
                try:
                    width_m = float(width.to(self.hunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if width_m != self._sel_ref.size()[0]:
                    updated['width'] = True
            # Otherwise get current width
            else:
                width_m = self._sel_ref.size()[0]

            # Calculate hight_m if height is provided
            if height is not None:
                try:
                    height_m = float(height.to(self.vunits).m)
                except DimensionalityError:
                    logger.error('Selected area cannot be set. Wrong units.')
                    return
                if height_m != self._sel_ref.size()[1]:
                    updated['height'] = True
            # Otherwise get current height
            else:
                height_m = self._sel_ref.size()[1]

            # check if new vals are correct

            if (x + width_m) > scan_width: # type: ignore
                # First try to correct position if it was changed
                if updated['left']:
                    x = scan_width - width_m # type: ignore
                # Correct size, if position was not changed
                else:
                    width_m = scan_width - x # type: ignore

            if (y + height_m) > scan_height: # type: ignore
                # First try to correct position if it was changed
                if updated['bottom']:
                    y = scan_height - height_m # type: ignore
                # Correct size, if position was not changed
                else:
                    height_m = scan_height - y # type: ignore

            # Update patch
            if updated['left'] or updated['bottom']:
                self._sel_ref.setPos((x, y))
            if updated['width'] or updated['height']:
                self._sel_ref.setSize((width_m, height_m))
        
            # draw and emit signal only if something was actualy changed
            if True in updated.values():
                self.selection_changed.emit(self.selected_area)
            return self.selected_area
        
        # if No selection is already present
        else:
            # All values should be provided for new sel area
            if None in [width, height, width, height]:
                logger.error('Selected area cannot be created.')
                return
            # Convert to map units
            try:
                x = left.to(self.hunits).m # type: ignore
                y = bottom.to(self.vunits).m # type: ignore
                width_m = width.to(self.hunits).m # type: ignore
                height_m = height.to(self.vunits).m # type: ignore
            except DimensionalityError:
                logger.error('Selected area cannot be set. Wrong units.')
                return
            bounds = QRectF(0, 0, scan_width, scan_height) # type: ignore
            self._sel_ref = pg.RectROI(
                (x,y),
                (width_m, height_m),
                maxBounds = bounds,
                removable = True)
            self._sel_ref.setPen(pg.mkPen('b', width = 3))
            self._sel_ref.hoverPen = pg.mkPen('k', width = 3)
            self.plot_item.addItem(self._sel_ref)
            # Emit signal
            self._sel_ref.sigRegionChanged.connect(
                lambda _: self.selection_changed.emit(self.selected_area)
            )
            return self.selected_area # type: ignore

    def set_def_selarea(self) -> tuple[PlainQuantity,...]:
        """
        Set default selected area on plot.
        
        The area is half size of scan range and positioned at the center.
        """

        x = self.width.m/2
        y = self.height.m/2
        width = self.width.m/2
        height = self.height.m/2

        # If No selection is already present
        if self._sel_ref is None:
            bounds = QRectF(0, 0, self.width.m, self.height.m) # type: ignore
            self._sel_ref = pg.RectROI(
                (x/2,y/2),
                (width, height),
                maxBounds = bounds,
                removable = True)
            self._sel_ref.setPen(pg.mkPen('b', width = 3))
            self._sel_ref.hoverPen = pg.mkPen('k', width = 3)
            self.plot_item.addItem(self._sel_ref)
            # connect signal
            self._sel_ref.sigRegionChanged.connect(
                lambda _: self.selection_changed.emit(self.selected_area)
            )
            
        # If selection was already present, try to resize it
        else:
            self._sel_ref.setSize(size=(width, height), center = (x, y))
        
        # Emit signal
        self.selection_changed.emit(self.selected_area)
        return self.selected_area # type: ignore

    def set_selarea_vis(self, state: bool) -> None:
        """
        Toggle visibility of the selected area.
        
        If selection did not exist, create default selection.
        """

        if self._sel_ref is not None:
            self._sel_ref.setVisible(state)
        else:
            self.set_def_selarea()

    def set_data(self, data: MapData) -> None:
        """Set scan data, which will lead to start displaying scan."""

        self.data = data
        # Set scan range
        self.set_scanrange(
            width = data.width,
            height = data.height
        )
        # Manualy remove selection to handle case, when scan is started
        # from selection, made in other scan.
        self._remove_sel()
        self.selection_changed.emit(None)

        if len(data.data):
            self.upd_scan()

    def clear_plot(self) -> None:
        """CLear plot and remove data."""

        if self._plot_ref is not None:
            self.plot_item.removeItem(self._plot_ref)
            self._plot_ref = None
        if self._bar is not None:
            self.removeItem(self._bar)
            self._bar = None
        if self.data is not None:
            self.data = None

    @Slot(ScanLine)
    def upd_scan(
        self,
        line: ScanLine | None=None
    ) -> None:
        """Update scan."""

        if self.data is None:
            logger.error('Scan line cannot be drawn. Data is None.')
            return
        # Convert units
        x, y, z = self.data.get_plot_data(signal='max_amp')
        self.hcoords = x.to(self.hunits).m
        self.vcoords = y.to(self.vunits).m
        self.signal = z.m
        # Calc relative or abs coords
        if self.abs_coords:
            dh, dv = self._rel_to_abs() # type: ignore
            hcoords = self.hcoords + dh
            vcoords = self.vcoords + dv
        else:
            hcoords = self.hcoords
            vcoords = self.vcoords
        # Plot data
        # New scanned line changes shape of data, therefore we have to
        # create new PColorMeshItem
        if self._plot_ref is not None:
            self.plot_item.removeItem(self._plot_ref)
        self._plot_ref = pg.PColorMeshItem(
            hcoords,
            vcoords,
            self.signal)
        self.plot_item.addItem(self._plot_ref)
        # Add colorbar
        if self._bar is not None:
            self.removeItem(self._bar)
            self._bar = None
        self._bar = pg.ColorBarItem(
            label = 'Signal amplitude',
            interactive=False,
            rounding=0.1) # type: ignore
        self._bar.setImageItem([self._plot_ref])
        self.addItem(self._bar, 0, 1)

    @Slot()
    def set_abs_scan_pos(self, state: bool) -> None:

        if self.data is None:
            logger.warning('Scan cannot be shifted. Data is missing.')
            return
        # Calculate shift values
        if state:
            dx, dy = self._rel_to_abs() # type: ignore
            if self.vcoords is None or self.hcoords is None:
                logger.warning(
                    'Scan cannot be shifted. Coord array is missing.'
            )
                return
            hcoords = self.hcoords + dx
            vcoords = self.vcoords + dy
        else:
            hcoords = self.hcoords
            vcoords = self.vcoords
        # Shift the data
        if self._plot_ref is None:
            logger.warning(
                'Scan cannot be shifted. Plot reference is mising.'
            )
            return
        self._plot_ref.setData(
            hcoords,
            vcoords,
            self.signal
        )

    def _rel_to_abs(
            self,
            quant: bool=False
        ) -> tuple[float,float] | tuple[PlainQuantity, PlainQuantity] | None:
        """
        Shift from relative to absolute coordinates.

        Attributes
        ----------
        `quant` if `True` return tuple of PlainQuantity, otherwise
        tuple of floats.\n
        Return
        ------
        Tuple containing (horizontal_shift, vertical_shift).
        """

        if self.data is None:
            logger.warning('Shift cannot be calcultaed. Data is missing.')
            return
        
        # Calculate shifts for both axes
        dx = getattr(self.data.blp, self.hlabel.lower())
        dy = getattr(self.data.blp, self.vlabel.lower())
        if quant:
            return (dx, dy)
        dx = dx.to(self.hunits).m
        dy = dy.to(self.vunits).m

        return (dx, dy)

    def _remove_sel(self) -> None:
        """Remove selection."""

        if self._sel_ref is not None:
            self.plot_item.removeItem(self._sel_ref)
            self._sel_ref = None

    @property
    def hlabel(self) -> str:
        "Horizontal plot label, do not include units."
        return self.plot_item.getAxis('bottom').labelText
    @hlabel.setter
    def hlabel(self, label: str) -> None:
        self.plot_item.setLabel('bottom', label, self.hunits)

    @property
    def vlabel(self) -> str:
        """Vertical plot label, do not include units."""
        return self.plot_item.getAxis('left').labelText
    @vlabel.setter
    def vlabel(self, label: str) -> None:
        self.plot_item.setLabel('left', label, self.vunits)

    @property
    def hunits(self) -> str:
        """Units of horizontal axis."""
        return self.plot_item.getAxis('bottom').labelUnits
    @hunits.setter
    def hunits(self, units: str) -> None:
        self.plot_item.setLabel('bottom', self.hlabel, units)

    @property
    def vunits(self) -> str:
        """Units of vertical axis."""
        return self.plot_item.getAxis('left').labelUnits
    @vunits.setter
    def vunits(self, units: str) -> None:
        self.plot_item.setLabel('left', self.vlabel, units)

    @property
    def selected_area(self) -> tuple[PlainQuantity,...]|None:
        """
        Currently selected area.
        
        Contains: ``x``, ``y``, ``width``, ``height``.
        """
        if self._sel_ref is None:
            return None
        x, y = Q_(self._sel_ref.pos(), self.hunits)
        width, height = Q_(self._sel_ref.size(), self.vunits)

        return (x, y, width, height)
    @selected_area.setter
    def selected_area(self, area: tuple[PlainQuantity|None,...]) -> None:
        self.set_selarea(area)

    @property
    def abs_coords(self) -> bool:
        "Flag for plotting data in absolute coordinates."
        return self._abs_coords
    @abs_coords.setter
    def abs_coords(self, state: bool) -> None:
        if state != self.abs_coords:
            self._remove_sel()
            self.selection_changed.emit(None)
            self._abs_coords = state


    def _plot_boundary(self) -> None:
        """Plot map boundary."""

        if None in [self.width, self.height]:
            return
        
        if self._boundary_ref is not None:
            self.plot_item.removeItem(self._boundary_ref)
            self._boundary_ref = None
        self._boundary_ref = QGraphicsRectItem(
            0,
            0,
            self.width.to(self.hunits).m,
            self.height.to(self.vunits).m
        )
        self._boundary_ref.setPen(pg.mkPen('r', width = 3))
        self.plot_item.addItem(self._boundary_ref)