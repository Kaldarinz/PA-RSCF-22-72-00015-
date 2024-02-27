"""
Base widgets, which are used as parts in other widgets.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

from typing import (
    Iterable,
    Literal, 
    cast,
    Sequence
)
import os
import logging
from dataclasses import fields
import time

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

from ..data_classes import (
    MapData,
    ScanLine,
    Position,
    MeasuredPoint,
    QuantRect,
    DataPoint,
    PointIndex,
    Plottable
)

from modules import ureg, Q_
rng = np.random.default_rng()
logger = logging.getLogger(__name__)

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
        if attrs is not None:
            self.fmd = QTreeWidgetItem(self, ['File attributes'])
            self.fmd.setExpanded(True)
            # Add file information
            self._set_items(self.fmd, attrs)

    def set_msmnt_md(self, attrs) -> None:
        """Set measurement metadata from a dataclass."""

        # clear existing information
        if self.mmd is not None:
            index = self.indexOfTopLevelItem(self.mmd)
            while self.topLevelItemCount() > index:
                self.takeTopLevelItem(index)
            self.gpmd = None
            self.spmd = None
        if attrs is not None:
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
        if attrs is not None:
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
        if attrs is not None:
            self.spmd = QTreeWidgetItem(self, ['Signal attributes'])
            self.spmd.setExpanded(True)
            # Add file information
            self._set_items(self.spmd, attrs)

    def _set_items(self, group: QTreeWidgetItem, attrs) -> None:
        """Fill group with info from a dataclass."""

        if group is None:
            return
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
    """
    2D map pyqtgraph widget.
    
    2D square map with fixed aspect ratio and fixed axes, which could 
    not be resized by mouse.
    Designed for plotting MapData instances with differently sized
    'pixels' along one axes. Amount of pixels along this line can also
    vary. Allows drawing selection region and optionally picking of
    individual 'data pixels' or positions. Picking of data pixels
    internally sets selected area to the boundaries of that pixel.\n
    Mouse clicking on the map has 2 mutually exclusive modes:
    * Picking data pixels - Allows one to pick scanned data pixels. 
    * Picking map points - Allows one to pick arbitrary point on the
    map.\n
    Also have optional scatter pointer, which is intended to represent
    current position on the map.\n
    For basic operation just set data using `set_data()`. To enable
    data pixel picking set `pick_pixel_enabled` attribute.

    Signals
    -------
    `selection_changed`:    Emitted, when `selected_area` property is 
                            changed (only when there are some real
                            changes). Send tuple of `PlainQuantity`,
                            which contain x, y, width, height of the
                            new selection. This is also emitted, when
                            selection is removed, in this case None is
                            sent.
    `pixel_selected`:       Emitted, when picking data pixels is
                            enabled (`pick_pixel_enabled` is True).
                            Send PointIndex named tuple, which contains
                            indexes of line and point in this line for
                            the data pixel, which was selected on the
                            map by mouse click.
    `pos_selected`:         Emitted, when picking map points is enabled
                            (`pick_pos_enabled` is True, which is
                            default option). Send position of the
                            picked point.

    Attributes
    ----------
    `data`:                 Optional `MapData` instance, which contain
                            scanned data.\n
    `width`:                `PlainQuantity` - Width of the displayed
                            map area. This value can be larger than
                            width of the scanned data.
    `height`:               `PlainQuantity` - Height of the displayed
                            map area. This value can be larger than
                            height of the scanned data.
    `hlabel`:               `str`. `Property` - horizontal map label,
                            does not contain units.
    `vlabel`:               `str`. `Property` - vertical map label,
                            does not contain units.
    `hunits`:               `str`. `Property` - horizontal axis units.\n
    `vunits`:               `str`. `Property` - vertical axis units.\n
    `selected_area`:        `tuple[PlainQuantity,...]`. `Property`.
                            Contain `x`, `y`, `width`, `height`. Redraw
                            selection and emit `selection_changed`,
                            when new value is set.\n
    `abs_coords`:           `bool`. `Property` - flag for plotting data
                            in absolute coordinates. Remove area
                            selection, when set to `False`.\n
    `pick_pixel_enabled`:   `bool`. `Property` - flag to allow data
                            pixel selection. Dasiables pos picking,
                            when set to `True`.
    `pick_pos_enabled`:     `bool`. `Property` - flag to allow position
                            selection. Disables pixel picking, when set
                            to `True`.
    """

    selection_changed = Signal(object)
    pixel_selected = Signal(PointIndex)
    pos_selected = Signal(Position)

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

        self.width: PlainQuantity = Q_(np.nan, 'mm')
        "Width of the displayed map area."
        self.height: PlainQuantity = Q_(np.nan, 'mm')
        "Height of the displayed map area."
        self._hcoords: npt.NDArray | None = None
        """2D array with horizontal axis coords of scanned points."""
        self._vcoords: npt.NDArray | None = None
        """2D array with vertical axis coords of scanned points."""
        self._signal: npt.NDArray | None = None
        """2D array with values of scanned points."""
        self._plot_ref: pg.PColorMeshItem | None = None
        "Reference to plotted data, which could be irregular map."
        self._pos_ref: pg.ScatterPlotItem | None = None
        "Current position pointer."
        self._sel_ref: pg.RectROI|None = None
        "Selected area for scanning."
        self._bar: pg.ColorBarItem | None = None
        "Reference to color bar."
        self._boundary_ref: QGraphicsRectItem | None = None
        "Reference to boundary."
        self._pick_pos_enabled: bool = True
        "Allow selection of positions."
        self._pick_pixel_enabled: bool = False
        "Allow selection of pixels."

        self.connectSignalSlots()
    
    def connectSignalSlots(self) -> None:
        """Default connection os signals and slots."""

        self.scene().sigMouseClicked.connect(self._select_pos)
  
    def set_data(self, data: MapData) -> None:
        """Set scan data, which will lead to start displaying scan."""

        self.clear_plot()
        self.data = data
        logger.info(f'{self.data.blp=}')
        # Set scan range
        self.set_scanrange(
            width = data.width,
            height = data.height
        )
        # Set labels
        self.hlabel = data.haxis
        self.vlabel = data.vaxis

        self.selection_changed.emit(None)
        if len(data.data):
            self.upd_scan()

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
        self._hcoords = x.to(self.hunits).m
        self._vcoords = y.to(self.vunits).m
        self._signal = z.m
        # Calc relative or abs coords
        if self.abs_coords:
            dh, dv = self._rel_to_abs() # type: ignore
            hcoords = self._hcoords + dh
            vcoords = self._vcoords + dv
        else:
            hcoords = self._hcoords
            vcoords = self._vcoords
        # Plot data
        # New scanned line changes shape of data, therefore we have to
        # create new PColorMeshItem
        if self._plot_ref is not None:
            self.plot_item.removeItem(self._plot_ref)
        self._plot_ref = pg.PColorMeshItem(
            hcoords,
            vcoords,
            self._signal)
        self.plot_item.addItem(self._plot_ref)
        # Add colorbar
        if self._bar is not None:
            self.removeItem(self._bar)
            self._bar = None
        self._bar = pg.ColorBarItem(
            label = 'Signal amplitude',
            #interactive=False,
            rounding=0.1) # type: ignore
        self._bar.setImageItem([self._plot_ref])
        self.addItem(self._bar, 0, 1)

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
                removable = True,
                movable = not self.pick_pixel_enabled,
                resizable = not self.pick_pixel_enabled)
            # Remove resize handles if pixel picking enabled
            if self.pick_pixel_enabled:
                        while len(self._sel_ref.handles) > 0:
                            self._sel_ref.removeHandle(
                                self._sel_ref.handles[0]['item']
                            )
            self._sel_ref.setPen(pg.mkPen('b', width = 3))
            self._sel_ref.hoverPen = pg.mkPen('k', width = 3)
            self.plot_item.addItem(self._sel_ref)
            # Emit signal
            self._sel_ref.sigRegionChanged.connect(
                lambda _: self.selection_changed.emit(self.selected_area)
            )
            return self.selected_area # type: ignore

    def set_cur_pos(self, pos: Position) -> None:
        """Set current position."""

        # Check if position is valid
        if not self.abs_coords:
            pos = pos - self.data.blp # type: ignore
        if ((hcoord:=getattr(pos, self.hlabel.lower())) is None
            or (vcoord:=getattr(pos, self.vlabel.lower())) is None):
            if self._pos_ref is not None:
                self.plot_item.removeItem(self._pos_ref)
                self._pos_ref = None
            return
        hpos = hcoord.to(self.hunits).m
        vpos = vcoord.to(self.vunits).m
        if self._pos_ref is None:
            self._pos_ref = pg.ScatterPlotItem(
                x = [hpos],
                y = [vpos],
                hoverable = False
            )
            self.plot_item.addItem(self._pos_ref)
        else:
            self._pos_ref.setData(
                x = [hpos],
                y = [vpos]
            )

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
        # Manualy remove selection to handle case, when scan is started
        # from selection, made in other scan.
        self._remove_sel()

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

    @Slot()
    def set_abs_scan_pos(self, state: bool) -> None:
        """RePlot data in absolute (State=True) or relative coords."""

        if self.data is None:
            logger.warning('Scan cannot be shifted. Data is missing.')
            return
        # Calculate shift values
        if state:
            dx, dy = self._rel_to_abs() # type: ignore
            if self._vcoords is None or self._hcoords is None:
                logger.warning(
                    'Scan cannot be shifted. Coord array is missing.'
            )
                return
            hcoords = self._hcoords + dx
            vcoords = self._vcoords + dy
        else:
            hcoords = self._hcoords
            vcoords = self._vcoords
        # Shift the data
        if self._plot_ref is None:
            logger.warning(
                'Scan cannot be shifted. Plot reference is mising.'
            )
            return
        self._plot_ref.setData(
            hcoords,
            vcoords,
            self._signal
        )

    def _pick_point(self, event: MouseClickEvent) -> Position:
        """Calculate position of a mouse click."""
        # This return correct coordinates

        click_pos = self.plot_item.getViewBox().mapSceneToView(event.scenePos()) # type: ignore
        click_pos = cast(QPointF, click_pos)
        pos = Position.from_tuples([
            (self.hlabel.lower(), Q_(click_pos.x(), self.hunits)),
            (self.vlabel.lower(), Q_(click_pos.y(), self.vunits))
        ])
        return pos

    def _select_pos(self, event: MouseClickEvent) -> None:
        """
        Select position at given mouse click.
        
        Emit `pos_selected`.
        """

        pos = self._pick_point(event)
        logger.info(f'New position selected {pos}')
        # Convert relative coordinates to absolute if necessary
        if not self.abs_coords:
            pos = pos + self.data.blp # type: ignore
        self.pos_selected.emit(pos)

    def _select_pixel(self, event: MouseClickEvent) -> None:
        """
        Select pixel at given mouse click.
        
        Draw boundary around selected pixel and Emit `pixel_selected`.
        """

        if self.data is None:
            logger.warning('Point cannot be selected. Data is missing.')
            return
        pos = self._pick_point(event)
        point = self.data.point_from_pos(pos)
        if point is None:
            logger.warning(f'Point for {pos} was not found.')
            return
        rect = self.data.point_rect(point)
        self.set_selarea(*rect)
        self.pixel_selected.emit(self.data.point_index(point))

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
        
        Contain: ``x``, ``y``, ``width``, ``height``.
        Redraw area selection and emit `selection_changed`, when new
        value is set.
        """
        if self._sel_ref is None:
            return None
        x, y = Q_(self._sel_ref.pos(), self.hunits)
        width, height = Q_(self._sel_ref.size(), self.vunits)

        return (x, y, width, height)
    @selected_area.setter
    def selected_area(self, area: tuple[PlainQuantity,...]) -> None:
        self.set_selarea(area) # type: ignore

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

    @property
    def pick_pixel_enabled(self) -> bool:
        """Whether data pixel can be selected."""
        return self._pick_pixel_enabled
    @pick_pixel_enabled.setter
    def pick_pixel_enabled(self, state: bool) -> None:
        if self._pick_pixel_enabled:
            if not state:
                self._pick_pixel_enabled = state
                self.scene().sigMouseClicked.disconnect(self._select_pixel)
        else:
            if state:
                self._pick_pixel_enabled = state
                self.pick_pos_enabled = False
                self.scene().sigMouseClicked.connect(self._select_pixel)

    @property
    def pick_pos_enabled(self) -> bool:
        """Whether data position can be selected."""
        return self._pick_pos_enabled
    @pick_pos_enabled.setter
    def pick_pos_enabled(self, state: bool) -> None:
        if self._pick_pos_enabled:
            if not state:
                self._pick_pos_enabled = state
                self.scene().sigMouseClicked.disconnect(self._select_pos)
        else:
            if state:
                self._pick_pos_enabled = state
                self.pick_pixel_enabled = False
                self.scene().sigMouseClicked.connect(self._select_pos)

class PgPlot(pg.PlotWidget):
    """
    Pyqtgraph plot widget.
    
    Implements a widget for plotting line, where data points can be 
    explicitly shown as points (this functionality can toggle on/off).
    Data points can optionally be picked, which will visually highlight
    them and fire `point_picked` signal. The picked point will be
    highlighted regardless of visibility of other points.
    Additionally SetPoint can be set, which will be painted as infinite
    horizontal line at specified y position.

    Signals
    -------
    `point_picked`:     Emitted if picking point is enabled by
                        `enable_pick` and a point on the plot was
                        picked. Send index of picked point. If several
                        points were picked by a single click, index of
                        the first point is sent.

    Attributes
    ----------
    `pick_enabled`:     `bool` if set to `True` allows to select points.\n
    `sel_ind`:          `int|None` index of currently selected point.
                        Marker of the selected point is always visible.\n
    `xdata`:            `NDArray`. `Property` - numerical values of
                        data along X axis. Can be set by any `Plottable`.
                        Automatically set `xunits` based on the dtype.\n
    `ydata`:            `NDArray`. `Property` - numerical values of
                        data along X axis. Can be set by any `Plottable`.
                        Automatically set `yunits` based on the dtype.\n
    `xerr`:             `NDArray|int|float|None`. `Property` - Error
                        along x axis. Can be set by a `Plottable|None`.
                        `None` or empty `NDArray` mean no error,
                        single value the same error bar for all values,
                        array type should have the same size as xdata
                        and provide individual error bars for all
                        points.\n
    `yerr`:             `NDArray|int|float|None`. `Property` - the same
                        as `xerr`, but for y axis.\n
    `xlabel`:           `str`. `Property` - x axis label, does not
                        contain units.\n
    `ylabel`:           `str`. `Property` - y axis label, does not
                        contain units.\n
    `xunits`:           `str`. `Property` - x axis units.\n
    `yunits`:           `str`. `Property` - y axis units.\n
    `sp`:               `PlainQuantity`. `Property` - set point value,
                        which is plotted as infinite horizontal line.                                            
    """

    point_picked = Signal(int)

    def __init__(self, parent=None, background='default', plotItem=None, **kargs):
        super().__init__(parent, background, plotItem, **kargs)

        plot_item = self.getPlotItem()
        if not isinstance(plot_item, pg.PlotItem):
            logger.warning('PgPlot cannot be created.')
            return
        self._plot_item = plot_item
        self._plot_ref: pg.PlotDataItem
        self._scat_ref: pg.ScatterPlotItem
        self._err_ref: pg.ErrorBarItem
        self._sp_ref: pg.InfiniteLine

        self.pick_enabled: bool = False
        'Flag of point selection by mouse.'
        self._sel_ind: npt.NDArray | None = None
        'Index of selected point'

        # Styles
        self._pen_def = pg.mkPen(color = 'l', width = 1)
        self._pen_line = pg.mkPen(color = '#1f77b4', width = 1)
        self._pen_sel = pg.mkPen(color='y', width = 3)
        self._pen_hover = pg.mkPen(color='r')

        self._xdata: np.ndarray | None = None
        self._ydata: np.ndarray | None = None
        self._xerr: npt.NDArray | int | float | None = None
        self._yerr: npt.NDArray | int | float | None = None
        self._points_visible: bool = True

        self.init_plot()

    def init_plot(self) -> None:
        """Initiate plot to default state."""

        # Line
        self._plot_ref = self._plot_item.plot(
            pen = self._pen_line,
            downsampleMethod = 'mean',
            autoDownsample = True
        )
        # Points
        self._scat_ref = pg.ScatterPlotItem(
                x = [],
                y = [],
                hoverable = True,
                hoverPen = self._pen_hover,
                tip = 'x:{x:.3g} y:{y:.3g}'.format
            )
        self._plot_item.addItem(self._scat_ref)
        self._scat_ref.sigClicked.connect(self.pick_point)
        # Error bar
        self._err_ref = pg.ErrorBarItem()
        self._plot_item.addItem(self._err_ref)
        # Setpoint
        self._sp_ref = pg.InfiniteLine(
            angle = 0,
            movable = False
        )

    def plot(
            self,
            ydata: Plottable,
            xdata: Plottable | None=None,
            yerr: Plottable | None=None,
            xerr: Plottable | None=None,
            ylabel: str | None=None,
            xlabel: str | None=None
        ) -> None:
        """
        Set plot data.
        
        Attributes
        ----------
        ``xdata`` - Iterable containing data for X axes. Should not
        be shorter than ``ydata``. If None, then enumeration of 
        ``ydata`` will be used.\n
        ``ydata`` - Iterable containing data for Y axes.\n
        ``yerr`` - Value of y error bar.\n
        ``xerr`` - Value of x error bar.\n
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
        self.xerr = xerr
        self.yerr = yerr
        if xlabel is not None:
            self.xlabel = xlabel
        if ylabel is not None:
            self.ylabel = ylabel
        # Reset marker
        self.sel_ind = None

        # Plot data
        self._plot_ref.setData(
            x = self.xdata,
            y = self.ydata)
        
        self._scat_ref.setData(
            x = self.xdata,
            y = self.ydata
        )
        self.set_pts_visible(self._points_visible)
        # Plot error bars
        self.set_errorbars()

    def set_errorbars(self) -> None:

        self._err_ref.setData(
            x = self.xdata,
            y = self.ydata,
            width = self.xerr,
            height = self.yerr
        )

    def set_marker(self, marker: int | npt.NDArray | None) -> None:
        """
        Add marker to the plot.
        
        Atrributes
        ----------
        `marker` - index of data.
        """

        self.sel_ind = marker
        pens = np.full(len(self.xdata), self._pen_def, dtype=object)
        sizes = np.full(len(self.xdata), 7)
        if (ind:=self.sel_ind) is not None:
            pens[ind] = self._pen_sel
            sizes[ind] = 20 
        if self._scat_ref is not None:
            self._scat_ref.setPen(pens)
            self._scat_ref.setSize(sizes)
        self.set_pts_visible(self._points_visible)

    def clear_plot(self) -> None:
        """Restore plot to dafault state."""

        self._plot_item.clear()
        self.init_plot()

    def pick_point(
            self,
            scat_item: pg.ScatterPlotItem,
            points: list[pg.SpotItem],
            ev: MouseClickEvent
        ) -> None:
        """Data pick event handler."""

        if self.pick_enabled:
            # Set index of the first point as selected index
            x_sel = points[0].pos().x()
            self.sel_ind = np.argwhere(self.xdata == x_sel)[0][0]
            self.point_picked.emit(self.sel_ind[0]) # type: ignore

    def set_pts_visible(self, visible: bool) -> None:
        """Set visibality of data points."""

        self._points_visible = visible
        if self.sel_ind is not None:
            vis_vals = np.full(len(self.xdata), None, dtype=object)
            vis_vals[self.sel_ind] = True
        else:
            vis_vals = visible
        self._scat_ref.setPointsVisible(vis_vals)

    def lock_scales(self) -> None:
        # Set square area
        self._plot_item.setAspectLocked() # type: ignore
        # Disabe axes movements
        self._plot_item.setMouseEnabled(x=False, y=False)
        # Hide autoscale button
        self._plot_item.hideButtons()

    @property
    def xdata(self) -> npt.NDArray:
        """
        Numerical values of data along X axis.
        
        Can be set by any `Plottable`. Automatically set `xunits` based
        on the dtype.
        """
        if self._xdata is None:
            logger.warning('xdata is not set!')
            return np.array([])
        return self._xdata
    @xdata.setter
    def xdata(self, data: Plottable) -> None:
        if isinstance(data, (int, float)):
            self.xunits = ''
            self._xdata = np.array(data)
        elif isinstance(data, PlainQuantity):
            data = data.to_preferred()
            try:
                data = data.to(self.xunits)
            except DimensionalityError:
                self.xunits = str(data.u)
            self._xdata = np.array(data.m)
        elif isinstance(data[0], PlainQuantity):
            data = Q_.from_list(list(data))
            data = data.to_preferred()
            try:
                data = data.to(self.xunits)
            except DimensionalityError:
                self.xunits = str(data.u)
            self._xdata = np.array(data.m)
        else:
            self.xunits = ''
            self._xdata = np.array(data)

    @property
    def ydata(self) -> npt.NDArray:
        """
        Numerical values of data along X axis.
        
        Can be set by any `Plottable`. Automatically set `xunits` based
        on the dtype.
        """
        if self._ydata is None:
            logger.warning('ydata is not set!')
            return np.array([])
        return self._ydata
    @ydata.setter
    def ydata(self, data: Plottable) -> None:
        if isinstance(data, (int, float)):
            self.yunits = ''
            self._ydata = np.array(data)
        elif isinstance(data, PlainQuantity):
            data = data.to_preferred()
            try:
                data = data.to(self.yunits)
            except DimensionalityError:
                self.yunits = str(data.u)
            self._ydata = np.array(data.m)
        elif isinstance(data[0], PlainQuantity):
            data = Q_.from_list(list(data))
            data = data.to_preferred()
            try:
                data = data.to(self.yunits)
            except DimensionalityError:
                self.yunits = str(data.u)
            self._ydata = np.array(data.m)
        else:
            self.yunits = ''
            self._ydata = np.array(data)

    @property
    def xerr(self) -> npt.NDArray | int | float | None:
        """
        Error along x axis. Can be set by a `Plottable|None`.
        
        `None` or empty `NDArray` mean no error,
        single value - the same error bar for all values,
        array type should have the same size as xdata
        and provide individual error bars for all
        points.
        """
        return self._xerr
    @xerr.setter
    def xerr(self, data: Plottable | None) -> None:
        if data is None or isinstance(data, (int, float, np.ndarray)):
            self._xerr = data
        elif isinstance(data, PlainQuantity) or isinstance(data[0], PlainQuantity):
            if isinstance(data, Sequence):
                data = Q_.from_list(list(data))
            try:
                data = data.to(self.xunits)
            except DimensionalityError:
                logger.warning('Wrong units for x error.')
                self._xerr = None
                return
            self._xerr = data.m

    @property
    def yerr(self) -> npt.NDArray | int | float | None:
        """
        Error along y axis. Can be set by a `Plottable|None`.
        
        `None` or empty `NDArray` mean no error,
        single value - the same error bar for all values,
        array type should have the same size as xdata
        and provide individual error bars for all
        points.
        """
        return self._yerr
    @yerr.setter
    def yerr(self, data: Plottable | None) -> None:
        if data is None or isinstance(data, (int, float, np.ndarray)):
            self._yerr = data
        elif isinstance(data, PlainQuantity) or isinstance(data[0], PlainQuantity):
            if isinstance(data, Sequence):
                data = Q_.from_list(list(data))
            try:
                data = data.to(self.yunits)
            except DimensionalityError:
                logger.warning('Wrong units for y error.')
                self._yerr = None
                return
            self._yerr = data.m

    @property
    def xlabel(self) -> str:
        "x label, does not include units."
        return self._plot_item.getAxis('bottom').labelText
    @xlabel.setter
    def xlabel(self, label: str) -> None:
        self._plot_item.setLabel('bottom', label, self.xunits)

    @property
    def ylabel(self) -> str:
        """y label, does not include units."""
        return self._plot_item.getAxis('left').labelText
    @ylabel.setter
    def ylabel(self, label: str) -> None:
        self._plot_item.setLabel('left', label, self.yunits)

    @property
    def xunits(self) -> str:
        """Units of x axis."""
        return self._plot_item.getAxis('bottom').labelUnits
    @xunits.setter
    def xunits(self, units: str) -> None:
        self._plot_item.setLabel('bottom', self.xlabel, units)

    @property
    def yunits(self) -> str:
        """Units of y axis."""
        return self._plot_item.getAxis('left').labelUnits
    @yunits.setter
    def yunits(self, units: str) -> None:
        self._plot_item.setLabel('left', self.ylabel, units)
    
    @property
    def sp(self) -> PlainQuantity | None:
        """Set point value."""
        if self._sp_ref is None:
            return None
        return Q_(self._sp_ref.value(), self.yunits)
    @sp.setter
    def sp(self, value: PlainQuantity) -> None:
        if not len(self.yunits):
            return
        self._sp_ref.setValue(value.to(self.yunits).m)

    @property
    def sel_ind(self) -> npt.NDArray | None:
        """Indexes of selected points."""
        return self._sel_ind
    @sel_ind.setter
    def sel_ind(self, val: int | npt.NDArray[np.int_] | None) -> None:
        if np.issubdtype(type(val), np.integer):
            val = np.array([val])
        self._sel_ind = val # type: ignore