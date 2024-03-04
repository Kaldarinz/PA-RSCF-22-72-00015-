"""
Contain old classes, which are not used any more.
"""

def upd_plot(
        base_widget: MplCanvas|MplNavCanvas,
        ydata: Iterable,
        xdata: Iterable | None=None,
        ylabel: str | None=None,
        xlabel: str | None=None,
        fmt: str | None=None,
        marker: Iterable | None=None,
        enable_pick: bool=False
    ) -> None:
    """
    Update a plot.
    
    ``widget`` - widget with axes, where to plot,\n
    ``xdata`` - Iterable containing data for X axes. Should not
    be shorter than ``ydata``. If None, then enumeration of 
    ``ydata`` will be used.\n
    ``ydata`` - Iterable containing data for Y axes.\n
    ``fmt`` - Format string.\n
    ``xlabel`` - Optional label for x data.\n
    ``ylabel`` - Optional label for y data.\n
    ``marker`` - Index of marker for data. If this option is
    omitted, then marker is removed.\n
    ``enable_pick`` - enabler picker for main data.
    """

    if isinstance(base_widget, MplNavCanvas):
        widget = base_widget.canvas
    else:
        widget = base_widget

    # Update data
    widget.ydata = ydata
    if xdata is None:
        widget.xdata = np.array(range(len(widget.ydata))) # type: ignore
    else:
        widget.xdata = xdata
    if xlabel is not None:
        widget.xlabel = xlabel
    if ylabel is not None:
        widget.ylabel = ylabel

    # Plot data
    # If plot is empty, create reference to 2D line
    if widget._plot_ref is None:
        # Set format
        if fmt is None:
            fmt = 'r'
        # Actual plot
        widget._plot_ref = widget.axes.plot(
            widget.xdata,
            widget.ydata,
            fmt,
            picker = enable_pick,
            pickradius = 10
        )[0]
    # otherwise just update data
    else:
        widget._plot_ref.set_xdata(widget.xdata) # type: ignore
        widget._plot_ref.set_ydata(widget.ydata) # type: ignore
    # Update SetPoint if necessary
    if widget.sp is not None:
        spdata = [widget.sp]*len(widget.xdata) # type: ignore
        if widget._sp_ref is None:
            widget._sp_ref = widget.axes.plot(
                widget.xdata,
                spdata,
                'b'
            )[0]
        else:
            widget._sp_ref.set_xdata(widget.xdata) # type: ignore
            widget._sp_ref.set_ydata(spdata)

    # Set markers
    if marker is not None:
        if widget._marker_ref is None:
            # Marker Style
            marker_style = {
                'marker': 'o',
                'alpha': 0.4,
                'ms': 12,
                'color': 'yellow',
                'linewidth': 0
            }
            widget._marker_ref, = widget.axes.plot(
                widget.xdata[marker], # type: ignore
                widget.ydata[marker], # type: ignore
                **marker_style
            )
        else:
            widget._marker_ref.set_data(
                [widget.xdata[marker]], # type: ignore
                [widget.ydata[marker]] # type: ignore
            )
    else:
        if widget._marker_ref is not None:
            widget._marker_ref.remove()
            widget._marker_ref = None
    # Set labels
    widget.axes.set_xlabel(widget.xlabel) # type: ignore
    widget.axes.set_ylabel(widget.ylabel) # type: ignore
    # Update scale limits
    if not widget.fixed_scales:
        widget.axes.relim()
        widget.axes.autoscale(True, 'both')
    widget.draw()

    # Reset history of navigation toolbar
    if isinstance(base_widget, MplNavCanvas):
        base_widget.toolbar.update()


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
            yerr: Iterable | None=None,
            xerr: Iterable | None=None,
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
            logger.info(yerr)
            self._plot_ref = self.axes.errorbar(
                x = self.xdata,
                y = self.ydata, # type: ignore
                yerr = yerr, # type: ignore
                xerr = xerr, # type: ignore
                fmt = fmt,
                picker = self.enable_pick,
                pickradius = 10
            )[0]
            if yerr is not None:
                lims = self.axes.set_ylim(min(self.ydata - yerr), max(self.ydata + yerr))
                logger.info(lims)
        # otherwise just update data
        else:
            self._plot_ref.set_data(self.xdata, self.ydata) # type: ignore
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
        self.axes.relim()
        self.axes.autoscale(True, 'both')
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
        self.draw()

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
            except DimensionalityError:
                logger.info(f'Setting xunits to {data.u}.')
                self.xunits = f'{data.u}'
            self._xdata = data.m
        else:
            self.xunits = ''
            self._xdata = np.array(data)
    
    @property
    def ydata(self) -> npt.NDArray:
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
            yerr: Iterable | None=None,
            xerr: Iterable | None=None,
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

        self.canvas.plot(
            ydata = ydata,
            xdata = xdata,
            yerr = yerr,
            xerr = xerr,
            ylabel = ylabel,
            xlabel = xlabel,
            fmt = fmt
        )
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
