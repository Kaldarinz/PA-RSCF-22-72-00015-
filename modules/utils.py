"""General purpose utility functions.
"""
import logging
from typing import Iterable

from PySide6.QtCore import (
    Slot
)
from PySide6.QtWidgets import (
    QPushButton
)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT # type: ignore
from pint.facets.plain.quantity import PlainQuantity
import numpy as np

from .gui.widgets import (
    MplCanvas,
    MplNavCanvas
)

logger = logging.getLogger(__name__)

def upd_plot(
        base_widget: MplCanvas|MplNavCanvas,
        ydata: Iterable,
        xdata: Iterable|None = None,
        ylabel: str|None = None,
        xlabel: str|None = None,
        fmt: str|None = None,
        marker: Iterable|None = None,
        enable_pick: bool = False
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

def form_quant(quant: PlainQuantity) -> str:
    """Format str representation of a quantity."""

    return f'{quant:~.2gP}'

def btn_set_silent (btn: QPushButton, state: bool) -> None:
    """Set state of a button without triggering signals."""

    btn.blockSignals(True)
    btn.setChecked(state)
    btn.blockSignals(False)

def __own_properties(cls: type) -> list[str]:
    return [
        key
        for key, value in cls.__dict__.items()
        if isinstance(value, property)
    ]

def properties(cls: type) -> list[str]:
    """Return list of all properties of a class."""
    props = []
    for kls in cls.mro():
        props += __own_properties(kls)
    
    return props

def propvals(instance: object) -> dict:
    """Return values off all properties of a class."""

    props = properties(type(instance))

    return {prop: getattr(instance, prop) for prop in props}