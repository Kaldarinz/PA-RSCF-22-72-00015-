"""General purpose utility functions.
"""
import logging
from typing import Iterable

from PySide6.QtCore import (
    Slot
)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT # type: ignore
from InquirerPy import inquirer
import numpy as np

from .gui.widgets import (
    MplCanvas,
    MplNavCanvas
)

logger = logging.getLogger(__name__)

def confirm_action(message: str='') -> bool:
    """CLI confirm action."""

    if not message:
        message = 'Are you sure?'
    confirm = inquirer.confirm(message=message).execute()
    logger.debug(f'"{message}" = {confirm}')
    return confirm

@Slot()
def upd_plot(
        base_widget: MplCanvas|MplNavCanvas,
        ydata: Iterable,
        xdata: Iterable|None = None,
        ylabel: str|None = None,
        xlabel: str|None = None,
        marker: Iterable|None = None,
        enable_pick: bool = False,
        toolbar: NavigationToolbar2QT|None = None
    ) -> None:
    """
    Update a plot.
    
    ``widget`` - widget with axes, where to plot,\n
    ``xdata`` - Iterable containing data for X axes. Should not
        be shorter than ``ydata``. If None, then enumeration of 
        ``ydata`` will be used.\n
        ``ydata`` - Iterable containing data for Y axes.\n
        ``xlabel`` - Optional label for x data.\n
        ``ylabel`` - Optional label for y data.\n
        ``marker`` - Index of marker for data. If this option is
        omitted, then marker is removed.\n
        ``enable_pick`` - enabler picker for main data.\n
        ``toolbar`` - navigation toolbar.
    """

    if isinstance(base_widget, MplNavCanvas):
        widget = base_widget.canvas
    else:
        widget = base_widget

    # Update data
    if xdata is None:
        widget.xdata = np.array(range(len(ydata))) # type: ignore
    else:
        widget.xdata = xdata
    widget.ydata = ydata
    if xlabel is not None:
        widget.xlabel = xlabel
    if ylabel is not None:
        widget.ylabel = ylabel

    # Plot data
    if widget._plot_ref is None:
        widget._plot_ref = widget.axes.plot(
            widget.xdata,
            widget.ydata,
            'r',
            picker = enable_pick,
            pickradius = 10
        )[0]
    else:
        widget._plot_ref.set_xdata(widget.xdata) # type: ignore
        widget._plot_ref.set_ydata(widget.ydata) # type: ignore
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
        widget._marker_ref = None
    widget.axes.set_xlabel(widget.xlabel) # type: ignore
    widget.axes.set_ylabel(widget.ylabel) # type: ignore
    widget.axes.relim()
    widget.axes.autoscale(True, 'both')
    widget.draw()

    # Reset history of navigation toolbar
    if isinstance(base_widget, MplNavCanvas):
        base_widget.toolbar.update()
