"""
Compound widgets, which are used as parts in other widgets.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""

from typing import Iterable, Literal, cast
import logging
from dataclasses import fields
import time

import numpy as np
import numpy.typing as npt

from PySide6.QtWidgets import (
    QWidget
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
    DataPoint
)
from .designer import (
    point_data_view_ui
)
from ..constants import (
    POINT_SIGNALS,
    MSMNTS_SIGNALS,
    CURVE_PARAMS
)
from ..pa_data import PaData

from modules import ureg, Q_
rng = np.random.default_rng()
logger = logging.getLogger(__name__)

class PointView(QWidget, point_data_view_ui.Ui_Point_view):
    """Plots for 1D data."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.dp: DataPoint | None = None

        self.connectSignalsSlots()

    def connectSignalsSlots(self) -> None:

        self.cb_detail_select.currentTextChanged.connect(
            lambda _: self.upd_plot()
        )
        self.cb_sample.currentTextChanged.connect(
            lambda _: self.upd_plot()
        )


    def reset_view(self) -> None:
        """Reset widget to default state."""

        self.plot_detail.clear_plot()
        self.cb_detail_select.blockSignals(True)
        self.cb_detail_select.clear()
        self.cb_detail_select.blockSignals(False)
        self.cb_sample.blockSignals(True)
        self.cb_sample.clear()
        self.cb_sample.blockSignals(False)

    def load_dp(self, point: DataPoint) -> None:
        """Load datapoint to the widget."""

        self.reset_view()
        self.dp = point
        # Load samples
        self.cb_detail_select.addItems(
            [key for key in POINT_SIGNALS.keys()]
        )
        self.cb_sample.addItems(
            [key for key in self.dp.raw_data.keys()]
        )
        # Plot data
        self.upd_plot()

    def upd_plot(self) -> None:

        if self.dp is None:
            logger.warning('No data to plot.')
            return
        # signal data type for plotting
        dtype =  POINT_SIGNALS[self.cb_detail_select.currentText()]
        # Sample for plotting
        sample_title = self.cb_sample.currentText()
        # Prepare data for plotting
        detail_data = PaData.point_data_plot(
            point = self.dp,
            dtype = dtype,
            sample = sample_title
        )
        self.plot_detail.plot(**detail_data._asdict())
