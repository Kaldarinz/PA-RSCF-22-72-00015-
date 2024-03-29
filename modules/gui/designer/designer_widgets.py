"""
Widgets, built in Qt Designer, and used as a relatively big parts of
the progamm.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""
import logging
from typing import Literal, cast, Optional
from collections import deque
from datetime import datetime
import math
import re

from PySide6.QtCore import (
    Signal,
    Qt,
    QStringListModel,
    Slot,
    QTimer,
    QRect,
    QItemSelectionModel,
    QItemSelection,
    QModelIndex,
    QThreadPool
)
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QDockWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QProgressDialog,
    QMessageBox,
    QLineEdit,
    QSpinBox
)
from PySide6.QtGui import (
    QPixmap
)
import numpy as np
import pyqtgraph as pg
from pint.facets.plain.quantity import PlainQuantity

from . import (
    verify_measure_ui,
    data_viewer_ui,
    log_dock_widget_ui,
    point_measure_widget_ui,
    curve_measure_widget_ui,
    map_measure_widget_ui,
    pm_monitor_ui,
    curve_data_view_ui,
    motor_control_ui,
    map_data_view_ui,
    tst_dialog_ui
)

from ... import Q_
from ...pa_data import PaData
from ...data_classes import (
    DataPoint,
    Position,
    QuantRect,
    StagesStatus,
    EnergyMeasurement,
    MapData,
    Worker,
    MeasuredPoint,
    Measurement
)
from ...constants import (
    POINT_SIGNALS,
    MSMNTS_SIGNALS,
    CURVE_PARAMS
)
from ...utils import (
    form_quant,
    btn_set_silent,
    set_layout_enabled,
    set_sb_silent,
    start_worker,
    start_cb_worker
)
from ...hardware import utils as hutils

from ..widgets import (
    PgPlot,
    QuantSpinBox
)
from ..compound_widgets import(
    PointView
)

logger = logging.getLogger(__name__)


class PAVerifierDialog(QDialog, verify_measure_ui.Ui_Dialog):
    """Dialog window for verification of a PA measurement."""

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

class DataViewer(QWidget, data_viewer_ui.Ui_Form):
    """Data viewer widget."""

    data_changed = Signal(bool)
    "Value is whether new data is valid PaData (not None)."
    data_opened = Signal()
    "Emitted when `data` property set from None to not None value."
    msmnt_selected = Signal()
    "Emitted when selected measurement is changed."
    point_selected = Signal()
    "Emitted when selected datapoint is changed."
    point_dtype_changed = Signal()
    "Emitted when type of displayed data in selected datapoint is changed."

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self._data: PaData | None=None
        # Create pages for 0D, 1D and 2D data
        self.p_2d = MapView(self)
        self.p_1d = CurveView(self)
        self.p_0d = PointView(self)
        self.sw_view.addWidget(self.p_2d)
        self.sw_view.addWidget(self.p_1d)
        self.sw_view.addWidget(self.p_0d)
        # Create empty page
        self.p_empty = QWidget()
        layout = QVBoxLayout()
        lbl = QLabel('No data to show')
        layout.addWidget(lbl)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.p_empty.setLayout(layout)
        self.sw_view.addWidget(self.p_empty)
        # By default empty page is shown
        self.sw_view.setCurrentWidget(self.p_empty)

        # Currently selected objects (start with s prefix)
        self._s_msmnt: Measurement | None = None # Private for property
        self._s_point: DataPoint | None = None # Private for property

        # Model for file content
        self.content_model = QStringListModel()
        self.lv_content.setModel(self.content_model)

        self.connect_signals_slots()

    def connect_signals_slots(self) -> None:
        """"Connect all signals and slots."""

        self.data_changed.connect(self.data_updated)
        self.data_opened.connect(self.load_data_to_view)
        self.msmnt_selected.connect(self.show_measurement)
        self.point_selected.connect(self.show_point)
        # Measurement selection from interface
        self.lv_content.selectionModel().selectionChanged.connect(
            self.new_sel_msmnt 
        )
        # Measurement selection from code
        self.msmnt_selected.connect(self.set_msmnt_selection)
        
        # Measurement renamed
        self.content_model.dataChanged.connect(
            self._msmnt_renamed
        )

        # Signal selection for point
        self.p_0d.cb_detail_select.currentTextChanged.connect(
            lambda _ : self.set_signal_md(self.p_0d))
        self.p_1d.pv.cb_detail_select.currentTextChanged.connect(
            lambda _ : self.set_signal_md(self.p_1d.pv))
        self.p_2d.pv.cb_detail_select.currentTextChanged.connect(
            lambda _ : self.set_signal_md(self.p_2d.pv))
        
        # Sample select
        self.p_0d.cb_sample.currentTextChanged.connect(
            lambda _ : self.set_signal_md(self.p_0d))
        self.p_1d.pv.cb_sample.currentTextChanged.connect(
            lambda _ : self.set_signal_md(self.p_1d.pv))
        self.p_2d.pv.cb_sample.currentTextChanged.connect(
            lambda _ : self.set_signal_md(self.p_2d.pv))

        # Signal selection for parameter
        self.p_1d.cb_curve_select.currentTextChanged.connect(
            lambda _ : self.plot_param()
        )

        # Data picker in curve plot
        self.p_1d.plot_curve.point_picked.connect(self.pick_event)
        # Data picker for map
        self.p_2d.plot_map.pixel_selected.connect(self.map_point_selected)
        self.p_2d.btn_hres.toggled.connect(self.p_2d.plot_map.set_highres)

    @Slot(bool)
    def data_updated(self, data_exist: bool) -> None:
        """
        Slot, which triggers when data is changed.
        
        Set marker of changed data for title and
        reload whole view.
        """

        # set marker of changed data for title
        data_title = self.lbl_data_title.text()
        if not data_title.endswith('*') and data_exist:
            data_title = data_title + '*'
            self.lbl_data_title.setText(data_title)

        if data_exist:
            self.load_data_to_view()
        else:
            self.clear_view()
            self.lbl_data_title.setText('')

    def load_data_to_view(self) -> None:
        """Load data to view."""

        if self.data is None:
            logger.warning(
                'Data cannot be displayed. Data is missing.'
            )
            return
        data = self.data
        logger.debug('Updating data in viewer...')
        # clear old data view
        self.clear_view()
        # Finish for empty data
        if not len(data.measurements):
            logger.debug('Data contain no measurements.')
            return
        # Set file description
        self.tv_info.set_file_md(data.attrs)
        # Load new file content
        self.content_model.setStringList(
            [key for key in data.measurements.keys()]
        )
        logger.debug(f'{self.content_model.rowCount()} msmnts set.')
        # Select the first msmnt
        self.s_msmnt = next(iter(data.measurements.values()))
        logger.debug('Data updated in viewer.')

    @Slot()
    def set_msmnt_selection(self) -> None:
        """Select `s_msmnt` in the list of measurements."""


        logger.debug('Starting selection of new msmnt in the view.')
        # Do not update selection if s_msmnt is not set
        if not len(s_msmnt_title:=self.s_msmnt_title):
            return
        # Get list of all msmsnts in the file content QListView
        msmnts = self.content_model.stringList()
        for ind, msmnt in enumerate(msmnts):
            if msmnt == s_msmnt_title:
                # QRect for selection must be in content coordinates
                _index = self.content_model.index(0,0)
                # Get reference rect
                ref_rect = self.lv_content.rectForIndex(_index)
                # Calc actual rect
                sel_rect = QRect(
                    0,
                    ind*ref_rect.height(),
                    ref_rect.width(),
                    ref_rect.height())
                # Set selection
                self.lv_content.setSelection(
                    sel_rect,
                    QItemSelectionModel.SelectionFlag.Select
                )
                break

    @Slot(QItemSelection, QItemSelection)
    def new_sel_msmnt(
            self,
            selected: QItemSelection,
            desecled: QItemSelection
        ) -> None:
        """Update `s_msmnt` if new measurement selected in the interface."""

        # Selected index
        try:
            ind = selected.indexes()[0]
        except IndexError:
            logger.info('No selection')
            return
        # Get selected measurement
        model = self.content_model
        msmnt_title = model.data(ind, Qt.ItemDataRole.EditRole)
        if self.data is None:
            logger.warning('Trying to load measurement from empty data.')
            return
        msmnt = self.data.measurements.get(msmnt_title, None)
        if msmnt is None:
            logger.warning(
                f'Trying to load invalid measurement: {msmnt_title}'
            )
            return
        self.s_msmnt = msmnt

    @Slot()
    def show_measurement(self) -> None:
        """
        Show selected measurement.
        
        Automatically called, when new measurement is seleced.
        """
        
        if self.s_msmnt is None:
            self.sw_view.setCurrentWidget(self.p_empty)
            return

        # Set measurement description
        self.tv_info.set_msmnt_md(self.s_msmnt.attrs)

        # Plot data
        if self.s_msmnt.attrs.measurement_dims == 0:
            self.set_point_view()
        elif self.s_msmnt.attrs.measurement_dims == 1:
            self.set_curve_view()
        elif self.s_msmnt.attrs.measurement_dims == 2:
            self.set_map_view()
        else:
            logger.error('Unsupported data dimensionality.')

    def set_point_view(self) -> None:
        """Load point measurement view."""

        # This method is called, when s_msmnt do exist
        msmnt = cast(Measurement, self.s_msmnt)
        # Load point view
        self.sw_view.setCurrentWidget(self.p_0d)
        # Set selected DataPoint
        self.s_point = next(iter(msmnt.data.values()))

    def set_curve_view(self) -> None:
        """Load curve view."""

        # This method is called, when s_msmnt do exist
        msmnt = cast(Measurement, self.s_msmnt)
        # Load curve view
        self.sw_view.setCurrentWidget(self.p_1d)
        # Set selected DataPoint
        self.s_point = next(iter(msmnt.data.values()))
        ## Set main plot ##
        self.plot_param()

    def set_map_view(self) -> None:
        """Load map view."""

        logger.debug('Starting map view loading')
        # Load curve view
        self.sw_view.setCurrentWidget(self.p_2d) 
        if self.data is not None:
            self.p_2d.plot_map.set_data(
                self.data.maps[self.s_msmnt_title]
            )
        self.s_point = None

    def show_point(self) -> None:
        """
        Show selected point.

        Automatically called, when new point is seleced.
        """

        view = self.sw_view.currentWidget()
        if isinstance(view, PointView):
            point_view = view
        elif isinstance(view, CurveView | MapView):
            point_view = view.pv
            point_view = cast(PointView, point_view)
        else:
            logger.warning('Trying to set point data into wrong widget')
            return
        # Set title
        point_view.lbl_title.setText('Title: ' + self.s_point_title)
        if self.s_point is None:
            point_view.reset_view()
            self.tv_info.set_gen_point_md(None)
            return
        
        # Set general point description
        self.tv_info.set_gen_point_md(self.s_point.attrs)
        point_view.load_dp(self.s_point)
        self.set_signal_md(point_view)

    def set_signal_md(self, pv: PointView) -> None:
        """Set signal metadata for the point view."""
        
        # signal data type for plotting
        dtype =  POINT_SIGNALS[pv.cb_detail_select.currentText()]
        # Sample for plotting
        sample_title = pv.cb_sample.currentText()
        # Get dict with dtype data for all samples in data point
        dtype_dct = getattr(self.s_point, dtype)
        # Get relative MD
        sig_attrs = dtype_dct.get(sample_title, None)
        self.tv_info.set_signal_md(sig_attrs)

    @Slot()
    def plot_param(self) -> None:
        """Plot param data."""

        view = self.sw_view.currentWidget()
        view = cast(CurveView, view)
        if self.s_point is None or self.s_msmnt is None:
            logger.debug('Signal cannot be shown. Some info is missing.')
            return
        # signal data type for plotting
        dtype =  MSMNTS_SIGNALS[view.cb_curve_select.currentText()]
        # Plot the data
        param_data = PaData.param_data_plot(
            msmnt = self.s_msmnt,
            dtype = dtype
        )
        view.plot_curve.plot(**param_data._asdict())
        view.plot_curve.set_marker(self.s_point_ind)

    def pick_event(self, sel_ind: int):
        """Callback method for processing data picking on plot."""

        # Update datamarker on plot
        self.p_1d.plot_curve.set_marker(sel_ind) # type: ignore
        # Update index of currently selected data
        dp_name = PaData._build_name(sel_ind + 1) # type: ignore
        self.s_point = self.s_msmnt.data[dp_name] # type: ignore

    def map_point_selected(self, params: list) -> None:
        """Set selected map pixel."""
        self.s_point = PaData.point_by_param(self.s_msmnt, params) # type: ignore
        
    def clear_view(self) -> None:
        """Clear whole viewer."""

        # Reset selected object attributes
        self.s_msmnt = None
        self.s_point = None
        # Clear file content
        self.content_model.setStringList([])
        # Clear information view
        self.tv_info.clear_md()
        # Set empty page for plot view
        self.sw_view.setCurrentWidget(self.p_empty)
        logger.debug('Data view cleared')

    @Slot(MapData)
    def add_map_to_data(self, scan: MapData) -> None:
        """Save scanned data."""

        # create new data structure if necessary
        if self.data is None:
            self.data = PaData()
        self.data.add_map(scan)
        logger.debug('Map data saved.')
        self.data_changed.emit(True)

    @Slot(Measurement)
    def add_curve_to_data(self, curve: Measurement) -> None:
        """Save measured curve."""

        # create new data structure if necessary
        if self.data is None:
            self.data = PaData()
        self.data.append_measurement(curve)
        logger.debug('Curve data saved.')
        self.data_changed.emit(True)

    @Slot(MeasuredPoint)
    def add_point_to_data(self, point: list[MeasuredPoint]) -> None:
        """SAve single point measurement."""

        # create new data structure if necessary
        if self.data is None:
            self.data = PaData()
        msmnt = self.data.add_measurement(dims = 0, params = [])
        if msmnt is not None:
            PaData.add_point(msmnt[1], point, [])
        self.data_changed.emit(True)

    @Slot(QModelIndex, QModelIndex, list)
    def _msmnt_renamed(
            self,
            top_left: QModelIndex,
            bot_right: QModelIndex,
            roles: list
        ) -> None:
        """Change measurement title."""

        # If new title is invalid, set old title back
        if not top_left.data():
            self.content_model.setData(
                top_left,
                self.s_msmnt_title,
                role = Qt.ItemDataRole.EditRole
            )
            return
        # Update data
        msmnts = self.data.measurements # type: ignore
        # I'm not sure if it will actually change title of s_msmnt
        ###### CHECK IT ####
        msmnts[top_left.data()] = msmnts.pop(self.s_msmnt_title)

    @Slot()
    def del_msmsnt(self) -> None:
        """Delete currently selected measurement."""

        # Get currently selected index
        val = self.lv_content.selectedIndexes()[0]
        # Remove measurement from data
        self.data.measurements.pop(val.data()) # type: ignore
        # Trigger update of widgets
        self.data_changed.emit()
        logger.info(f'Measurement: {val.data()} removed.')

    @property
    def s_msmnt(self) -> Measurement | None:
        """
        Selected measurement.

        If new value is set, emit `msmnt_selected` signal.\n
        """
        return self._s_msmnt
    @s_msmnt.setter
    def s_msmnt(self, new_val: Measurement | None) -> None:
        if self._s_msmnt is not None:
            if new_val is None or self._s_msmnt != new_val:
                self._s_msmnt = new_val
                self.msmnt_selected.emit()
        elif self._s_msmnt is None and new_val is not None:
            self._s_msmnt = new_val
            self.msmnt_selected.emit()
        else:
            self._s_msmnt = new_val

    @property
    def s_msmnt_title(self) -> str:
        """
        Title of the selected measurement.
        
        Read only.
        """
        title = ''
        if self.s_msmnt is not None and self.data is not None:
            for key, val in self.data.measurements.items():
                if val == self.s_msmnt:
                    title = key
                    break
        return title

    @property
    def s_point(self) -> DataPoint | None:
        """
        Selected data point.

        If new value is set, emit `point_selected` signal.\n
        """
        return self._s_point
    @s_point.setter
    def s_point(self, new_val: DataPoint | None) -> None:
        if self._s_point is not None:
            if new_val is None or self._s_point != new_val:
                self._s_point = new_val
                self.point_selected.emit()
        elif self._s_point is None and new_val is not None:
            self._s_point = new_val
            self.point_selected.emit()
        else:
            self._s_point = new_val

    @property
    def s_point_title(self) -> str:
        """
        Title of the selected point.
        
        Read only.
        """
        title = ''
        if self.s_point is not None and self.s_msmnt is not None:
            for key, val in self.s_msmnt.data.items():
                if val == self.s_point:
                    title = key
                    break
        return title

    @property
    def s_point_ind(self) -> int:
        """
        Index of the selected point.
        
        Read only.
        """
        ind = 0
        if self.s_point is not None and self.s_msmnt is not None:
            for key, val in self.s_msmnt.data.items():
                if val == self.s_point:
                    # Extract number from point title, which is spected
                    # to be Point001, etc
                    ind = int(re.findall(r'\d+', key)[0]) - 1
                    break
        return ind

    @property
    def data(self) -> PaData | None:
        "PA data."
        return self._data
    @data.setter
    def data(self, value: PaData | None) -> None:
        """Set new PaData and triggers data_viewer update."""
        if isinstance(value, PaData):
            if self._data is not None and self._data != value:
                self._data = value
                self.data_changed.emit(True)
                logger.debug(f'data attribute is set to {value}')
            elif self._data is None:
                self._data = value
                self.data_opened.emit()
                logger.debug(f'data attribute is set to {value}')
        elif value is None and self._data is not None:
            self._data = None
            self.data_changed.emit(False)
            logger.debug(f'data attribute is set to {value}')
        else:
            return

class LoggerWidget(QDockWidget, log_dock_widget_ui.Ui_d_log):
    """Logger dock widget."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class MeasureWidget(QWidget):
    """
    Base class for Curve and Point Measure widgets.
    
    Subclasses should implement the following elements:\n
    `btn_measure`: `QPushButton` - btn, which launches measurement of PA signal.\n
    `sb_sample_sp`: `QuantSpinBox` - value of setpoint energy on the sample.\n
    `sb_pm_sp`: `QuantSpinBox` - value of setpoint energy on the sample.\n
    `pm_monitor`: `PowerMeterMonitor` - Widget for energy monitoring.\n
    `le_sample_en`: `QLineEdit` - Current energy at sample.\n
    `le_pm_en`: `QLineEdit` - Current energy at power meter.\n
    """
    point_measured = Signal(MeasuredPoint)
    "Value indicates whether measured point"
    measure_point = Signal(PlainQuantity)

    btn_measure: QPushButton
    sb_sample_sp: QuantSpinBox
    sb_pm_sp: QuantSpinBox
    pm_monitor: 'PowerMeterMonitor'
    le_sample_en: QLineEdit
    le_pm_en: QLineEdit
    sb_aver: QSpinBox
    cur_rep: int
    sb_cur_param: QuantSpinBox
    data_point: list[MeasuredPoint]

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)

    def verify_msmnt(self, data: MeasuredPoint | None) -> None:
        """Verify measured datapoint."""

        # Check if data was obtained
        if data is None:
            info_dial = QMessageBox()
            info_dial.setWindowTitle('Bad data')
            info_dial.setText('Error during data reading!')
            info_dial.exec()
            self.btn_measure.setEnabled(True)
            return

        self.ver = PAVerifierDialog()
        plot_pa = self.ver.plot_pa
        plot_pm = self.ver.plot_pm

        # Power meter plot
        ydata=data.pm_signal
        xdata=Q_(
            np.arange(len(data.pm_signal))*data.dt_pm.m, # type: ignore
            data.dt_pm.u
        )
        plot_pm.plot(ydata = ydata, xdata = xdata)

        # PA signal plot
        start = data.start_time
        stop = data.stop_time.to(start.u)
        plot_pa.plot(
            ydata = data.pa_signal,
            xdata = Q_(
                np.linspace(
                    start.m,
                    stop.m,
                    len(data.pa_signal), # type: ignore
                    endpoint=True), # type: ignore
                start.u
            ) # type: ignore
        )
        if self.ver.exec():
            self.data_point.append(data)
            if self.cur_rep == self.sb_aver.value(): 
                self.point_measured.emit(self.data_point.copy()) # type: ignore
                self.data_point = []
                self.cur_rep = 1
                return
            else:
                self.cur_rep += 1
                self.measure()
        else:
            logger.info('Point rejected')
            self.measure()
            return

    def measure(self) -> None:
        param = self.sb_cur_param.quantity
        self.measure_point.emit(param) # type: ignore

    def upd_sp(
        self,
        caller: QuantSpinBox) -> None:
        """Update Set Point data."""

        # Update PM SetPoint if Sample SetPoint was set
        if caller == self.sb_sample_sp:
            new_sp = hutils.glan_calc_reverse(caller.quantity)
            if new_sp is None:
                logger.error('New set point cannot be calculated')
                return
            set_sb_silent(self.sb_pm_sp, new_sp)
        # Update Sample SetPoint if PM SetPoint was set   
        elif caller == self.sb_pm_sp:
            new_sp = hutils.glan_calc(caller.quantity)
            if new_sp is None:
                logger.error('New set point cannot be calculated')
                return
            set_sb_silent(self.sb_sample_sp, new_sp)

        # Update SetPoint on plot
        self.pm_monitor.plot_right.sp = self.sb_pm_sp.quantity

    def setup_sp(self) -> None:
        """Setup setpoint controls."""

        # Setpoint initial value
        self.upd_sp(self.sb_sample_sp)
        # Sample energy SetPoint 
        self.sb_sample_sp.valueChanged.connect(
            lambda x: self.upd_sp(self.sb_sample_sp)
        )
        # Power meter energy SetPoint
        self.sb_pm_sp.valueChanged.connect(
            lambda x: self.upd_sp(self.sb_pm_sp)
        )
    
    def setup_pm(self) -> None:
        """Setup energy measurements."""

        # Current PM energy value
        self.pm_monitor.le_cur_en.textChanged.connect(
            self.le_pm_en.setText
        )
        # Current Sample energy value
        self.pm_monitor.le_cur_en.textChanged.connect(
            lambda val: self.le_sample_en.setText(
                str(hutils.glan_calc(Q_(val)))
            )
        )

class CurveMeasureWidget(MeasureWidget, curve_measure_widget_ui.Ui_Curve_measure_widget):
    """1D PhotoAcoustic measurements widget."""

    cur_p_changed = Signal()
    "Signal change of current data point."
    curve_obtained = Signal(Measurement)

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.pm_monitor = PowerMeterMonitor(self)
        self.lo_measure.replaceWidget(
            0,
            self.pm_monitor
        )
        self.points_num: int = 1
        "Amount of data points in measurement."
        self.step: PlainQuantity
        "Parameter step value."
        self.param_points: PlainQuantity
        "Array with all param values."
        self._current_point: int = 0
        self.s_point: DataPoint
        'Currently selected datapoint.'
        self.cur_rep = 1
        self.data_point = []
        self.measurement: Measurement

        self.setup_widgets()
        self.connect_signals_slots()
        self.setup_sp()
        self.setup_pm()

    def setup_widgets(self) -> None:
        """Setup widgets."""

        # Plot params
        self.plot_measurement.pick_enabled = True
        # Set available params to measure
        self.cb_mode.clear()
        self.cb_mode.addItems(list(CURVE_PARAMS))
        self.set_param_widget(self.cb_mode.currentText())

    def connect_signals_slots(self) -> None:
        """Connect signals and slots for the widget."""

        # Auto update widgets when current_point is changed
        self.cur_p_changed.connect(self.upd_widgets)
        # Set param btn
        self.btn_run.clicked.connect(self.setup_measurement)
        # Measure btn clicked
        self.btn_measure.clicked.connect(self.measure)
        # Point measured
        self.point_measured.connect(self.add_point)
        # New parameter choosen
        self.cb_mode.currentTextChanged.connect(self.set_param_widget)
        # Stop btn
        self.btn_stop.clicked.connect(self.curve_finished)
        # Restart btn
        self.btn_redo.clicked.connect(self.redo_msmnt)
        # Data picker in curve plot
        self.plot_measurement.point_picked.connect(
            self.pick_event
        )

    def measure(self) -> None:
        """Start measurement."""
        
        # block measure button
        self.btn_measure.setEnabled(False)
        # Launch measurement
        self.measure_point.emit(self.sb_cur_wl.quantity)

    def redo_msmnt(self) -> None:

        self.current_point -= 1
        self.measure()

    def add_point(self, point: list[MeasuredPoint]) -> None:
        """Add measured point to msmnt."""

        PaData.add_point(
            measurement = self.measurement,
            data = point,
            param_val = [self.param_points[self.current_point]] # type: ignore
        )
        # Enable measure button
        self.btn_measure.setEnabled(True)
        # update current point attribute, which will trigger 
        # update of realted widgets on parent
        self.current_point += 1

    def setup_measurement(self) -> None:
        """
        Start 1D measurement.
        
        Launch energy adjustment.\n
        Set ``p_curve.step``, ``p_curve.spectral_points`` attributes.
        Also set progress bar and realted widgets.
        """

        # Activate curve measure button
        set_layout_enabled(
            self.lo_measure_ctrls,
            True
        )
        # But keep restart deactivated
        self.btn_redo.setEnabled(False)

        # Deactivate paramter controls
        set_layout_enabled(
            self.lo_parameter,
            False
        )

        # calculate amount of data points
        start = self.sb_from.quantity
        stop = self.sb_to.quantity
        step = self.sb_step.quantity
        delta = abs(stop - start)
        self.points_num = math.ceil(delta/step) + 1
        # create array with all parameter values
        points = []
        if start < stop:
            self.step = step
            for i in range(self.points_num - 1):
                points.append(start + i*step)
            points.append(stop)
        else:
            self.step = -1*step
            for i in range(self.points_num - 1):
                points.append(start - i*step)
            points.append(stop)
        self.param_points = Q_.from_list(points)
        self.current_point = 0
        # Set enable state for current wavelength
        if self.cb_mode.currentText() == 'Wavelength':
            self.sb_cur_wl.setEnabled(False)
        else:
            self.sb_cur_wl.setEnabled(True)
        # Set param label
        self.lbl_cur_param.setText('Set ' + self.cb_mode.currentText())
        # Manually call to update widgets for the first point.
        # Create measurement
        self.measurement = PaData.create_measurement(
            dims = 1,
            params = [self.cb_mode.currentText()]
        )
        self.upd_widgets()

    def upd_widgets(self) -> None:
        """
        Update widgets, related to current scan point.
        
        Update progress bar, its label and current param."""

        # progress bar
        pb = int(self.current_point/self.points_num*100)
        self.pb.setValue(pb)

        # progress bar label
        text = f'{self.current_point}/{self.points_num}'
        self.lbl_pb.setText(text)

        if self.current_point:
            # Set current data point
            dp_title = PaData._build_name(self.current_point)
            self.s_point = self.measurement.data[dp_title]
            self.btn_redo.setEnabled(True)
            self.pv.load_dp(self.s_point)
            # Plot data
            data = PaData.param_data_plot(
                msmnt = self.measurement,
                dtype = 'raw_data'
            )
            self.plot_measurement.plot(**data._asdict())
            self.plot_measurement.set_marker(self.current_point - 1)
        
        if self.current_point < self.points_num:
            # current param value
            new_param = self.param_points[self.current_point] # type: ignore
            self.sb_cur_param.quantity = new_param
            if self.cb_mode.currentText() == 'Wavelength':
                self.sb_cur_wl.quantity = new_param
            # Measure next btn
            self.btn_measure.setEnabled(True)
        # If the last point
        else:
            self.btn_measure.setEnabled(False)

    def pick_event(self, sel_ind: int):
        """Callback method for processing data picking on plot."""

        # Update datamarker on plot
        self.current_point = sel_ind + 1

    def curve_finished(self) -> None:
        
        # Disable measure btn
        self.btn_measure.setEnabled(False)
        # Set state of controls
        set_layout_enabled(self.lo_parameter, True)
        set_layout_enabled(self.lo_measure_ctrls, False)
        self.curve_obtained.emit(self.measurement)

    def set_param_widget(self, param: str) -> None:

        if (vals:=CURVE_PARAMS.get(param, None)) is None:
            logger.warning('Wrong parameter!')
            return
        self.sb_from.quantity = vals['start']
        self.sb_to.quantity = vals['stop']
        self.sb_step.quantity = vals['step']

    @property
    def current_point(self) -> int:
        """
        Index of current parameter value.
        
        Automatically updates related widgets."""

        return self._current_point
    @current_point.setter
    def current_point(self, val: int) -> None:
        if val != self._current_point:
            self._current_point = val
            self.cur_p_changed.emit()

class PointMeasureWidget(MeasureWidget, point_measure_widget_ui.Ui_Form):
    """0D PhotoAcoustic measurements widget."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.pm_monitor = PowerMeterMonitor(self)
        self.lo_measure.replaceWidget(
            self.placeholder_pm_monitor,
            self.pm_monitor
        )

        self.cur_rep = 1
        self.data_point = []

        self.connect_signals_slots()
        self.setup_sp()
        self.setup_pm()

    def connect_signals_slots(self) -> None:
        """Connect signals and slots for the widget."""

        # Measure btn clicked
        self.btn_measure.clicked.connect(self.measure)

class MapMeasureWidget(QWidget,map_measure_widget_ui.Ui_map_measure):
    """2D PhotoAcoustic measurements widget."""

    calc_astepClicked = Signal(QProgressDialog)
    astep_calculated = Signal(bool)
    "Emitted, when auto step is calculated. Always return `True`."
    scan_started = Signal(MapData)
    scan_obtained = Signal(MapData)
    "Emiited when scanning finished and some data were measured."
    scan_worker: Worker | None = None

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        # Zoom state of plot
        self._zoomed_out = False
        # max scan range
        self.XMAX = Q_(25, 'mm')
        self.YMAX = Q_(25, 'mm')

        # Current position
        self.cur_pos = Position(Q_(0, 'mm'), Q_(0, 'mm'), Q_(0, 'mm'))

        # Set Axis titles
        self._new_scan_plane()

        # Set default range
        self.set_zoomed_out(True)

        # Set default selection
        self.plot_scan.set_def_selarea()
        # We should manually call this slot, since it is not yet connected
        self._sel_changed(self.plot_scan.selected_area)

        # Set enable state for axis controls
        self._new_dir()

        # Set default vals
        self.sb_cur_speed.quantity = Q_(0, 'm/s')
        self.sb_tpp.quantity = Q_(0, 's')

        # Scan timer
        self.scan_timer: QTimer | None = None
        self._scan_started: datetime | None = None

        self.connect_signals_slots()

    def connect_signals_slots(self):
        """Connect signals and slots."""

        # Scan size X changed
        self.sb_sizeX.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(width = val)
        )
        # Scan size Y changed
        self.sb_sizeY.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(height = val)
        )
        # Scan center X changed
        self.sb_centerX.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(
                left = val - self.sb_sizeX.quantity/2)
        )
        # Scan center Y changed
        self.sb_centerY.quantSet.connect(
            lambda val: self.plot_scan.set_selarea(
                bottom = val - self.sb_sizeY.quantity/2)
        )
        # Scan points changed
        self.sb_pointsX.quantSet.connect(lambda val: self._points_set())
        self.sb_pointsY.quantSet.connect(lambda val: self._points_set())
        # Selected area changed from plot
        self.plot_scan.selection_changed.connect(self._sel_changed)
        # Toogle selected area visibility
        self.btn_plot_area_sel.toggled.connect(
            self.plot_scan.set_selarea_vis
        )
        # Set maximum plot size visible
        self.btn_plot_full_area.pressed.connect(lambda: self.set_zoomed_out(True))
        # Expand scan to full plot area
        self.btn_plot_fit.pressed.connect(lambda: self.set_zoomed_out(False))
        # Clear plot
        self.btn_plot_clear.clicked.connect(self.clear_data)
        # Data smoothing
        self.btn_hres.toggled.connect(self.smooth_data)
        # Scan plane changed
        self.cb_scanplane.currentTextChanged.connect(
            lambda _: self._new_scan_plane()
        )
        # Scan direction changed
        self.cb_scandir.currentTextChanged.connect( lambda _: self._new_dir())
        # Auto step calculation
        self.btn_astep_calc.clicked.connect(lambda _: self.calc_astep())
        # Current speed changed
        self.sb_cur_speed.valueChanged.connect(lambda _: self._recalc_points())
        # Scan
        self.btn_start.toggled.connect(self.scan)
        # Stop scan
        self.btn_stop.clicked.connect(self.stop_scan)
        
    def scan(self, state: bool) -> None:
        """Launch scanning by emitting ``scan_started``."""

        if state:
            # Check if scan step was already calculated
            if self.sb_tpp.quantity.m == 0:
                self.calc_astep()
                self.astep_calculated.connect(self.scan)
                return
            else:
                try:
                    self.astep_calculated.disconnect(self.scan)
                except:
                    pass
            # Set buttons state
            self.btn_start.setEnabled(False)
            self.btn_plot_clear.setEnabled(False)
            self.btn_hres.setEnabled(True)
            self.btn_plot_fit.setEnabled(True)
            # Create MapData object
            scan = MapData(
                center = self.center,
                width = self.sb_sizeX.quantity,
                height = self.sb_sizeY.quantity,
                hpoints = int(self.sb_pointsX.value()),
                vpoints = int(self.sb_pointsY.value()),
                scan_plane = self.cb_scanplane.currentText(), # type: ignore
                scan_dir = self.cb_scandir.currentText(),
                mode = self.cb_mode.currentText(),
                wavelength = self.sb_wl.quantity
            )
            # Load scan data to plot
            self.plot_scan.set_data(data=scan)
            self.set_zoomed_out(False)
            # Initiate scan timer
            if self.scan_timer is None:    
                self.scan_timer = QTimer(self)
                self.scan_timer.timeout.connect(self._upd_scan_time)
            self._scan_started = datetime.now()
            self.scan_timer.start(100)
            # Emit signal to start actual scanning from main window
            self.scan_started.emit(scan)

    @Slot()
    def stop_scan(self) -> None:
        """Stop current scan."""

        if self.scan_worker is None:
            return
        # Stop actual measurement
        self.scan_worker.flags['is_running'] = False
        
    @Slot()
    def scan_finished(self) -> None:
        """Scan finished procedures."""

        # Change state of buttons
        self.btn_start.setChecked(False)
        self.btn_start.setEnabled(True)
        self.btn_plot_clear.setEnabled(True)
        # Stop scan timer
        if self.scan_timer is None:
            logger.warning('Scan timer is missing.')
        else:
            self.scan_timer.stop()
        if self.plot_scan.data is not None:
            lines = len(self.plot_scan.data.lines)
            dur = self.le_dur.text()
            logger.info(f'{lines} lines scanned. Scan duration {dur}')
            # Save scan data
            if lines:
                self.scan_obtained.emit(self.plot_scan.data)
        logger.info('Scanning finished.')

    @Slot(bool)
    def smooth_data(self, state: bool) -> None:
        if self.plot_scan.data is not None:
            self.plot_scan.set_highres(state)

    @Slot()
    def clear_data(self) -> None:
        """Clear scanned data."""

        self.plot_scan.clear_plot()
        self.le_dur.setText('')
        self.set_zoomed_out(True)
        self.btn_hres.setEnabled(False)
        self.btn_plot_fit.setEnabled(False)

    def calc_astep(self, count: int=10) -> QProgressDialog:
        """Launch auto step calculation."""

        # Create progress bar dialog
        pb = QProgressDialog(self)
        #pb.setAutoReset(False)
        pb.setAutoClose(True)
        pb.setLabelText('Calculating scan step size...')
        pb.setWindowTitle('Auto step calculator')
        pb.setRange(0, count)
        # Set appear immediatelly
        pb.setMinimumDuration(0)
        pb.setValue(0)
        # Emit signal to launch worker for measuring.
        self.calc_astepClicked.emit(pb)
        pb.show()
        return pb

    @Slot()
    def set_astep(self, data: list[EnergyMeasurement]) -> None:
        """
        Set auto step and points from data list.
        
        ``data`` should contain results of performance test of signal
        acquisition system with at least 4 EnergyMeasurements, but
        for better precision it is better to provide 30-50 EnergyMeasurements.

        The algorithm will calculate average dt between signal measurements
        and use it to calculate step size and amount of points.
        Calculated dt is set to sb_tpp spinbox.
        """

        if len(data) < 4:
            logger.warning(
                'Auto step cannot be calculated. Too few data.'
            )
            return

        # Calculate duration of all measurements
        delta = data[-1].datetime - data[0].datetime
        dt = Q_(delta.total_seconds(), 's')
        # Force update of the current stage speed. Do not remove it!!!
        self.sb_speed.quantSet.emit(self.sb_speed.quantity)
        # Average time per 1 step
        time_per_p = dt/(len(data) - 1)
        ### Set calculated value
        self.sb_tpp.quantity = time_per_p
        self._recalc_points()
        # Emit signal
        self.astep_calculated.emit(True)
    
    def set_zoomed_out(self, state: bool) -> None:
        """Set whether maximum scan area is visible."""
        if state and self._zoomed_out:
            return
        if not state and not self._zoomed_out:
            return
        if not state and self.plot_scan.data is None:
            return
        
        self._zoomed_out = state
        self.plot_scan.abs_coords = state
        if state:
            self.plot_scan.set_scanrange(self.XMAX, self.YMAX)
            self.plot_scan.set_abs_scan_pos(True)
        else:
            self.plot_scan.set_scanrange(
                self.plot_scan.data.width, # type: ignore
                self.plot_scan.data.height # type: ignore
            )
            self.plot_scan.set_abs_scan_pos(False)

    @Slot(Position)
    def set_cur_pos(self, pos: Position) -> None:
        """Set current position."""

        if not self.isVisible():
            return
        
        self.cur_pos = pos
        self.plot_scan.set_cur_pos(pos)

        hpos = getattr(pos, self.cb_scanplane.currentText()[0].lower())
        vpos = getattr(pos, self.cb_scanplane.currentText()[1].lower())
        if hpos is None or vpos is None:
            return
        self.le_posX.setText(f'{hpos.to(self.plot_scan.hunits):~.2fP}')
        self.le_posY.setText(f'{vpos.to(self.plot_scan.vunits):~.2fP}')

    def _set_est_dur(self) -> None:
        """Set estimated duration of scan."""

        if self.sb_tpp.quantity.m == 0:
            return
        # Get current speed
        speed = self.sb_cur_speed.quantity
        # Line size depends on orientation of fast scan direction
        if self.cb_scandir.currentText().startswith('H'):
            line_size = self.sb_sizeX.quantity
            spoints = self.sb_pointsY.value()
            est_dur = (line_size*spoints + self.sb_sizeY.quantity)/speed
        else:
            line_size = self.sb_sizeY.quantity
            spoints = self.sb_pointsX.value()
            est_dur = (line_size*spoints + self.sb_sizeX.quantity)/speed
        # Calcualte starting point
        if self.cb_scandir.currentText()[1] == 'L':
            hs_coord = self.sb_centerX.quantity - self.sb_sizeX.quantity/2
        else:
            hs_coord = self.sb_centerX.quantity + self.sb_sizeX.quantity/2
        if self.cb_scandir.currentText()[2] == 'B':
            vs_coord = self.sb_centerY.quantity - self.sb_sizeY.quantity/2
        else:
            vs_coord = self.sb_centerY.quantity + self.sb_sizeY.quantity/2
        s_pos = Position.from_tuples([
            (self.cb_scanplane.currentText()[0].lower(), hs_coord), # type: ignore
            (self.cb_scanplane.currentText()[1].lower(), vs_coord) # type: ignore
        ])
        est_dur += s_pos.dist(self.cur_pos)/speed
        tot_s = est_dur.to('s').m # type: ignore
        hours = int(tot_s // 3600)
        mins = int((tot_s % 3600) // 60)
        secs = (tot_s % 3600) % 60
        strdur = f'{hours:02d}:{mins:02d}:{secs:04.1f}'
        self.le_estdur.setText(strdur)

    @Slot()
    def _sel_changed(self, new_sel: QuantRect | None) -> None:
        """This slot is always called when selection is changed."""
        # Uncheck selection button if selection is not visible
        if new_sel is None:
            btn_set_silent(self.btn_plot_area_sel, False)
            return
        # Otherwise update spin boxes
        x, y, width, height = new_sel
        cx = x + width/2 # type: ignore
        cy = y + height/2 # type: ignore
        if self.sb_centerX.quantity != cx:
            self.sb_centerX.quantity = cx # type: ignore
        if self.sb_centerY.quantity != cy:
            self.sb_centerY.quantity = cy # type: ignore
        if self.sb_sizeX.quantity != width:
            self.sb_sizeX.quantity = width # type: ignore
        if self.sb_sizeY.quantity != height:
            self.sb_sizeY.quantity = height # type: ignore

        # Step size is determined by scan speed and ADC rate, therefore
        # when scan size is changed, points number, but step size 
        # should be changed.
        self._recalc_points()

    @Slot()
    def _recalc_points(self) -> None:
        """Recalculate amount of points."""

        # Get current speed
        speed = self.sb_cur_speed.quantity
        if (time_per_p:=self.sb_tpp.quantity).m == 0:
            return
        step = speed * time_per_p
        # Line size depends on orientation of fast scan direction
        if self.cb_scandir.currentText().startswith('H'):
            line_size = self.sb_sizeX.quantity
        else:
            line_size = self.sb_sizeY.quantity
        # Fast axis points
        points = int(line_size/step) + 1
        # Step
        x_units = self.sb_stepX.quantity.u
        self.sb_stepX.quantity = step.to(x_units)
        y_units = self.sb_stepY.quantity.u
        self.sb_stepY.quantity = step.to(y_units)
        # Points
        if self.cb_scandir.currentText().startswith('H'):
            self.sb_pointsX.setValue(points)
            self.sb_pointsY.setValue(
               int(self.sb_sizeY.quantity/step) + 1 
            )
        else:
            self.sb_pointsX.setValue(
               int(self.sb_sizeX.quantity/step) + 1 
            )
            self.sb_pointsY.setValue(points)
        # Set estimated scan duration
        self._set_est_dur()

    @Slot()
    def _points_set(self) -> None:
        """Update scan step, when amount of points is changed."""

        xstep = self.sb_sizeX.quantity/(self.sb_pointsX.value() - 1)
        ystep = self.sb_sizeY.quantity/(self.sb_pointsY.value() - 1)
        if self.sb_stepX.quantity != xstep:
            self.sb_stepX.quantity = xstep
        if self.sb_stepY.quantity != ystep:
            self.sb_stepY.quantity = ystep
        # Update estimated scan duration
        #self._set_est_dur()

    @Slot()
    def _new_scan_plane(self) -> None:
        """Slot, called when new scan plane is selected."""

        sel_plane = self.cb_scanplane.currentText()
        self.plot_scan.hlabel = sel_plane[0]
        self.plot_scan.vlabel = sel_plane[1]
        # Labels for scan area
        self.lbl_areaX.setText(sel_plane[0])
        self.lbl_areaY.setText(sel_plane[1])
        # Labels for current position
        self.lbl_posX.setText(sel_plane[0])
        self.lbl_posY.setText(sel_plane[1])

    @Slot()
    def _new_dir(self) -> None:
        """Callback for new scan direction."""
        
        # Update est scan duration
        self._set_est_dur()
        # Check if auto step calculation is enabled
        if not self.chb_astep.isChecked():
            return
        # Set enable state for axis controls
        if self.cb_scandir.currentText()[0] == 'H':
            self.sb_pointsX.setEnabled(False)
            self.sb_pointsY.setEnabled(True)
        else:
            self.sb_pointsX.setEnabled(True)
            self.sb_pointsY.setEnabled(False)

    @Slot()
    def _upd_scan_time(self) -> None:
        """Update scan duration."""

        if self._scan_started is not None:
            dt = datetime.now() - self._scan_started
            s, ms = str(dt).split('.')
            self.le_dur.setText(f'{s}.{ms[:1]}')
        else:
            logger.warning('Scan timer cannot be updated. Start time is None.')

    @property
    def center(self) -> Position:
        """Absolute position of scan center."""

        # Get titles of horizontal and vertical axes
        haxis = self.cb_scanplane.currentText()[0]
        vaxis = self.cb_scanplane.currentText()[1]
        # Get values of horizontal and vertical coordinates
        hcoord = self.sb_centerX.quantity
        vcoord = self.sb_centerY.quantity
        # Covert relative coordinate to absolute
        if not self.zoomed_out:
            if (shift:= self.plot_scan._rel_to_abs(quant = True)) is not None:
                hcoord += shift[0]
                vcoord += shift[1]
        # Generate position
        center = Position.from_tuples([
            (haxis, hcoord), (vaxis, vcoord)
        ])
        return center

    @property
    def zoomed_out(self) -> bool:
        """Whether maximum scan area is visible."""
        return self._zoomed_out
    @zoomed_out.setter
    def zoomed_out(self, state: bool) -> None:
        logger.warning('Zoomed out attribute is read only!')

class PowerMeterMonitor(QWidget,pm_monitor_ui.Ui_Form):
    """Power meter monitor.
    
    Contains two plot areas:\n
    ``plot_left`` and ``plot_right``.
    """

    def __init__(
            self,
            parent: QWidget | None=None,
            tune_width: int=50,
            aver: int=10  
        ) -> None:
        """
        Initialization of Power meter monitor.

        ``tune_width`` - size of energy adjustment plot (right).\n
        ``aver`` - amount of measurements used to calc mean and std.
        """
        super().__init__(parent)
        self.setupUi(self)
        self.tune_width = tune_width
        self.qu = deque(maxlen = self.tune_width)
        self.aver = aver
        self.connect_signals_slots()

        self.plot_left.set_pts_visible(False)
        self.plot_right.set_pts_visible(False)

    def connect_signals_slots(self):
        """Connect signals and slots."""

        # Start button
        self.btn_start.toggled.connect(
            lambda enable: self.btn_pause.setChecked(not enable)
        )
        # Pause button
        self.btn_pause.toggled.connect(
            lambda enable: self.btn_start.setChecked(not enable)
        )

        ### Stop button
        # Reset buttons
        self.btn_stop.clicked.connect(
            lambda: btn_set_silent(self.btn_start, False)
        )
        self.btn_stop.clicked.connect(
            lambda: btn_set_silent(self.btn_pause, False)
        )
        # Clear plots
        self.btn_stop.clicked.connect(
            lambda: self.plot_left.clear_plot()
        )
        self.btn_stop.clicked.connect(
            lambda: self.plot_right.clear_plot()
        )

    def add_msmnt(self, measurement: EnergyMeasurement) -> None:
        """"Add energy measurement."""

        # Update information only if Start btn is checked 
        # and widget is visible
        if (not self.btn_start.isChecked()) or (not self.isVisible()):
            return
        # Check if measurement is not empty
        if measurement.energy.m is np.nan:
            return
        # Skip duplicate read
        if np.array_equal(
            measurement.signal.m,
            self.plot_left.ydata
        ):
            return
        # Plot signal
        self.plot_left.plot(measurement.signal)
        self.plot_left.set_marker(measurement.istart)

        # Plot tune graph
        ytune = self.plot_right.ydata
        # Check if some data already present
        if not len(ytune):
            self.plot_right.plot(Q_.from_list([measurement.energy]))
        else:
            energy = measurement.energy.to(self.plot_right.yunits)
            #qu = deque(ytune, maxlen = self.tune_width)
            self.qu.append(energy.m)
            ytune = np.array(self.qu)
            self.plot_right.plot(Q_(ytune, energy.u))
        
        # Update line edits with energy info
        units = self.plot_right.yunits
        ytune = self.plot_right.ydata
        self.le_cur_en.setText(form_quant(measurement.energy))
        aver = Q_(ytune.mean(), units)
        self.le_aver_en.setText(form_quant(aver))
        std = Q_(ytune.std(), units)
        self.le_std_en.setText(form_quant(std))

class MapView(QWidget, map_data_view_ui.Ui_Map_view):
    """Plots for 2D data."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        # Param signals
        self.cb_curve_select.addItems(
            [key for key in MSMNTS_SIGNALS.keys()]
        )

        # Enable pixel selection for map
        self.plot_map.pick_pixel_enabled = True

class CurveView(QWidget,curve_data_view_ui.Ui_Curve_view):
    """Plots for 1D data."""

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.cb_curve_select.addItems(
            list(MSMNTS_SIGNALS.keys())
        )

        #Enable picker
        self.plot_curve.pick_enabled = True
        
class MotorView(QDockWidget, motor_control_ui.Ui_DockWidget):
    """Mechanical positioning widget."""

    pos_selected = Signal(Position)

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.x: PlainQuantity = Q_(0, 'mm')
        self.y: PlainQuantity = Q_(0, 'mm')
        self.z: PlainQuantity = Q_(0, 'mm')

        # Set fixed scales for all plots.
        self.plot_xy.abs_coords = True
        self.plot_xz.abs_coords = True
        self.plot_yz.abs_coords = True

        self.pixmap_c = QPixmap(
            u":/icons/qt_resources/plug-connect.png"
        )
        "Connected state icon."
        self.pixmap_dc = QPixmap(
            u":/icons/qt_resources/plug-disconnect-prohibition.png"
        )
        "Disconnected state icon."

        # Set default range
        self.set_range(
            (Q_(25, 'mm'), Q_(25, 'mm'))
        )

        # Set initial position of plots
        self.plot_xy.set_cur_pos(Position(x = self.x, y = self.y))
        self.plot_xz.set_cur_pos(Position(x = self.x, z = self.z))
        self.plot_yz.set_cur_pos(Position(y = self.y, z = self.z))

    def connectSignalsSlots(self) -> None:
        """Connect default signals and slots."""

        self.btn_go.clicked.connect(
            lambda: self.pos_selected.emit(self.current_pos())
        )
        self.plot_xy.pos_selected.connect(self.pos_selected.emit)
        self.plot_xz.pos_selected.connect(self.pos_selected.emit)
        self.plot_yz.pos_selected.connect(self.pos_selected.emit)

    @Slot(Position)
    def set_position(self, position: Position) -> None:
        """Update current coordinate."""

        if not self.isVisible():
            return
        if position is None:
            logger.warning('Invalid coordinate in set position.')
        # Update current position line edits and attributes.
        if position.x is not None and position.x != self.x:
            self.x = position.x.to('mm')
            self.le_x_cur_pos.setText(f'{self.x:.3f~P}')
            self.sb_x_new_pos.setValue(self.x.m)
        if position.y is not None and position.y != self.y:
            self.y = position.y.to('mm')
            self.le_y_cur_pos.setText(f'{self.y:.3f~P}')
            self.sb_y_new_pos.setValue(self.y.m)
        if position.z is not None and position.z != self.z:
            self.z = position.z.to('mm')
            self.le_z_cur_pos.setText(f'{self.z:.3f~P}')
            self.sb_z_new_pos.setValue(self.z.m)
        
        # Update plots if necessary
        self.plot_xy.set_cur_pos(position)
        self.plot_xz.set_cur_pos(position)
        self.plot_yz.set_cur_pos(position)

    def set_range(
            self,
            range: tuple[PlainQuantity, PlainQuantity]
        ) -> None:
        """Set range of valid coordinates."""

        min_val = range[0].to('mm').m
        max_val = range[1].to('mm').m
        
        self.plot_xy.set_scanrange(*range)
        self.plot_xz.set_scanrange(*range)
        self.plot_yz.set_scanrange(*range)

        # Set plot titles
        self.plot_xy.hlabel = 'X'
        self.plot_xy.vlabel = 'Y'
        self.plot_xz.hlabel = 'X'
        self.plot_xz.vlabel = 'Z'
        self.plot_yz.hlabel = 'Z'
        self.plot_yz.vlabel = 'Y'
        # Set range for spin boxes
        for sb in iter([
            self.sb_x_new_pos,
            self.sb_y_new_pos,
            self.sb_z_new_pos
        ]):
            sb.setMinimum(min_val)
            sb.setMaximum(max_val)

    def upd_status(
            self,
            status: StagesStatus
        ) -> None:
        """Update status of motors."""

        if not self.isVisible():
            return
        self._upd_axes_status(
            status.x_open,
            status.x_status,
            self.icon_x_status,
            self.lbl_x_status
        )
        self._upd_axes_status(
            status.y_open,
            status.y_status,
            self.icon_y_status,
            self.lbl_y_status
        )
        self._upd_axes_status(
            status.z_open,
            status.z_status,
            self.icon_z_status,
            self.lbl_z_status
        )

    def _upd_axes_status(
            self,
            is_opened: bool | None,
            status: list[str],
            icon: QLabel,
            lbl: QLabel
        ) -> None:
        """Update status widgets for an axes."""

        if is_opened is None or len(status) == 0:
            return
        if is_opened:
            icon.setPixmap(self.pixmap_c)
            lbl.setText(','.join(status))
        else:
            icon.setPixmap(self.pixmap_dc)
            lbl.setText('Disconnected')

    def current_pos(self) -> Position:
        """Get position to send stages."""

        pos = Position()
        pos.x = Q_(self.sb_x_new_pos.value(), 'mm')
        pos.y = Q_(self.sb_y_new_pos.value(), 'mm')
        pos.z = Q_(self.sb_z_new_pos.value(), 'mm')
        return pos
        