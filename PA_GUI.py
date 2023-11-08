"""
Latest developing GUI version
"""

from __future__ import annotations

import os, os.path
import sys
import time
import logging, logging.config
from datetime import datetime
import traceback
from typing import Iterable, Optional, cast, Callable
from dataclasses import field, fields

import pint
import yaml
from pylablib.devices.Thorlabs import KinesisMotor
from pint.facets.plain.quantity import PlainQuantity
from matplotlib.backend_bases import PickEvent
from PySide6.QtCore import (
    Qt,
    QObject,
    QThread,
    QThreadPool,
    Slot,
    Signal,
    QItemSelection,
    QRect,
    QItemSelectionModel
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMessageBox,
    QLineEdit,
    QLayout,
    QComboBox,
    QTreeWidgetItem,
    QFileDialog
)
from PySide6.QtGui import (
    QColor,
    QPalette,
    QStandardItem,
    QStandardItemModel
)
import numpy as np

from modules import ureg, Q_
import modules.validators as vd
from modules.pa_data import PaData
import modules.pa_logic as pa_logic
from modules.exceptions import StageError, HardwareError
import modules.data_classes as dc
from modules.data_classes import (
    Worker,
    MeasuredPoint,
    Measurement,
    DataPoint,
    DetailedSignals,
    ParamSignals
)
from modules.gui.widgets import (
    MplCanvas,
    QuantSpinBox,
)
from modules.gui.designer.PA_main_window_ui import Ui_MainWindow

from modules.gui.designer.custom_widgets import (
    PAVerifierDialog,
    DataViewer,
    LoggerWidget,
    PowerMeterWidget,
    PointMeasureWidget,
    CurveMeasureWidget,
    MapMeasureWidget,
    PowerMeterMonitor,
    CurveView
)



def init_logs() -> tuple[logging.Logger, QLogHandler]:
    """Initiate logging"""

    with open('rsc/log_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        log_config = config['official']
        max_files = int(config['additional']['maxfiles'])
    filesnames = [f for f in os.listdir('./logs') if f.endswith('log')]
    filesnames.sort(key = lambda x: os.path.getctime('./logs/' + x))
    while len(filesnames) > max_files:
        old_file = filesnames.pop(0)
        os.remove('./logs/' + old_file)

    # adding current data to name of log files
    for i in (log_config["handlers"].keys()):
        # Check if filename in handler.
        # Console handlers might be present in config file
        if 'filename' in log_config['handlers'][i]:
            log_filename = log_config["handlers"][i]["filename"]
            base, extension = os.path.splitext(log_filename)
            today = datetime.today().strftime("_%Y_%m_%d-%H-%M-%S")
            log_filename = f"{base}{today}{extension}"
            log_config["handlers"][i]["filename"] = log_filename

    #adding wdiget handler. I cannot handle it with file config.
    q_log_handler = QLogHandler()
    q_log_handler.thread.start()
    logging.config.dictConfig(log_config)
    logging.getLogger('root').addHandler(q_log_handler)
    logging.getLogger('modules').addHandler(q_log_handler)

    return logging.getLogger('pa_cli'), q_log_handler

def init_hardware() -> None:
    """CLI for hardware init"""

    try:
        pa_logic.init_hardware()
    except HardwareError:
        logger.error('Hardware initialization failed')

class Window(QMainWindow,Ui_MainWindow,):
    """Application MainWindow."""

    worker_pm: Worker|None = None
    "Thread for measuring energy."
    worker_curve_adj: Worker|None = None
    "Thread for curve adjustments."

    _data: PaData|None = None

    data_changed = Signal()

    @property
    def data(self) -> PaData|None:
        "PA data."
        return self._data
    
    @data.setter
    def data(self, value: PaData|None) -> None:
        if isinstance(value, PaData) or value is None:
            self._data = value
        else:
            logger.warning('Attempt to set wrong object to data.')
            return
        self.data_changed.emit()

    def __init__(
            self,
            parent: QMainWindow|None = None,
        ) -> None:
        super().__init__(parent)
        self.setupUi(self)
        
        #Long-running threads
        self.q_logger = q_log_handler

        #Threadpool
        self.pool = QThreadPool()

        self.set_widgets()

        #Connect custom signals and slots
        self.connectSignalsSlots()

    def set_widgets(self):
        """Add and configure custom widgets."""

        #Measurement verifuer dialog
        self.pa_verifier = PAVerifierDialog()

        #Data viewer
        self.data_viwer = DataViewer()
        self.sw_central.addWidget(self.data_viwer)

        #Log viewer
        self.d_log = LoggerWidget()
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.d_log)

        # Power Meter
        # Set parent in constructor makes 
        # dock widget undocked by default
        self.d_pm = PowerMeterWidget(self)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.d_pm
        )
        self.d_pm.hide()

        # 0D measurements
        self.p_point = PointMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_point)

        # 1D measurements
        self.p_curve = CurveMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_curve)
        self.p_curve.w_measure.hide()
        setpoint = self.p_curve.sb_sample_sp.quantity
        self.p_curve.pm_monitor.plot_right.sp = setpoint
        self.upd_sp(self.p_curve.sb_sample_sp)

        # 2D measurements
        self.p_map = MapMeasureWidget(self.sw_central)
        self.sw_central.addWidget(self.p_map)

        # Scan mode selector
        cb = QComboBox()
        cb.addItems(
            ('Single point', 'Curve', 'Map')
        )
        cb.setEditable(False)
        cb.setCurrentIndex(self.sw_central.currentIndex())
        self.upd_mode(cb.currentText())
        self.tb_mode.addWidget(cb)
        self.cb_mode_select = cb
        self.upd_mode(self.cb_mode_select.currentText())

    def connectSignalsSlots(self):
        """Connect signals and slots."""

        #logs
        self.q_logger.thread.log.connect(self.d_log.te_log.appendPlainText)

        #Various actions
        self.action_Init.triggered.connect(self.init_hardware)
        self.data_changed.connect(self.data_updated)
        self.action_Close.triggered.connect(self.close_file)
        self.actionSave_As.triggered.connect(self.save_as_file)

        #Mode selection
        self.cb_mode_select.currentTextChanged.connect(self.upd_mode)
        self.action_measure_PA.toggled.connect(self.activate_measure_widget)

        #1D measurements
        self.p_curve.btn_run.toggled.connect(self.run_curve)
        self.p_curve.btn_measure.clicked.connect(self.measure_curve)
        self.p_curve.sb_sample_sp.valueChanged.connect(
            lambda x: self.upd_sp(self.p_curve.sb_sample_sp)
        )
        self.p_curve.sb_pm_sp.valueChanged.connect(
            lambda x: self.upd_sp(self.p_curve.sb_pm_sp)
        )

        #Data
        self.action_Data.toggled.connect(self.activate_data_viwer)
        self.action_Open.triggered.connect(self.open_file)

        #Logs
        self.action_Logs.toggled.connect(self.d_log.setVisible)
        self.d_log.visibilityChanged.connect(self.action_Logs.setChecked)

        #PowerMeter
        self.d_pm.btn_start.clicked.connect(self.track_power)
        self.d_pm.btn_stop.clicked.connect(
            lambda x: self.stop_worker(self.worker_pm)
        )
        #### Stopped here
        self.d_pm.btn_pause.toggled.connect(
            lambda state: self.pause_worker(state, self.worker_pm)
        )
        self.action_Power_Meter.toggled.connect(self.d_pm.setVisible)
        self.d_pm.visibilityChanged.connect(
            self.action_Power_Meter.setChecked
        )

    def data_updated(self) -> None:
        """Slot, which triggers when data is changed."""

        # Set Save enabled
        if self.data is None:
            is_exist = False
        else:
            is_exist = True
        self.action_Save.setEnabled(is_exist)
        self.actionSave_As.setEnabled(is_exist)

        # Update data viewer
        self.load_data_view(self.data)

    def open_file(self) -> None:
        """Load data file."""

        logger.debug('Start loading data...')
        filename = self.get_filename()
        if filename:
            self.close_file()
            self.data = PaData()
            self.data.load(filename)
            logger.info(f'Data with {self.data.attrs.measurements_count} '
                        + 'PA measurements loaded!')
            basename = os.path.basename(filename)
            self.data_viwer.lbl_data_title.setText(basename)
            self.load_data_view(self.data)

    def save_as_file(self) -> None:
        """Save data to file."""

        path = os.path.dirname(__file__)
        initial_dir = os.path.join(path, 'measuring results')
        filename = QFileDialog.getSaveFileName(
            self,
            caption='Choose a file',
            dir=initial_dir,
            filter='Hierarchical Data Format (*.hdf5)'
        )[0]
        if self.data is not None:
            if self.data.attrs.version < 1.2:
                self.data.attrs.version = PaData.VERSION
            self.data.save(filename)

    def close_file(self) -> None:
        """Close current file."""

        if self.data is not None and self.data.changed:
            if self.confirm_action('Data was changed.'
                                + ' Do you really want to close the file?'):
                self._file_close_proc()
        else:
            self._file_close_proc()

    def _file_close_proc(self) -> None:
        """Actual file close."""

        self.data = None
        self._clear_data_view()

    def _clear_data_view(self) -> None:
        """Clear data viewer."""

        self.data_viwer.measurement = None
        self.data_viwer.datapoint = None
        self.data_viwer.tv_content.setModel(QStandardItemModel())
        tv_detail = self.data_viwer.tv_info
        while tv_detail.topLevelItemCount():
            tv_detail.takeTopLevelItem(0)
        self.data_viwer.sw_view.setCurrentWidget(self.data_viwer.p_empty)

    def new_sel_msmnt(
            self,
            selected: QItemSelection,
            desecled: QItemSelection
        ) -> None:
        """Load new selected measurement."""

        # Selected index
        ind = selected.indexes()[0]

        model = self.data_viwer.tv_content.model()
        msmnt_title = next(iter(model.itemData(ind).values()))
        msmnt = self.data.measurements[msmnt_title] # type: ignore
        data_index = self.data_viwer.data_index
        self.show_data(msmnt, data_index)

    def load_data_view(self, data: PaData|None = None) -> None:
        """Load data into DataView widget."""

        logger.debug('Starting plotting data...')
        if data is None:
            if self.data is None:
                logger.error('No data to show!')
                return
            data = self.data

        # clear old data view
        self._clear_data_view()

        # Update content representation
        self.set_data_content_view(data)

        # choose a measurement to show data from
        try:
            msmnt_title, msmnt = next(iter(data.measurements.items()))
        except StopIteration:
            logger.debug('Data is empty.')
            return
        self.data_viwer.tv_content.setSelection(
            QRect(0,0,1,1), QItemSelectionModel.SelectionFlag.Select)
        self.data_viwer.tv_content.selectionModel().selectionChanged.connect(
            self.new_sel_msmnt
        )

        self.show_data(
            measurement = msmnt,
            data_index = 0
        )
        logger.debug('Data plotting complete.')

    def show_data(
            self,
            measurement: Measurement,
            data_index: int
    ) -> None:
        """
        Show data, which was already loaded.
        
        Set 3 attributes of ``data_viwer``:
        ``measurement``, ``data_index``, ``datapoint``.\n
        ``data_index`` - index of datapoint to be displayed.
        """

        self.data_viwer.measurement = measurement
        self.data_viwer.data_index = data_index
        dp_name = PaData._build_name(data_index + 1)
        self.data_viwer.datapoint = measurement.data[dp_name]

        if measurement.attrs.measurement_dims == 1:
            self.set_curve_view()
        elif measurement.attrs.measurement_dims == 0:
            self.set_point_view()
        else:
            logger.error(f'Data dimension is not supported.')
            
        # Update description
        self.set_data_description_view(self.data) # type: ignore

    def set_data_content_view(self, data: PaData) -> None:
        """"Set tv_content."""

        content = self.data_viwer.tv_content

        treemodel = QStandardItemModel()
        rootnode = treemodel.invisibleRootItem()

        for msmnt_title, msmnt in data.measurements.items():
            name = QStandardItem(msmnt_title)
            rootnode.appendRow(name)
        content.setModel(treemodel)

    def set_point_view(self) -> None:
        """Set central part of data_viewer for point view."""

        self.data_viwer.sw_view.setCurrentWidget(self.data_viwer.p_0d)
        # Set combobox
        view = self.data_viwer.p_0d
        self.set_cb_detail_view(view)
        # Set plot
        detail_data = data.point_data_plot(
            msmnt = self.data_viwer.measurement,
            index = self.data_viwer.data_index,
            dtype = self.data_viwer.dtype_point
        )
        self.upd_plot(view.plot_detail, *detail_data)

    def set_curve_view(self) -> None:
        """Set central part of data_viewer for curve view."""

        self.data_viwer.sw_view.setCurrentWidget(self.data_viwer.p_1d)
        view = self.data_viwer.p_1d
        ## Set comboboxes ##
        # Detail combobox
        self.set_cb_detail_view(view)
        # Curve combobox
        view.cb_curve_select.blockSignals(True)
        view.cb_curve_select.clear()
        view.cb_curve_select.blockSignals(False)
        view.cb_curve_select.addItems(
            [key for key in ParamSignals.keys()]
        )
        view.cb_curve_select.currentTextChanged.connect(
            lambda new_val: self.upd_plot(
                view.plot_curve,
                *data.param_data_plot(
                    self.data_viwer.measurement,
                    ParamSignals[new_val]
                ),
                marker = self.data_viwer.data_index,
                enable_pick = True 
            )
        )
        # Set main plot
        main_data = data.param_data_plot(
            msmnt = self.data_viwer.measurement,
            dtype = ParamSignals[view.cb_curve_select.currentText()]
        )
        self.upd_plot(
            view.plot_curve,
            *main_data,
            marker = self.data_viwer.data_index,
            enable_pick = True
        )
        # Set additional plot
        detail_data = data.point_data_plot(
            msmnt = self.data_viwer.measurement,
            index = self.data_viwer.data_index,
            dtype = self.data_viwer.dtype_point
        )
        self.upd_plot(view.plot_detail, *detail_data)
        # Picker event
        view.plot_curve.fig.canvas.mpl_connect(
            'pick_event',
            lambda event: self.pick_event(
                view.plot_curve,
                event
            )
        )

    def set_cb_detail_view(self, parent: QWidget) -> None:
        """
        Set combobox with signals for detail view.
        
        Implies that combobox is ``cb_detail_select``.
        ``parent`` should have attribute with this name.
        """
        parent.cb_detail_select.blockSignals(True) # type: ignore
        parent.cb_detail_select.clear() # type: ignore
        parent.cb_detail_select.blockSignals(False) # type: ignore
        parent.cb_detail_select.addItems( # type: ignore
            [key for key in DetailedSignals.keys()]
        )
        self.data_viwer.dtype_point = DetailedSignals[
            parent.cb_detail_select.currentText() # type: ignore
        ]
        #Disconnect old slots
        try:
            parent.cb_detail_select.currentTextChanged.disconnect() # type: ignore
        except:
            logger.debug('No slots were connected to cb_detail_select.')
        parent.cb_detail_select.currentTextChanged.connect(self.upd_point_info) # type: ignore

    def set_data_description_view(self, data: PaData) -> None:
        """Set data description tv_info."""

        tv = self.data_viwer.tv_info
        if self.data_viwer.measurement is None:
            logger.error('Describption cannot be build. Measurement is not set.')
            return
        measurement = self.data_viwer.measurement

        # file attributes
        # clear if existed
        if getattr(tv, 'fmd', None) is not None:
            index = tv.indexOfTopLevelItem(tv.fmd)
            tv.takeTopLevelItem(index)
        tv.fmd = QTreeWidgetItem(tv, ['File attributes'])
        tv.fmd.setExpanded(True)
        for field in fields(data.attrs):
            QTreeWidgetItem(
                tv.fmd,
                [field.name, str(getattr(data.attrs, field.name))]
            )

        # measurement attributes
        # clear if existed
        if getattr(tv, 'mmd', None) is not None:
            index = tv.indexOfTopLevelItem(tv.mmd)
            tv.takeTopLevelItem(index)
        tv.mmd = QTreeWidgetItem(tv, ['Measurement attributes'])
        tv.mmd.setExpanded(True)
        msmnt_title = ''
        for key, val in data.measurements.items():
            if val == measurement:
                msmnt_title = key
                break
        QTreeWidgetItem(tv.mmd, ['Title', msmnt_title])
        for field in fields(measurement.attrs):
            QTreeWidgetItem(
                tv.mmd,
                [field.name, str(getattr(
                    measurement.attrs,
                    field.name
                    ))]
            )
        
        # point attributes
        self.upd_point_info()

    def upd_point_info(self, new_text: str|None = None) -> None:
        """Update plot and text information about current datapoint."""

        if new_text is not None:
            self.data_viwer.dtype_point = DetailedSignals[new_text]

        # update plot
        active_dview = self.data_viwer.sw_view.currentWidget()
        self.upd_plot(
            active_dview.plot_detail, # type: ignore
            *data.point_data_plot(
                msmnt = self.data_viwer.measurement,
                index = self.data_viwer.data_index,
                dtype = self.data_viwer.dtype_point
            )
        )
        # Update description
        tv = self.data_viwer.tv_info
        exp_state = False
        # clear existing description 
        # but save expanded state
        if tv.pmd is not None:
            index = tv.indexOfTopLevelItem(tv.pmd)
            tv.takeTopLevelItem(index)
        tv.pmd = QTreeWidgetItem(tv, ['Point attributes'])
        tv.pmd.setExpanded(True)
        dp_title = PaData._build_name(self.data_viwer.data_index + 1)
        dp = self.data_viwer.measurement.data[dp_title] #type: ignore
        QTreeWidgetItem(tv.pmd, ['Title', dp_title])
        for field in fields(dp.attrs):
            QTreeWidgetItem(
                tv.pmd,
                [field.name, str(getattr(
                    dp.attrs,
                    field.name
                    ))]
            )
        dtype = getattr(dp, self.data_viwer.dtype_point)
        for field in fields(dtype):
            if field.name in ['data', 'data_raw']:
                continue
            QTreeWidgetItem(
                tv.pmd,
                [field.name, str(getattr(dtype, field.name))]
            )

    def pick_event(
            self,
            widget:MplCanvas,
            event: PickEvent,
            parent: CurveView|None = None,
            measurement: Measurement|None = None):
        """Callback method for processing data picking on plot."""

        # Update datamarker on plot
        if widget._marker_ref is None:
            logger.warning('Marker is None')
            return
        widget._marker_ref.set_data(
            widget.xdata[event.ind[0]], # type: ignore
            widget.ydata[event.ind[0]] # type: ignore
        )
        widget.draw()
        # Update index of currently selected data
        self.data_viwer.data_index = event.ind[0] # type: ignore
        dp_name = PaData._build_name(event.ind[0] + 1) # type: ignore
        self.data_viwer.datapoint = self.data_viwer.measurement.data[dp_name]
        # update text info
        self.upd_point_info()

    def get_filename(self) -> str:
        """Launch a dialog to open a file."""

        path = os.path.dirname(__file__)
        initial_dir = os.path.join(path, 'measuring results')
        filename = QFileDialog.getOpenFileName(
            self,
            caption='Choose a file',
            dir=initial_dir,
            filter='Hierarchical Data Format (*.hdf5)'
        )[0]
        return filename

    def activate_data_viwer(self, activated: bool) -> None:
        """Toogle data viewer."""

        if activated:
            self.sw_central.setCurrentWidget(self.data_viwer)
            self.action_measure_PA.setChecked(False)
        else:
            self.upd_mode(self.cb_mode_select.currentText())

    def activate_measure_widget(self, activated: bool) -> None:
        """Toggle measre widget."""

        if activated:
            self.upd_mode(self.cb_mode_select.currentText())
        else:
            self.action_Data.setChecked(True)

    def run_curve(self, activated: bool) -> None:
        """
        Start 1D measurement.
        
        Launch energy adjustment.\n
        Set ``p_curve.step``, ``p_curve.spectral_points`` attributes.
        Also set progress bar and realted widgets.
        """

        self.p_curve.w_measure.setVisible(activated)

        page = self.p_curve
        #launch energy adjustment
        if activated and pa_logic.osc_open():
            if self.worker_curve_adj is None:
                self.worker_curve_adj = Worker(pa_logic.track_power)
                self.worker_curve_adj.signals.progess.connect(
                    lambda measure: self.upd_measure_monitor(
                        page,
                        measure
                    )
                )
                self.pool.start(self.worker_curve_adj)

            # calculate amount of paramter values
            param_start = page.sb_from.quantity
            param_end = page.sb_to.quantity
            step = page.sb_step.quantity
            delta = abs(param_end-param_start)
            if delta%step:
                max_steps = int(delta/step) + 2
            else:
                max_steps = int(delta/step) + 1
            page.max_steps = max_steps

            # set paramter name attribute
            page.parameter = [page.cb_mode.currentText()]

            # create array with all parameter values
            # and set it as attribute along with step
            spectral_points = []
            if param_start < param_end:
                page.sb_cur_param.setMinimum(param_start.m)
                page.sb_cur_param.setMaximum(param_end.m)
                page.step = step
                for i in range(max_steps-1):
                    spectral_points.append(param_start+i*step)
                spectral_points.append(param_end)
            else:
                page.sb_cur_param.setMinimum(param_end.m)
                page.sb_cur_param.setMaximum(param_start.m)
                page.step = -step
                for i in range(max_steps-1):
                    spectral_points.append(param_end+i*step)
                spectral_points.append(param_start)
            page.param_points = Q_.from_list(spectral_points)
            page.current_point = 0
        elif not pa_logic.osc_open():
            logger.warning('Oscilloscope is not initialized.')
        else:
            self.stop_worker(self.worker_curve_adj)

    def measure_curve(self) -> None:
        """Measure a point for 1D PA measurement."""

        #Stop adjustment measurements
        if self.worker_curve_adj is not None:
            adj_flag = self.worker_curve_adj.kwargs['flags']['pause']
            self.worker_curve_adj.kwargs['flags']['pause'] = True
        if self.worker_pm is not None:
            self.worker_pm.kwargs['flags']['pause'] = True

        wl = self.p_curve.sb_cur_param.quantity
        worker = Worker(pa_logic._measure_point, wl)
        worker.signals.result.connect(
            lambda data: self.verify_measurement(
                self.p_curve,
                data
            )
        )
        self.pool.start(worker)

    def verify_measurement(
            self,
            parent: CurveMeasureWidget,
            data: MeasuredPoint
        ) -> None:
        """
        Verify a measurement.
        
        ``parent`` is widget, which stores current point index.
        """

        plot_pa = self.pa_verifier.plot_pa
        plot_pm = self.pa_verifier.plot_pm

        # Power meter plot
        ydata=data.pm_signal
        xdata=Q_(
            np.arange(len(data.pm_signal))*data.dt_pm.m, # type: ignore
            data.dt_pm.u
        )
        self.upd_plot(
            plot_pm,
            ydata=ydata,
            xdata=xdata
        )

        # PA signal plot
        start = data.start_time
        stop = data.stop_time.to(start.u)
        self.upd_plot(
            plot_pa,
            ydata=data.pa_signal,
            xdata=Q_(
                np.linspace(
                    start.m,
                    stop.m,
                    len(data.pa_signal), # type: ignore
                    endpoint=True), # type: ignore
                start.u
            ) # type: ignore
        )

        # pa_verifier.exec return 1 if data is ok and 0 otherwise
        if self.pa_verifier.exec():
            # create new measurement if it is the first point
            if parent.current_point == 0:
                if self.data is None:
                    self.data = PaData()
                dims = 0
                if isinstance(parent, CurveMeasureWidget):
                    dims = 1
                msmnt_title = self.data.add_measurement(
                    dims = dims,
                    parameters = parent.parameter
                )
            else:
                if self.data is None:
                    logger.error('Trying to add not starting point '
                                 + 'while data is None.')
                    return
                msmnt_title = PaData._build_name(
                    self.data.attrs.measurements_count,
                    'measurement'
                )
            
            # add datapoint to the measurement
            msmnt = self.data.measurements[msmnt_title]
            cur_param = []
            if isinstance(parent,CurveMeasureWidget):
                cur_param.append(parent.sb_cur_param.quantity)
            self.data.add_point(
                measurement = msmnt,
                data = data,
                param_val = cur_param.copy()
            )
            # update parent attributes
            parent.current_point += 1
            # Update data_viewer
            self.data_changed.emit()

        # Resume energy adjustment
        if self.worker_curve_adj is not None:
            self.worker_curve_adj.kwargs['flags']['pause'] = False

    def upd_measure_monitor(
            self,
            page: CurveMeasureWidget,
            measurement: dict) -> None:
        """
        Update ``monitor`` widget.

        Expect that ``measurement`` match return of 
        ``pa_logic.track_power``, i.e. it should contain:\n 
        ``data``: list[PlainQuantity] - a list with data\n
        ``signal``: PlainQuantity - measured PM signal(full data)\n
        ``sbx``: int - Signal Begining X
        ``sby``: PlainQuantity - Signal Begining Y
        ``sex``: int - Signal End X
        ``sey``: PlainQuantity - Signal End Y
        ``energy``: PlainQuantity - just measured energy value\n
        ``aver``: PlainQuantity - average energy\n
        ``std``: PlainQuantity - standart deviation of the measured data\n
        """

        pm_energy = measurement['energy']
        sample_energy = pa_logic.glan_calc(pm_energy)
        if sample_energy is None:
            logger.warning('Sample energy cannot be calculated')
            sample_energy = Q_(-1, 'uJ')
        page.sb_pm_en.quantity = pm_energy
        page.sb_sample_en.quantity = sample_energy
        self.upd_plot(
            page.pm_monitor.plot_left,
            ydata=measurement['signal']
        )
        self.upd_plot(
            page.pm_monitor.plot_right,
            ydata=measurement['data']
        )

    def upd_sp(self, caller: QuantSpinBox) -> None:
        """Update Set Point data for currently active page."""

        # only for curve widget?

        page = self.p_curve
        if caller == page.sb_sample_sp:
            new_sp = pa_logic.glan_calc_reverse(caller.quantity)
            if new_sp is None:
                logger.error('New set point cannot be calculated')
                return
            self.set_spinbox_silent(page.sb_pm_sp, new_sp)
                
        elif caller == page.sb_pm_sp:
            new_sp = pa_logic.glan_calc(caller.quantity)
            if new_sp is None:
                logger.error('New set point cannot be calculated')
                return
            self.set_spinbox_silent(page.sb_sample_sp, new_sp)

        page.pm_monitor.plot_right.sp = page.sb_pm_sp.quantity
                
    def set_spinbox_silent(
            self,
            sb: QuantSpinBox,
            value: PlainQuantity
        ) -> None:
        """Set ``value`` to ``sb`` without firing events."""

        sb.blockSignals(True)
        sb.quantity = value.to(sb.quantity.u)
        sb.blockSignals(False)

    def upd_mode(self, value: str) -> None:
        """Change sw_main for current measuring mode."""

        if value == 'Single point':
            self.sw_central.setCurrentWidget(self.p_point)
        elif value == 'Curve':
            self.sw_central.setCurrentWidget(self.p_curve)
        elif value == 'Map':
            self.sw_central.setCurrentWidget(self.p_map)
        else:
            logger.warning(f'Unsupported mode selected: {value}')

        self.action_measure_PA.setChecked(True)
        self.action_Data.setChecked(False)

    def closeEvent(self, event) -> None:
        """Exis routines."""

        if self.confirm_action(
            title = 'Confirm Close',
            message = 'Do you really want to exit?'
        ):
            logger.info('Stopping application')
            for stage in dc.hardware.stages:
                stage.close()
        else:
            event.ignore()

    def init_hardware(self) -> None:
        """Hardware initiation."""
        worker = Worker(pa_logic.init_hardware)
        self.pool.start(worker)

    def track_power(self) -> None:
        """Start energy measurements."""

        if self.worker_pm is None:
            self.worker_pm = Worker(pa_logic.track_power)
            self.worker_pm.signals.progess.connect(self.update_pm)
            self.pool.start(self.worker_pm)

    def stop_worker(self, worker: Worker|None) -> None:
        
        if worker is not None:
            worker.kwargs['flags']['is_running'] = False
            for key, value in self.__dict__.items():
                if value is worker:
                    self.__dict__[key] = None

    def pause_worker(
            self,
            is_paused: bool,
            worker: Worker|None
        ) -> None:
        """Pause or resume worker."""

        if worker is not None:
            worker.kwargs['flags']['pause'] = is_paused

    def update_pm(
            self,
            measurement: dict
        ) -> None:
        """
        Update power meter widget.

        Expect that ``measurement`` match return of 
        ``pa_logic.track_power``, i.e. it should contain:\n 
        ``data``: list[PlainQuantity] - a list with data\n
        ``signal``: PlainQuantity - measured PM signal(full data)\n
        ``sbx``: int - Signal Begining X
        ``sby``: PlainQuantity - Signal Begining Y
        ``sex``: int - Signal End X
        ``sey``: PlainQuantity - Signal End Y
        ``energy``: PlainQuantity - just measured energy value\n
        ``aver``: PlainQuantity - average energy\n
        ``std``: PlainQuantity - standart deviation of the measured data\n
        """

        self.upd_plot(self.d_pm.plot_left, ydata=measurement['signal'])
        self.upd_plot(self.d_pm.plot_right, ydata=measurement['data'])
        self.d_pm.le_cur_en.setText(self.form_quant(measurement['energy']))
        self.d_pm.le_aver_en.setText(self.form_quant(measurement['aver']))
        self.d_pm.le_std_en.setText(self.form_quant(measurement['std']))
    
    def form_quant(self, quant: PlainQuantity) -> str:
        """Format str representation of a quantity."""

        return f'{quant:~.2gP}'

    def upd_plot(
            self,
            widget: MplCanvas,
            ydata: Iterable,
            xdata: Iterable|None = None,
            ylabel: str|None = None,
            xlabel: str|None = None,
            marker: int|None = None,
            enable_pick: bool = False
        ) -> None:
        """
        Update a plot.
        
        ``name`` - name of the plot widget,\n
        ``xdata`` - Iterable containing data for X axes. Should not
         be shorter than ``ydata``. If None, then enumeration of 
         ``ydata`` will be used.\n
         ``ydata`` - Iterable containing data for Y axes.\n
         ``xlabel`` - Optional label for x data.\n
         ``ylabel`` - Optional label for y data.\n
         ``marker`` - Index of marker for data. If this option is
         omitted, then marker is removed.\n
         ``picker`` - enabler picker for main data.
        """

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
        # if marker is not None:
        #     widget.marker = marker

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
        if marker is not None:
            if widget._marker_ref is None:
                marker_style = {
                    'marker': 'o',
                    'alpha': 0.4,
                    'ms': 12,
                    'color': 'yellow'
                }
                widget._marker_ref, = widget.axes.plot(
                    widget.xdata[marker], # type: ignore
                    widget.ydata[marker], # type: ignore
                    **marker_style
                )
            else:
                widget._marker_ref.set_data(
                    widget.xdata[marker], # type: ignore
                    widget.ydata[marker] # type: ignore
                )
        else:
            widget._marker_ref = None
        widget.axes.set_xlabel(widget.xlabel) # type: ignore
        widget.axes.set_ylabel(widget.ylabel) # type: ignore
        widget.axes.relim()
        widget.axes.autoscale_view(True,True,True)
        widget.draw()

    def get_layout(self, widget: QWidget) -> QLayout|None:
        """Return parent layout for a widget."""

        parent = widget.parent()
        child_layouts = self._unpack_layout(parent.children())
        for layout in child_layouts:
            if layout.indexOf(widget) > -1:
                return layout

    def _unpack_layout(self, childs: list[QObject]) -> list[QLayout]:
        """Recurrent function to find all layouts in a list of widgets."""

        result = []
        for item in childs:
            if isinstance(item, QLayout):
                result.append(item)
                result.extend(self._unpack_layout(item.children()))
        return result

    def upd_le(self, widget: QWidget, val: str|PlainQuantity) -> None:
        """
        Update text of a line edit.
        
        ``widget`` - line edit widget,\n
        ``val`` - value to be set.
        """

        if not isinstance(widget, QLineEdit):
            logger.warning(
                'Trying to set text of a LINE EDIT = '
                + f'{widget.objectName()}, which actually is '
                + f'{type(widget)}.')
            return
        widget.setText(str(val))

    def confirm_action(self,
            title: str = 'Confirm',
            message: str = 'Are you sure?'
        ) -> bool:
        """Dialog to confirm an action."""

        answer = QMessageBox.question(
            self,
            title,
            message
        )
        if answer == QMessageBox.StandardButton.Yes:
            result = True
        else:
            result = False
        logger.debug(f'{message} = {result}')
        return result

class LoggerThread(QThread):
    """Thread for logging."""

    log = Signal(str)
    data = []

    @Slot()
    def run(self) -> None:
        self.waiting_for_data = True
        while True:
            while self.waiting_for_data:
                time.sleep(0.01)
            while len(self.data):
                self.log.emit(self.data.pop(0))
            self.waiting_for_data = True

    def get_data(self, log_entry: str) -> None:
        """Store a log entry for displaying."""
        self.data.append(log_entry)
        self.waiting_for_data = False

class QLogHandler(logging.Handler):
    """Log handler for GUI."""

    def __init__(self):
        super().__init__()
        self.thread = LoggerThread()
        self.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                '%I:%M:%S'
            )
        )
        self.setLevel(logging.INFO)

    def emit(self, record):
        msg = self.format(record)
        self.thread.get_data(msg)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    logger, q_log_handler = init_logs()
    win = Window()
    
    logger.info('Starting application')

    ureg = pint.UnitRegistry(auto_reduce_dimensions=True) # type: ignore
    ureg.default_format = '~P'
    logger.debug('Pint activated (measured values with units)')
    ureg.setup_matplotlib(True)
    logger.debug('MatplotLib handlers for pint activated')
    pa_logic.load_config()
    logger.debug('Initializing PaData class for storing data')
    data = PaData()
    win.showMaximized()
    sys.exit(app.exec())