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
    Signal
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
    DetailedSignals
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

    data: PaData|None = None
    "PA data."

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
        self.action_Power_Meter.toggled.connect(self.d_pm.setVisible)
        self.d_pm.visibilityChanged.connect(self.action_Power_Meter.setChecked)

    def open_file(self) -> None:
        """Load data file."""

        logger.debug('Start loading data...')
        filename = self.get_filename()
        self.data = PaData()
        self.data.load(filename)
        logger.info(f'Data with {self.data.attrs.measurements_count} '
                    + 'PA measurements loaded!')
        self.show_data(self.data)
        
    def show_data(self, data: PaData|None = None) -> None:
        """Load data into DataView widget."""


        logger.debug('Starting plotting data...')
        if data is None:
            if self.data is None:
                logger.error('No data to show!')
                return
            data = self.data

        # choose a measurement to show data from
        msmnt_title, msmnt = next(iter(data.measurements.items()))

        # Update content representation
        self.set_data_content_view(data)

        # Data plotting
        # Set measurement and datapoint to be plotted as attributed 
        # to the data_view widget
        self.data_viwer.measurement = msmnt
        dp_name = PaData._build_name(self.data_viwer.p_1d.marker_ind + 1)
        self.data_viwer.datapoint = msmnt.data[dp_name]
        if msmnt.attrs.measurement_dims == 1:
            view = self.data_viwer.p_1d
            # Set comboboxes
            view.cb_detail_select.addItems(
                [val.name for val in DetailedSignals]
            )
            view.cb_detail_select.currentTextChanged.connect(
                lambda signal: self.upd_plot(
                    view.plot_detail,
                    *data.point_data_plot(
                        msmnt,
                        view.marker_ind,
                        signal
                    )
                )
            )
            view.cb_detail_select.currentTextChanged.connect(self.upd_point_info)
            # Set main plot
            main_data = data.param_data_plot(msmnt)
            self.upd_plot(
                view.plot_curve,
                *main_data,
                marker = view.marker_ind,
                enable_pick = True
            )
            # Set additional plot
            detail_data = data.point_data_plot(
                msmnt = msmnt,
                index = view.marker_ind,
                dtype = view.cb_detail_select.currentText()
            )
            self.upd_plot(view.plot_detail, *detail_data)
            view.plot_curve.fig.canvas.mpl_connect(
                'pick_event',
                lambda event: self.pick_event(
                    view.plot_curve,
                    event,
                    view,
                    measurement = msmnt
                )
            )
            
        # Update description
        self.set_data_description_view(data)
        logger.debug('Data plotting complete.')

    def set_data_content_view(self, data: PaData) -> None:
        """"Set tv_content."""

        content = self.data_viwer.tv_content

        treemodel = QStandardItemModel()
        rootnode = treemodel.invisibleRootItem()

        for msmnt_title, msmnt in data.measurements.items():
            name = QStandardItem(msmnt_title)
            rootnode.appendRow(name)
        content.setModel(treemodel)

    def set_data_description_view(self, data: PaData) -> None:
        """Set data description tv_info."""

        tv = self.data_viwer.tv_info
        if self.data_viwer.measurement is None:
            logger.error('Describption cannot be build. Measurement is not set.')
            return
        measurement = self.data_viwer.measurement
        dp = self.data_viwer.datapoint

        # file attributes
        tv.fmd = QTreeWidgetItem(tv, ['File attributes'])
        for field in fields(data.attrs):
            QTreeWidgetItem(
                tv.fmd,
                [field.name, str(getattr(data.attrs, field.name))]
            )
        tv.mmd = QTreeWidgetItem(tv, ['Measurement attributes'])
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
        self.upd_point_info()
        
    def upd_point_info(self) -> None:
        """Update text information about current datapoint."""

        tv = self.data_viwer.tv_info
        exp_state = False
        if tv.pmd is not None:
            exp_state = tv.pmd.isExpanded()
            index = tv.indexOfTopLevelItem(tv.pmd)
            tv.takeTopLevelItem(index)
        tv.pmd = QTreeWidgetItem(tv, ['Point attributes'])
        tv.pmd.setExpanded(exp_state)
        dp_title = PaData._build_name(self.data_viwer.p_1d.marker_ind + 1)
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
            #### тут надо енам, чтобы из комбобокса доставать нужно значение
        point_signal = self.data_viwer.p_1d.cb_detail_select.currentText()
        dtype_name = getattr(DetailedSignals, point_signal).value
        dtype = getattr(dp, dtype_name)
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
        # Update index of currently selected data on CurveView
        if parent is not None:
            parent.marker_ind = event.ind[0] # type: ignore
            if measurement is not None:
                dp_name = PaData._build_name(event.ind[0] + 1) # type: ignore
                self.data_viwer.datapoint = measurement.data[dp_name]
                detailed_data = PaData.point_data_plot(
                    measurement,
                    event.ind[0], # type: ignore
                    parent.cb_detail_select.currentText()
                )
                self.upd_plot(parent.plot_detail,*detailed_data)
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

    def measure_curve(self) -> None:
        """Measure a point for 1D PA measurement."""

        #Stop adjustment measurements
        if self.worker_curve_adj is not None:
            self.worker_curve_adj.kwargs['flags']['is_running'] = False

        wl = self.p_curve.sb_cur_param.quantity
        worker = Worker(pa_logic._measure_point, wl)
        worker.signals.result.connect(self.verify_measurement)
        self.pool.start(worker)

    def verify_measurement(self, data: MeasuredPoint) -> MeasuredPoint|None:
        """Verifies a measurement."""

        plot_pa = self.pa_verifier.plot_pa
        plot_pm = self.pa_verifier.plot_pm

        ydata=data.pm_signal,
        xdata=Q_(
            np.arange(len(data.pm_signal)*data.dt_pm.m),
            data.dt_pm.u
        )
        print(f'{len(xdata)=}')
        print(f'{len(ydata)=}')
        self.upd_plot(
            plot_pm,
            ydata=ydata,
            xdata=xdata
        )
        start = data.start_time
        stop = data.stop_time.to(start.u)
        step = data.dt.to(start.u)
        self.upd_plot(
            plot_pa,
            ydata=data.pa_signal,
            xdata=Q_(
                np.arange(start.m, stop.m, step.m),
                start.u
            )
        )
        print(self.pa_verifier.exec())

    def run_curve(self, activated: bool) -> None:
        """
        Start 1D measurement.
        
        Launch energy adjustment.\n
        Set ``p_curve.step`` and ``p_curve.spectral_points`` attributes.
        Also set progress bar and realted widgets.
        """

        self.p_curve.w_measure.setVisible(activated)

        page = self.p_curve
        #launch energy adjustment
        if activated:
            print(self.worker_curve_adj)
            if self.worker_curve_adj is None:
                self.worker_curve_adj = Worker(pa_logic.track_power)
                self.worker_curve_adj.signals.progess.connect(
                    lambda measure: self.upd_measure_monitor(
                        page,
                        measure
                    )
                )
                self.pool.start(self.worker_curve_adj)

            #set progress bar and related widgets
            param_start = page.sb_from.quantity
            param_end = page.sb_to.quantity
            step = page.sb_step.quantity
            delta = abs(param_end-param_start)
            if delta%step:
                spectral_points = int(delta/step) + 2
            else:
                spectral_points = int(delta/step) + 1
            page.pb.setValue(0)
            page.lbl_pb.setText(f'0/{spectral_points}')
            page.spectral_points = spectral_points

            if param_start < param_end:
                page.sb_cur_param.setMinimum(param_start.m)
                page.sb_cur_param.setMaximum(param_end.m)
                page.step = step
            else:
                page.sb_cur_param.setMinimum(param_end.m)
                page.sb_cur_param.setMaximum(param_start.m)
                page.step = -step
            page.sb_cur_param.quantity = page.sb_from.quantity
        else:
            self.stop_worker(self.worker_curve_adj)

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

        #Curve
        if self.sw_central.currentWidget() == self.p_curve:
            page = self.p_curve
            if caller == page.sb_sample_sp:
                new_sp = pa_logic.glan_calc_reverse(caller.quantity)
                self.set_spinbox_silent(page.sb_pm_sp, new_sp)
                   
            elif caller == page.sb_pm_sp:
                new_sp = pa_logic.glan_calc(caller.quantity)
                self.set_spinbox_silent(page.sb_sample_sp, new_sp)
                    
            page.plot_adjust.sp = caller.quantity
                
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