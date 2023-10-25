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

import pint
import yaml
from pylablib.devices.Thorlabs import KinesisMotor
from pint.facets.plain.quantity import PlainQuantity
from PySide6.QtCore import (
    Qt,
    QObject,
    QThread,
    QRunnable,
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
    QSpinBox,
    QVBoxLayout,
    QDialog,
    QDockWidget
)
from PySide6.QtGui import (
    QColor,
    QPalette
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
    DataPoint
)
from modules.gui.widgets import (
    MplCanvas,
    QuantSpinBox,
)
from modules.gui.designer import (
    PA_main_window_ui,
    verify_measure_ui,
    data_viewer_ui,
    log_dock_widget_ui,
    pm_dock_widget_ui
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

class Window(QMainWindow,PA_main_window_ui.Ui_MainWindow,):
    """Application MainWindow."""

    worker_pm: Worker|None = None
    "Thread for measuring energy."
    worker_curve_adj: Worker|None = None
    "Thread for curve adjustments."

    def __init__(
            self,
            parent:QMainWindow = None,
        ) -> None:
        super().__init__(parent)
        self.setupUi(self)
        
        #Long-running threads
        self.q_logger = q_log_handler

        #Threadpool
        self.pool = QThreadPool()

        self.set_custom_widgets()
        
        self.w_curve_measure.hide()

        #Set mode selection combo box
        self.set_mode_selector()

        #Set layout for central widget
        #By some reason it is not possible to set in designer
        self.clayout = QVBoxLayout()
        self.clayout.addWidget(self.sw_central)
        self.centralwidget.setLayout(self.clayout)

        #Set initial values
        self.plot_curve_adjust.sp = self.sb_curve_sample_sp.quantity
        self.upd_sp(self.sb_curve_sample_sp)
        self.upd_mode(self.cb_mode_select.currentText())

        #Connect custom signals and slots
        self.connectSignalsSlots()

    def set_custom_widgets(self):
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
        self.d_log.hide()

        #Power Meter
        #Parent in constructor makes it undocked by default
        self.d_pm = PowerMeterWidget(self)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.d_pm
        )
        self.d_pm.hide()

    def connectSignalsSlots(self):
        """Connect signals and slots."""

        #logs
        self.q_logger.thread.log.connect(self.d_log.te_log.appendPlainText)

        #Init
        self.action_Init.triggered.connect(self.init_hardware)

        self.btn_curve_run.clicked.connect(self.run_curve)
        self.btn_curve_measure.clicked.connect(self.measure_curve)

        #interface
        self.cb_mode_select.currentTextChanged.connect(self.upd_mode)
        self.btn_curve_run.toggled.connect(self.w_curve_measure.setVisible)
        
        self.action_measure_PA.toggled.connect(self.activate_measure_widget)
        self.sb_curve_sample_sp.valueChanged.connect(
            lambda x: self.upd_sp(self.sb_curve_sample_sp)
        )
        self.sb_curve_pm_sp.valueChanged.connect(
            lambda x: self.upd_sp(self.sb_curve_pm_sp)
        )

        #Data
        self.action_Data.toggled.connect(self.activate_data_viwer)

        #Logs
        self.action_Logs.toggled.connect(self.d_log.setVisible)
        self.d_log.visibilityChanged.connect(self.action_Logs.setChecked)

        #PowerMeter
        self.d_pm.btn_start.clicked.connect(self.track_power)
        self.d_pm.btn_stop.clicked.connect(self.stop_track_power)
        self.action_Power_Meter.toggled.connect(self.d_pm.setVisible)
        self.d_pm.visibilityChanged.connect(self.action_Power_Meter.setChecked)

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

        wl = self.sb_curve_cur_param.quantity
        worker = Worker(pa_logic._measure_point, wl)
        worker.signals.result.connect(self.verify_measurement)
        self.pool.start(worker)

    def verify_measurement(self, data: DataPoint) -> DataPoint|None:
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

    def run_curve(self) -> None:
        """
        Start 1D measurement.
        
        Launch energy adjustment.\n
        Set ``step`` and ``spectral_points`` attributes.
        Also set progress bar and realted widgets.
        """

        #launch energy adjustment
        if self.worker_curve_adj is None:
            self.worker_curve_adj = Worker(pa_logic.track_power)
            self.worker_curve_adj.signals.progess.connect(self.upd_curve_adj)
            self.pool.start(self.worker_curve_adj)

        #set progress bar and related widgets
        param_start = self.sb_curve_from.quantity
        param_end = self.sb_curve_to.quantity
        step = self.sb_curve_step.quantity
        delta = abs(param_end-param_start)
        if delta%step:
            spectral_points = int(delta/step) + 2
        else:
            spectral_points = int(delta/step) + 1
        self.pb_curve.setValue(0)
        self.lbl_curve_progr.setText(f'0/{spectral_points}')
        self.spectral_points = spectral_points

        if param_start < param_end:
            self.sb_curve_cur_param.setMinimum(param_start.m)
            self.sb_curve_cur_param.setMaximum(param_end.m)
            self.step = step
        else:
            self.sb_curve_cur_param.setMinimum(param_end.m)
            self.sb_curve_cur_param.setMaximum(param_start.m)
            self.step = -step
        self.sb_curve_cur_param.quantity = self.sb_curve_from.quantity

    def upd_curve_adj(self, measurement: dict) -> None:
        """
        Update information on measure widgets and adj plots.

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
        self.sb_curve_pm_energy.quantity = pm_energy
        self.sb_curve_sample_energy.quantity = sample_energy
        self.upd_plot(self.plot_curve_signal, ydata=measurement['signal'])
        self.upd_plot(self.plot_curve_adjust, ydata=measurement['data'])

    def upd_sp(self, caller: QuantSpinBox) -> None:
        """Update Set Point data for currently active page"""

        #Curve
        if self.sw_central.currentIndex() == 1:
            if caller == self.sb_curve_sample_sp:
                new_sp = pa_logic.glan_calc_reverse(caller.quantity).to('uJ')
                if new_sp.m > self.sb_curve_pm_sp.maximum():
                    new_sp.m = self.sb_curve_pm_sp.maximum()
                #prevent infinite mutual update
                self.sb_curve_pm_sp.blockSignals(True)
                self.sb_curve_pm_sp.quantity = new_sp
                self.sb_curve_pm_sp.blockSignals(False)
                   
            elif caller == self.sb_curve_pm_sp:
                new_sp = pa_logic.glan_calc(caller.quantity).to('uJ')
                if new_sp.m > self.sb_curve_sample_sp.maximum():
                    new_sp.m = self.sb_curve_sample_sp.maximum()
                #prevent infinite mutual update
                self.sb_curve_sample_sp.blockSignals(True)
                self.sb_curve_sample_sp.quantity = new_sp
                self.sb_curve_sample_sp.blockSignals(False)
                    
            self.plot_curve_adjust.sp = caller.quantity
                
    def upd_mode(self, value: str) -> None:
        """Change sw_main for current measuring mode."""

        if value == 'Single point':
            self.sw_central.setCurrentIndex(0)
        elif value == 'Curve':
            self.sw_central.setCurrentIndex(1)
        elif value == 'Map':
            self.sw_central.setCurrentIndex(2)
        else:
            logger.warning(f'Unsupported mode selected: {value}')

        self.action_measure_PA.setChecked(True)
        self.action_Data.setChecked(False)
    
    def set_mode_selector(self):
        """Set mode selection view."""

        cb = QComboBox()
        cb.addItems(
            ('Single point', 'Curve', 'Map')
        )
        cb.setEditable(False)
        cb.setCurrentIndex(self.sw_central.currentIndex())
        self.tb_mode.addWidget(cb)
        self.cb_mode_select = cb

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

    def stop_track_power(self) -> None:
        
        worker = self.worker_pm
        if worker is not None:
            worker.kwargs['flags']['is_running'] = False
            self.worker_pm = None

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
            ydata: Iterable|None,
            xdata: Iterable|None = None
        ) -> None:
        """
        Update a plot.
        
        ``name`` - name of the plot widget,\n
        ``xdata`` - Iterable containing data for X axes. Should not
         be shorter than ``ydata``. If None, then enumeration of 
         ``ydata`` will be used.\n
         ``ydata`` - Iterable containing data for Y axes.
        """

        if xdata is None:
            widget.xdata = np.array(range(len(ydata)))
        else:
            widget.xdata = xdata
        widget.ydata = ydata
        if widget._plot_ref is None:
            widget._plot_ref = widget.axes.plot(
                widget.xdata, widget.ydata, 'r'
            )[0]
        else:
            widget._plot_ref.set_xdata(widget.xdata)
            widget._plot_ref.set_ydata(widget.ydata)
        if widget.sp is not None:
            spdata = [widget.sp]*len(widget.xdata)
            if widget._sp_ref is None:
                widget._sp_ref = widget.axes.plot(
                    widget.xdata, spdata, 'b'
                )[0]
            else:
                widget._sp_ref.set_xdata(widget.xdata)
                widget._sp_ref.set_ydata(spdata)
        widget.axes.set_xlabel(widget.xlabel)
        widget.axes.set_ylabel(widget.ylabel)
        widget.axes.relim()
        widget.axes.autoscale_view(True,True,True)
        widget.draw()

    def get_layout(self, widget: QWidget) -> QLayout:
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

class PAVerifierDialog(QDialog, verify_measure_ui.Ui_Dialog):
    """Dialog window for verification of a PA measurement."""

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

class DataViewer(QWidget, data_viewer_ui.Ui_Form):
    """Data viewer widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

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

class LoggerWidget(QDockWidget, log_dock_widget_ui.Ui_d_log):
    """Logger dock widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

class PowerMeterWidget(QDockWidget, pm_dock_widget_ui.Ui_d_pm):
    """Power meter dock widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

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