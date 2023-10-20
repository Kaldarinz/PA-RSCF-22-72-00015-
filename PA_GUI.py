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
from typing import Iterable, cast, Callable

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
    QLayout
)
from PySide6.QtGui import (
    QColor,
    QPalette
)
import numpy as np

from modules.PA_main_window_ui import Ui_MainWindow
from modules import ureg, Q_
import modules.validators as vd
from modules.pa_data import PaData
import modules.pa_logic as pa_logic
from modules.exceptions import StageError, HardwareError
import modules.data_classes as dc
from modules.data_classes import (
    Worker
)
from modules.utils import confirm_action
from modules.widgets import MplCanvas

def init_logs(q_log_handler: QLogHandler) -> logging.Logger:
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

    #adding textbox handler. I cannot handle it with file config.
    logging.config.dictConfig(log_config)
    logging.getLogger('root').addHandler(q_log_handler)
    logging.getLogger('modules').addHandler(q_log_handler)

    return logging.getLogger('pa_cli')

def init_hardware() -> None:
    """CLI for hardware init"""

    try:
        pa_logic.init_hardware()
    except HardwareError:
        logger.error('Hardware initialization failed')

class Window(QMainWindow, Ui_MainWindow):
    """Application MainWindow."""

    worker_pm: Worker|None = None
    "Thread for measuring energy."

    def __init__(self, parent:QMainWindow = None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        
        #Long-running threads
        self.q_logger = QLogHandler()
        self.q_logger.thread.start()

        #Threadpool
        self.pool = QThreadPool()

        #Connect custom signals and slots
        self.connectSignalsSlots()

        #Set visibility of some widgets
        self.dock_pm.hide()
        self.dock_log.hide()

    def connectSignalsSlots(self):
        """Connect signals and slots."""

        self.action_Init.triggered.connect(self.init_hardware)
        self.q_logger.thread.log.connect(self.te_log.appendPlainText)
        self.btn_pm_stop.clicked.connect(self.stop_track_power)
        self.btn_pm_start.clicked.connect(self.track_power)

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

        self.upd_plot('plot_pm_left', ydata=measurement['signal'])
        self.upd_plot('plot_pm_right', ydata=measurement['data'])
        self.le_cur_en.setText(self.form_quant(measurement['energy']))
        self.le_aver_en.setText(self.form_quant(measurement['aver']))
        self.le_std_en.setText(self.form_quant(measurement['std']))
    
    def form_quant(self, quant: PlainQuantity) -> str:
        """Format str representation of a quantity."""

        return f'{quant:~.2gP}'

    def upd_plot(
            self,
            name: str,
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

        widget = getattr(self, name, None)
        if widget is None:
            logger.warning(f'Trying to update non-existing PLOT={name}')
        if not isinstance(widget, MplCanvas):
            logger.warning('Trying to update a PLOT, '
                           + f'which actually is {type(widget)}.')
            return
        widget = cast(MplCanvas, widget)
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
    win = Window()
    logger = init_logs(win.q_logger)
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