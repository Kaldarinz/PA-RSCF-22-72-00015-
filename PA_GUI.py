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

from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import pint
import yaml
from pylablib.devices.Thorlabs import KinesisMotor
from PySide6.QtCore import (
    QObject,
    QThread,
    QRunnable,
    QThreadPool,
    Slot,
    Signal
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow
)

from gui.PA_main_window_ui import Ui_MainWindow
from modules import ureg, Q_
import modules.validators as vd
from modules.pa_data import PaData
import modules.pa_logic as pa_logic
from modules.exceptions import StageError, HardwareError
import modules.data_classes as dc
from modules.utils import confirm_action

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
    logging.getLogger('root').setLevel(logging.INFO)
    logging.getLogger('modules').addHandler(q_log_handler)
    logging.getLogger('modules').setLevel(logging.INFO)

    return logging.getLogger()

def init_hardware() -> None:
    """CLI for hardware init"""

    try:
        pa_logic.init_hardware()
    except HardwareError:
        logger.error('Hardware initialization failed')

class Window(QMainWindow, Ui_MainWindow):
    """Application MainWindow."""

    def __init__(self, parent:QMainWindow = None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        
        #Long-running threads
        self.q_logger = QLogHandler()
        self.q_logger.thread.start()

        #Threadpool
        self.pool = QThreadPool()

        self.connectSignalsSlots()

    def connectSignalsSlots(self):
        """Connect signals and slots."""

        self.action_Init.triggered.connect(self.init_hardware)
        self.q_logger.thread.log.connect(self.LogTextEdit.appendPlainText)

    def init_hardware(self) -> None:
        """Hardware initiation."""
        worker = Worker(pa_logic.init_hardware)
        self.pool.start(worker)
        

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
                self.log.emit(self.data.pop())
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

    def emit(self, record):
        msg = self.format(record)
        self.thread.get_data(msg)

class WorkerSignals(QObject):
    """
    Signals available from a running worker thred.
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)

class Worker(QRunnable):
    """
    Generic working thread for backend functions.
    """

    def __init__(self, func, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """Actually run a func."""

        try:
            self.result = self.func(*self.args, **self.kwargs)
        except:
            exctype, value = sys.exc_info()[:2]
            logger.warning(f'An error occured: {exctype}, {value}')
            self.signals.error.emit(
                (exctype, value, traceback.format_exc())
            )
        else:
            self.signals.result.emit(self.result)
        finally:
            self.signals.finished.emit()

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
    win.show()
    sys.exit(app.exec())